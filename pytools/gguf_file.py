"""GGUF (v2/v3) reader: typed metadata, tensor directory, lazy tensor byte access."""
import mmap
import struct
from dataclasses import dataclass, field

import numpy as np

MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32

# metadata value types
U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)
_SCALAR = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i", F32: "<f", BOOL: "<B",
           U64: "<Q", I64: "<q", F64: "<d"}
_NP = {U8: "<u1", I8: "<i1", U16: "<u2", I16: "<i2", U32: "<u4", I32: "<i4", F32: "<f4", U64: "<u8",
       I64: "<i8", F64: "<f8"}

# ggml tensor type id -> (name, elements per block, bytes per block)
GEOMETRY = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 42: ("Q2_0", 64, 18),
}
TYPE_BY_NAME = {v[0]: k for k, v in GEOMETRY.items()}


def type_name(t):
    return GEOMETRY[t][0] if t in GEOMETRY else "UNKNOWN(%d)" % t


def block_geometry(t):
    """(elements per block, bytes per block); raises KeyError for an unknown type."""
    g = GEOMETRY[t]
    return g[1], g[2]


def tensor_nbytes(t, n_elements):
    el, by = block_geometry(t)
    if n_elements % el:
        raise ValueError("%d elements is not a whole number of %s blocks" % (n_elements, type_name(t)))
    return n_elements // el * by


class GGUFError(ValueError):
    pass


class _Cursor:
    def __init__(self, buf, pos=0):
        self.buf, self.pos = buf, pos

    def take(self, n):
        if n < 0 or self.pos + n > len(self.buf):
            raise GGUFError("truncated GGUF data")
        v = self.buf[self.pos:self.pos + n]
        self.pos += n
        return v

    def unpack(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def u32(self):
        return self.unpack("<I")

    def u64(self):
        return self.unpack("<Q")

    def string(self):
        return bytes(self.take(self.u64())).decode("utf-8", errors="replace")


def _read_value(c, t, depth=0):
    if t in _SCALAR:
        v = c.unpack(_SCALAR[t])
        if t == BOOL:
            if v > 1:
                raise GGUFError("bad bool")
            return bool(v)
        return v
    if t == STRING:
        return c.string()
    if t == ARRAY:
        if depth >= 4:
            raise GGUFError("array nesting too deep")
        et, n = c.u32(), c.u64()
        if et > F64:
            raise GGUFError("bad array element type")
        if et in _NP:
            sz = np.dtype(_NP[et]).itemsize
            if n > (len(c.buf) - c.pos) // sz:
                raise GGUFError("array larger than file")
            a = np.frombuffer(bytes(c.take(n * sz)), dtype=_NP[et])
            return a.astype(bool) if et == BOOL else a
        if et == BOOL:
            return [_read_value(c, BOOL) for _ in range(n)]
        return [_read_value(c, et, depth + 1) for _ in range(n)]
    raise GGUFError("bad metadata type %d" % t)


@dataclass
class TensorInfo:
    name: str
    shape: tuple          # GGUF order: fastest-varying axis first
    type: int
    offset: int           # relative to the data section
    n_elements: int = field(init=False)
    nbytes: int = field(init=False)

    def __post_init__(self):
        n = 1
        for d in self.shape:
            n *= int(d)
        self.n_elements = n
        self.nbytes = tensor_nbytes(self.type, n) if self.type in GEOMETRY else 0

    @property
    def type_name(self):
        return type_name(self.type)


class GGUFFile:
    """Parsed GGUF file. Tensor bytes are read lazily through mmap."""

    def __init__(self, path):
        self.path = str(path)
        self._f = open(self.path, "rb")
        try:
            self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self._f.close()
            raise
        try:
            self._parse()
        except Exception:
            self.close()
            raise

    def _parse(self):
        m = self._mm
        if len(m) < 24 or m[:4] != MAGIC:
            raise GGUFError("not a GGUF file")
        c = _Cursor(m, 4)
        self.version = c.u32()
        if self.version not in (2, 3):
            raise GGUFError("unsupported GGUF version %d" % self.version)
        n_tensors, n_kv = c.u64(), c.u64()
        if n_tensors > (1 << 32) or n_kv > (1 << 32):
            raise GGUFError("implausible counts")
        self.metadata = {}
        for _ in range(n_kv):
            key = c.string()
            self.metadata[key] = _read_value(c, c.u32())
        self.tensors = []
        for _ in range(n_tensors):
            name = c.string()
            nd = c.u32()
            if nd > 8:
                raise GGUFError("too many dims")
            shape = tuple(c.u64() for _ in range(nd))
            t = c.u32()
            off = c.u64()
            self.tensors.append(TensorInfo(name, shape, t, off))
        self.alignment = int(self.metadata.get("general.alignment", DEFAULT_ALIGNMENT))
        if self.alignment <= 0:
            raise GGUFError("bad alignment")
        self.data_offset = (c.pos + self.alignment - 1) // self.alignment * self.alignment
        self.by_name = {t.name: t for t in self.tensors}
        if len(self.by_name) != len(self.tensors):
            raise GGUFError("duplicate tensor name")
        for t in self.tensors:
            if t.type not in GEOMETRY:
                continue
            if t.offset % self.alignment:
                raise GGUFError("tensor %s offset not aligned" % t.name)
            if self.data_offset + t.offset + t.nbytes > len(m):
                raise GGUFError("tensor %s extends past end of file" % t.name)

    def tensor_bytes(self, name):
        t = self.by_name[name]
        a = self.data_offset + t.offset
        return bytes(self._mm[a:a + t.nbytes])

    def tensor_view(self, name):
        t = self.by_name[name]
        a = self.data_offset + t.offset
        return memoryview(self._mm)[a:a + t.nbytes]

    def close(self):
        try:
            self._mm.close()
        finally:
            self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
