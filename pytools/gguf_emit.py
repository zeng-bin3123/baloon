"""Minimal GGUF v3 writer and Q2_0 quantiser (clamp to {-1..2}, LSB-first 2-bit codes, 18-byte blocks)."""
import struct
from dataclasses import dataclass

import numpy as np

from . import gguf_file as gf

Q2_0 = 42
QK2_0 = 64
Q2_0_BYTES = 18


def _pad(a, n):
    return (a - n % a) % a


@dataclass
class _Tensor:
    name: str
    shape: tuple          # GGUF order (fastest axis first)
    type: int
    data: bytes

    def check(self):
        if self.type not in gf.GEOMETRY or not self.shape:
            raise ValueError("tensor %r has unknown type or empty shape" % self.name)
        n = 1
        for d in self.shape:
            n *= int(d)
        if len(self.data) != gf.tensor_nbytes(self.type, n):
            raise ValueError("tensor %r has inconsistent size/shape/type" % self.name)


def _p32(v):
    return struct.pack("<I", v)


def _p64(v):
    return struct.pack("<Q", v)


def _pstr(s):
    b = s.encode("utf-8")
    return _p64(len(b)) + b


class GGUFWriter:
    def __init__(self, alignment=gf.DEFAULT_ALIGNMENT):
        if alignment <= 0:
            raise ValueError("alignment must be positive")
        self.alignment = alignment
        self.kv = []
        self.tensors = []

    def add_kv(self, key, value):
        """str -> STRING, bool -> BOOL, float -> F32, int -> U32 (U64 when it does not fit),
        non-empty list of str -> STRING array, of int -> I32 array."""
        if any(k == key for k, _ in self.kv):
            raise ValueError("duplicate key %r" % key)
        self.kv.append((key, value))

    def add_tensor(self, name, shape, type, data):
        if any(t.name == name for t in self.tensors):
            raise ValueError("duplicate tensor %r" % name)
        t = _Tensor(name, tuple(int(d) for d in shape), type, bytes(data))
        t.check()
        self.tensors.append(t)
        return t

    def _enc_value(self, v):
        if isinstance(v, bool):
            return _p32(gf.BOOL) + bytes([1 if v else 0])
        if isinstance(v, str):
            return _p32(gf.STRING) + _pstr(v)
        if isinstance(v, float):
            return _p32(gf.F32) + struct.pack("<f", v)
        if isinstance(v, int):
            if 0 <= v < (1 << 32):
                return _p32(gf.U32) + _p32(v)
            return _p32(gf.U64) + _p64(v)
        if isinstance(v, (list, tuple)) and v and all(isinstance(e, str) for e in v):
            return (_p32(gf.ARRAY) + _p32(gf.STRING) + _p64(len(v))) + b"".join(_pstr(e) for e in v)
        if isinstance(v, (list, tuple)) and v and all(isinstance(e, int) and not isinstance(e, bool) for e in v):
            return (_p32(gf.ARRAY) + _p32(gf.I32) + _p64(len(v))) + struct.pack("<%di" % len(v), *v)
        raise TypeError("unsupported metadata value %r" % (v,))

    def to_bytes(self):
        a = self.alignment
        kv = list(self.kv)
        if a != gf.DEFAULT_ALIGNMENT and not any(k == "general.alignment" for k, _ in kv):
            kv.append(("general.alignment", a))
        offs, cur = [], 0
        for t in self.tensors:
            t.check()
            offs.append(cur)
            cur += len(t.data) + _pad(a, len(t.data))
        o = bytearray(gf.MAGIC + _p32(3) + _p64(len(self.tensors)) + _p64(len(kv)))
        for k, v in kv:
            o += _pstr(k) + self._enc_value(v)
        for t, off in zip(self.tensors, offs):
            o += _pstr(t.name) + _p32(len(t.shape))
            for d in t.shape:
                o += _p64(d)
            o += _p32(t.type) + _p64(off)
        o += b"\0" * _pad(a, len(o))
        for t in self.tensors:
            o += t.data + b"\0" * _pad(a, len(t.data))
        return bytes(o)

    def write(self, path):
        data = self.to_bytes()
        with open(path, "wb") as f:
            f.write(data)
        return len(data)


def quantize_q2_0(x):
    """float array (length multiple of 64) -> Q2_0 bytes. d = fp16(amax); code = clip(rint(x/d), -1, 2) + 1."""
    x = np.asarray(x, dtype=np.float32).ravel()
    if x.size % QK2_0:
        raise ValueError("length must be a multiple of 64")
    b = x.reshape(-1, QK2_0)
    amax = np.abs(b).max(axis=1)
    h = amax.astype(np.float16)
    d = h.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(d[:, None] != 0, np.rint(b.astype(np.float64) / d[:, None].astype(np.float64)), 0.0)
    codes = (np.clip(s, -1, 2) + 1).astype(np.uint8)
    packed = (codes.reshape(-1, 16, 4) << np.array([0, 2, 4, 6], dtype=np.uint8)).sum(axis=2).astype(np.uint8)
    out = np.empty((b.shape[0], Q2_0_BYTES), dtype=np.uint8)
    out[:, :2] = h.view(np.uint16)[:, None].view(np.uint8).reshape(-1, 2)
    out[:, 2:] = packed
    return out.tobytes()


def dequantize_q2_0(raw):
    """Q2_0 bytes -> float32 values: d * (code - 1)."""
    r = np.frombuffer(bytes(raw), dtype=np.uint8)
    if r.size % Q2_0_BYTES:
        raise ValueError("size is not a whole number of Q2_0 blocks")
    r = r.reshape(-1, Q2_0_BYTES)
    d = np.ascontiguousarray(r[:, :2]).view(np.float16).astype(np.float32).reshape(-1)
    q = (r[:, 2:, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3
    codes = q.reshape(-1, 64).astype(np.int32)
    return (d[:, None] * (codes - 1).astype(np.float32)).reshape(-1)


def _selfcheck():
    """Round-trip invariants; raises AssertionError on failure, returns True."""
    rng = np.random.default_rng(37)
    # packing: LSB-first, four codes per byte, code = symbol + 1
    blk = np.zeros(64, dtype=np.float32)
    blk[0], blk[1], blk[2], blk[3] = -1.0, 0.0, 1.0, 2.0
    blk[63] = 2.0
    raw = quantize_q2_0(blk)
    assert len(raw) == 18
    d = np.frombuffer(raw[:2], dtype=np.float16)[0]
    assert float(d) == 2.0                                   # amax
    # symbols are rint(x/2) = {-0.5->-0, 0, 0.5->0, 1}: codes 1,1,1,2 (RNE ties to even)
    assert raw[2] == (1 | (1 << 2) | (1 << 4) | (2 << 6))
    # dequantisation never leaves the {-1,0,1,2} * d grid
    x = rng.standard_normal(64 * 7).astype(np.float32)
    y = dequantize_q2_0(quantize_q2_0(x))
    assert y.shape == x.shape
    for i in range(7):
        dd = float(np.frombuffer(quantize_q2_0(x[i * 64:(i + 1) * 64])[:2], dtype=np.float16)[0])
        syms = np.rint(y[i * 64:(i + 1) * 64] / dd)
        assert syms.min() >= -1 and syms.max() <= 2
    # writer -> reader round trip with alignment 64
    import os
    import tempfile
    w = GGUFWriter(alignment=64)
    w.add_kv("general.architecture", "test")
    w.add_kv("test.count", 7)
    w.add_kv("test.flag", True)
    w.add_tensor("a", (64, 3), Q2_0, quantize_q2_0(x[:192]))
    w.add_tensor("b", (4,), 0, np.arange(4, dtype=np.float32).tobytes())
    fd, p = tempfile.mkstemp(suffix=".gguf")
    os.close(fd)
    try:
        w.write(p)
        with gf.GGUFFile(p) as g:
            assert g.alignment == 64 and g.metadata["test.count"] == 7 and g.metadata["test.flag"] is True
            assert g.data_offset % 64 == 0
            assert all(t.offset % 64 == 0 for t in g.tensors)
            assert g.tensor_bytes("a") == quantize_q2_0(x[:192])
            assert np.frombuffer(g.tensor_bytes("b"), dtype=np.float32).tolist() == [0, 1, 2, 3]
    finally:
        os.unlink(p)
    return True


if __name__ == "__main__":
    print("selfcheck:", _selfcheck())
