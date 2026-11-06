"""Canonical code-plane forms (S2 / S4 / S8 / P16 / P32) of GGUF quantisation types and the cross-check that every
canonical form decodes bit-exactly to the straight-line ggml dequantiser.

Decode rule: (code - bias) * scale  (bias applied to the INTEGER code before the single float multiply),
scale * code - offset for Q4_K / Q5_K, scale * kvalues_iq4nl[code] for the IQ4 types."""
import sys
from collections import namedtuple

import numpy as np

from . import gguf_file as gf
from . import planes

KVALUES_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.int8)

# canonical block set: codes (nb, elems) uint8, scales (nb, groups) float32, offsets (nb, groups) float32
Canon = namedtuple("Canon", "codes scales offsets")

Spec = namedtuple("Spec", "name type kind bits block_elems block_bytes group_elems bias has_offset codebook")
_S = Spec
MAPPINGS = {s.name: s for s in [
    _S("Q2_0", 42, "S2", 2, 64, 18, 64, 1, False, False),
    _S("Q3_K", 11, "S4", 4, 256, 110, 16, 4, False, False),
    _S("Q4_0", 2, "S4", 4, 32, 18, 32, 8, False, False),
    _S("Q4_K", 12, "S4", 4, 256, 144, 32, 0, True, False),
    _S("Q5_0", 6, "S8", 8, 32, 22, 32, 16, False, False),
    _S("Q5_K", 13, "S8", 8, 256, 176, 32, 0, True, False),
    _S("Q6_K", 14, "S8", 8, 256, 210, 16, 32, False, False),
    _S("Q8_0", 8, "S8", 8, 32, 34, 32, 128, False, False),
    _S("IQ4_NL", 20, "S4", 4, 32, 18, 32, 0, False, True),
    _S("IQ4_XS", 23, "S4", 4, 256, 136, 32, 0, False, True),
]}
PASSTHROUGH = {"F16": (1, "P16", 2), "BF16": (30, "P32", 2), "F32": (0, "P32", 4)}
_SPEC_BY_TYPE = {s.type: s for s in MAPPINGS.values()}


def reference_name(t):
    """ggml type name for a type id (or a name, normalised)."""
    if isinstance(t, str):
        u = t.upper()
        if u in MAPPINGS or u in PASSTHROUGH:
            return u
        raise KeyError(t)
    return gf.type_name(t)


# ---- helpers ------------------------------------------------------------------------------------------------------
def _blocks(raw, size):
    r = np.frombuffer(bytes(raw), dtype=np.uint8)
    if r.size == 0 or r.size % size:
        raise ValueError("raw size %d is not a positive multiple of the %d-byte block" % (r.size, size))
    return r.reshape(-1, size)


def _f16(b2):
    return np.ascontiguousarray(b2).view(np.float16).astype(np.float32).reshape(-1)


def get_scale_min_k4(j, q):
    """6-bit (scale, min) pair of group j (0..7) from the 12 packed bytes q (works on [..., 12] arrays)."""
    q = np.asarray(q).astype(np.int64)
    if j < 4:
        return q[..., j] & 63, q[..., j + 4] & 63
    return (q[..., j + 4] & 0xF) | ((q[..., j - 4] >> 6) << 4), (q[..., j + 4] >> 4) | ((q[..., j] >> 6) << 4)


def unpack_q3_k_scales(packed12):
    """16 stored 6-bit Q3_K scales (before the -32 bias) as int8 from the 12 packed bytes ([..., 12] -> [..., 16])."""
    p = np.ascontiguousarray(np.asarray(packed12, dtype=np.uint8))
    a = p.view("<u4").astype(np.uint32)                       # [..., 3]
    k1, k2 = np.uint32(0x03030303), np.uint32(0x0F0F0F0F)
    a0, a1, tmp = a[..., 0], a[..., 1], a[..., 2]
    o2 = ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4)
    o3 = ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4)
    o0 = (a0 & k2) | (((tmp >> 0) & k1) << 4)
    o1 = (a1 & k2) | (((tmp >> 2) & k1) << 4)
    out = np.stack([o0, o1, o2, o3], axis=-1).astype("<u4")
    return np.ascontiguousarray(out).view(np.int8).reshape(a.shape[:-1] + (16,))


# ---- encode: raw ggml blocks -> canonical -------------------------------------------------------------------------
def _nib_split(qs):
    return np.concatenate([qs & 0xF, qs >> 4], axis=1)


def _enc_q2_0(b):
    codes = ((b[:, 2:, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3).reshape(-1, 64)
    return Canon(codes.astype(np.uint8), _f16(b[:, :2])[:, None], np.zeros((len(b), 1), np.float32))


def _enc_q4_like(b):
    return Canon(_nib_split(b[:, 2:18]).astype(np.uint8), _f16(b[:, :2])[:, None], np.zeros((len(b), 1), np.float32))


def _enc_q5_0(b):
    qh = np.ascontiguousarray(b[:, 2:6]).view("<u4").reshape(-1).astype(np.int64)
    qs = b[:, 6:22].astype(np.int64)
    j = np.arange(16, dtype=np.int64)
    lo = (qs & 0xF) | (((qh[:, None] >> j) << 4) & 0x10)
    hi = (qs >> 4) | ((qh[:, None] >> (j + 12)) & 0x10)
    return Canon(np.concatenate([lo, hi], axis=1).astype(np.uint8), _f16(b[:, :2])[:, None],
                 np.zeros((len(b), 1), np.float32))


def _enc_q8_0(b):
    codes = (b[:, 2:34].view(np.int8).astype(np.int32) + 128).astype(np.uint8)
    return Canon(codes, _f16(b[:, :2])[:, None], np.zeros((len(b), 1), np.float32))


def _enc_q6_k(b):
    nb = len(b)
    ql, qh = b[:, 0:128].astype(np.int64), b[:, 128:192].astype(np.int64)
    sc = np.ascontiguousarray(b[:, 192:208]).view(np.int8).astype(np.float32)
    d = _f16(b[:, 208:210])
    codes = np.empty((nb, 256), dtype=np.int64)
    for h in range(2):
        e = 128 * h
        L, H = ql[:, 64 * h:64 * h + 64], qh[:, 32 * h:32 * h + 32]
        codes[:, e:e + 32] = (L[:, :32] & 0xF) | (((H >> 0) & 3) << 4)
        codes[:, e + 32:e + 64] = (L[:, 32:] & 0xF) | (((H >> 2) & 3) << 4)
        codes[:, e + 64:e + 96] = (L[:, :32] >> 4) | (((H >> 4) & 3) << 4)
        codes[:, e + 96:e + 128] = (L[:, 32:] >> 4) | (((H >> 6) & 3) << 4)
    return Canon(codes.astype(np.uint8), d[:, None] * sc, np.zeros((nb, 16), np.float32))


def _enc_q45_k(b, q5):
    nb = len(b)
    d, dmin = _f16(b[:, 0:2]), _f16(b[:, 2:4])
    scales = b[:, 4:16]
    if q5:
        qh, qs = b[:, 16:48].astype(np.int64), b[:, 48:176].astype(np.int64)
    else:
        qh, qs = None, b[:, 16:144].astype(np.int64)
    codes = np.empty((nb, 256), dtype=np.int64)
    sc = np.empty((nb, 8), np.float32)
    off = np.empty((nb, 8), np.float32)
    for jj in range(4):
        for k in (0, 1):
            s, m = get_scale_min_k4(2 * jj + k, scales)
            sc[:, 2 * jj + k] = d * s.astype(np.float32)
            off[:, 2 * jj + k] = dmin * m.astype(np.float32)
        ql = qs[:, 32 * jj:32 * jj + 32]
        lo, hi = ql & 0xF, ql >> 4
        if q5:
            u1, u2 = 1 << (2 * jj), 2 << (2 * jj)
            lo = lo + np.where((qh & u1) != 0, 16, 0)
            hi = hi + np.where((qh & u2) != 0, 16, 0)
        codes[:, 64 * jj:64 * jj + 32] = lo
        codes[:, 64 * jj + 32:64 * jj + 64] = hi
    return Canon(codes.astype(np.uint8), sc, off)


def _enc_q3_k(b):
    nb = len(b)
    hm, q = b[:, 0:32].astype(np.int64), b[:, 32:96].astype(np.int64)
    sc = unpack_q3_k_scales(b[:, 96:108]).astype(np.int32)
    dall = _f16(b[:, 108:110])
    codes = np.empty((nb, 256), dtype=np.int64)
    scale = np.empty((nb, 16), np.float32)
    for h in range(2):
        for j in range(4):
            m = 1 << (4 * h + j)
            for c in range(2):
                g = 8 * h + 2 * j + c
                e = 128 * h + 32 * j + 16 * c
                qq = q[:, 32 * h + 16 * c:32 * h + 16 * c + 16]
                hh = hm[:, 16 * c:16 * c + 16]
                codes[:, e:e + 16] = ((qq >> (2 * j)) & 3) + np.where((hh & m) != 0, 4, 0)
                scale[:, g] = dall * (sc[:, g] - 32).astype(np.float32)
    return Canon(codes.astype(np.uint8), scale, np.zeros((nb, 16), np.float32))


def _enc_iq4_xs(b):
    nb = len(b)
    d = _f16(b[:, 0:2])
    sh = np.ascontiguousarray(b[:, 2:4]).view("<u2").reshape(-1).astype(np.int64)
    sl, qs = b[:, 4:8].astype(np.int64), b[:, 8:136].astype(np.int64)
    codes = np.empty((nb, 256), dtype=np.int64)
    scale = np.empty((nb, 8), np.float32)
    for ib in range(8):
        ls = ((sl[:, ib // 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4)
        scale[:, ib] = d * (ls - 32).astype(np.float32)
        q = qs[:, 16 * ib:16 * ib + 16]
        codes[:, ib * 32:ib * 32 + 16] = q & 0xF
        codes[:, ib * 32 + 16:ib * 32 + 32] = q >> 4
    return Canon(codes.astype(np.uint8), scale, np.zeros((nb, 8), np.float32))


_ENCODERS = {
    "Q2_0": _enc_q2_0, "Q4_0": _enc_q4_like, "IQ4_NL": _enc_q4_like, "Q5_0": _enc_q5_0, "Q8_0": _enc_q8_0,
    "Q6_K": _enc_q6_k, "Q4_K": lambda b: _enc_q45_k(b, False), "Q5_K": lambda b: _enc_q45_k(b, True),
    "Q3_K": _enc_q3_k, "IQ4_XS": _enc_iq4_xs,
}


def _encode(name, raw):
    s = MAPPINGS[name]
    c = _ENCODERS[name](_blocks(raw, s.block_bytes))
    assert c.codes.shape[1] == s.block_elems and c.codes.max() < (1 << s.bits)
    return c


def _decode(name, c):
    """canonical -> float32 values: integer bias first, then one float multiply."""
    s = MAPPINGS[name]
    if c.codes.shape[1] != s.block_elems:
        raise ValueError("canonical block has %d elements, %s needs %d" % (c.codes.shape[1], name, s.block_elems))
    g = np.arange(s.block_elems) // s.group_elems
    sc = c.scales[:, g]
    if s.codebook:
        return sc * KVALUES_IQ4NL[c.codes & 0xF].astype(np.float32)
    if s.has_offset:
        return sc * c.codes.astype(np.float32) - c.offsets[:, g]
    return sc * (c.codes.astype(np.int32) - s.bias).astype(np.float32)


def to_s2(raw): return _encode("Q2_0", raw)
def to_s4_q4_0(raw): return _encode("Q4_0", raw)
def to_s4_iq4_nl(raw): return _encode("IQ4_NL", raw)
def to_s4_iq4_xs(raw): return _encode("IQ4_XS", raw)
def to_s4_q3_k(raw): return _encode("Q3_K", raw)
def to_s4_q4_k(raw): return _encode("Q4_K", raw)
def to_s8_q5_0(raw): return _encode("Q5_0", raw)
def to_s8_q5_k(raw): return _encode("Q5_K", raw)
def to_s8_q6_K(raw): return _encode("Q6_K", raw)
def to_s8_q8_0(raw): return _encode("Q8_0", raw)


def from_s2(c): return _decode("Q2_0", c)
def from_s4_q4_0(c): return _decode("Q4_0", c)
def from_s4_iq4_nl(c): return _decode("IQ4_NL", c)
def from_s4_iq4_xs(c): return _decode("IQ4_XS", c)
def from_s4_q3_k(c): return _decode("Q3_K", c)
def from_s4_q4_k(c): return _decode("Q4_K", c)
def from_s8_q5_0(c): return _decode("Q5_0", c)
def from_s8_q5_k(c): return _decode("Q5_K", c)
def from_s8_q6_K(c): return _decode("Q6_K", c)
def from_s8_q8_0(c): return _decode("Q8_0", c)


def s2_to_raw(c):
    """Lossless inverse of to_s2: canonical Q2_0 -> raw 18-byte blocks."""
    nb = c.codes.shape[0]
    out = np.empty((nb, 18), np.uint8)
    out[:, :2] = c.scales[:, 0].astype(np.float16).view(np.uint16)[:, None].view(np.uint8).reshape(-1, 2)
    cc = c.codes.reshape(nb, 16, 4).astype(np.int64)
    out[:, 2:] = (cc[:, :, 0] | (cc[:, :, 1] << 2) | (cc[:, :, 2] << 4) | (cc[:, :, 3] << 6)).astype(np.uint8)
    return out.tobytes()


# ---- passthrough planes -------------------------------------------------------------------------------------------
def to_p16_f16(raw):
    """F16 stays a 16-bit plane: uint16 words."""
    r = bytes(raw)
    if len(r) % 2:
        raise ValueError("odd byte count")
    return np.frombuffer(r, dtype="<u2").copy()


def from_p16_f16(words):
    return np.asarray(words, dtype="<u2").view(np.float16).astype(np.float32)


def to_p32_bf16(raw):
    """BF16 is promoted to a 32-bit plane holding the float32 bit pattern (bf16 bits << 16)."""
    r = bytes(raw)
    if len(r) % 2:
        raise ValueError("odd byte count")
    return np.frombuffer(r, dtype="<u2").astype(np.uint32) << np.uint32(16)


def from_p32_bf16(words):
    return np.asarray(words, dtype="<u4").view(np.float32).copy()


def p32_bf16_to_raw(words):
    w = np.asarray(words, dtype="<u4")
    if (w & np.uint32(0xFFFF)).any():
        raise ValueError("low half of a promoted bf16 word must be zero")
    return (w >> np.uint32(16)).astype("<u2").tobytes()


def to_p32_f32(raw):
    r = bytes(raw)
    if len(r) % 4:
        raise ValueError("byte count is not a multiple of 4")
    return np.frombuffer(r, dtype="<u4").copy()


def from_p32_f32(words):
    return np.asarray(words, dtype="<u4").view(np.float32).copy()


# ---- straight-line reference dequantisers (the oracle) --------------------------------------------------------------
def _rd_q2_0(b):
    d = np.float32(np.frombuffer(b[:2].tobytes(), np.float16)[0])
    y = np.empty(64, np.float32)
    for j in range(64):
        y[j] = d * np.float32(((int(b[2 + j // 4]) >> (2 * (j % 4))) & 3) - 1)
    return y


def _rd_q4_0(b):
    d = np.float32(np.frombuffer(b[:2].tobytes(), np.float16)[0])
    y = np.empty(32, np.float32)
    for j in range(16):
        y[j] = np.float32((int(b[2 + j]) & 0xF) - 8) * d
        y[j + 16] = np.float32((int(b[2 + j]) >> 4) - 8) * d
    return y


def _rd_iq4_nl(b):
    d = np.float32(np.frombuffer(b[:2].tobytes(), np.float16)[0])
    y = np.empty(32, np.float32)
    for j in range(16):
        y[j] = d * np.float32(KVALUES_IQ4NL[int(b[2 + j]) & 0xF])
        y[j + 16] = d * np.float32(KVALUES_IQ4NL[int(b[2 + j]) >> 4])
    return y


def _rd_q5_0(b):
    d = np.float32(np.frombuffer(b[:2].tobytes(), np.float16)[0])
    qh = int.from_bytes(b[2:6].tobytes(), "little")
    y = np.empty(32, np.float32)
    for j in range(16):
        x0 = ((int(b[6 + j]) & 0xF) | (((qh >> j) << 4) & 0x10)) - 16
        x1 = ((int(b[6 + j]) >> 4) | ((qh >> (j + 12)) & 0x10)) - 16
        y[j] = np.float32(x0) * d
        y[j + 16] = np.float32(x1) * d
    return y


def _rd_q8_0(b):
    d = np.float32(np.frombuffer(b[:2].tobytes(), np.float16)[0])
    return d * b[2:34].view(np.int8).astype(np.float32)


def _rd_q6_k(b):
    ql, qh = b[0:128], b[128:192]
    sc = b[192:208].view(np.int8)
    d = np.float32(np.frombuffer(b[208:210].tobytes(), np.float16)[0])
    y = np.empty(256, np.float32)
    for n in range(2):
        for l in range(32):
            is_ = l // 16
            L0, L1, H = int(ql[64 * n + l]), int(ql[64 * n + l + 32]), int(qh[32 * n + l])
            q = [((L0 & 0xF) | (((H >> 0) & 3) << 4)) - 32, ((L1 & 0xF) | (((H >> 2) & 3) << 4)) - 32,
                 ((L0 >> 4) | (((H >> 4) & 3) << 4)) - 32, ((L1 >> 4) | (((H >> 6) & 3) << 4)) - 32]
            for k in range(4):
                y[128 * n + l + 32 * k] = d * np.float32(sc[8 * n + is_ + 2 * k]) * np.float32(q[k])
    return y


def _rd_q45_k(b, q5):
    d = np.float32(np.frombuffer(b[0:2].tobytes(), np.float16)[0])
    mn = np.float32(np.frombuffer(b[2:4].tobytes(), np.float16)[0])
    scales = b[4:16]
    qh = b[16:48] if q5 else None
    ql = b[48:176] if q5 else b[16:144]
    y = np.empty(256, np.float32)
    u1, u2 = 1, 2
    for jj in range(4):
        s0, m0 = get_scale_min_k4(2 * jj, scales)
        s1, m1 = get_scale_min_k4(2 * jj + 1, scales)
        d1, mm1 = d * np.float32(s0), mn * np.float32(m0)
        d2, mm2 = d * np.float32(s1), mn * np.float32(m1)
        for l in range(32):
            v = int(ql[32 * jj + l])
            lo = (v & 0xF) + (16 if (q5 and (int(qh[l]) & u1)) else 0)
            hi = (v >> 4) + (16 if (q5 and (int(qh[l]) & u2)) else 0)
            y[64 * jj + l] = d1 * np.float32(lo) - mm1
            y[64 * jj + 32 + l] = d2 * np.float32(hi) - mm2
        u1, u2 = (u1 << 2) & 0xFF, (u2 << 2) & 0xFF
    return y


def _rd_q3_k(b):
    hm, q = b[0:32], b[32:96]
    sc = unpack_q3_k_scales(b[96:108])
    dall = np.float32(np.frombuffer(b[108:110].tobytes(), np.float16)[0])
    y = np.empty(256, np.float32)
    m, is_, e = 1, 0, 0
    for n in range(2):
        shift = 0
        for j in range(4):
            for c in range(2):
                dl = dall * np.float32(int(sc[is_]) - 32)
                is_ += 1
                for l in range(16):
                    v = ((int(q[32 * n + 16 * c + l]) >> shift) & 3) - (0 if (int(hm[16 * c + l]) & m) else 4)
                    y[e] = dl * np.float32(v)
                    e += 1
            shift += 2
            m = (m << 1) & 0xFF
    return y


def _rd_iq4_xs(b):
    d = np.float32(np.frombuffer(b[0:2].tobytes(), np.float16)[0])
    sh = int.from_bytes(b[2:4].tobytes(), "little")
    y = np.empty(256, np.float32)
    for ib in range(8):
        ls = ((int(b[4 + ib // 2]) >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4)
        dl = d * np.float32(ls - 32)
        for j in range(16):
            v = int(b[8 + 16 * ib + j])
            y[32 * ib + j] = dl * np.float32(KVALUES_IQ4NL[v & 0xF])
            y[32 * ib + j + 16] = dl * np.float32(KVALUES_IQ4NL[v >> 4])
    return y


_REFERENCE = {"Q2_0": _rd_q2_0, "Q4_0": _rd_q4_0, "IQ4_NL": _rd_iq4_nl, "Q5_0": _rd_q5_0, "Q8_0": _rd_q8_0,
              "Q6_K": _rd_q6_k, "Q4_K": lambda b: _rd_q45_k(b, False), "Q5_K": lambda b: _rd_q45_k(b, True),
              "Q3_K": _rd_q3_k, "IQ4_XS": _rd_iq4_xs}


def reference_values(t, raw):
    """float32 values of raw blocks of type t (id or name) via the block-by-block reference dequantiser."""
    name = reference_name(t)
    if name in PASSTHROUGH:
        return {"F16": from_p16_f16(to_p16_f16(raw)), "BF16": from_p32_bf16(to_p32_bf16(raw)),
                "F32": from_p32_f32(to_p32_f32(raw))}[name]
    s = MAPPINGS[name]
    b = _blocks(raw, s.block_bytes)
    return np.concatenate([_REFERENCE[name](b[i]) for i in range(len(b))])


# ---- shard access and the full cross-check ---------------------------------------------------------------------------
def open_shard(path):
    return gf.GGUFFile(path)


def data_section_offset(g):
    """Absolute offset of the tensor data section of a GGUFFile (or of a path), always a multiple of the alignment."""
    if isinstance(g, (str, bytes)) or hasattr(g, "__fspath__"):
        with open_shard(g) as f:
            return data_section_offset(f)
    assert g.data_offset % g.alignment == 0
    return g.data_offset


def _bits_equal(a, b):
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def check_block_set(name, raw):
    """Cross-check one raw block run; returns a list of failure strings (empty = every invariant holds)."""
    bad = []
    s = MAPPINGS[name]
    c = _encode(name, raw)
    ref = reference_values(name, raw)
    got = _decode(name, c).reshape(-1)
    if not _bits_equal(got.astype(np.float32), ref.astype(np.float32)):
        bad.append("%s: canonical decode differs from the reference dequantiser" % name)
    packed = planes.pack_codes(c.codes, s.bits)
    if not np.array_equal(planes.unpack_codes(packed, s.bits).reshape(c.codes.shape), c.codes):
        bad.append("%s: code plane does not round-trip through %d-bit packing" % (name, s.bits))
    if name == "Q2_0" and s2_to_raw(c) != bytes(raw):
        bad.append("Q2_0: S2 form is not lossless on the raw bytes")
    return bad


def run_full(path, blocks=8):
    """Cross-check the first `blocks` blocks of every tensor of the GGUF at `path`. Returns (n_checked, failures)."""
    checked, fails = 0, []
    with open_shard(path) as g:
        for t in g.tensors:
            tn = gf.type_name(t.type)
            if tn in MAPPINGS:
                s = MAPPINGS[tn]
                nb = min(blocks, t.n_elements // s.block_elems)
                if nb <= 0:
                    continue
                raw = bytes(g.tensor_view(t.name)[:nb * s.block_bytes])
                bad = check_block_set(tn, raw)
            elif tn in PASSTHROUGH:
                tid, kind, w = PASSTHROUGH[tn]
                raw = bytes(g.tensor_view(t.name)[:min(blocks * 32, t.n_elements) * w])
                bad = []
                if tn == "BF16" and p32_bf16_to_raw(to_p32_bf16(raw)) != raw:
                    bad.append("BF16: P32 promotion is not lossless")
                if tn == "F16" and not np.array_equal(from_p16_f16(to_p16_f16(raw)), reference_values("F16", raw)):
                    bad.append("F16: passthrough mismatch")
            else:
                continue
            checked += 1
            fails += ["%s: %s" % (t.name, m) for m in bad]
    return checked, fails


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    blocks = 8
    if "--blocks" in argv:
        i = argv.index("--blocks")
        blocks = int(argv[i + 1])
        del argv[i:i + 2]
    if not argv:
        print("usage: canon_xcheck.py [--blocks N] model.gguf [...]", file=sys.stderr)
        return 2
    rc = 0
    for p in argv:
        n, fails = run_full(p, blocks)
        print("%s: %d tensors checked, %d failures" % (p, n, len(fails)))
        for f in fails:
            print("  FAIL", f)
        rc |= 1 if fails else 0
    return rc


if __name__ == "__main__":
    sys.exit(main())
