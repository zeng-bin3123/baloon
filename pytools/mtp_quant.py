"""Quantisation of the MTP block's fused routed experts into Q2_0 / Q4_0 / Q8_0 blocks (same arithmetic order as the
C++ quantiser: float32 scale math, float64 error accumulation, round-to-nearest-even)."""
import json
import struct
import sys

import numpy as np

from . import mtp_download as dl

E, GU_OUT, HID, INTER = 512, 1280, 2560, 640
FORMATS = {"q2_0": (64, 18), "q4_0": (32, 18), "q8_0": (32, 34)}
SCALE_CANDIDATES = 17


def _check(x, block):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim < 1 or x.shape[-1] % block:
        raise ValueError("inner dimension %s is not a multiple of %d" % (x.shape[-1:], block))
    return x.reshape(-1, block)


def _half_bytes(h):
    return h.view(np.uint16)[:, None].view(np.uint8).reshape(-1, 2)


def q2_0(x):
    """Per 64-block: try d = fp16(amax * (16+k)/32), k = 0..16; keep the first minimum of the squared error.
    code = clip(rint(x/d), -1, 2) + 1 (LSB-first, four per byte)."""
    b = _check(x, 64)
    amax = np.abs(b).max(axis=1)
    bd = b.astype(np.float64)
    best_err = np.full(len(b), -1.0)
    best_h = np.zeros(len(b), np.float16)
    best_c = np.zeros(b.shape, np.int64)
    for k in range(SCALE_CANDIDATES):
        h = (amax * np.float32((16 + k) / 32.0)).astype(np.float16)
        d = h.astype(np.float32).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(d[:, None] != 0, np.clip(np.rint(bd / d[:, None]), -1, 2), 0.0)
        err = np.zeros(len(b))
        for j in range(64):                                  # sequential accumulation, as in the scalar code
            e = bd[:, j] - d * s[:, j]
            err = err + e * e
        better = (best_err < 0) | (err < best_err)
        best_err = np.where(better, err, best_err)
        best_h = np.where(better, h, best_h).astype(np.float16)
        best_c = np.where(better[:, None], s.astype(np.int64) + 1, best_c)
    out = np.empty((len(b), 18), np.uint8)
    out[:, :2] = _half_bytes(best_h)
    c = best_c.reshape(-1, 16, 4)
    out[:, 2:] = (c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)).astype(np.uint8)
    return out.tobytes()


def q4_0(x):
    """d = (value with the largest |.|) / -8; nibble = min(15, trunc(x/d + 8.5)); low half then high half."""
    b = _check(x, 32)
    idx = np.abs(b).argmax(axis=1)
    mx = b[np.arange(len(b)), idx]
    d = mx / np.float32(-8.0)
    with np.errstate(divide="ignore"):
        idv = np.where(d != 0, np.float32(1.0) / d, np.float32(0.0)).astype(np.float32)
    t = b * idv[:, None] + np.float32(8.5)
    nib = np.clip(np.trunc(t), 0, 15).astype(np.int64)
    out = np.empty((len(b), 18), np.uint8)
    out[:, :2] = _half_bytes(d.astype(np.float16))
    out[:, 2:] = (nib[:, :16] | (nib[:, 16:] << 4)).astype(np.uint8)
    return out.tobytes()


def q8_0(x):
    """d = amax/127; q = clip(rint(x * (1/d)), -127, 127)."""
    b = _check(x, 32)
    d = np.abs(b).max(axis=1) / np.float32(127.0)
    with np.errstate(divide="ignore"):
        idv = np.where(d != 0, np.float32(1.0) / d, np.float32(0.0)).astype(np.float32)
    q = np.clip(np.rint(b * idv[:, None]), -127, 127).astype(np.int8)
    out = np.empty((len(b), 34), np.uint8)
    out[:, :2] = _half_bytes(d.astype(np.float16))
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


_QUANT = {"q2_0": q2_0, "q4_0": q4_0, "q8_0": q8_0}


def dequant(fmt, raw):
    """Raw blocks of `fmt` -> float32 values (flat)."""
    block, by = FORMATS[fmt]
    r = np.frombuffer(bytes(raw), dtype=np.uint8)
    if r.size % by:
        raise ValueError("size is not a whole number of %s blocks" % fmt)
    r = r.reshape(-1, by)
    d = np.ascontiguousarray(r[:, :2]).view(np.float16).astype(np.float32).reshape(-1)[:, None]
    if fmt == "q2_0":
        c = ((r[:, 2:, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3).reshape(-1, 64).astype(np.int32)
        return (d * (c - 1).astype(np.float32)).reshape(-1)
    if fmt == "q4_0":
        c = np.concatenate([r[:, 2:] & 0xF, r[:, 2:] >> 4], axis=1).astype(np.int32)
        return ((c - 8).astype(np.float32) * d).reshape(-1)
    return (d * r[:, 2:].view(np.int8).astype(np.float32)).reshape(-1)


def load_bf16(path, name):
    """Read one BF16 tensor of a local safetensors file as float32 (bf16 bits << 16), with its shape."""
    hdr, start = dl.shard_header(path)
    meta = hdr[name]
    if meta["dtype"] != "BF16":
        raise ValueError("%s is %s, not BF16" % (name, meta["dtype"]))
    b, e = meta["data_offsets"]
    n = 1
    for s in meta["shape"]:
        n *= s
    if e - b != 2 * n:
        raise ValueError("size of %s does not match its shape" % name)
    w = np.frombuffer(dl.get(path, start + b, start + e), dtype="<u2").astype(np.uint32) << np.uint32(16)
    return w.view(np.float32).reshape(meta["shape"])


class _Experts:
    """A fused expert tensor [experts, out, in] quantised expert by expert along the last axis."""

    def __init__(self, w):
        w = np.asarray(w, dtype=np.float32)
        if w.ndim != 3:
            raise ValueError("expected [experts, out, in]")
        self.w = w

    def expert_bytes(self, fmt):
        block, by = FORMATS[fmt]
        if self.w.shape[2] % block:
            raise ValueError("inner dim not divisible by %d" % block)
        return self.w.shape[1] * (self.w.shape[2] // block) * by

    def quantize(self, fmt):
        """Yield the bytes of each expert in order."""
        self.expert_bytes(fmt)
        for e in range(self.w.shape[0]):
            yield _QUANT[fmt](self.w[e])

    def total_bytes(self, fmt):
        return self.w.shape[0] * self.expert_bytes(fmt)


def total_bytes(fmt):
    """Bytes of both fused tensors of the MTP block (gate_up [E,GU_OUT,HID] + down [E,HID,INTER]) in `fmt`."""
    block, by = FORMATS[fmt]
    return E * GU_OUT * (HID // block * by) + E * HID * (INTER // block * by)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["totals"]:
        print(json.dumps({f: total_bytes(f) for f in FORMATS}))
        return 0
    if len(argv) == 4 and argv[0] == "quantize":      # quantize SAFETENSORS NAME FMT -> raw bytes on stdout size only
        w = load_bf16(argv[1], argv[2])
        ex = _Experts(w if w.ndim == 3 else w[None])
        n = sum(len(b) for b in ex.quantize(argv[3]))
        print("quantised %s to %s: %d bytes" % (argv[2], argv[3], n))
        return 0
    print("usage: mtp_quant.py totals | quantize FILE.safetensors NAME q2_0|q4_0|q8_0", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
