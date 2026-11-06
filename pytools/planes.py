"""Bit packing of code planes (LSB-first) and 64-byte plane alignment."""
import numpy as np

ALIGN = 64


def valid_bits(bits):
    return bits in (2, 4, 8)


def pack_codes(codes, bits):
    """Pack unsigned codes LSB-first, 8/bits per byte. Raises ValueError on a bad width, length or out-of-range code."""
    if not valid_bits(bits):
        raise ValueError("bits must be 2, 4 or 8")
    c = np.asarray(codes).astype(np.int64).ravel()
    per = 8 // bits
    if c.size % per:
        raise ValueError("code count must be a multiple of %d" % per)
    if c.size and (c.min() < 0 or c.max() >= (1 << bits)):
        raise ValueError("code does not fit in %d bits" % bits)
    c = c.reshape(-1, per)
    out = np.zeros(c.shape[0], dtype=np.int64)
    for k in range(per):
        out |= c[:, k] << (bits * k)
    return out.astype(np.uint8)


def unpack_codes(packed, bits):
    if not valid_bits(bits):
        raise ValueError("bits must be 2, 4 or 8")
    p = np.asarray(packed, dtype=np.uint8).ravel().astype(np.int64)
    per = 8 // bits
    out = np.empty((p.size, per), dtype=np.uint8)
    for k in range(per):
        out[:, k] = (p >> (bits * k)) & ((1 << bits) - 1)
    return out.ravel()


def pad64(n):
    return (ALIGN - n % ALIGN) % ALIGN


class PlaneWriter:
    """Appends planes so each one starts on a 64-byte boundary."""

    def __init__(self):
        self.buf = bytearray()

    def append_plane(self, data):
        self.buf.extend(b"\0" * pad64(len(self.buf)))
        off = len(self.buf)
        self.buf.extend(bytes(data))
        return off
