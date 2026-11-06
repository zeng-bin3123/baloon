"""Runtime-side helpers of the MTP path: expert blob from Q2_0 roles, and the Q8_0 activation quantiser."""
import sys

import numpy as np

from . import layer_pack


def blob_of(gate, up, down):
    """Assemble the expert blob from three raw Q2_0 role byte strings (checked: roles must rebuild exactly)."""
    blob = layer_pack.build(gate, up, down)
    if not layer_pack.verify(blob, gate, up, down):
        raise AssertionError("blob does not rebuild its roles")
    return blob


def q8_0(x):
    """Activation quantiser: per 32-chunk d = amax/127 (float32), q = clip(rint(x * (1/d)), -127, 127).
    Returns (q int8 [n], d float32 [n/32], sums int32 [n/32])."""
    x = np.asarray(x, dtype=np.float32).ravel()
    if x.size % 32:
        raise ValueError("length must be a multiple of 32")
    c = x.reshape(-1, 32)
    amax = np.abs(c).max(axis=1)
    d = amax / np.float32(127.0)
    with np.errstate(divide="ignore"):
        idv = np.where(d != 0, np.float32(1.0) / d, np.float32(0.0)).astype(np.float32)
    q = np.clip(np.rint(c * idv[:, None]), -127, 127).astype(np.int8)
    return q.reshape(-1), d.astype(np.float32), q.astype(np.int32).sum(axis=1).astype(np.int32)


def main(argv=None):
    """Self-check: random roles -> blob -> roles; prints the blob size."""
    rng = np.random.default_rng(0)
    g, u, d = (rng.integers(0, 256, layer_pack.ROLE_BYTES, dtype=np.uint8).tobytes() for _ in range(3))
    blob = blob_of(g, u, d)
    print("blob bytes:", len(blob))
    return 0 if len(blob) == layer_pack.BLOB_BYTES else 1


if __name__ == "__main__":
    sys.exit(main())
