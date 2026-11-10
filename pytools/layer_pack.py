"""Expert blob layout: planes in order gu_codes, d_codes, gu_scales, d_scales (offsets 0, 819200, 1228800, 1331200); every plane 64-aligned.

One expert blob (H=2560, FF=640, 64-weight Q2_0 blocks) is 1,382,400 bytes = 3 roles x 25600 blocks x 18 bytes."""
import numpy as np

H, FF, QK = 2560, 640, 64
SC_GU = H // QK                    # blocks per gate/up row (40)
SC_D = FF // QK                    # blocks per down row (10)
BLOCK = 18
GU_CODES = 2 * FF * SC_GU * 16
GU_SCALES = 2 * FF * SC_GU * 2
D_CODES = H * SC_D * 16
D_SCALES = H * SC_D * 2
BLOB_BYTES = GU_CODES + GU_SCALES + D_CODES + D_SCALES
ROLE_BYTES = H * FF // QK * BLOCK
EXPERTS_PER_LAYER = 512
ROLES = ("gate", "up", "down")


def blob_offsets():
    """name -> (offset, nbytes) of the four planes inside a blob; asserts the 64-byte alignment of each."""
    off, cur = {}, 0
    for name, n in (("gu_codes", GU_CODES), ("d_codes", D_CODES), ("gu_scales", GU_SCALES), ("d_scales", D_SCALES)):
        assert cur % 64 == 0, name
        off[name] = (cur, n)
        cur += n
    assert cur == BLOB_BYTES
    return off


def expert_offset(layer, expert):
    """Byte offset of expert blob (layer, expert) in the flat expert arena."""
    if not (0 <= expert < EXPERTS_PER_LAYER) or layer < 0:
        raise ValueError("expert index out of range")
    return (layer * EXPERTS_PER_LAYER + expert) * BLOB_BYTES


def _role(raw, rows, sc):
    a = np.frombuffer(bytes(raw), dtype=np.uint8)
    if a.size != rows * sc * BLOCK:
        raise ValueError("role has %d bytes, expected %d" % (a.size, rows * sc * BLOCK))
    a = a.reshape(rows, sc, BLOCK)
    return a[:, :, :2], a[:, :, 2:]          # scales (rows, sc, 2), codes (rows, sc, 16)


def build(gate, up, down):
    """Raw Q2_0 role bytes (block order row*SC + b) -> expert blob bytes."""
    gs, gc = _role(gate, FF, SC_GU)
    us, uc = _role(up, FF, SC_GU)
    ds, dc = _role(down, H, SC_D)
    gu_s = np.stack([gs, us], axis=1).reshape(2 * FF, SC_GU, 2)      # gate is slot 2r, up is slot 2r+1
    gu_c = np.stack([gc, uc], axis=1).reshape(2 * FF, SC_GU, 16)
    blob = b"".join(np.ascontiguousarray(p).tobytes() for p in (gu_c, dc, gu_s, ds))
    assert len(blob) == BLOB_BYTES
    return blob


def rebuild_role(blob, role):
    """Inverse of build for one role: blob -> raw Q2_0 bytes."""
    if len(blob) != BLOB_BYTES:
        raise ValueError("blob size")
    o = blob_offsets()
    b = np.frombuffer(bytes(blob), dtype=np.uint8)

    def plane(name, shape):
        s, n = o[name]
        return b[s:s + n].reshape(shape)
    if role == "down":
        sc, cd = plane("d_scales", (H, SC_D, 2)), plane("d_codes", (H, SC_D, 16))
    elif role in ("gate", "up"):
        k = 0 if role == "gate" else 1
        sc = plane("gu_scales", (FF, 2, SC_GU, 2))[:, k]
        cd = plane("gu_codes", (FF, 2, SC_GU, 16))[:, k]
    else:
        raise ValueError("role must be gate, up or down")
    return np.concatenate([sc, cd], axis=2).tobytes()


def verify(blob, gate, up, down):
    return (rebuild_role(blob, "gate") == bytes(gate) and rebuild_role(blob, "up") == bytes(up)
            and rebuild_role(blob, "down") == bytes(down))
