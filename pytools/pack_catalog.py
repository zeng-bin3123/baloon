"""Catalog of a pack directory: dense.bin / embd.bin / experts.bin sizes and the expert arena geometry."""
import json
import os
import sys

from . import layer_pack as lp

FILES = ("dense.bin", "embd.bin", "experts.bin")


def describe(directory):
    """Inspect a pack directory; returns the catalog dict. Raises ValueError if the arena is not whole blobs."""
    info = {"version": 1, "blob_bytes": lp.BLOB_BYTES, "experts_per_layer": lp.EXPERTS_PER_LAYER,
            "planes": {k: list(v) for k, v in lp.blob_offsets().items()}, "files": {}}
    for f in FILES:
        p = os.path.join(directory, f)
        if os.path.isfile(p):
            info["files"][f] = os.path.getsize(p)
    arena = info["files"].get("experts.bin")
    if arena is not None:
        layer_bytes = lp.EXPERTS_PER_LAYER * lp.BLOB_BYTES
        if arena % layer_bytes:
            raise ValueError("experts.bin (%d bytes) is not a whole number of %d-byte layers" % (arena, layer_bytes))
        info["layers"] = arena // layer_bytes
    return info


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: pack_catalog.py PACK_DIR [OUT.json]", file=sys.stderr)
        return 2
    try:
        info = describe(argv[0])
    except (ValueError, OSError) as e:
        print("error:", e, file=sys.stderr)
        return 1
    out = argv[1] if len(argv) > 1 else os.path.join(argv[0], "pack_index.json")
    with open(out, "w") as f:
        json.dump(info, f, indent=1, sort_keys=True)
    print("wrote", out, "layers:", info.get("layers", "n/a"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
