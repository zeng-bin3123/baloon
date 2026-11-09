"""Canonical pack of a quantised checkpoint: every tensor becomes code/scale/offset planes (64-byte aligned, codes bit-packed
LSB-first) plus an index entry carrying the plane geometry and a sha256 of each plane, so a pack can be re-verified without
the source file.

Entry = {name, type, shape, expert, kind, bits, scale_dtype, planes: {role: {offset, nbytes, count, sha256}}, sha256}
where `sha256` is the digest of the ORIGINAL raw tensor bytes. Roles: codes, scales, offsets (passthrough types: one
`codes` plane holding the 16/32-bit words)."""
import hashlib
import json
import os
import sys

import numpy as np

from . import canon_xcheck as cx
from . import gguf_file as gf
from . import iq_index
from . import planes

VERSION = 1
ROLES = ("codes", "scales", "offsets")
_WORD = {"F16": ("P16", 16), "BF16": ("P32", 32), "F32": ("P32", 32)}


def sha256(data):
    return hashlib.sha256(bytes(data)).hexdigest()


def is_expert(name):
    return iq_index.is_expert(name)


def code_bits(name):
    """Bits per stored code of a type (id or name): 2/4/8 for quantised types, 16/32 for passthrough words."""
    n = cx.reference_name(name)
    if n in _WORD:
        return _WORD[n][1]
    return cx.MAPPINGS[n].bits


def choose_scale_dtype(scales, offsets=None):
    """'f16' when every scale/offset survives a float16 round trip bit-exactly, else 'f32'."""
    for a in (scales, offsets):
        if a is None:
            continue
        a = np.asarray(a, dtype=np.float32)
        if not np.array_equal(a.astype(np.float16).astype(np.float32).view(np.uint32), a.view(np.uint32)):
            return "f32"
    return "f16"


def canonicalise(type_, raw):
    """raw ggml bytes -> {'name', 'bits', 'codes' (nb, elems | words), 'scales', 'offsets'} (scales/offsets None for words)."""
    n = cx.reference_name(type_)
    if n in _WORD:
        w = {"F16": cx.to_p16_f16, "BF16": cx.to_p32_bf16, "F32": cx.to_p32_f32}[n](raw)
        return {"name": n, "bits": _WORD[n][1], "codes": w, "scales": None, "offsets": None}
    c = cx._encode(n, raw)
    return {"name": n, "bits": cx.MAPPINGS[n].bits, "codes": c.codes, "scales": c.scales, "offsets": c.offsets}


def pack_codes(codes, bits):
    """Codes -> plane bytes: bit-packed for 2/4/8, little-endian words for 16/32."""
    if bits in (16, 32):
        return np.ascontiguousarray(np.asarray(codes, dtype="<u%d" % (bits // 8))).tobytes()
    return planes.pack_codes(codes, bits).tobytes()


def unpack_codes(data, bits):
    if bits in (16, 32):
        return np.frombuffer(bytes(data), dtype="<u%d" % (bits // 8)).copy()
    return planes.unpack_codes(np.frombuffer(bytes(data), dtype=np.uint8), bits)


def _fdt(d):
    if d not in ("f16", "f32"):
        raise ValueError("scale dtype must be f16 or f32")
    return "<f2" if d == "f16" else "<f4"


def tensor_entry(name, type_, shape, raw, writer=None):
    """Canonicalise one tensor, append its planes to `writer` (a planes.PlaneWriter; fresh one if None).
    Returns (entry, writer); plane offsets are relative to the writer's buffer."""
    n = int(np.prod([int(d) for d in shape]))
    if len(raw) != gf.tensor_nbytes(cx_type_id(type_), n):
        raise ValueError("tensor %s: %d bytes do not match type/shape" % (name, len(raw)))
    w = writer if writer is not None else planes.PlaneWriter()
    c = canonicalise(type_, raw)
    sd = None
    pl = {}
    pl["codes"] = _plane(w, pack_codes(c["codes"], c["bits"]), c["codes"].size)
    if c["scales"] is not None:
        sd = choose_scale_dtype(c["scales"], c["offsets"])
        pl["scales"] = _plane(w, np.ascontiguousarray(c["scales"].astype(_fdt(sd))).tobytes(), c["scales"].size)
        pl["offsets"] = _plane(w, np.ascontiguousarray(c["offsets"].astype(_fdt(sd))).tobytes(), c["offsets"].size)
    shape_l = [int(d) for d in shape]
    e = {"name": name, "type": c["name"], "shape": shape_l, "expert": is_expert(name),
         "kind": cx.MAPPINGS[c["name"]].kind if c["name"] in cx.MAPPINGS else _WORD[c["name"]][0],
         "bits": c["bits"], "scale_dtype": sd, "planes": pl, "sha256": sha256(raw)}
    e["blocks"] = int(c["codes"].shape[0]) if c["codes"].ndim == 2 else 0
    return e, w


def cx_type_id(t):
    n = cx.reference_name(t)
    return cx.PASSTHROUGH[n][0] if n in cx.PASSTHROUGH else cx.MAPPINGS[n].type


def _plane(w, data, count):
    off = w.append_plane(data)
    return {"offset": off, "nbytes": len(data), "count": int(count), "sha256": sha256(data)}


def _slice(entry, blob, role):
    p = entry["planes"][role]
    d = bytes(blob[p["offset"]:p["offset"] + p["nbytes"]])
    if len(d) != p["nbytes"]:
        raise ValueError("%s: plane %s is truncated" % (entry["name"], role))
    return d


def rebuild_role(entry, blob, role):
    """One canonical plane back from the blob: codes -> unpacked array, scales/offsets -> float32 (nb, groups)."""
    if role not in entry["planes"]:
        raise ValueError("%s has no %s plane" % (entry["name"], role))
    d = _slice(entry, blob, role)
    if role == "codes":
        c = unpack_codes(d, entry["bits"])
        if entry["bits"] in (16, 32):
            return c
        return c.reshape(entry["blocks"], -1)
    a = np.frombuffer(d, dtype=_fdt(entry["scale_dtype"])).astype(np.float32)
    return a.reshape(entry["blocks"], -1)


def decode_entry(entry, blob):
    """Float32 values of a packed tensor (same numbers as the reference dequantiser)."""
    n = entry["type"]
    codes = rebuild_role(entry, blob, "codes")
    if n in _WORD:
        return {"F16": cx.from_p16_f16, "BF16": cx.from_p32_bf16, "F32": cx.from_p32_f32}[n](codes)
    c = cx.Canon(codes, rebuild_role(entry, blob, "scales"), rebuild_role(entry, blob, "offsets"))
    return cx._decode(n, c).reshape(-1)


def build(tensors):
    """tensors: iterable of (name, type, shape, raw). Returns (index dict, blob bytes) with unique names."""
    w = planes.PlaneWriter()
    ents, seen = [], set()
    for name, type_, shape, raw in tensors:
        if name in seen:
            raise ValueError("duplicate tensor %s" % name)
        seen.add(name)
        e, _ = tensor_entry(name, type_, shape, raw, w)
        ents.append(e)
    return {"version": VERSION, "tensors": ents, "blob_bytes": len(w.buf)}, bytes(w.buf)


def _check_entry(e, blob):
    for role, p in e["planes"].items():
        if p["offset"] % planes.ALIGN:
            raise ValueError("%s: plane %s is not 64-byte aligned" % (e["name"], role))
        if sha256(_slice(e, blob, role)) != p["sha256"]:
            raise ValueError("%s: plane %s digest mismatch" % (e["name"], role))


def verify(index, blob):
    """Raises ValueError unless every plane of every entry is aligned, in range and matches its digest. Returns #entries."""
    if index.get("version") != VERSION:
        raise ValueError("unsupported pack version")
    if index.get("blob_bytes") != len(blob):
        raise ValueError("blob has %d bytes, index says %s" % (len(blob), index.get("blob_bytes")))
    names = [e["name"] for e in index["tensors"]]
    if len(set(names)) != len(names):
        raise ValueError("duplicate tensor in pack")
    for e in index["tensors"]:
        _check_entry(e, blob)
    return len(names)


def verify_experts(index, blob, expected=None):
    """Verify only the routed-expert entries: flags consistent with names, planes intact; optionally exactly `expected` of them."""
    n = 0
    for e in index["tensors"]:
        if e["expert"] != is_expert(e["name"]):
            raise ValueError("%s: wrong expert flag" % e["name"])
        if e["expert"]:
            _check_entry(e, blob)
            n += 1
    if expected is not None and n != expected:
        raise ValueError("%d expert tensors, expected %d" % (n, expected))
    return n


def info(index):
    """Summary: tensor count, expert count, plane bytes by role, type histogram."""
    out = {"tensors": len(index["tensors"]), "experts": 0, "types": {}, "plane_bytes": {r: 0 for r in ROLES},
           "blob_bytes": index.get("blob_bytes", 0)}
    for e in index["tensors"]:
        out["experts"] += 1 if e["expert"] else 0
        out["types"][e["type"]] = out["types"].get(e["type"], 0) + 1
        for r, p in e["planes"].items():
            out["plane_bytes"][r] += p["nbytes"]
    return out


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    try:
        if len(a) == 3 and a[0] == "pack":
            def gen():
                with gf.GGUFFile(a[1]) as g:
                    for t in g.tensors:
                        n = cx.reference_name(t.type) if t.type in gf.GEOMETRY else None
                        if n in cx.MAPPINGS or n in cx.PASSTHROUGH:
                            yield t.name, n, t.shape, g.tensor_bytes(t.name)
            index, blob = build(gen())
            os.makedirs(a[2], exist_ok=True)
            with open(os.path.join(a[2], "planes.bin"), "wb") as f:
                f.write(blob)
            with open(os.path.join(a[2], "pack.json"), "w") as f:
                json.dump(index, f, indent=1)
            print("packed", len(index["tensors"]), "tensors,", len(blob), "bytes")
        elif len(a) == 2 and a[0] == "verify":
            with open(os.path.join(a[1], "pack.json")) as f:
                index = json.load(f)
            with open(os.path.join(a[1], "planes.bin"), "rb") as f:
                blob = f.read()
            print("ok:", verify(index, blob), "tensors,", verify_experts(index, blob), "experts")
        elif len(a) == 2 and a[0] == "info":
            with open(os.path.join(a[1], "pack.json")) as f:
                print(json.dumps(info(json.load(f)), indent=1, sort_keys=True))
        else:
            print("usage: ref_pack.py pack MODEL.gguf OUT_DIR | verify DIR | info DIR", file=sys.stderr)
            return 2
    except (ValueError, KeyError, OSError, gf.GGUFError) as e:
        print("error:", e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
