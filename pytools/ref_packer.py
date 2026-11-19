"""Reference-format weight packer — utility layer.

sha256, code_bits, is_expert, choose_scale_dtype: the four building-block
helpers that every higher-level operation in this module depends on.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

from . import canon_xcheck as cx
from . import planes

MAGIC = b"RPAK"
VERSION = 1
ALIGN = 64


def sha256(data):
    """Return the hex-encoded SHA-256 digest of *data* (bytes, bytearray, or a file path).

    When *data* is a path the file is read in 1 MiB chunks so arbitrarily large
    files can be hashed without loading them entirely into memory."""
    if isinstance(data, (str, Path)) or hasattr(data, "__fspath__"):
        h = hashlib.sha256()
        with open(data, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    return hashlib.sha256(bytes(data)).hexdigest()


def code_bits(type_name):
    """Return bits per code element for *type_name* (e.g. 'Q4_0' → 4, 'F16' → 16)."""
    name = cx.reference_name(type_name)
    if name in cx.MAPPINGS:
        return cx.MAPPINGS[name].bits
    _, kind, _ = cx.PASSTHROUGH[name]        # KeyError if unknown
    return 16 if kind == "P16" else 32


def is_expert(name):
    """Return True if *name* identifies a MoE expert tensor.

    Matches the naming conventions used by DeepSeek / Qwen MoE checkpoints:
    '.experts.' for the dense-expert weight block and '_exps' for the iq-packed
    overlay files produced by tools/iq_pack.py."""
    return ".experts." in name or "_exps" in name


def choose_scale_dtype(type_name):
    """Return 'f32' for float pass-through planes, 'f16' for quantised types.

    Pass-through planes (F16, BF16, F32) store their values directly — scales
    are meaningless.  Quantised types carry fp16 block scales."""
    name = cx.reference_name(type_name)
    return "f32" if name in cx.PASSTHROUGH else "f16"


def canonicalise(raw, type_name):
    """Convert raw GGUF bytes for *type_name* to the canonical (codes, scales, offsets) triple.

    For pass-through types (F16/BF16/F32) *scales* and *offsets* are None and
    *codes* holds the word array returned by the matching to_p* converter.
    For quantised types the returned Canon namedtuple has numpy arrays of dtype
    uint8 (codes), float32 (scales), float32 (offsets)."""
    name = cx.reference_name(type_name)
    if name in cx.PASSTHROUGH:
        _, kind, _ = cx.PASSTHROUGH[name]
        if kind == "P16":
            return cx.to_p16_f16(raw), None, None
        if name == "F32":
            return cx.to_p32_f32(raw), None, None
        return cx.to_p32_bf16(raw), None, None          # BF16
    canon = cx._encode(name, bytes(raw))
    return canon.codes, canon.scales, canon.offsets


def pack_codes(codes, bits):
    """Pack a 1-D or 2-D uint8 *codes* array into a byte array, *bits* bits per element.

    Thin wrapper around planes.pack_codes so callers that only import ref_packer
    do not also need to import planes."""
    arr = np.asarray(codes, dtype=np.uint8)
    return planes.pack_codes(arr.reshape(-1), bits)


def unpack_codes(packed, bits, shape=None):
    """Inverse of pack_codes.  Optionally reshape the output to *shape*."""
    out = planes.unpack_codes(packed, bits)
    if shape is not None:
        out = out.reshape(shape)
    return out


def tensor_entry(name, type_name, n_elements, raw_sha256, offset, nbytes_packed):
    """Create a tensor-table entry dict for the ref-pack format.

    Fields:
      name          – tensor name as it appears in the GGUF metadata
      type          – normalised type name (e.g. 'Q4_0', 'BF16')
      n_elements    – total number of scalar values
      sha256        – hex SHA-256 of the raw GGUF block bytes (for verify())
      offset        – byte offset of this tensor's packed data inside the pack file
      nbytes        – byte length of the packed data
      expert        – True if this is a MoE expert tensor
      scale_dtype   – 'f16' for quantised types, 'f32' for pass-through planes
      bits          – code plane width in bits per element
    """
    return {
        "name": str(name),
        "type": cx.reference_name(type_name),
        "n_elements": int(n_elements),
        "sha256": str(raw_sha256),
        "offset": int(offset),
        "nbytes": int(nbytes_packed),
        "expert": is_expert(str(name)),
        "scale_dtype": choose_scale_dtype(type_name),
        "bits": code_bits(type_name),
    }


def decode_entry(entry, data):
    """Decode packed *data* back to float32 values using the metadata in *entry*.

    *data* must be the raw GGUF block bytes for the tensor (the same bytes that
    were fed to sha256() when the entry was created).  The return value is a
    flat float32 numpy array of length entry['n_elements']."""
    type_name = entry["type"]
    name = cx.reference_name(type_name)
    if name in cx.PASSTHROUGH:
        _, kind, _ = cx.PASSTHROUGH[name]
        if kind == "P16":
            return cx.from_p16_f16(np.frombuffer(bytes(data), dtype=np.uint16))
        if name == "F32":
            return cx.from_p32_f32(np.frombuffer(bytes(data), dtype=np.uint32))
        return cx.from_p32_bf16(np.frombuffer(bytes(data), dtype=np.uint32))
    return cx.reference_values(type_name, data).reshape(-1)


def rebuild_role(blob, role):
    """Extract a single role (gate | up | down) from an MTP expert blob.

    The blob is the concatenation of three equal-sized role byte strings in the
    order gate | up | down — matching the layout produced by layer_pack.build.
    Returns the bytes for *role* only.

    Raises ValueError if the blob length is not divisible by 3 or *role* is not
    one of 'gate', 'up', 'down'."""
    blob = bytes(blob)
    n = len(blob)
    if n % 3 != 0:
        raise ValueError("blob length %d is not divisible by 3" % n)
    role_bytes = n // 3
    idx = {"gate": 0, "up": 1, "down": 2}
    if role not in idx:
        raise ValueError("unknown role %r; expected 'gate', 'up', or 'down'" % role)
    start = idx[role] * role_bytes
    return blob[start:start + role_bytes]


def info(src_gguf):
    """Return a dict describing the quantised tensors in *src_gguf*.

    The dict has keys:
      path       – absolute path
      n_tensors  – number of tensors in the file
      n_quant    – number of quantised (non-passthrough) tensors
      n_expert   – number of MoE expert tensors
      types      – {type_name: count} frequency table
    """
    from . import gguf_file as gf
    type_counts = {}
    n_quant = 0
    n_expert = 0
    with gf.GGUFFile(src_gguf) as g:
        tensors = list(g.tensors)
        for t in tensors:
            tname = gf.type_name(t.type)
            type_counts[tname] = type_counts.get(tname, 0) + 1
            if cx.reference_name(tname) not in cx.PASSTHROUGH:
                n_quant += 1
            if is_expert(t.name):
                n_expert += 1
    return {
        "path": str(src_gguf),
        "n_tensors": len(tensors),
        "n_quant": n_quant,
        "n_expert": n_expert,
        "types": type_counts,
    }


# ---------------------------------------------------------------------------
# Pack-level operations: build, verify, verify_experts, main
# ---------------------------------------------------------------------------

def build(src_gguf, tensor_names=None):
    """Return a list of tensor_entry dicts for all (or selected) tensors in *src_gguf*.

    *tensor_names*: optional iterable of names; default is every tensor in the GGUF.
    Each entry has the sha256 of the raw GGUF block bytes so verify() can later
    confirm that the source file has not changed.

    The function does not write any file; callers that need a serialised form
    should JSON-dump the returned list."""
    from . import gguf_file as gf
    want = set(tensor_names) if tensor_names is not None else None
    entries = []
    cursor = 0
    with gf.GGUFFile(src_gguf) as g:
        for t in g.tensors:
            if want is not None and t.name not in want:
                continue
            raw = g.tensor_bytes(t.name)
            tname = gf.type_name(t.type)
            nbytes = len(raw)
            e = tensor_entry(t.name, tname, t.n_elements, sha256(raw), cursor, nbytes)
            entries.append(e)
            # advance cursor to next ALIGN boundary
            cursor += nbytes
            pad = (ALIGN - cursor % ALIGN) % ALIGN
            cursor += pad
    return entries


def verify(src_gguf, entries):
    """Verify that the GGUF at *src_gguf* still matches a previously built entry list.

    Reads every tensor listed in *entries* and re-computes its SHA-256; mismatches
    are reported as strings in the returned list (empty list = everything matches)."""
    from . import gguf_file as gf
    fails = []
    with gf.GGUFFile(src_gguf) as g:
        by_name = {t.name: t for t in g.tensors}
        for e in entries:
            name = e["name"]
            if name not in by_name:
                fails.append("%s: tensor not found in GGUF" % name)
                continue
            raw = g.tensor_bytes(name)
            got = sha256(raw)
            if got != e["sha256"]:
                fails.append("%s: sha256 mismatch (stored %s, computed %s)" % (name, e["sha256"], got))
    return fails


def verify_experts(src_gguf, entries):
    """Like verify() but restricted to entries where entry['expert'] is True.

    Useful for a quick sanity check after an expert overlay has been applied
    without having to re-verify all the dense weights."""
    return verify(src_gguf, [e for e in entries if e.get("expert")])


def main(argv=None):
    """CLI: build a ref-pack entry list from a GGUF and print or save it as JSON."""
    import json
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: ref_packer.py <src.gguf> [out.json]", file=sys.stderr)
        print("  Builds the tensor entry table for <src.gguf> and writes JSON.", file=sys.stderr)
        return 2
    src = argv[0]
    entries = build(src)
    out = json.dumps(entries, indent=2)
    if len(argv) >= 2:
        with open(argv[1], "w") as f:
            f.write(out + "\n")
        print("wrote %d entries to %s" % (len(entries), argv[1]))
    else:
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
