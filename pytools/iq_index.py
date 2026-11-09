"""Tensor index of an IQ-quantised checkpoint: where every tensor's bytes live (file, absolute offset, size).

A standalone index describes one GGUF. A layered index starts from a base GGUF and lets an overlay GGUF replace
tensors (typically the routed experts re-quantised to an IQ type); every tensor ends up with exactly one source."""
import json
import sys

from . import gguf_file as gf

VERSION = 1


def is_expert(name):
    """Routed-expert tensors: fused `_exps` GGUF names or `.experts.` checkpoint names."""
    return ".experts." in name or "_exps" in name


def tensor_bytes(type_id, shape):
    """Byte size of a tensor of ggml type `type_id` and `shape`; ValueError if not whole blocks."""
    n = 1
    for d in shape:
        n *= int(d)
    return gf.tensor_nbytes(type_id, n)


def _entries(path, fileno):
    out = {}
    with gf.GGUFFile(path) as g:
        for t in g.tensors:
            out[t.name] = {"name": t.name, "type": t.type, "shape": [int(d) for d in t.shape],
                           "file": fileno, "offset": g.data_offset + t.offset,
                           "nbytes": tensor_bytes(t.type, t.shape), "expert": is_expert(t.name)}
    return out


def index_standalone(path):
    """Index of a single GGUF."""
    ents = _entries(path, 0)
    return {"version": VERSION, "files": [str(path)], "tensors": list(ents.values())}


def index_from_base(base, overlay):
    """Index over base with overlay tensors replacing same-named base tensors (shapes must agree)."""
    b, o = _entries(base, 0), _entries(overlay, 1)
    for name, e in o.items():
        if name in b and b[name]["shape"] != e["shape"]:
            raise ValueError("overlay tensor %s has shape %s, base has %s" % (name, e["shape"], b[name]["shape"]))
    merged = dict(b)
    merged.update(o)
    return {"version": VERSION, "files": [str(base), str(overlay)], "tensors": list(merged.values())}


def validate(index):
    """Raises ValueError unless the index is self-consistent."""
    if index.get("version") != VERSION:
        raise ValueError("unsupported index version")
    seen = set()
    for e in index["tensors"]:
        if e["name"] in seen:
            raise ValueError("duplicate tensor %s" % e["name"])
        seen.add(e["name"])
        if not (0 <= e["file"] < len(index["files"])):
            raise ValueError("tensor %s names a missing file" % e["name"])
        if e["nbytes"] != tensor_bytes(e["type"], e["shape"]):
            raise ValueError("tensor %s size does not match its type/shape" % e["name"])
        if e["expert"] != is_expert(e["name"]):
            raise ValueError("tensor %s has a wrong expert flag" % e["name"])


def write_index(path, index):
    validate(index)
    with open(path, "w") as f:
        json.dump(index, f, indent=1)


def read_index(path):
    with open(path) as f:
        index = json.load(f)
    validate(index)
    return index


class Model:
    """Read-only view of a validated tensor index: lookup by name, expert/dense split, byte totals per source file."""

    def __init__(self, index):
        validate(index)
        self.files = list(index["files"])
        self.tensors = {e["name"]: e for e in index["tensors"]}

    @classmethod
    def load(cls, path):
        return cls(read_index(path))

    def __len__(self):
        return len(self.tensors)

    def __contains__(self, name):
        return name in self.tensors

    def get(self, name):
        """Index entry of `name`; KeyError if absent."""
        return self.tensors[name]

    def experts(self):
        return [e for e in self.tensors.values() if e["expert"]]

    def dense(self):
        return [e for e in self.tensors.values() if not e["expert"]]

    def bytes_by_file(self):
        out = [0] * len(self.files)
        for e in self.tensors.values():
            out[e["file"]] += e["nbytes"]
        return out

    def total_bytes(self):
        return sum(self.bytes_by_file())

    def read(self, name):
        """Raw bytes of tensor `name`, read from the file the index assigns to it."""
        e = self.tensors[name]
        with open(self.files[e["file"]], "rb") as f:
            f.seek(e["offset"])
            data = f.read(e["nbytes"])
        if len(data) != e["nbytes"]:
            raise ValueError("tensor %s is truncated in %s" % (name, self.files[e["file"]]))
        return data


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    try:
        if len(a) == 3 and a[0] == "standalone":
            write_index(a[2], index_standalone(a[1]))
        elif len(a) == 4 and a[0] == "base":
            write_index(a[3], index_from_base(a[1], a[2]))
        else:
            print("usage: iq_index.py standalone MODEL.gguf OUT.json | base BASE.gguf OVERLAY.gguf OUT.json",
                  file=sys.stderr)
            return 2
    except (ValueError, OSError, gf.GGUFError) as e:
        print("error:", e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
