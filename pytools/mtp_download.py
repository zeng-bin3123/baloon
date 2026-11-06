"""Fetching safetensors shards (local path, file:// or http(s) with Range) and inventorying their tensors."""
import json
import os
import struct
import sys
import urllib.request

DEFAULT_FILTER = "mtp"
INDEX_NAME = "model.safetensors.index.json"


def _is_url(s):
    return s.startswith("http://") or s.startswith("https://")


def _local(s):
    return s[7:] if s.startswith("file://") else s


def get(src, start=0, end=None, timeout=60):
    """Bytes [start, end) of src (end=None: to EOF). Local paths are read directly; URLs use an HTTP Range request."""
    if start < 0 or (end is not None and end < start):
        raise ValueError("bad range")
    if end is not None and end == start:
        return b""
    if _is_url(src):
        hdr = {"Range": "bytes=%d-%s" % (start, "" if end is None else end - 1)}
        with urllib.request.urlopen(urllib.request.Request(src, headers=hdr), timeout=timeout) as r:
            data = r.read()
            if r.status == 200 and (start or end is not None):      # server ignored Range: slice locally
                data = data[start:end]
        if end is not None and len(data) != end - start:
            raise IOError("short read from %s: %d of %d bytes" % (src, len(data), end - start))
        return data
    with open(_local(src), "rb") as f:
        f.seek(start)
        data = f.read(-1 if end is None else end - start)
    if end is not None and len(data) != end - start:
        raise IOError("short read from %s" % src)
    return data


def shard_header(src):
    """Safetensors header of a shard: (tensors dict, data_start). Tensor offsets are relative to data_start."""
    (n,) = struct.unpack("<Q", get(src, 0, 8))
    if n > (1 << 28):
        raise ValueError("implausible safetensors header length %d" % n)
    hdr = json.loads(get(src, 8, 8 + n).decode("utf-8"))
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


def inventory(base, name_filter=DEFAULT_FILTER):
    """Tensors (whose name contains name_filter) of a checkpoint: name -> dict(shard, dtype, shape, begin, end).

    `base` is a directory/URL prefix holding model.safetensors.index.json, or a single .safetensors file."""
    if base.endswith(".safetensors"):
        shards = {base: None}
        join = lambda s: s
    else:
        sep = "" if base.endswith("/") else "/"
        join = lambda s: base + sep + s
        try:
            wm = json.loads(get(join(INDEX_NAME)).decode("utf-8"))["weight_map"]
        except (IOError, OSError, ValueError, KeyError):
            wm = None
        if wm is None:
            raise IOError("no %s under %s" % (INDEX_NAME, base))
        shards = {}
        for tname, shard in wm.items():
            if name_filter in tname:
                shards.setdefault(join(shard), set()).add(tname)
    inv = {}
    for src, wanted in shards.items():
        hdr, data_start = shard_header(src)
        for tname, meta in hdr.items():
            if name_filter in tname and (wanted is None or tname in wanted):
                b, e = meta["data_offsets"]
                inv[tname] = {"shard": src, "dtype": meta["dtype"], "shape": meta["shape"],
                              "begin": data_start + b, "end": data_start + e}
    return inv


def fetch(src, dest, chunk=1 << 22, begin=0, end=None):
    """Copy [begin, end) of src to dest, resuming a partial dest.part; renames to dest when complete. Returns bytes."""
    if end is None:
        if _is_url(src):
            with urllib.request.urlopen(urllib.request.Request(src, method="HEAD"), timeout=60) as r:
                end = int(r.headers["Content-Length"])
        else:
            end = os.path.getsize(_local(src))
    total = end - begin
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have > total:
        have = 0
    with open(part, "r+b" if have else "wb") as f:
        f.seek(have)
        pos = begin + have
        while pos < end:
            n = min(chunk, end - pos)
            f.write(get(src, pos, pos + n))
            pos += n
        f.truncate(total)
    os.replace(part, dest)
    return total


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 2 and argv[0] == "inventory":
        inv = inventory(argv[1], argv[2] if len(argv) > 2 else DEFAULT_FILTER)
        for k in sorted(inv):
            v = inv[k]
            print("%s %s %s %d" % (k, v["dtype"], v["shape"], v["end"] - v["begin"]))
        return 0
    if len(argv) == 3 and argv[0] == "fetch":
        print("fetched", fetch(argv[1], argv[2]), "bytes")
        return 0
    print("usage: mtp_download.py inventory BASE [FILTER] | fetch SRC DEST", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
