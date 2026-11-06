"""Byte-level BPE tokenizer extracted from GGUF metadata (tokenizer.ggml.*), GPT-2 style."""
import json
import re
import sys

from . import gguf_file as gf

TYPE_NORMAL, TYPE_CONTROL, TYPE_USER_DEFINED = 1, 3, 4

# approximation of the GPT-2 pre-tokeniser with the stdlib `re` (unicode classes via \w)
_PRETOK = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+", re.UNICODE)


def bytes_to_unicode():
    """GPT-2 byte -> printable code point map: bytes 33..126, 161..172, 174..255 map to themselves, the other 68 to
    256 + n in increasing byte order."""
    keep = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    m, n = {}, 0
    for b in range(256):
        if b in keep:
            m[b] = chr(b)
        else:
            m[b] = chr(256 + n)
            n += 1
    return m


class Tokenizer:
    def __init__(self, tokens, merges, types=None):
        self.tokens = list(tokens)
        self.types = list(types) if types is not None else [TYPE_NORMAL] * len(self.tokens)
        if len(self.types) != len(self.tokens):
            raise ValueError("types and tokens differ in length")
        self.id_of = {t: i for i, t in enumerate(self.tokens)}
        self.rank = {}
        for r, m in enumerate(merges):
            a, _, b = m.partition(" ")
            self.rank.setdefault((a, b), r)
        self.b2u = bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}
        self.special = sorted((t for t, ty in zip(self.tokens, self.types) if ty in (TYPE_CONTROL, TYPE_USER_DEFINED)),
                              key=len, reverse=True)

    def _bpe(self, word):
        parts = list(word)
        while len(parts) > 1:
            best, bi = None, -1
            for i in range(len(parts) - 1):
                r = self.rank.get((parts[i], parts[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if bi < 0:
                break
            parts[bi:bi + 2] = [parts[bi] + parts[bi + 1]]      # each step shortens the word by exactly one
        return parts

    def _segments(self, text, parse_special):
        if not parse_special:
            specials = [t for t, ty in zip(self.tokens, self.types) if ty == TYPE_USER_DEFINED]
            specials.sort(key=len, reverse=True)
        else:
            specials = self.special
        if not specials:
            return [(text, False)]
        pat = re.compile("|".join(re.escape(s) for s in specials))
        out, pos = [], 0
        for m in pat.finditer(text):
            if m.start() > pos:
                out.append((text[pos:m.start()], False))
            out.append((m.group(0), True))
            pos = m.end()
        if pos < len(text):
            out.append((text[pos:], False))
        return out

    def encode(self, text, parse_special=False):
        ids = []
        for seg, is_special in self._segments(text, parse_special):
            if is_special:
                ids.append(self.id_of[seg])
                continue
            for piece in _PRETOK.findall(seg):
                mapped = "".join(self.b2u[b] for b in piece.encode("utf-8"))
                for p in self._bpe(mapped):
                    if p not in self.id_of:
                        raise KeyError("token %r missing from vocabulary" % p)
                    ids.append(self.id_of[p])
        return ids

    def decode_bytes(self, ids):
        out = bytearray()
        for i in ids:
            t = self.tokens[i]
            if self.types[i] in (TYPE_CONTROL, TYPE_USER_DEFINED):
                out += t.encode("utf-8")
            else:
                for ch in t:
                    out.append(self.u2b[ch])
        return bytes(out)

    def decode(self, ids):
        return self.decode_bytes(ids).decode("utf-8", errors="replace")


def extract(path):
    """Read tokenizer tables from a GGUF: dict(tokens, merges, types, ...). Raises ValueError if absent."""
    with gf.GGUFFile(path) as g:
        md = g.metadata
        if "tokenizer.ggml.tokens" not in md:
            raise ValueError("no tokenizer.ggml.tokens in %s" % path)
        n = len(md["tokenizer.ggml.tokens"])
        types = md.get("tokenizer.ggml.token_type")
        out = {"model": md.get("tokenizer.ggml.model"), "tokens": list(md["tokenizer.ggml.tokens"]),
               "merges": list(md.get("tokenizer.ggml.merges", [])),
               "types": [int(t) for t in types] if types is not None else [TYPE_NORMAL] * n}
        for k in ("bos_token_id", "eos_token_id", "padding_token_id", "unknown_token_id"):
            if "tokenizer.ggml." + k in md:
                out[k] = int(md["tokenizer.ggml." + k])
    if len(out["merges"]) >= len(out["tokens"]):
        raise ValueError("fewer merges than tokens expected")
    return out


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) < 1:
        print("usage: bpe_extract.py MODEL.gguf [OUT.json | --encode TEXT]", file=sys.stderr)
        return 2
    try:
        d = extract(a[0])
    except (ValueError, OSError, gf.GGUFError) as e:
        print("error:", e, file=sys.stderr)
        return 1
    if len(a) == 3 and a[1] == "--encode":
        t = Tokenizer(d["tokens"], d["merges"], d["types"])
        print(json.dumps(t.encode(a[2])))
    elif len(a) == 2:
        with open(a[1], "w") as f:
            json.dump(d, f)
    print("tokens: %d merges: %d" % (len(d["tokens"]), len(d["merges"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
