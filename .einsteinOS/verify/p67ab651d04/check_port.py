import re, sys
kinds = {}
for l in open(sys.argv[2]):
    p = l.rstrip("\n").split("\t")
    if len(p) == 3: kinds[(p[0], p[1])] = p[2]
def strip(t):
    t = re.sub(r"/\*.*?\*/", "", t, flags=re.S)
    return re.sub(r"//[^\n]*", "", t)
def pat(name, k):
    e = re.escape(name)
    if k == "macro": return r"^\s*#\s*define\s+" + e + r"\b"
    if k == "type": return r"\b(struct|class)\s+" + e + r"\b"
    if k == "const": return r"\b" + e + r"\s*(=|\{)"
    if k == "kernel": return r"__global__[^;{]*\b" + e + r"\s*\("
    return r"\b" + e + r"\s*\("
bad = 0; n = 0; seen = set()
for line in open(sys.argv[1]):
    f = line.rstrip("\r\n").split("\t")
    if len(f) != 4 or not f[0]: print("port: malformed row", repr(line)); bad += 1; continue
    n += 1; seen.add((f[0], f[1]))
    if not f[0].startswith("src/kernels/cuda/") or not f[0].endswith(".cu"): print("port: unexpected reference file", f[0]); bad += 1
    try: t = strip(open(f[2], errors="ignore").read())
    except OSError: print("port: missing file", f[2]); bad += 1; continue
    k = kinds.get((f[0], f[1]), "func")
    if not re.search(pat(f[3], k), t, re.M | re.S): print("port: %s not defined in %s" % (f[3], f[2])); bad += 1
if n != 111 or len(seen) != 111: print("port table has %d rows (%d distinct), expected 111" % (n, len(seen))); bad += 1
print("port rows:", n, "bad:", bad)
sys.exit(1 if bad else 0)
