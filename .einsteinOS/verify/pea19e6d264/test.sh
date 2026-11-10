#!/bin/bash
# Unit pea19e6d264: port table + dequant CLI invariants (plain formats, block formats, error codes) + mutants.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_pea1_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -Iinclude"
rc=0
echo "== port table =="
python3 - .einsteinOS/port/pea19e6d264.tsv <<'PY' || rc=1
import re, sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip()]
bad = 0
if rows != [["src/artifact/dequant.cpp", "main", "src/artifact/dequant.cpp", "main"]]:
    print("port table wrong:", rows); bad = 1
for f in rows:
    t = open(f[2], errors="ignore").read()
    if not re.search(r'\bint\s+' + re.escape(f[3]) + r'\s*\(', t): print("port: not defined", f); bad = 1
sys.exit(bad)
PY
build() { $HOSTCXX $HF "$1/src/artifact/dequant.cpp" -o "$2"; }
echo "== build CLI =="
build "$ROOT" "$OUT/dequant" || { echo "BUILD FAILED"; exit 1; }
cat > "$OUT/check.py" <<'PY'
import struct, subprocess, sys, os
exe, d = sys.argv[1], sys.argv[2]
fails = 0
def run(t, data):
    i, o = os.path.join(d, "in.bin"), os.path.join(d, "out.f32")
    open(i, "wb").write(data)
    if os.path.exists(o): os.remove(o)
    p = subprocess.run([exe, t, i, o], capture_output=True)
    return p.returncode, (open(o, "rb").read() if os.path.exists(o) else None), p.stdout.decode()
def check(name, cond):
    global fails
    if not cond: print("FAIL:", name); fails += 1
vals = [1.5, -2.25, 0.0, 1e-3, 3.0e5]
rcode, out, so = run("f32", struct.pack("<5f", *vals))
check("f32 passthrough", rcode == 0 and out == struct.pack("<5f", *vals) and so.strip() == "5 values")
h = [0x3C00, 0xC000, 0x0000, 0x3555, 0xBC00]
exp = [1.0, -2.0, 0.0, struct.unpack("<f", struct.pack("<f", 0.333251953125))[0], -1.0]
rcode, out, _ = run("f16", struct.pack("<5H", *h))
check("f16 values", rcode == 0 and list(struct.unpack("<5f", out)) == exp)
b = [0x3F80, 0xC000, 0x0000, 0x3F00]
rcode, out, _ = run("bf16", struct.pack("<4H", *b))
check("bf16 values", rcode == 0 and list(struct.unpack("<4f", out)) == [1.0, -2.0, 0.0, 0.5])
# q8_0: 34-byte blocks, d (fp16) * int8; two blocks, output 64 floats in block order
blk = lambda dh, q: struct.pack("<H", dh) + bytes((x & 255) for x in q)
q1 = [i - 16 for i in range(32)]; q2 = [(-1) ** i * i for i in range(32)]
rcode, out, so = run("q8_0", blk(0x3800, q1) + blk(0x4000, q2))
want = [0.5 * x for x in q1] + [2.0 * x for x in q2]
check("q8_0 values", rcode == 0 and list(struct.unpack("<64f", out)) == want and so.strip() == "64 values")
rcode, out, _ = run("q8_0", blk(0x3C00, q1) + b"\0")
check("q8_0 partial block rejected", rcode == 1)
rcode, out, _ = run("q4_0", b"\0" * 18)
check("q4_0 zero-d block -> 32 zeros", rcode == 0 and list(struct.unpack("<32f", out)) == [0.0] * 32)
rcode, out, _ = run("f32", b"\0\0\0")
check("f32 size not multiple rejected", rcode == 1)
rcode, out, _ = run("f16", b"\0")
check("f16 size not multiple rejected", rcode == 1)
rcode, out, _ = run("nosuch", b"\0" * 8)
check("unknown type exit 2", rcode == 2)
rcode, out, _ = run("f32", b"")
check("empty input -> 0 values", rcode == 0 and out == b"")
p = subprocess.run([exe, "f32"], capture_output=True); check("usage exit 2", p.returncode == 2)
p = subprocess.run([exe, "f32", os.path.join(d, "missing"), os.path.join(d, "o")], capture_output=True)
check("missing input exit 1", p.returncode == 1)
sys.exit(1 if fails else 0)
PY
echo "== CLI invariants =="
python3 "$OUT/check.py" "$OUT/dequant" "$OUT" || { echo "CLI invariants FAILED"; rc=1; }
echo "== mutants =="
mutate() {
  local name="$1" expr="$2" M="$OUT/mut_$1"
  rm -rf "$M"; mkdir -p "$M"; cp -r include src "$M/"
  sed -i "$expr" "$M/src/artifact/dequant.cpp"
  if cmp -s src/artifact/dequant.cpp "$M/src/artifact/dequant.cpp"; then echo "mutation $name did not apply"; rc=1; return; fi
  if ! (cd "$M" && $HOSTCXX $HF src/artifact/dequant.cpp -o "$M/d") >/dev/null 2>&1; then echo "mutant $name failed to build"; rc=1; return; fi
  if python3 "$OUT/check.py" "$M/d" "$M" >/dev/null 2>&1; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant $name detected"; fi
}
mutate blocksize 's/out.resize(nb \* bt->elems)/out.resize(nb * bt->elems + 0); nb2: ;/;s/in.data() + b \* bt->bytes/in.data() + b * (bt->bytes + 0) + 1/'
mutate partial   's/if (in.size() % bt->bytes)/if (0)/'
mutate unknown   's/return 2; }\n        if (in.size() % bt/X/;s/unknown type %s\\n", type); return 2;/unknown type %s\\n", type); return 1;/'
mutate usage     's/argc != 4/argc != 5/'
mutate elem      's/elem_bytes = 2; plain = dequantize_f16/elem_bytes = 4; plain = dequantize_f16/'
exit $rc
