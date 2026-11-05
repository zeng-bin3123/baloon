#!/bin/bash
# Unit p984d0d954f: port table (names defined in the file named) + file-backed GGUF loader invariants + mutation self-tests.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_p984_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread"
rc=0
echo "== port table =="
python3 - .einsteinOS/port/p984d0d954f.tsv <<'PY' || rc=1
import re, sys
bad = 0; seen = set()
for line in open(sys.argv[1]):
    f = line.rstrip("\n").split("\t")
    if len(f) < 4 or not f[0]: continue
    seen.add(f[1])
    try: t = open(f[2], errors="ignore").read()
    except OSError: print("port: missing file", f[2]); bad += 1; continue
    if f[1] == "Gguf": ok = re.search(r'(struct|class)\s+' + re.escape(f[3]) + r'(?![A-Za-z0-9_])', t)
    else: ok = re.search(r'inline\s+[\w:<>\s]+?\b' + re.escape(f[3]) + r'\s*\(', t)
    if not ok: print("port: %s not defined in %s" % (f[3], f[2])); bad += 1
if seen != {"Gguf", "_bf16_to_f32", "_dequant_flat", "_discover_shards"}: print("port table wrong:", seen); bad += 1
sys.exit(1 if bad else 0)
PY
bt() { (cd "$1" && $HOSTCXX $HF -Iinclude tests/k26_gguf_file/test_gguf_file.cpp -o "$2"); }
echo "== unit test =="
bt "$ROOT" "$OUT/t26" || { echo "BUILD FAILED (unit test)"; exit 1; }
"$OUT/t26" | tee "$OUT/t26.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "unit tests FAILED"; rc=1; }
grep -q "UNIT 26 TESTS PASSED" "$OUT/t26.log" || { echo "suite did not report PASSED"; rc=1; }
echo "== mutation self-tests =="
mutate() {
  local name="$1" file="$2" expr="$3" M="$OUT/mut_$1"
  rm -rf "$M"; mkdir -p "$M"; cp -r include tests src "$M/"
  sed -i "$expr" "$M/$file"
  if cmp -s "$file" "$M/$file"; then echo "mutation $name did not apply"; rc=1; return; fi
  if ! bt "$M" "$M/t" >/dev/null 2>&1; then echo "mutant $name failed to build"; rc=1; return; fi
  if "$M/t" >/dev/null 2>&1; then echo "MUTANT SURVIVED: $name"; rc=1; else echo "mutant $name detected"; fi
}
H=include/baloon/gguf_file_loader.hpp
L=include/baloon/gguf_lazy_loader.hpp
mutate bf16shift  $H 's|const uint32_t u = (uint32_t)src\[i\] << 16;|const uint32_t u = (uint32_t)src[i] << 15;|'
mutate bf16flat   $H 's|const uint32_t u = (uint32_t)artifact::read_u16(src + 2 \* i) << 16;|const uint32_t u = (uint32_t)artifact::read_u16(src + 2 * i) << 15;|'
mutate partial    $H 's/ || elements % g.el != 0) return false;/) return false;/'
mutate nullsrc    $H 's/if (g.el == 0 || !src || !out/if (g.el == 0 || !out/'
mutate blkstride  $H 's|fn(src + b \* g.by, out + b \* g.el)|fn(src + b * g.el, out + b * g.el)|'
mutate f16path    $H 's|out\[i\] = artifact::fp16_to_fp32(artifact::read_u16(src + 2 \* i));|out[i] = artifact::bf16_to_fp32(artifact::read_u16(src + 2 * i));|'
mutate q2type     $H 's|case gguf::kQ2_0: fn = artifact::dequantize_q2_0; break;|case gguf::kQ2_0: fn = artifact::dequantize_q4_0; break;|'
mutate shardstat  $H 's|if (::stat(s.c_str(), \&st) != 0 \|\| !S_ISREG(st.st_mode)) return std::nullopt;||'
mutate dupname    $H 's|if (where_.count(t.name)) {|if (false) {|'
mutate rowwidth   $H 's|if (!gguf::whole_blocks(t->shape\[0\], g.el)) return false;||'
mutate rowsf32n   $H 's|const uint64_t n = idx.size() \* t->shape\[0\];|const uint64_t n = idx.size() * t->shape[0] / 2;|'
mutate expf32n    $H 's|const uint64_t n = sp->len / g.by \* g.el;|const uint64_t n = sp->len / g.by * g.el / 2;|'
mutate closeclear $H 's|shards_.clear(); where_.clear();|shards_.clear();|'
mutate rowperm    $H 's|auto sp = shards_\[where_.at(name)\]->img->read_rows(name, r, 1);|auto sp = shards_[where_.at(name)]->img->read_rows(name, 0, 1);|'
mutate expstart   $L 's|return e \* expert_bytes(per, el, by); }|return e * (per / el) * by + by; }|'
mutate expbytes   $L 's|inline uint64_t expert_bytes(uint64_t per, uint64_t el, uint64_t by) { return per / el \* by; }|inline uint64_t expert_bytes(uint64_t per, uint64_t el, uint64_t by) { return per / el * by + 1; }|'
mutate q2geom     $L 's|case kQ2_0: return {64, 18};|case kQ2_0: return {64, 16};|'
mutate shardname  $L 's|"-%05u-of-%05u.gguf"|"-%04u-of-%04u.gguf"|'
mutate shardmiss  $L 's|if (!ok) return std::nullopt;  // missing shard -> rejected|if (!ok) continue;|'
[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
