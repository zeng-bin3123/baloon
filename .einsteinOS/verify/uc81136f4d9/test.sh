#!/bin/bash
# Unit uc81136f4d9: builds and runs tests/unit18 (host). It asserts the LAYER 1 statements of the eight modules:
#  ExpertCache   - layer ranges tile [0,slots), admission never evicts / stays in its range, flat index recovery, 256-aligned
#                  sized slots, fill length bound, profile validation;
#  ExpertSource  - expert axis checked separately (no cross-layer alias), exact file size, arena slack, jobs + gpu == k, PCIe
#                  share = last m misses, 16-expert GGUF chunks tiling [gate|up|down];
#  LayerComposition - 16-byte carving, Q8_K sizing, plane partition, activation dispatch, stage prefixes, verify-window
#                  conjunction, fused write-back, PLE gating, 12/36 layer split, dump layout;
#  MtpDrafter    - window/cap rules, step-row staging, probability-gated chain vs the fuel model, chunked prefill, name+kind lookup;
#  SessionGraphs - align256 sizing and carving, GDN/QSA state indices, telescoping stage times, ring numbering, spin boundary;
#  Verifier      - ring/group tiling, monotone flag under contention, accepted prefix, commit vector, fetch budget;
#  WeightLoader  - segment totals, fp16 widening, promoted bf16/f16 rows, chunk loop, placement/compaction, kind mapping;
#  Prefill       - chunk partition, bump allocator exactness, staging rings, counting-sort grouping, GDN state layout;
#  ServeEngine   - admission, T/DONE protocol, 64-entry cache, authorization, finish reasons.
# It also checks the port table. Exits non-zero if any build or check fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_uc81_build}"
mkdir -p "$OUT"
HOSTCXX=""
for c in /usr/local/corex/bin/clang++ g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
rc=0
HF="-std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off -pthread -Iinclude"

echo "== library sources (warnings are errors) =="
for f in src/core/*.cpp src/prefill/prefill_host.cpp; do
  $HOSTCXX $HF -c "$f" -o "$OUT/obj_$(basename "$f").o" || { echo "BUILD FAILED ($f)"; exit 1; }
done

echo "== unit18 (host) =="
$HOSTCXX $HF -Itests/unit18 tests/unit18/*.cpp -o "$OUT/t18" || { echo "BUILD FAILED (unit18)"; exit 1; }
timeout 900 "$OUT/t18" || { echo "unit18 FAILED (exit $?)"; rc=1; }

printf 'int main(){return 0;}\n' > "$OUT/san_probe.cpp"
if $HOSTCXX -fsanitize=address,undefined "$OUT/san_probe.cpp" -o "$OUT/san_probe" >/dev/null 2>&1 && "$OUT/san_probe" >/dev/null 2>&1; then
  echo "== unit18 (ASan/UBSan) =="
  $HOSTCXX -std=c++17 -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=undefined -pthread \
    -Iinclude -Itests/unit18 tests/unit18/*.cpp -o "$OUT/t18_san" || { echo "SAN BUILD FAILED"; exit 1; }
  timeout 1500 "$OUT/t18_san" || { echo "unit18 sanitizer run FAILED (exit $?)"; rc=1; }
else
  echo "sanitizers unavailable with $HOSTCXX: skipped"
fi

echo "== port table =="
TSV=.einsteinOS/port/uc81136f4d9.tsv
[ "$(wc -l < "$TSV")" -eq 285 ] || { echo "port table must have 285 lines"; rc=1; }
while IFS=$'\t' read -r rf rn lf ln; do
  [ -n "$ln" ] && [ -f "$lf" ] && grep -qw -- "$ln" "$lf" || { echo "port MISSING: $rn -> $lf:$ln"; rc=1; }
done < "$TSV"

[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
