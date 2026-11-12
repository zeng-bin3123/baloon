#!/bin/bash
# Unit u883f305326 tests: the 6 required q2_avx2 rows exist and every port row names a definition in its file; the CPU suite (Q2Avx2Fallback and
# Q2ExpertDotKernel, AVX-512 sections only where the CPU has VNNI/VBMI); the GPU suite, which launches
# bf16_gemv_fp32_mmvf in all 8 block-size cases and checks the host contract (Bf16MmvfBlockSelect).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/build/u883f305326"
export CUDA_VISIBLE_DEVICES="${EINSTEINOS_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
rc=0

echo "== port table =="
TSV=.einsteinOS/port/u883f305326.tsv
# The 6 definitions the unit requires (reference q2_avx2.cpp) must each have a row; extra rows (bf16_native.cu) are allowed.
for req in act_quant_q8_1_avx2 h2f q2_0_gguf_rows_multi_avx2 row_multi rows unpack64; do
  awk -F'\t' -v r="$req" '$1=="src/kernels/cpu/q2_avx2.cpp" && $2==r { f=1 } END { exit !f }' "$TSV" \
    || { echo "port: required q2_avx2.cpp definition $req has no row"; rc=1; }
done
while IFS="$(printf '\t')" read -r rf rn mf mn; do
  [ -n "$rf" ] || continue
  if [ ! -f "$mf" ] || ! grep -qwF -- "$mn" "$mf"; then echo "port: $mn not defined in $mf"; rc=1; fi
done < "$TSV"

echo "== kernel_unported (local) =="
# Same rule as the kernel's gate: a non-test file this unit changed that defines __global__ kernels must be a file the
# port map writes a counterpart into (column 3). The changed files come from changed_files.txt (works in shallow
# clones), plus git diff against unit 5's verified commit bff4eba7 when that commit is in history. Never skipped.
MANIFEST=.einsteinOS/verify/u883f305326/changed_files.txt
BASE="${U883_BASE:-bff4eba7}"
unit_files() {
  { [ -f "$MANIFEST" ] && cat "$MANIFEST"
    git rev-parse -q --verify "$BASE^{commit}" >/dev/null 2>&1 && git diff --name-only "$BASE" -- src include
  } | sed '/^$/d' | sort -u
}
# unported <tsv>: prints every unit file that defines __global__ and is not a column-3 target of <tsv>.
unported() {
  local targets; targets=$(awk -F'\t' 'NF>=4 { print $3 }' "$1" | sort -u)
  while read -r f; do
    [ -f "$f" ] || continue
    case "$f" in tests/*|*test*) continue ;; esac
    grep -q '__global__' "$f" || continue
    grep -qxF "$f" <<<"$targets" || echo "$f"
  done <<<"$UNIT_FILES"
}
UNIT_FILES=$(unit_files)
# The manifest is what the kernel's shallow clone sees, so it must not drift: when the base is in history, every file
# git diff reports must also be listed in the manifest (an empty or stale manifest fails here, not only on the kernel).
if git rev-parse -q --verify "$BASE^{commit}" >/dev/null 2>&1; then
  for f in $(git diff --name-only "$BASE" -- src include); do
    grep -qxF "$f" "$MANIFEST" 2>/dev/null || { echo "manifest: $f changed since $BASE but is missing from $MANIFEST"; rc=1; }
  done
fi
if [ -z "$UNIT_FILES" ]; then
  echo "kernel_unported: no changed-file list ($MANIFEST missing and $BASE not in history)"; rc=1
else
  echo "unit files checked: $(wc -l <<<"$UNIT_FILES")"
  for f in $(unported "$TSV"); do echo "kernel_unported: $f defines __global__ kernels and has no port row"; rc=1; done
fi
# Negative control: with only the q2_avx2.cpp rows the check must flag bf16_native.cu; if it does not, the check is inert.
NEG=$(mktemp); awk -F'\t' '$1 == "src/kernels/cpu/q2_avx2.cpp"' "$TSV" > "$NEG"
if unported "$NEG" | grep -qxF src/kernels/gpu/bf16_native.cu; then echo "negative control: q2-only table flags bf16_native.cu (ok)"
else echo "negative control FAILED: kernel_unported check did not flag bf16_native.cu with a q2-only table"; rc=1; fi
rm -f "$NEG"

echo "== ISA content (EINSTEINOS_ISA_FLAGS: ${EINSTEINOS_ISA_FLAGS:--mavx2 -mf16c -mfma}) =="
# The unit requirement: the binaries carry AVX2 (ymm), F16C (vcvtph2ps/vcvtps2ph) and FMA (vfmadd/vfmsub/vfnm*) code.
OBJDUMP=""
for c in /usr/local/corex/bin/llvm-objdump objdump llvm-objdump; do command -v "$c" >/dev/null 2>&1 && { OBJDUMP="$c"; break; }; done
if [ -z "$OBJDUMP" ]; then echo "no objdump; ISA check failed"; rc=1; else
  for b in "$OUT/q2_avx2.o" "$OUT/test_cpu"; do
    d=$("$OBJDUMP" -d "$b" 2>/dev/null)
    a=$(grep -c 'ymm' <<<"$d"); h=$(grep -cE 'vcvtph2ps|vcvtps2ph' <<<"$d"); m=$(grep -cE 'vf(n)?m(add|sub)[0-9]{3}' <<<"$d")
    echo "$(basename "$b"): avx2(ymm)=$a f16c=$h fma=$m"
    [ "$a" -gt 0 ] && [ "$h" -gt 0 ] && [ "$m" -gt 0 ] || { echo "ISA: $b lacks AVX2/F16C/FMA instructions"; rc=1; }
  done
fi

echo "== CPU tests =="
"$OUT/test_cpu" | tee "$OUT/test_cpu.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "CPU tests FAILED"; rc=1; }
grep -q "U883F305326 CPU TESTS PASSED" "$OUT/test_cpu.log" || { echo "CPU suite did not report PASSED"; rc=1; }

echo "== device tests (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) =="
ls /dev | grep -qE '^(iluvatar|nvidia)[0-9]+$' || { echo "no GPU device node"; rc=1; }
"$OUT/test_gpu" | tee "$OUT/test_gpu.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "GPU tests FAILED"; rc=1; }
grep -q "U883F305326 GPU TESTS PASSED" "$OUT/test_gpu.log" || { echo "GPU suite did not report PASSED"; rc=1; }

[ $rc -eq 0 ] && echo "unit u883f305326 tests OK"
exit $rc
