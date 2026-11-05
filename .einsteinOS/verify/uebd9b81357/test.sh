#!/bin/bash
# Unit uebd9b81357 tests. Asserts the LAYER 1 statements of F16BitsRounding (rneStep: below half keeps, above half bumps,
# tie lands even, never more than one ulp; inf/NaN versus finite overflow; subnormal normalisation and exponent re-bias;
# bf16 round-to-nearest-even) exhaustively / against a double-precision oracle on the host, and on the GPU checks that the
# device conversions are bit-identical to the host ones and that the Q8 KV cache, activation quantizers and QSA step
# kernels (step state, fp16 KV, indexer pooling and scoring, top-k, attention, gate) satisfy their invariants.
# Also: dequant CLI round trip, a mutation self-test (a broken rounding rule MUST make the host suite fail), port table.
# Exits non-zero if any build or any check fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_BUILD_DIR:-/tmp/baloon_unit_uebd9_build}"
mkdir -p "$OUT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
HOSTCXX=""
for c in "$CLANG" g++ clang++ c++; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then HOSTCXX="$c"; break; fi
done
[ -n "$HOSTCXX" ] || { echo "no C++ compiler found"; exit 2; }
HF="-std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off -pthread -Iinclude"
rc=0

echo "== library sources (warnings are errors) =="
for f in src/core/weights.cpp src/artifact/dequant.cpp; do
  $HOSTCXX $HF -c "$f" -o "$OUT/obj_$(basename "$f").o" || { echo "BUILD FAILED ($f)"; exit 1; }
done

echo "== host invariants (F16 bits, rounding oracle, bf16, dequantizers) =="
$HOSTCXX $HF tests/unit_f16bits/test_f16_bits.cpp -o "$OUT/t_f16bits" || { echo "BUILD FAILED (test_f16_bits)"; exit 1; }
timeout 1200 "$OUT/t_f16bits" | tee "$OUT/t_f16bits.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "host f16bits tests FAILED"; rc=1; }
grep -q "F16BITS UNIT TESTS PASSED" "$OUT/t_f16bits.log" || { echo "host suite did not report PASSED"; rc=1; }

echo "== mutation self-test: breaking the tie rule must be detected =="
MUT="$OUT/mut"; rm -rf "$MUT"; mkdir -p "$MUT"; cp -r include "$MUT/include"
sed -i 's/(rem == half \&\& (q \& 1u) == 1u)/(rem == half \&\& (q \& 1u) == 0u)/' "$MUT/include/baloon/f16_bits.hpp"
if cmp -s include/baloon/f16_bits.hpp "$MUT/include/baloon/f16_bits.hpp"; then echo "mutation did not apply"; rc=1
else
  $HOSTCXX -std=c++17 -O2 -ffp-contract=off -I"$MUT/include" tests/unit_f16bits/test_f16_bits.cpp -o "$OUT/t_f16bits_mut" 2>/dev/null \
    || { echo "mutant failed to build"; rc=1; }
  if [ -x "$OUT/t_f16bits_mut" ] && timeout 1200 "$OUT/t_f16bits_mut" >/dev/null 2>&1; then echo "MUTANT SURVIVED: tests cannot fail"; rc=1; else echo "mutant detected"; fi
fi

echo "== dequant CLI round trip =="
$HOSTCXX $HF src/artifact/dequant.cpp -o "$OUT/dequant" || { echo "BUILD FAILED (dequant CLI)"; exit 1; }
python3 - "$OUT" <<'PY' || rc=1
import struct, subprocess, sys, os, math
out = sys.argv[1]
def f16(x): return struct.unpack('<H', struct.pack('<e', x))[0]
def run(t, data):
    i, o = os.path.join(out, 'cli_in.bin'), os.path.join(out, 'cli_out.bin')
    open(i, 'wb').write(data)
    r = subprocess.run([os.path.join(out, 'dequant'), t, i, o], capture_output=True)
    if r.returncode != 0: return None
    b = open(o, 'rb').read()
    return list(struct.unpack('<%df' % (len(b) // 4), b))
bad = 0
# q8_0: two blocks
blk = lambda d, q: struct.pack('<H', f16(d)) + bytes((v & 0xFF) for v in q)
q1, q2 = list(range(-16, 16)), [(-1) ** i * i * 4 for i in range(32)]
got = run('q8_0', blk(0.25, q1) + blk(2.0, q2))
want = [0.25 * v for v in q1] + [2.0 * v for v in q2]
if got != want: print('q8_0 mismatch'); bad += 1
# f16 / bf16 / f32
vals = [0.0, 1.0, -2.5, 65504.0, 2.0 ** -24, 0.333251953125]
got = run('f16', b''.join(struct.pack('<H', f16(v)) for v in vals))
if got != vals: print('f16 mismatch', got); bad += 1
bf = [0.0, 1.0, -2.5, 3.0e38, 0.0078125]
got = run('bf16', b''.join(struct.pack('<H', struct.unpack('<I', struct.pack('<f', v))[0] >> 16) for v in bf))
want = [struct.unpack('<f', struct.pack('<I', (struct.unpack('<I', struct.pack('<f', v))[0] >> 16) << 16))[0] for v in bf]
if got != want: print('bf16 mismatch', got); bad += 1
got = run('f32', struct.pack('<3f', 1.5, -0.0, 1e-30))
if got is None or got[0] != 1.5 or got[2] != struct.unpack('<f', struct.pack('<f', 1e-30))[0]: print('f32 mismatch'); bad += 1
# rejections: partial block, unknown type
if run('q8_0', b'\x00' * 33) is not None: print('partial block accepted'); bad += 1
if run('nope', b'\x00' * 34) is not None: print('unknown type accepted'); bad += 1
print('cli checks done, bad =', bad)
sys.exit(1 if bad else 0)
PY

echo "== GPU invariants =="
if [ ! -x "$CLANG" ]; then
  echo "corex toolchain missing at $CLANG: cannot run the GPU tests"; rc=1
else
  CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Itests -Iinclude -O2"
  objs=()
  for f in src/ilu_kv_q8.cu src/ilu_qsa_step.cu src/ilu_quantize_act.cu tests/gpu_unit_f16bits/test_unit7_gpu.cu; do
    o="$OUT/gpu_$(basename "$f").o"
    "$CLANG" $CUFLAGS -c "$f" -o "$o" || { echo "BUILD FAILED ($f)"; exit 1; }
    objs+=("$o")
  done
  "$CLANG" "${objs[@]}" -o "$OUT/test_unit7_gpu" -L"$COREX/lib64" -lcudart -lpthread || { echo "LINK FAILED"; exit 1; }
  timeout 1500 "$OUT/test_unit7_gpu" | tee "$OUT/gpu.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "GPU tests FAILED"; rc=1; }
  grep -q "UNIT7 GPU TESTS PASSED" "$OUT/gpu.log" || { echo "GPU suite did not report PASSED"; rc=1; }
fi

echo "== port table =="
TSV=.einsteinOS/port/uebd9b81357.tsv
[ "$(wc -l < "$TSV")" -eq 90 ] || { echo "port table must have 90 lines"; rc=1; }
while IFS=$'\t' read -r rf rn lf ln; do
  [ -n "$ln" ] && [ -f "$lf" ] && grep -qw -- "$ln" "$lf" || { echo "port MISSING: $rn -> $lf:$ln"; rc=1; }
done < "$TSV"

[ $rc -eq 0 ] && echo "ALL UNIT TESTS PASSED" || echo "UNIT TESTS FAILED"
exit $rc
