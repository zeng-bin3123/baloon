#!/bin/bash
# Integration unit (reference 6801c77ff6f7): (1) builds the WHOLE project via the previous unit's full build (headers, libraries, every host/CUDA test,
# CLIs, python checks); (2) builds every test executable the integration suite runs into build/ (named as test.sh calls them).
# Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }

echo "== whole-project build (previous unit's build script) =="
BALOON_BUILD_DIR="${TMPDIR:-/tmp}/baloon_integration_whole" bash .einsteinOS/verify/p17652417b0/build.sh

OUT="$ROOT/build"
rm -rf "$OUT"; mkdir -p "$OUT"
OBJ="$OUT/.obj"; mkdir -p "$OBJ"
HFLAGS="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude -Itests/k39_integration -Itests/k60_integration"
ISA512="-mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma"
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Isrc/cu -Itests -Itests/pk_unit3 -Iinclude -Itests/k39_integration -Itests/k60_integration -O2"
LIBS="-rdynamic -L$COREX/lib64 -lcudart -lpthread"
T=tests/k60_integration; T39=tests/k39_integration

cuobj() { local f="$1" o="$2"; shift 2; echo "[cuda] $f"; "$CLANG" $CUFLAGS "$@" -c "$f" -o "$o"; }
link()  { local o="$1"; shift; echo "[link] $o"; "$CLANG" "$@" -o "$o" $LIBS; }

# ---- host executables --------------------------------------------------------------------------------------------------------
hostexe() { local name="$1"; shift; echo "[host] $name"; "$CLANG" $HFLAGS "$@" -o "$OUT/$name"; }
hostexe controller_test -Itests/unit9 $T/controller_test.cpp tests/unit9/test_spec_controller.cpp tests/unit9/test_spec_theory.cpp \
        tests/unit13/test_spec_pm.cpp tests/unit13/test_spec_port.cpp
hostexe suffix_drafter_test -Itests/unit9 $T/suffix_drafter_test.cpp tests/unit9/test_suffix_drafter.cpp tests/unit9/test_spec_theory.cpp \
        tests/unit13/test_suffix_theory.cpp tests/unit13/test_spec_port.cpp
hostexe graph_registry_test -Itests/unit9 $T39/graph_registry_test.cpp tests/unit11/test_graph_registry.cpp
hostexe platform_memory_test $T/platform_memory_test.cpp src/platform/direct_file.cpp src/platform/memory.cpp
hostexe ple_reader_test $T/ple_reader_test.cpp src/ngram/ple_reader.cpp
hostexe pool_test $ISA512 $T/pool_test.cpp src/cpu/expert_q2.cpp src/cpu/iq_rows_avx512.cpp src/cpu/expert_pool.cpp
hostexe expert_multi_test $ISA512 $T/expert_multi_test.cpp src/cpu/native_expert.cpp src/cpu/expert_q2.cpp src/cpu/iq_rows_avx512.cpp
hostexe expert_parity $ISA512 $T/expert_parity.cpp src/cpu/expert_q2.cpp src/cpu/iq_rows_avx512.cpp

# ---- src/kernels/gpu library (kg) and its parity executables -------------------------------------------------------------
KG=""
for f in src/kernels/gpu/*.cu; do o="$OBJ/kg__$(basename "$f").o"; cuobj "$f" "$o" -Isrc/kernels/gpu; KG="$KG $o"; done
# flash_short.cu delegates to native_flash_attn_step, which lives in src/native_flash_attn.cu
cuobj src/native_flash_attn.cu "$OBJ/native_flash_attn.o"; KG="$KG $OBJ/native_flash_attn.o"
for n in bf16_gemv_parity elementwise_parity gdn_parity gr_parity router_top10_parity s2_gemv_parity rope_parity s_gemv_q8k_parity; do
  cuobj $T/$n.cu "$OBJ/$n.o"; link "$OUT/$n" "$OBJ/$n.o" $KG
done

# ---- src/cu library (misc kernels) and its parity executables -----------------------------------------------------------
CU=""
for f in src/cu/*.cu; do o="$OBJ/cu__$(basename "$f").o"; cuobj "$f" "$o"; CU="$CU $o"; done
echo "[host-isa] src/cpu/expert_q2.cpp"
"$CLANG" $HFLAGS $ISA512 -c src/cpu/expert_q2.cpp -o "$OBJ/expert_q2.o"
for n in dequant_s2_parity quantize_act_parity s_gemv_parity sampler_parity s2_gemv_q8_parity ple_parity; do
  cuobj $T/$n.cu "$OBJ/$n.o"; link "$OUT/$n" "$OBJ/$n.o" $CU "$OBJ/expert_q2.o"
done
# native_mmvq_multi: the unit-14 mmvq test (kernel dispatch tables, validation, multi-column exactness)
cuobj tests/k14gpu_mmvq/test_mmvq.cu "$OBJ/native_mmvq_multi.o"
link "$OUT/native_mmvq_multi" "$OBJ/native_mmvq_multi.o" "$OBJ/cu__native_mmvq.cu.o"

# ---- ilu_* kernels ------------------------------------------------------------------------------------------------------------
ILU=""
for f in src/ilu_*.cu; do o="$OBJ/ilu__$(basename "$f").o"; cuobj "$f" "$o"; ILU="$ILU $o"; done
cuobj $T/qsa_parity.cu "$OBJ/qsa_parity.o"
link "$OUT/qsa_parity" "$OBJ/qsa_parity.o" "$OBJ/ilu__ilu_qsa_decode.cu.o" "$OBJ/ilu__ilu_qsa_select.cu.o" \
     "$OBJ/ilu__ilu_qsa_score.cu.o" "$OBJ/ilu__ilu_qsa_append.cu.o"
cuobj $T/kv_q8_parity.cu "$OBJ/kv_q8_parity.o"
link "$OUT/kv_q8_parity" "$OBJ/kv_q8_parity.o" $ILU

# ---- shared expert (the test includes its kernel source) ------------------------------------------------------------------------
cuobj $T/shared_expert_parity.cu "$OBJ/shared_expert_parity.o"
link "$OUT/shared_expert_parity" "$OBJ/shared_expert_parity.o"

# ---- device runtime (src/rt) ------------------------------------------------------------------------------------------------------
RT=""
for f in src/rt/device.cu src/rt/pinned.cu src/rt/kernels.cu; do o="$OBJ/rt__$(basename "$f").o"; cuobj "$f" "$o"; RT="$RT $o"; done
for f in src/rt/graph.cpp src/rt/layout_check.cpp src/rt/dense_bank.cpp src/rt/embed_head.cpp; do o="$OBJ/rt__$(basename "$f").o"; cuobj "$f" "$o"; RT="$RT $o"; done
cuobj src/rt/device_main.cpp "$OBJ/device_main_lib.o" -DRT_DEVICE_MAIN_DISABLED
cuobj src/rt/device_main.cpp "$OBJ/device_main.o"
cuobj src/core/weights.cpp "$OBJ/core_weights.o"
link "$OUT/ref-device" "$OBJ/device_main.o" $RT
cuobj $T39/pinned_capture.cu "$OBJ/pinned_capture.o" -DRT_DEVICE_MAIN_DISABLED
link "$OUT/pinned_capture" "$OBJ/pinned_capture.o" "$OBJ/device_main_lib.o" "$OBJ/core_weights.o" $RT

rm -rf "$OBJ"
echo "BUILD OK ($(find "$OUT" -maxdepth 1 -type f -perm -u+x | wc -l) executables in $OUT)"
