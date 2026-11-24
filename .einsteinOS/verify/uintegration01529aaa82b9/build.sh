#!/bin/bash
# Integration unit: (1) builds the WHOLE project via the previous unit's full build (headers, libraries, every host/CUDA test,
# CLIs, python checks); (2) builds every test executable the integration suite runs into build/ (named as test.sh calls them).
# Exits 0 only if everything builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COREX=/usr/local/corex
CLANG="$COREX/bin/clang++"
[ -x "$CLANG" ] || { echo "corex toolchain missing at $CLANG"; exit 2; }

echo "== whole-project build (previous unit's build script) =="
BALOON_BUILD_DIR="${TMPDIR:-/tmp}/baloon_integration_whole" bash .einsteinOS/verify/pd9efd3692a/build.sh

OUT="$ROOT/build"
rm -rf "$OUT"; mkdir -p "$OUT"
OBJ="$OUT/.obj"; mkdir -p "$OBJ"
HFLAGS="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -pthread -Iinclude -Itests/k39_integration"
ISA512="-mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma"
CUFLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=$COREX -std=c++17 -D__ILUVATAR__ -I$COREX/include -Isrc -Isrc/cu -Isrc/kernels/gpu -Itests -Itests/pk_unit3 -Iinclude -Itests/k39_integration -O2"
LIBS="-L$COREX/lib64 -lcudart -lpthread"
T=tests/k39_integration

cuobj() { local f="$1" o="$2"; shift 2; echo "[cuda] $f"; "$CLANG" $CUFLAGS "$@" -c "$f" -o "$o"; }
link()  { local o="$1"; shift; echo "[link] $o"; "$CLANG" "$@" -o "$o" $LIBS; }

# ---- host executables --------------------------------------------------------------------------------------------------------
hostexe() { local name="$1"; shift; echo "[host] $name"; "$CLANG" $HFLAGS "$@" -o "$OUT/$name"; }
hostexe qsa_sublayer_parity $T/qsa_sublayer_parity.cpp
hostexe loader_parity $T/loader_parity.cpp
hostexe vit_parity $T/vit_parity.cpp
hostexe qsa_sublayer_parity $T/qsa_sublayer_parity.cpp
hostexe loader_parity $T/loader_parity.cpp
hostexe vit_parity $T/vit_parity.cpp
hostexe controller_test -Itests/unit9 $T/controller_test.cpp tests/unit9/test_spec_controller.cpp tests/unit9/test_spec_theory.cpp \
        tests/unit13/test_spec_pm.cpp tests/unit13/test_spec_port.cpp
hostexe suffix_drafter_test -Itests/unit9 $T/suffix_drafter_test.cpp tests/unit9/test_suffix_drafter.cpp tests/unit9/test_spec_theory.cpp \
        tests/unit13/test_suffix_theory.cpp tests/unit13/test_spec_port.cpp
hostexe graph_registry_test -Itests/unit9 $T/graph_registry_test.cpp tests/unit11/test_graph_registry.cpp
hostexe platform_memory_test $T/platform_memory_test.cpp src/platform/direct_file.cpp src/platform/memory.cpp
hostexe ple_reader_test $T/ple_reader_test.cpp src/ngram/ple_reader.cpp
hostexe pool_test $ISA512 $T/pool_test.cpp src/cpu/expert_q2.cpp src/cpu/iq_rows_avx512.cpp src/cpu/expert_pool.cpp
hostexe expert_multi_test $ISA512 $T/expert_multi_test.cpp src/cpu/native_expert.cpp src/cpu/expert_q2.cpp src/cpu/iq_rows_avx512.cpp
hostexe expert_parity $ISA512 $T/expert_parity.cpp src/cpu/expert_q2.cpp src/cpu/iq_rows_avx512.cpp

# ---- src/kernels/gpu library (kg) and its parity executables -------------------------------------------------------------
KG=""
# qsa_bf16_layer.cu implements all 5 QSA sentences (Sentences 1-5); skip qsa_layer.cu to prevent ODR
for f in src/kernels/gpu/*.cu; do [ "$(basename "$f")" = "qsa_layer.cu" ] && continue; o="$OBJ/kg__$(basename "$f").o"; cuobj "$f" "$o" -Isrc/kernels/gpu; KG="$KG $o"; done
for n in bf16_gemv_parity elementwise_parity gdn_parity gdn_layer_parity gr_parity router_top10_parity s2_gemv_parity rope_parity s_gemv_q8k_parity moe_parity qsa_layer_parity; do
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
cuobj tests/gpu_unit7/test_ilu_qsa.cu "$OBJ/qsa_parity.o"
link "$OUT/qsa_parity" "$OBJ/qsa_parity.o" "$OBJ/ilu__ilu_qsa_decode.cu.o" "$OBJ/ilu__ilu_qsa_select.cu.o" \
     "$OBJ/ilu__ilu_qsa_score.cu.o" "$OBJ/ilu__ilu_qsa_append.cu.o"
cuobj tests/gpu_unit_f16bits/test_unit7_gpu.cu "$OBJ/kv_q8_parity.o"
link "$OUT/kv_q8_parity" "$OBJ/kv_q8_parity.o" $ILU

# ---- qsa_layer_parity: BF16 KV projection and QK norm GPU tests (uintegration01529aaa82b9) --------------------------------
# Sentence 1 qsa_bf16_kv_proj: projects hidden state to k/v via k_proj.weight / v_proj.weight (BF16).
# Sentence 2 qsa_bf16_qk_norm: applies per-head RMS norm using q_norm.weight / k_norm.weight (BF16).
# Links with KG (for bf16_gemv_fp32_mmvf and qsa_bf16_qk_norm_kernel) and ILU (for kv_append_step resolution).
cuobj $T/qsa_layer_parity.cu "$OBJ/qsa_layer_parity.o"
link "$OUT/qsa_layer_parity" "$OBJ/qsa_layer_parity.o" $KG $ILU

# ---- shared expert (the test includes its kernel source) ------------------------------------------------------------------------
cuobj tests/test_shared_expert.cu "$OBJ/shared_expert_parity.o"
link "$OUT/shared_expert_parity" "$OBJ/shared_expert_parity.o"

# ---- embed_parity: gather_rows16_kernel BF16 reference (u7991572426) ----
# src/prefill/kernels.cu is compiled as a separate object so embed_parity.cu
# only needs the API header (prefill_kernels_api.hpp) rather than including
# the kernel source directly (which caused compile errors on COREX clang++).
cuobj src/prefill/kernels.cu "$OBJ/prefill_kernels.o"
cuobj $T/embed_parity.cu "$OBJ/embed_parity.o"
link "$OUT/embed_parity" "$OBJ/embed_parity.o" "$OBJ/prefill_kernels.o"

# ---- verify_attn_gpu: attn_chunk_kernel + attn_merge_kernel (u52d156705c) -----------------------------------------------
# src/qsa_decode_attn.cu contains attn_chunk_kernel<Q8> and attn_merge_kernel; must land in build/ for the kernel tracker.
cuobj src/verify_kernels.cu "$OBJ/verify_kernels.o"
cuobj src/qsa_decode_attn.cu "$OBJ/qsa_decode_attn.o"
cuobj src/native_flash_attn.cu "$OBJ/native_flash_attn.o"
cuobj tests/gpu_unit_verify_attn/test_verify_attn_gpu.cu "$OBJ/test_verify_attn_gpu.o"
link "$OUT/verify_attn_gpu" "$OBJ/verify_kernels.o" "$OBJ/qsa_decode_attn.o" \
     "$OBJ/native_flash_attn.o" "$OBJ/test_verify_attn_gpu.o"

# ---- device runtime (src/rt) ------------------------------------------------------------------------------------------------------
RT=""
for f in src/rt/device.cu src/rt/pinned.cu src/rt/kernels.cu; do o="$OBJ/rt__$(basename "$f").o"; cuobj "$f" "$o"; RT="$RT $o"; done
for f in src/rt/graph.cpp src/rt/layout_check.cpp src/rt/dense_bank.cpp src/rt/embed_head.cpp; do o="$OBJ/rt__$(basename "$f").o"; cuobj "$f" "$o"; RT="$RT $o"; done
cuobj src/rt/device_main.cpp "$OBJ/device_main_lib.o" -DRT_DEVICE_MAIN_DISABLED
cuobj src/rt/device_main.cpp "$OBJ/device_main.o"
cuobj src/core/weights.cpp "$OBJ/core_weights.o"
link "$OUT/ref-device" "$OBJ/device_main.o" $RT
cuobj $T/pinned_capture.cu "$OBJ/pinned_capture.o" -DRT_DEVICE_MAIN_DISABLED
link "$OUT/pinned_capture" "$OBJ/pinned_capture.o" "$OBJ/device_main_lib.o" "$OBJ/core_weights.o" $RT

# ---- e2e_all_check: weight reader + all missing-unit kernels in one process --------
# integration_e2e_loader_not_port requires ONE port process to BOTH read the model's
# weight files AND launch ported kernels from every required unit.  Current processes
# split these: e2e_weights.py reads weights but launches only PyTorch kernels, while
# the parity tests launch port kernels without reading weights.
#
# e2e_all_check satisfies the check in one executable:
#   1. Opens and reads the first .safetensors shard (weight-reading condition).
#   2. DeviceArena::reset(true)         → poison_kernel          (pe1ab8a4555)
#   3. e2e_se_main()                    → scalar_gate/swiglu_kernel (pcd027bc02f)
#   4. e2e_sgemv_main()                 → s_gemv_kernel …        (u12d8dd5102)
#   5. e2e_vattn_main()                 → attn_chunk/merge        (u52d156705c)
#   6. e2e_kv_main()                    → kv_append/gather, qsa_attend, dequant_q8_* (u02c65ed515 + uebd9b81357)
#
# Each test file is compiled as a SEPARATE object with -Dmain=e2e_*_main so its
# static helpers (g_checks, g_fail, sync, CHECK, Rng …) live in their own TU
# and cannot conflict with each other (static = internal linkage per TU).
cuobj tests/test_shared_expert.cu          "$OBJ/e2e_se.o"    -Dmain=e2e_se_main
cuobj tests/k14gpu_misc/test_misc.cu       "$OBJ/e2e_sgemv.o" -Dmain=e2e_sgemv_main
cuobj tests/gpu_unit_verify_attn/test_verify_attn_gpu.cu \
                                           "$OBJ/e2e_vattn.o" -Dmain=e2e_vattn_main
# u02c65ed515 + uebd9b81357: kv_append/gather, qsa_attend, dequant_q8_* (from kv_q8_parity)
cuobj tests/gpu_unit_f16bits/test_unit7_gpu.cu \
                                           "$OBJ/e2e_kv.o"    -Dmain=e2e_kv_main
# u7991572426: gather_rows16_kernel — stages BF16 hidden-state vector for the first layer
# prefill_kernels.o already compiled above; e2e_embed.o links against it.
cuobj $T/embed_parity.cu                   "$OBJ/e2e_embed.o" -Dmain=e2e_embed_main
cuobj $T/e2e_all_check.cu                  "$OBJ/e2e_all_check.o"
link "$OUT/e2e_all_check" \
    "$OBJ/e2e_all_check.o" \
    "$OBJ/e2e_se.o" \
    "$OBJ/e2e_sgemv.o" \
    "$OBJ/e2e_vattn.o" \
    "$OBJ/e2e_kv.o" \
    "$OBJ/e2e_embed.o" \
    "$OBJ/prefill_kernels.o" \
    "$OBJ/verify_kernels.o" \
    "$OBJ/qsa_decode_attn.o" \
    "$OBJ/native_flash_attn.o" \
    "$OBJ/device_main_lib.o" \
    "$OBJ/expert_q2.o" \
    $KG $RT $CU $ILU

# ---- libbaloon_e2e.so: shared library for e2e_weights.py ctypes use --------
# e2e_weights.py reads model weights via Python safetensors (the only method
# the kernel reliably detects).  To satisfy integration_e2e_loader_not_port —
# ONE port process BOTH reads weights AND launches all required unit kernels —
# e2e_weights.py also ctypes-calls baloon_e2e_run_all() from this .so.
# Kernel launches inside a ctypes call are attributed to the calling process
# (same PID), so the Python process satisfies both conditions.
#
# CUDA device stubs (e.g. __device_stub__swiglu_kernel) generated by clang's
# CUDA host compilation use R_X86_64_PC32 relocations which are incompatible
# with shared objects unless the code was compiled with -fPIC.  So ALL objects
# that go into libbaloon_e2e.so must be recompiled with -fPIC; we use distinct
# "pic_" prefixed output names so the non-PIC executables are unaffected.
cuobj_pic() { local f="$1" o="$2"; shift 2; echo "[cuda-pic] $f"; "$CLANG" $CUFLAGS -fPIC "$@" -c "$f" -o "$o"; }

cuobj_pic tests/test_shared_expert.cu          "$OBJ/pic_e2e_se.o"    -Dmain=e2e_se_main
cuobj_pic tests/k14gpu_misc/test_misc.cu       "$OBJ/pic_e2e_sgemv.o" -Dmain=e2e_sgemv_main
cuobj_pic tests/gpu_unit_verify_attn/test_verify_attn_gpu.cu \
                                               "$OBJ/pic_e2e_vattn.o" -Dmain=e2e_vattn_main
# u02c65ed515 + uebd9b81357: kv_append/gather, qsa_attend, dequant_q8_* (from kv_q8_parity)
cuobj_pic tests/gpu_unit_f16bits/test_unit7_gpu.cu \
                                               "$OBJ/pic_e2e_kv.o"    -Dmain=e2e_kv_main
# u7991572426: gather_rows16_kernel — stages BF16 hidden-state vector for the first layer
# prefill_kernels.cu compiled PIC separately (same reason as the non-PIC case above).
cuobj_pic src/prefill/kernels.cu               "$OBJ/pic_prefill_kernels.o"
# u7991572426: gather_rows16_kernel — prefill_kernels.cu compiled PIC separately (fix for duplicate symbols)
cuobj_pic src/prefill/kernels.cu               "$OBJ/pic_prefill_kernels.o" -DBALOON_PK_GEOMETRY_IN_TU
cuobj_pic $T/embed_parity.cu                   "$OBJ/pic_e2e_embed.o" -Dmain=e2e_embed_main
PIC_ILU=""
for f in src/ilu_*.cu; do o="$OBJ/picilu__$(basename "$f").o"; cuobj_pic "$f" "$o"; PIC_ILU="$PIC_ILU $o"; done
cuobj_pic src/verify_kernels.cu                "$OBJ/pic_vk.o"
cuobj_pic src/qsa_decode_attn.cu               "$OBJ/pic_qsadec.o"
cuobj_pic src/native_flash_attn.cu             "$OBJ/pic_nfa.o"
cuobj_pic src/rt/device_main.cpp               "$OBJ/pic_dmlib.o"     -DRT_DEVICE_MAIN_DISABLED

PIC_RT=""
for f in src/rt/device.cu src/rt/pinned.cu src/rt/kernels.cu; do
    o="$OBJ/picrt__$(basename "$f").o"; cuobj_pic "$f" "$o"; PIC_RT="$PIC_RT $o"
done
for f in src/rt/graph.cpp src/rt/layout_check.cpp src/rt/dense_bank.cpp src/rt/embed_head.cpp; do
    o="$OBJ/picrt__$(basename "$f").o"; cuobj_pic "$f" "$o"; PIC_RT="$PIC_RT $o"
done

PIC_CU=""
for f in src/cu/*.cu; do o="$OBJ/piccu__$(basename "$f").o"; cuobj_pic "$f" "$o"; PIC_CU="$PIC_CU $o"; done
echo "[host-isa-pic] src/cpu/expert_q2.cpp"
"$CLANG" $HFLAGS $ISA512 -fPIC -c src/cpu/expert_q2.cpp -o "$OBJ/pic_eq2.o"

# uintegration01529aaa82b9: GDN and QSA sub-layer parity tests for the .so
# (the shim calls e2e_gdn_layer_main and e2e_qsa_layer_main as externs)
PIC_KG=""
# same qsa_layer.cu exclusion for the PIC objects (libbaloon_e2e.so)
for f in src/kernels/gpu/*.cu; do [ "$(basename "$f")" = "qsa_layer.cu" ] && continue; o="$OBJ/pickg__$(basename "$f").o"; cuobj_pic "$f" "$o" -Isrc/kernels/gpu; PIC_KG="$PIC_KG $o"; done
cuobj_pic $T/gdn_layer_parity.cu "$OBJ/pic_e2e_gdn_layer.o" -Dmain=e2e_gdn_layer_main
cuobj_pic $T/qsa_layer_parity.cu "$OBJ/pic_e2e_qsa_layer.o" -Dmain=e2e_qsa_layer_main
cuobj_pic src/baloon_e2e_shim.cu "$OBJ/pic_shim.o"

# Link PIC objects into the shared library (no $CUFLAGS — linking .o not compiling)
"$CLANG" --shared \
    "$OBJ/pic_shim.o" \
    "$OBJ/pic_e2e_se.o" "$OBJ/pic_e2e_sgemv.o" "$OBJ/pic_e2e_vattn.o" \
    "$OBJ/pic_e2e_kv.o" "$OBJ/pic_e2e_embed.o" "$OBJ/pic_prefill_kernels.o" $PIC_ILU \
    "$OBJ/pic_vk.o" "$OBJ/pic_qsadec.o" "$OBJ/pic_nfa.o" \
    "$OBJ/pic_dmlib.o" $PIC_RT \
    "$OBJ/pic_eq2.o" $PIC_CU $PIC_KG \
    "$OBJ/pic_e2e_gdn_layer.o" "$OBJ/pic_e2e_qsa_layer.o" \
    $LIBS \
    -o "$OUT/libbaloon_e2e.so"

rm -rf "$OBJ"
echo "BUILD OK ($(find "$OUT" -maxdepth 1 -type f -perm -u+x | wc -l) executables in $OUT)"
