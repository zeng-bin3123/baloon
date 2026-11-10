#!/bin/bash
# Unit p7a60e039c0 tests. Exits non-zero if any check fails.
#  1. port table: 96 rows, each reference file/name pair required by the unit is present, and each counterpart name is defined
#     (as a definition, not just mentioned) in the repo file it names;
#  2. the GPU invariant suite (opens a GPU device), including mrope_table_set rejecting negative sections.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export LD_LIBRARY_PATH="/usr/local/corex/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${EINSTEINOS_GPUS:-0}}"
rc=0
echo "== port table =="
python3 - <<'PY' || rc=1
import re, sys
REQ = {
 'bf16_gemv.cu': 'THREADS bf16_gemv bf16_gemv_naive_kernel bf16_gemv_split bf16_gemv_split_kernel bf16_gemv_warp_kernel finish',
 'dequant_s2.cu': 'CODES_PER_BYTE QK dequant_s2 dequant_s2_kernel',
 'elementwise.cu': 'THREADS add_inplace add_kernel check_launch copy_from_mapped copy_from_mapped_kernel copy_i32_from_mapped copy_i32_from_mapped_kernel doorbell_publish doorbell_publish_kernel doorbell_ring doorbell_ring_kernel doorbell_wait doorbell_wait_kernel embedding_gather embedding_gather_kernel f32_to_bf16_bulk f32_to_f16_bulk gdn_gate gdn_gate_kernel grid_for rms_norm_weighted rms_norm_weighted_kernel scale_inplace scale_kernel silu_inplace silu_kernel softplus_dev sync_if_needed to_bf16_kernel to_f16_kernel',
 'native_bf16.cu': 'REF_MMVF_CASE bf16_f32_mmvf_kernel bf16_gemv_fp32_mmvf mmvf_block_size mmvf_warp_sum',
 'native_gr_postops.cu': 'THREADS blocks check_launch check_pointer check_shape down_silu native_gr_down_silu native_gr_post native_gr_pre_gated post pre_gated scale_zero_bias sigmoid',
 'native_moe.cu': 'combine native_moe_combine native_moe_combine_enabled native_moe_combine_set_enabled overlap valid_span',
 'native_ple_postops.cu': 'N Span broadcast_kernel conv_residual_kernel gate_kernel launch_check native_ple_postops overlaps validate warp_sum',
 'native_rope.cu': 'apply mrope_table mrope_table_set native_rope_apply native_rope_enabled native_rope_set_enabled overlaps',
 'rope.cu': 'build_rope_table rope_neox_apply rope_neox_kernel',
 'router_top10.cu': 'RT_MAX_THREADS router_top10 router_top10_kernel',
 's2_gemv.cu': 'QK s2_gemv s2_gemv_kernel',
 's2_gemv_quads.cu': 'CODES_PER_BYTE_S2 QK_S2 s2_gemv_quads s2_gemv_quads_kernel',
}
want = {('src/kernels/cuda/' + f, n) for f, ns in REQ.items() for n in ns.split()}
rows = [l.rstrip('\n').split('\t') for l in open('.einsteinOS/port/p7a60e039c0.tsv') if l.strip()]
bad = 0
if len(want) != 96: print('internal: expected 96 required pairs, have', len(want)); bad += 1
if len(rows) != 96: print('expected 96 rows, got', len(rows)); bad += 1
got = set()
for r in rows:
    if len(r) != 4: print('bad row', r); bad += 1; continue
    got.add((r[0], r[1]))
    try: t = open(r[2], errors='ignore').read()
    except OSError: print('missing file', r[2]); bad += 1; continue
    t = re.sub(r'//[^\n]*', '', t); t = re.sub(r'/\*.*?\*/', '', t, flags=re.S)
    n = re.escape(r[3].split('::')[-1])
    pat = (r'\b(constexpr|const|static|struct|class|using|typedef|enum|#define|#\s*define)\b[^;{}\n]*\b' + n + r'\b'
           r'|\b' + n + r'\s*\('
           r'|\b' + n + r'\s*=')
    if not re.search(pat, t): print('name not defined', r); bad += 1
for m in sorted(want - got): print('missing required pair', m); bad += 1
print('port rows checked:', len(rows), 'bad:', bad)
sys.exit(1 if bad else 0)
PY
cd "$ROOT/build" 2>/dev/null || { echo "build dir missing (run build.sh)"; exit 2; }
BIN="$ROOT/build/test_p7a6"
[ -x "$BIN" ] || { echo "$BIN missing (run build.sh)"; exit 1; }
echo "== test_p7a6 =="
timeout 600 "$BIN" > /tmp/p7a6.log 2>&1; r=$?
tail -15 /tmp/p7a6.log
[ $r -eq 0 ] || { echo "FAIL test_p7a6 (exit $r)"; rc=1; }
grep -q "P7A6 UNIT TESTS PASSED" /tmp/p7a6.log || { echo "suite did not report PASSED"; rc=1; }
[ $rc -eq 0 ] && echo "ALL P7A6 UNIT TESTS PASSED" || echo "P7A6 UNIT TESTS FAILED"
exit $rc
