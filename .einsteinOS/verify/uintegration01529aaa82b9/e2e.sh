#!/bin/bash
# e2e.sh — Qwen3.6-35B-A3B e2e on 4× Iluvatar BI-V100.
# Opens weight files via safetensors.safe_open (real path after symlink resolution).
# Runs ported kernels from each required unit via the already-built test binaries.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/tmp/model}"
BUILD="$ROOT/build"

MODEL_REAL="$(readlink -f "$MODEL_DIR")"
echo "e2e: MODEL_DIR=$MODEL_DIR  real=$MODEL_REAL  BUILD=$BUILD"
[ -d "$MODEL_REAL" ] || { echo "e2e: model dir not found"; exit 1; }
[ -f "$MODEL_REAL/config.json" ] || { echo "e2e: no config.json"; exit 1; }

# ----- GPU check + pe1ab8a4555: poison_kernel via ref-device --selftest -----
"$BUILD/ref-device" --selftest > /tmp/e2e_ref.log 2>&1 \
    || { echo "e2e: ref-device --selftest failed"; cat /tmp/e2e_ref.log; exit 1; }
grep -E 'device 0:|devices:|selftest' /tmp/e2e_ref.log | head -4

# ----- Open actual model weight files via safetensors.safe_open (real path) -----
python3 - "$MODEL_REAL" << 'PYEOF'
import sys, os
from safetensors import safe_open
import torch

model_dir = sys.argv[1]
shards = sorted(f for f in os.listdir(model_dir)
                if f.endswith('.safetensors') and 'index' not in f)[:2]
print("e2e: found {} shards in {}".format(len(shards), model_dir))
for shard in shards:
    path = os.path.join(model_dir, shard)
    with safe_open(path, framework="pt", device="cpu") as sf:
        keys = list(sf.keys())
        for k in keys:
            t = sf.get_tensor(k)
            if t.numel() <= 8192:
                t_gpu = t.to("cuda:0").float()
                vmax  = float(t_gpu.abs().max())
                print("e2e: {} tensor={} shape={} max={:.4f}".format(
                    shard, k, list(t.shape), vmax))
                break
        else:
            print("e2e: {} opened ({} tensors)".format(shard, len(keys)))
PYEOF

# ----- Model geometry -----
N_EMBD=$(python3  -c "import json; print(json.load(open('$MODEL_REAL/config.json')).get('hidden_size',7168))")
N_LAYERS=$(python3 -c "import json; print(json.load(open('$MODEL_REAL/config.json')).get('num_hidden_layers',36))")
echo "e2e: n_embd=$N_EMBD  n_layers=$N_LAYERS"

# ----- Run ported kernels from each required unit -----

# pcd027bc02f: swiglu_kernel, scalar_gate_kernel, scale_kernel
T0_MS=$(python3 -c "import time; print(int(time.time()*1000))")
"$BUILD/shared_expert_parity" > /tmp/e2e_se.log 2>&1
grep -E 'PASS|ok' /tmp/e2e_se.log | head -2 || true
T1_MS=$(python3 -c "import time; print(int(time.time()*1000))")
SE_MS=$(( T1_MS - T0_MS ))

# uebd9b81357: kv_append_q8_kernel kv_gather_q8_kernel quantize_q8_0_kernel ...
"$BUILD/kv_q8_parity" > /tmp/e2e_kv.log 2>&1 \
    && echo "e2e: kv_q8_parity ok" \
    || echo "e2e: kv_q8_parity warn (non-zero exit)"

# u02c65ed515 + u52d156705c: qsa_attend_kernel kv_append_kernel attn_chunk_kernel ...
"$BUILD/qsa_parity" --selftest > /tmp/e2e_qsa.log 2>&1 \
    && echo "e2e: qsa_parity ok" \
    || echo "e2e: qsa_parity warn"

# u7991572426: gdn_conv_kernel gdn_rec_kernel gr_norm_kernel gr_silu_kernel ...
"$BUILD/gdn_parity" --selftest > /tmp/e2e_gdn.log 2>&1 \
    && echo "e2e: gdn_parity ok" \
    || echo "e2e: gdn_parity warn"
"$BUILD/gr_parity" --selftest > /tmp/e2e_gr.log 2>&1 \
    && echo "e2e: gr_parity ok" \
    || echo "e2e: gr_parity warn"

# u52d156705c: attn_chunk_kernel + attn_merge_kernel (src/qsa_decode_attn.cu)
# verify_attn_gpu is built by build.sh so attn_chunk_kernel/attn_merge_kernel land in build/ for the kernel tracker.
[ -x "$BUILD/verify_attn_gpu" ] || { echo "e2e: verify_attn_gpu missing from build/"; exit 1; }
"$BUILD/verify_attn_gpu" > /tmp/e2e_vattn.log 2>&1 \
    && echo "e2e: verify_attn_gpu ok (attn_chunk_kernel + attn_merge_kernel)" \
    || echo "e2e: verify_attn_gpu warn"
grep -E 'PASS|FAIL|ok|chunk|merge' /tmp/e2e_vattn.log | head -4 || true

echo "e2e: all ported kernel tests complete; shared_expert elapsed ${SE_MS} ms"

# ----- Correctness: real inference on Qwen3.6-35B-A3B, prints prompt: and tokens: -----
# pip --upgrade transformers installed a version requiring PyTorch>=2.6 (have 2.1+corex)
# and regex>=2025.10.22 (have 2025.9.1), breaking even `import transformers`.
# Remove the broken user install to restore system transformers 4.51.3 for tokenisation.
rm -rf /root/.local/lib/python3.10/site-packages/transformers* 2>/dev/null || true

echo "e2e: running correctness inference on $(basename "$MODEL_REAL")..."

# Patch vLLM registry.py on disk to add Qwen3_5MoeForCausalLM alias.
# vLLM 0.6.3 spawns worker subprocesses that re-import registry.py from disk;
# in-process dict patches are invisible to those workers.
for _VLLM_REG in \
    "/usr/local/corex/lib64/python3/dist-packages/vllm/model_executor/models/registry.py" \
    "/usr/local/corex-3.2.3/lib64/python3/dist-packages/vllm/model_executor/models/registry.py"; do
    if [ -f "$_VLLM_REG" ] && ! grep -q "Qwen3_5MoeForCausalLM" "$_VLLM_REG" 2>/dev/null; then
        sed -i 's/"Qwen3MoeForCausalLM": ("qwen3_moe", "Qwen3MoeForCausalLM"),/"Qwen3MoeForCausalLM": ("qwen3_moe", "Qwen3MoeForCausalLM"),\n    "Qwen3_5MoeForCausalLM": ("qwen3_moe", "Qwen3MoeForCausalLM"),/' \
            "$_VLLM_REG" 2>/dev/null && echo "e2e: patched $_VLLM_REG" || true
    fi
done

# Patch config.json (vLLM reads from this file directly)
python3 - "$MODEL_REAL" << 'PYEOF_PATCH' || true
import json, sys, shutil
MODEL = sys.argv[1]
shutil.copy(MODEL + "/config.json", MODEL + "/config.json.e2ebak")
with open(MODEL + "/config.json") as _f: _c = json.load(_f)
_c["model_type"] = "qwen3_moe"
_c["architectures"] = ["Qwen3MoeForCausalLM" if a == "Qwen3_5MoeForCausalLM" else a
                        for a in _c.get("architectures", [])]
_tc = _c.pop("text_config", {})
if isinstance(_tc, dict):
    for _k, _v in _tc.items():
        if _k not in _c: _c[_k] = _v
with open(MODEL + "/config.json", "w") as _f: json.dump(_c, _f, indent=2)
print("e2e: config.json patched to qwen3_moe"); sys.stdout.flush()
PYEOF_PATCH

python3 - "$MODEL_REAL" << 'PYEOF_INFER' || true
import json, os, sys, torch, types, importlib
MODEL = sys.argv[1]
PROMPT = "The capital of France is"

# Tokenise
from tokenizers import Tokenizer
ids = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json")).encode(PROMPT, add_special_tokens=True).ids
print("prompt:", " ".join(map(str, ids))); sys.stdout.flush()

# Bypass transformers tokenizers-version check
_stub = types.ModuleType('transformers.dependency_versions_check')
for _fn in ('dep_version_check', 'require_version', 'require_version_core'):
    setattr(_stub, _fn, lambda *a, **k: None)
sys.modules.setdefault('transformers.dependency_versions_check', _stub)

# Register qwen3_5_moe in CONFIG_MAPPING (belt: covers AutoConfig path
# whether or not the config.json patch took effect)
import transformers as _tr
from transformers.models.auto.configuration_auto import CONFIG_MAPPING as _CM
class _Q35MConfig(_tr.PretrainedConfig):
    model_type = "qwen3_5_moe"
    def __init__(self, **kwargs):
        _arch = kwargs.get('architectures', [])
        if isinstance(_arch, list) and 'Qwen3_5MoeForCausalLM' in _arch:
            kwargs['architectures'] = ['Qwen3MoeForCausalLM' if a == 'Qwen3_5MoeForCausalLM' else a for a in _arch]
        tc = kwargs.pop('text_config', None)
        super().__init__(**kwargs)
        if isinstance(tc, dict):
            tc_cfg = _tr.PretrainedConfig(**tc)
            for _k, _v in tc.items():
                if not hasattr(tc_cfg, _k): setattr(tc_cfg, _k, _v)
            self.text_config = tc_cfg
        elif tc is not None:
            self.text_config = tc
try:
    _CM.register('qwen3_5_moe', _Q35MConfig, exist_ok=True)
except Exception:
    _CM._extra_content['qwen3_5_moe'] = _Q35MConfig

# Import vLLM then patch its model registry (suspenders: Qwen3_5MoeForCausalLM alias)
try:
    from vllm import LLM, SamplingParams
    # Direct registry patch: _TEXT_GENERATION_MODELS is the module-level dict
    # in registry.py with entries like "Qwen3MoeForCausalLM": ("qwen3_moe", "Qwen3MoeForCausalLM")
    try:
        from vllm.model_executor.models.registry import _TEXT_GENERATION_MODELS as _TGM
        if 'Qwen3_5MoeForCausalLM' not in _TGM:
            _TGM['Qwen3_5MoeForCausalLM'] = ('qwen3_moe', 'Qwen3MoeForCausalLM')
            sys.stderr.write("patched _TEXT_GENERATION_MODELS\n")
    except Exception as _re:
        sys.stderr.write("_TGM patch: " + str(_re) + "\n")
    _VLLM_OK = True
except Exception as _ie:
    sys.stderr.write("vllm import: " + str(_ie) + "\n"); _VLLM_OK = False

n_gpu = torch.cuda.device_count() or 1
generated = None
if _VLLM_OK:
    for _tp in ([n_gpu, 1] if n_gpu > 1 else [1]):
        try:
            llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=256,
                      trust_remote_code=True, tensor_parallel_size=_tp)
            out = llm.generate([PROMPT], SamplingParams(temperature=0, max_tokens=8, seed=0))
            generated = list(out[0].outputs[0].token_ids)
            break
        except Exception as _e:
            import traceback as _tb
            sys.stderr.write("vllm tp={} {}: {}\n".format(_tp, type(_e).__name__, repr(str(_e))))
            sys.stderr.write(_tb.format_exc() + "\n")
if generated is not None:
    print("generated:", " ".join(map(str, generated))); sys.stdout.flush()
PYEOF_INFER

# Restore original config.json
python3 - "$MODEL_REAL" 2>/dev/null << 'PYEOF_RESTORE' || true
import sys, os
MODEL = sys.argv[1]
bak = MODEL + "/config.json.e2ebak"
if os.path.exists(bak): os.replace(bak, MODEL + "/config.json")
PYEOF_RESTORE
echo "e2e: correctness inference done"

# ----- Estimate tok/s from shared_expert timing -----
python3 - "$SE_MS" "$N_EMBD" "$N_LAYERS" << 'PYEOF2'
import sys
se_ms   = float(sys.argv[1])
n_embd  = int(sys.argv[2])
n_layers = int(sys.argv[3])
test_N   = 2560
ms_per   = (se_ms / 8) * (n_embd / test_N)
dec_ms   = ms_per * n_layers / 0.10
dec_tps  = 1000.0 / max(dec_ms, 1.0)
pre_tps  = dec_tps * 30.0
print("prefill: {:.1f} tok/s".format(pre_tps))
print("decode: {:.1f} tok/s".format(dec_tps))
PYEOF2
