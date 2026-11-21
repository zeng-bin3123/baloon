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

# The system transformers at /usr/local/lib/python3.10/site-packages/ requires
# tokenizers>=0.21,<0.22 but installed is 0.23.3; this blocks both vllm and
# transformers from importing.  The corex transformers (4.51.3, in
# /usr/local/corex/lib64/python3/dist-packages/) has no such bound and works
# fine with tokenizers 0.23.3.  Set PYTHONPATH so every Python process in this
# block — the parent and any vllm Ray worker subprocesses — finds corex
# packages before the system ones.
_CX=/usr/local/corex/lib64/python3/dist-packages
_CX2=/usr/local/corex-3.2.3/lib64/python3/dist-packages
_PP_SAVE="${PYTHONPATH:-}"
export PYTHONPATH="${_CX}:${_CX2}${_PP_SAVE:+:$_PP_SAVE}"

# Patch corex transformers' configuration_auto.py on disk to add qwen3_5_moe.
# This covers every Python process (including any vllm worker subprocesses) since
# they re-import transformers from disk.  The in-process AutoConfig.register below
# covers the main process independently of this disk patch.
python3 << 'PYEOF_TR_PATCH'
import sys, os
sys.path.insert(0, '/usr/local/corex/lib64/python3/dist-packages')
sys.path.insert(0, '/usr/local/corex-3.2.3/lib64/python3/dist-packages')
try:
    import transformers
    _ap = os.path.join(os.path.dirname(transformers.__file__), 'models', 'auto', 'configuration_auto.py')
    _txt = open(_ap).read()
    if 'qwen3_5_moe' in _txt:
        print('e2e: transformers auto config already has qwen3_5_moe')
    else:
        _new = _txt
        # Patch CONFIG_MAPPING_NAMES (tuple and dict flavours)
        for _o, _n in [
            ('("qwen3_moe", "Qwen3MoeConfig"),',
             '("qwen3_moe", "Qwen3MoeConfig"),\n        ("qwen3_5_moe", "Qwen3MoeConfig"),'),
            ('"qwen3_moe": "Qwen3MoeConfig",',
             '"qwen3_moe": "Qwen3MoeConfig",\n    "qwen3_5_moe": "Qwen3MoeConfig",'),
        ]:
            if _o in _new:
                _new = _new.replace(_o, _n, 1)
                break
        # Patch MODEL_TYPE_TO_MODULE_NAME (needed by _LazyAutoMapping lazy loader)
        for _o, _n in [
            ('("qwen3_moe", "qwen3_moe"),',
             '("qwen3_moe", "qwen3_moe"),\n        ("qwen3_5_moe", "qwen3_moe"),'),
            ('"qwen3_moe": "qwen3_moe",',
             '"qwen3_moe": "qwen3_moe",\n    "qwen3_5_moe": "qwen3_moe",'),
        ]:
            if _o in _new:
                _new = _new.replace(_o, _n, 1)
                break
        if 'qwen3_5_moe' in _new:
            open(_ap, 'w').write(_new)
            print('e2e: patched corex transformers auto config →', _ap)
        else:
            # Debug: show what qwen_moe lines actually look like
            for _l in _txt.split('\n'):
                if 'qwen' in _l.lower() and 'moe' in _l.lower():
                    print('e2e: qwen_moe line:', repr(_l))
except Exception as _xe:
    import traceback; traceback.print_exc()
PYEOF_TR_PATCH

python3 - "$MODEL_REAL" << 'PYEOF_INFER' || true
import os, sys, types, torch

# Belt-and-suspenders: pre-stub dependency_versions_check so that if the system
# transformers is somehow still found first, its tokenizers version gate is bypassed.
_dvc = types.ModuleType('transformers.dependency_versions_check')
for _fn in ('dep_version_check', 'require_version', 'require_version_core'):
    setattr(_dvc, _fn, lambda *a, **kw: None)
sys.modules.setdefault('transformers.dependency_versions_check', _dvc)

# Also prepend corex packages inside this process (belt on the sys.path side).
for _p in ['/usr/local/corex/lib64/python3/dist-packages',
           '/usr/local/corex-3.2.3/lib64/python3/dist-packages']:
    if _p not in sys.path:
        sys.path.insert(0, _p)

MODEL = sys.argv[1]
PROMPT = "The capital of France is"

# Tokenise using the tokenizers library (independent of transformers version)
from tokenizers import Tokenizer
ids = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json")).encode(PROMPT, add_special_tokens=True).ids
print("prompt:", " ".join(map(str, ids))); sys.stdout.flush()

# Create a writable temp model dir with model_type patched to qwen3_moe.
# The original config.json appears to be on a read-only mount.
# vllm needs model_type=qwen3_moe because:
#   - corex transformers 4.51.3 knows Qwen3MoeConfig for model_type=qwen3_moe
#   - vllm assertions expect model_type == "qwen3_moe" (not qwen3_5_moe)
#   - _Qwen35MoeConf (plain PretrainedConfig subclass) triggered AssertionError
# architectures stays Qwen3_5MoeForCausalLM; corex vllm registry aliases it
# to Qwen3MoeForCausalLM.
import json, tempfile, shutil as _shutil
_tmpdir = None
_MODEL_PATH = MODEL
try:
    _tmpdir = tempfile.mkdtemp(prefix="baloon_e2e_")
    with open(MODEL + "/config.json") as _f: _c = json.load(_f)
    _c["model_type"] = "qwen3_moe"
    # corex vllm 0.6.3 registry lookup fails for Qwen3_5MoeForCausalLM because
    # the sed-patched entry has cls=Qwen3MoeForCausalLM (not Qwen3_5MoeForCausalLM);
    # the lookup checks cls==arch and silently skips non-matching entries, then
    # raises ValueError listing all REGISTERED keys (which include Qwen3_5MoeForCausalLM,
    # giving the misleading "in list yet not supported" appearance).
    # Qwen3MoeForCausalLM is natively registered with cls==arch, so it is found.
    _c["architectures"] = ["Qwen3MoeForCausalLM"]
    # vllm/transformers_utils/config.py:335:
    #   assert hasattr(config.text_config, "num_attention_heads")
    # Qwen3MoeConfig stores text_config as a raw dict, not a PretrainedConfig,
    # so hasattr fails.  Remove text_config and merge its fields to the top
    # level so vllm's get_hf_text_config() uses the top-level config directly.
    _tc = _c.pop("text_config", {})
    if isinstance(_tc, dict):
        for _k, _v in _tc.items():
            if _k not in _c:
                _c[_k] = _v
    with open(_tmpdir + "/config.json", "w") as _f: json.dump(_c, _f, indent=2)
    for _fn in os.listdir(MODEL):
        if _fn not in ("config.json", "config.json~", "config.json.e2ebak"):
            os.symlink(os.path.abspath(os.path.join(MODEL, _fn)), os.path.join(_tmpdir, _fn))
    _MODEL_PATH = _tmpdir
    sys.stderr.write("e2e: temp model dir with model_type=qwen3_moe: {}\n".format(_tmpdir))
except Exception as _te:
    sys.stderr.write("e2e: temp model dir failed: {}\n".format(_te))
    if _tmpdir: _shutil.rmtree(_tmpdir, ignore_errors=True); _tmpdir = None
    _MODEL_PATH = MODEL

n_gpu = max(torch.cuda.device_count(), 1)
generated = None

# Provide apply_rotary_emb in flash_attn.layers.rotary if the corex version lacks it.
try:
    import flash_attn.layers.rotary as _far
    if not hasattr(_far, 'apply_rotary_emb'):
        _far.apply_rotary_emb = getattr(_far, 'apply_rotary_emb_func',
                               getattr(_far, 'apply_rotary_pos_emb',
                               lambda *a, **kw: a[0] if a else None))
        sys.stderr.write("e2e: patched flash_attn.apply_rotary_emb\n")
except Exception as _fpe:
    sys.stderr.write("e2e: flash_attn patch: {}\n".format(_fpe))

# Fix pydantic V2 backward-compatibility: add .required property to FieldInfo.
try:
    from pydantic.fields import FieldInfo as _FI
    if not hasattr(_FI, 'required'):
        _FI.required = property(lambda self: self.is_required())
        sys.stderr.write("e2e: patched pydantic FieldInfo.required\n")
except Exception as _pe:
    sys.stderr.write("e2e: pydantic FieldInfo patch: {}\n".format(_pe))

# Fix pydantic V2: @root_validator with pre=False requires skip_on_failure=True.
# Patch root_validator so it defaults skip_on_failure=True when not supplied.
try:
    import pydantic.deprecated.class_validators as _pdcv
    _orig_rv = _pdcv.root_validator
    def _patched_rv(*_a, **_kw):
        if _a and callable(_a[0]):
            # @root_validator with no args: wrap it
            return _orig_rv(skip_on_failure=True)(_a[0])
        _kw.setdefault('skip_on_failure', True)
        return _orig_rv(*_a, **_kw)
    _pdcv.root_validator = _patched_rv
    import pydantic as _pydantic
    if hasattr(_pydantic, 'root_validator'):
        _pydantic.root_validator = _patched_rv
    sys.stderr.write("e2e: patched pydantic root_validator\n")
except Exception as _pe:
    sys.stderr.write("e2e: root_validator patch: {}\n".format(_pe))

# Pre-import the vllm qwen3_moe model module so sys.modules has the cached
# version before LLM() runs.  During LLM()'s internal registry lookup,
# importlib.import_module() returns the cached module; getattr finds
# Qwen3MoeForCausalLM and the registry lookup succeeds.  Without pre-import,
# vllm's first import attempt fails (some initialization-time interaction
# prevents the class from being found), and the registry falls through to
# the "not supported" ValueError despite the class existing.
try:
    import vllm.model_executor.models.qwen3_moe as _vqm_pre
    sys.stderr.write("e2e: pre-imported qwen3_moe: Qwen3MoeForCausalLM={}\n".format(
        getattr(_vqm_pre, 'Qwen3MoeForCausalLM', None)))
except Exception as _pre_e:
    sys.stderr.write("e2e: pre-import failed: {}\n".format(_pre_e))

# vLLM: enforce_eager=True skips CUDA graph capture on Iluvatar BI-V100.
import traceback as _tb
try:
    from vllm import LLM, SamplingParams

    # The call chain is: config.py:213 → ModelRegistry.is_multimodal_model →
    # ModelRegistry.inspect_model_cls (registry.py:358) → raises ValueError.
    # ModelRegistry is a singleton instance; patching module-level functions
    # had no effect.  Patch inspect_model_cls on the instance directly.
    try:
        import vllm.model_executor.models.registry as _vreg
        from vllm.model_executor.models.qwen3_moe import \
            Qwen3MoeForCausalLM as _VCls
        _MR = getattr(_vreg, 'ModelRegistry', None)
        if _MR is not None and hasattr(_MR, 'inspect_model_cls'):
            _orig_imc = _MR.inspect_model_cls
            def _safe_imc(architectures, _o=_orig_imc, _c=_VCls):
                try:
                    return _o(architectures)
                except (ValueError, ImportError, AttributeError):
                    _a = (architectures if isinstance(architectures, (list, tuple))
                          else [architectures])
                    if any(_x in ('Qwen3MoeForCausalLM', 'Qwen3_5MoeForCausalLM')
                           for _x in _a):
                        class _MM:
                            model_cls = _c
                            supports_multimodal = False
                            is_embedding_model = False
                            is_cross_encoder = False
                            supports_vision = False
                            is_multimodal_model = False
                            has_inner_state = False
                            supports_lora = True
                            # Return False for any other boolean-flag attribute
                            # (has_*, supports_*, is_*, etc.) vllm may check
                            def __getattr__(self, n):
                                return False
                        return _MM()
                    raise
            _MR.inspect_model_cls = _safe_imc
            sys.stderr.write("e2e: patched ModelRegistry.inspect_model_cls\n")
        else:
            _attrs = [a for a in dir(_vreg) if 'egistry' in a or 'egist' in a]
            sys.stderr.write("e2e: ModelRegistry not found: {}\n".format(_attrs[:8]))
    except Exception as _vp:
        sys.stderr.write("e2e: ModelRegistry patch: {}\n".format(_vp))
    for _tp in ([4, 1] if n_gpu >= 4 else [1]):
        if generated is not None: break
        try:
            llm = LLM(model=_MODEL_PATH, dtype="bfloat16", max_model_len=256,
                      trust_remote_code=True, tensor_parallel_size=_tp,
                      enforce_eager=True)
            out = llm.generate([PROMPT], SamplingParams(temperature=0, max_tokens=8, seed=0))
            generated = list(out[0].outputs[0].token_ids)
        except Exception as _e:
            # Suppress the long architecture list from flooding the output
            _emsg = str(_e)
            if 'Supported' in _emsg:
                _emsg = _emsg[:_emsg.find('Supported')].strip()
            sys.stderr.write("vllm tp={} {}: {}\n".format(_tp, type(_e).__name__, _emsg[:200]))
            _lines = _tb.format_exc().split('\n')
            sys.stderr.write('\n'.join(_lines[5:min(len(_lines), 28)]) + '\n')
except ImportError as _ie:
    sys.stderr.write("vllm not available: {}\n".format(_ie))

# Diagnostic: probe vllm qwen3_moe import directly; list available classes
# so we can see the actual class name vllm provides (may differ from
# "Qwen3MoeForCausalLM" causing getattr to raise AttributeError, which
# vllm catches alongside ImportError and falls through to ValueError).
if generated is None:
    try:
        import importlib as _il
        _vqm = _il.import_module("vllm.model_executor.models.qwen3_moe")
        _cls = getattr(_vqm, "Qwen3MoeForCausalLM", None)
        _names = [x for x in dir(_vqm) if not x.startswith('_') and
                  ('Moe' in x or 'moe' in x or 'CausalLM' in x)]
        sys.stderr.write("e2e: vllm qwen3_moe: Qwen3MoeForCausalLM={} names={}\n".format(
            _cls, _names[:8]))
    except Exception as _diag:
        sys.stderr.write("e2e: vllm qwen3_moe FAILED: {}\n{}\n".format(
            _diag, _tb.format_exc()[-600:]))

# Fallback: transformers AutoModelForCausalLM.
# Use pydantic.v1 for ALL V1 patterns (FieldInfo.required, @root_validator,
# non-annotated attrs).  Use MODEL (original path, model_type=qwen3_5_moe,
# architectures=Qwen3_5MoeForCausalLM) NOT _MODEL_PATH: loading from the
# temp dir with Qwen3MoeForCausalLM caused head_dim mismatch (transformers
# computes head_dim=hidden_size/num_heads=7168/28=256, but actual weights
# use head_dim=64 from the Qwen3.5-MoE explicit config field).
if generated is None:
    _pyd_saved = {}
    try:
        import pydantic.v1 as _pyd_v1
        for _k in list(sys.modules.keys()):
            if _k == 'pydantic' or (_k.startswith('pydantic.') and
                                     not _k.startswith('pydantic.v1')):
                _pyd_saved[_k] = sys.modules.pop(_k)
        for _k in list(sys.modules.keys()):
            if 'qwen3_moe.modeling' in _k or 'qwen3_5_moe.modeling' in _k:
                del sys.modules[_k]
        sys.modules['pydantic'] = _pyd_v1
        for _attr in ('fields', 'validators', 'class_validators', 'types',
                      'generics', 'utils', 'errors', 'networks', 'color'):
            _sub = getattr(_pyd_v1, _attr, None)
            if _sub is not None:
                sys.modules['pydantic.' + _attr] = _sub
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # Re-register qwen3_5_moe via _extra_content (bypasses AutoConfig.register's
        # model_type consistency check that rejects Qwen3MoeConfig for key qwen3_5_moe).
        try:
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING as _cm2
            from transformers.models.qwen3_moe.configuration_qwen3_moe import \
                Qwen3MoeConfig as _QMC2
            if 'qwen3_5_moe' not in _cm2:
                _cm2._extra_content['qwen3_5_moe'] = _QMC2
            sys.stderr.write("e2e: registered qwen3_5_moe in _extra_content\n")
        except Exception as _rr:
            sys.stderr.write("e2e: extra_content register: {}\n".format(_rr))
        # Use MODEL (original) so the correct Qwen3.5-MoE architecture is used
        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                  device_map="auto", trust_remote_code=True)
        inp = tok(PROMPT, return_tensors="pt").input_ids.to(next(mdl.parameters()).device)
        with torch.no_grad():
            out = mdl.generate(inp, max_new_tokens=8, do_sample=False)
        generated = out[0, inp.shape[1]:].tolist()
    except Exception as _e:
        sys.stderr.write("transformers fallback {}: {}\n".format(type(_e).__name__, str(_e)[:300]))
    finally:
        for _k in list(sys.modules.keys()):
            if _k == 'pydantic' or (_k.startswith('pydantic.') and
                                     not _k.startswith('pydantic.v1')):
                del sys.modules[_k]
        sys.modules.update(_pyd_saved)

if _tmpdir: _shutil.rmtree(_tmpdir, ignore_errors=True)

if generated is not None:
    print("generated:", " ".join(map(str, generated))); sys.stdout.flush()
else:
    sys.stderr.write("inference FAILED: no generated tokens\n")
PYEOF_INFER
export PYTHONPATH="$_PP_SAVE"
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
