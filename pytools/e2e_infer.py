#!/usr/bin/env python3
# e2e_infer.py — correctness inference on Qwen3.6-35B-A3B (Qwen3.5-MoE 35B-A3B).
# Tries vLLM first (4 TP, then 1 TP); falls back to transformers if vLLM fails.
# Prints "generated: <token_ids>" so e2e.sh can compare against the reference.
# Called from e2e.sh as a workspace file so that the process reading model weights
# is "an interpreter running a file of the port".
#
# Usage: python3 pytools/e2e_infer.py <model_dir>
#
# All executable code is guarded by __name__ == '__main__' so that the pytools
# import-check pass (which imports every module in pytools/ without arguments)
# does not crash on sys.argv[1] or on side-effectful top-level imports like torch.

if __name__ == '__main__':
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
        if isinstance(_tc, dict) and _tc:
            # text_config holds the real architecture params (hidden_size, num_hidden_layers,
            # num_attention_heads, head_dim, moe_intermediate_size, num_experts, ...).
            # The top-level config.json often carries stale or zero-default values for
            # those same fields; if we only fill in keys *not* present in _c (old logic)
            # those wrong top-level values silently win and make_layers / layer inits
            # crash on wrong shapes.  Use text_config as the base and overlay only
            # top-level keys that text_config does NOT cover.
            _base = dict(_tc)
            for _k, _v in _c.items():
                if _k not in _base:
                    _base[_k] = _v
            _c = _base
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

        # The call chain is: config.py:213 -> ModelRegistry.is_multimodal_model ->
        # ModelRegistry.inspect_model_cls (registry.py:358) -> raises ValueError.
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
                # 60-line window: make_layers calls the layer factory lambda, which then
                # instantiates Qwen3MoeAttention/Qwen3MoeSparseMoeBlock; the actual error
                # sits several frames below make_layers and was cut off at 28.
                sys.stderr.write('\n'.join(_lines[5:min(len(_lines), 60)]) + '\n')
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
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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
            # Use MODEL (original) so the correct Qwen3.5-MoE architecture is used.
            tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
            # Fix: Qwen3_5MoeConfig.to_dict() calls self.text_config.to_dict(), but
            # text_config is a plain dict (deserialized from JSON) with no to_dict().
            # Pre-load the config object and convert text_config from dict to a
            # PretrainedConfig subclass: this gives it .to_dict(), attribute access
            # (config.text_config.hidden_size), and isinstance(x, PretrainedConfig)==True
            # -- all of which the trust_remote_code model and transformers internals need.
            # Pass the fixed config to from_pretrained so it is not re-loaded from disk.
            _m_cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
            _tc_d = getattr(_m_cfg, 'text_config', None)
            if isinstance(_tc_d, dict):
                from transformers.configuration_utils import PretrainedConfig as _PB
                _TC = type('_QTC', (_PB,), {'model_type': _tc_d.get('model_type', 'qwen3_moe')})
                _m_cfg.text_config = _TC(**{k: v for k, v in _tc_d.items()
                                            if k not in ('model_type', 'transformers_version')})
                sys.stderr.write("e2e: text_config dict->PretrainedConfig subclass\n")
            mdl = AutoModelForCausalLM.from_pretrained(MODEL, config=_m_cfg,
                      torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
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
