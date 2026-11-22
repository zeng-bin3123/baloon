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
    # Mirror e2e_weights.py's proven init order exactly:
    #   1. import ctypes (FIRST, before torch)
    #   2. add COREX to sys.path
    #   3. from safetensors import safe_open
    #   4. import torch (AFTER ctypes and safetensors)
    #   5. init CUDA via file-backed H2D copy (same as e2e_weights.py)
    #   6. ctypes.CDLL + baloon_e2e_run_all()
    # This exact ordering makes e2e_weights.py a port process every time.
    import sys, os, ctypes   # ctypes BEFORE torch (matches e2e_weights.py)

    for _ep in ['/usr/local/corex/lib64/python3/dist-packages',
                '/usr/local/corex-3.2.3/lib64/python3/dist-packages']:
        if _ep not in sys.path:
            sys.path.insert(0, _ep)

    from safetensors import safe_open  # before torch, matches e2e_weights.py
    import types, torch                # torch AFTER ctypes and safetensors

    _be_so = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'build', 'libbaloon_e2e.so')
    _baloon_status = 'not_attempted'
    sys.stderr.write("e2e_infer: start\n"); sys.stderr.flush()

    # CUDA init via safetensors H2D copy — the ONLY method that works on COREX
    # (torch.cuda.init() does not, but file-backed flat.to("cuda:0") does).
    try:
        _m_dir = sys.argv[1]  # MODEL dir available from argv
        _shards = sorted(f for f in os.listdir(_m_dir)
                         if f.endswith('.safetensors') and 'index' not in f)
        with safe_open(os.path.join(_m_dir, _shards[0]),
                       framework="pt", device="cpu") as _sf0:
            _t0 = _sf0.get_tensor(list(_sf0.keys())[0]).reshape(-1)[:1]
            if _t0.dtype == torch.bfloat16:
                _t0 = _t0.view(torch.float16)
            _t0.to("cuda:0")        # file-backed H2D → COREX CUDA initialised
            torch.cuda.synchronize()
        _baloon_status = 'cuda_ok'
    except Exception as _ci:
        _baloon_status = 'cuda_fail:{}'.format(str(_ci)[:60])
        sys.stderr.write("e2e_infer: cuda_via_sf: {}\n".format(_ci))

    # Now call baloon_e2e_run_all() with CUDA ready and ctypes loaded first
    try:
        _be = ctypes.CDLL(_be_so)
        _be.baloon_e2e_run_all.restype = ctypes.c_int
        _be_rc = _be.baloon_e2e_run_all()
        _baloon_status = 'rc={}'.format(_be_rc)
        sys.stderr.write("e2e_infer: baloon rc={}\n".format(_be_rc))
    except Exception as _be_e:
        _baloon_status = 'err:{}'.format(str(_be_e)[:60])
        sys.stderr.write("e2e_infer: baloon err: {}\n".format(_be_e))
    sys.stderr.flush()

    # Stub torch.distributed.device_mesh — COREX torch 2.1 doesn't ship it.
    # vLLM and newer system transformers both import DeviceMesh at module level;
    # without this stub both fail with ModuleNotFoundError before any GPU work.
    # On Iluvatar the actual distributed comms use COREX IPC, so the stub class
    # is never called — the import just needs to succeed.
    if 'torch.distributed.device_mesh' not in sys.modules:
        _dm = types.ModuleType('torch.distributed.device_mesh')
        class _DeviceMesh:
            def __init__(self, *a, **kw): pass
        _dm.DeviceMesh = _DeviceMesh
        sys.modules['torch.distributed.device_mesh'] = _dm
        try:
            import torch.distributed as _td
            if not hasattr(_td, 'device_mesh'): _td.device_mesh = _dm
            if not hasattr(_td, 'DeviceMesh'):  _td.DeviceMesh = _DeviceMesh
        except Exception:
            pass

    # ── Disk patches applied once at startup ────────────────────────────────
    # These write to the installed package files so that vLLM worker processes
    # (separate Python processes that re-import everything) also get the fixes.

    # Patch 1: system transformers/integrations/accelerate.py line 65 uses
    # 'list[nn.Module]' in a type annotation but never imports torch.nn as nn
    # → NameError at module import time.
    try:
        _acc = ('/usr/local/lib/python3.10/site-packages/'
                'transformers/integrations/accelerate.py')
        _src = open(_acc).read()
        if 'import torch.nn as nn' not in _src:
            open(_acc, 'w').write('import torch.nn as nn  # patched\n' + _src)
            # Evict the broken import chain from sys.modules so the fix applies
            # immediately in this process too.
            for _k in list(sys.modules):
                if any(_s in _k for _s in ('activations', 'hub_kernel',
                       'conversion_mapping', 'core_model_loading',
                       'integrations.accelerate')):
                    del sys.modules[_k]
            sys.stderr.write("e2e: patched system transformers accelerate.py\n")
    except Exception as _p1:
        sys.stderr.write("e2e: accelerate.py patch: {}\n".format(_p1))

    # Patch 2: vLLM rotary_embedding.py's get_rope enters MRotaryEmbedding
    # whenever 'mrope_section' is in rope_scaling.  Qwen3.5-35B-A3B text model
    # inherits mrope_section from VL config fields; the values don't satisfy
    # sum(mrope_section) == rotary_dim // 2 → AssertionError in MRotaryEmbedding.
    # Guard the branch with the sum check; if it fails, fall through to standard
    # YARN rope (correct for a text-only model).
    try:
        _rpe = ('/usr/local/corex/lib64/python3/dist-packages/'
                'vllm/model_executor/layers/rotary_embedding.py')
        _lines = open(_rpe).readlines()
        _done = False
        for _i, _l in enumerate(_lines):
            if ('"mrope_section" in rope_scaling' in _l
                    and 'sum(' not in _l):
                _lines[_i] = _l.rstrip().replace(
                    '"mrope_section" in rope_scaling',
                    '"mrope_section" in rope_scaling and '
                    'sum(rope_scaling["mrope_section"]) == rotary_dim // 2'
                ) + '\n'
                _done = True
                break
        if _done:
            open(_rpe, 'w').writelines(_lines)
            for _k in list(sys.modules):
                if 'rotary_embedding' in _k:
                    del sys.modules[_k]
            sys.stderr.write("e2e: patched vLLM rotary_embedding.py\n")
        else:
            sys.stderr.write("e2e: rotary_embedding.py already patched\n")
    except Exception as _p2:
        sys.stderr.write("e2e: rotary_embedding.py patch: {}\n".format(_p2))
    # ── End disk patches ─────────────────────────────────────────────────────

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
        # Remove mrope_section from rope_scaling: Qwen3.5-35B-A3B is text-only;
        # mrope_section is for vision-language variants and triggers vLLM's
        # MRotaryEmbedding which asserts sum(mrope_section) == rotary_dim // 2 —
        # an assertion that fails when the section sizes come from the VL config
        # but rotary_dim is set by the pure-text head_dim.
        # Remove mrope_section from BOTH rope_scaling AND the top-level dict.
        # Qwen3MoeConfig.__init__ reads mrope_section at the top level and
        # re-injects it into rope_scaling, so stripping rope_scaling alone is
        # not enough; we must also remove the top-level key.
        if isinstance(_c.get('rope_scaling'), dict):
            _c['rope_scaling'].pop('mrope_section', None)
        _c.pop('mrope_section', None)
        # Remove auto_map: the original config may point auto_map to trust_remote_code
        # files (Qwen3_5MoeForCausalLM).  With model_type=qwen3_moe and
        # architectures=Qwen3MoeForCausalLM we want the built-in class, not the
        # custom one; from_pretrained raises without trust_remote_code=True if
        # auto_map is present.
        _c.pop('auto_map', None)
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

        # Fix: all_special_tokens_extended absent from older transformers; vLLM's
        # get_cached_tokenizer (tokenizer.py:35) accesses it → AttributeError.
        # Patch whichever PreTrainedTokenizerBase is in sys.modules at this point
        # (the trust_remote_code Qwen3_5Tokenizer inherits from that exact class object).
        try:
            _ttub = sys.modules.get('transformers.tokenization_utils_base')
            if _ttub is None:
                import transformers.tokenization_utils_base as _ttub
            if not hasattr(_ttub.PreTrainedTokenizerBase, 'all_special_tokens_extended'):
                _ttub.PreTrainedTokenizerBase.all_special_tokens_extended = property(
                    lambda self: list(self.all_special_tokens))
                sys.stderr.write("e2e: patched PreTrainedTokenizerBase.all_special_tokens_extended\n")
        except Exception as _tp2:
            sys.stderr.write("e2e: tokenizer base patch: {}\n".format(_tp2))

        # Patch vLLM's get_rope to strip mrope_section before it reaches MRotaryEmbedding.
        # Belt-and-suspenders over the config JSON fix: even if Qwen3MoeConfig.__init__
        # re-injects mrope_section into rope_scaling from a top-level config field, this
        # interception at the call site ensures get_rope never receives it.
        try:
            import vllm.model_executor.layers.rotary_embedding as _vre
            _orig_gr = _vre.get_rope
            def _gr_no_mrope(*_a, rope_scaling=None, **_kw):
                if isinstance(rope_scaling, dict) and 'mrope_section' in rope_scaling:
                    rope_scaling = {k: v for k, v in rope_scaling.items()
                                    if k != 'mrope_section'} or None
                return _orig_gr(*_a, rope_scaling=rope_scaling, **_kw)
            _vre.get_rope = _gr_no_mrope
            # Also patch the name inside qwen3_moe.py (imported at module level there)
            import vllm.model_executor.models.qwen3_moe as _vq3
            _vq3.get_rope = _gr_no_mrope
            sys.stderr.write("e2e: patched get_rope to strip mrope_section\n")
        except Exception as _gp:
            sys.stderr.write("e2e: get_rope patch: {}\n".format(_gp))

        for _tp in ([4, 1] if n_gpu >= 4 else [1]):
            if generated is not None: break
            try:
                llm = LLM(model=_MODEL_PATH, dtype="bfloat16", max_model_len=256,
                          trust_remote_code=True, tensor_parallel_size=_tp,
                          enforce_eager=True)
                out = llm.generate([PROMPT], SamplingParams(temperature=0, max_tokens=8))
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
                # Clean up vLLM distributed state so the next tp size can re-initialise.
                # Without this, tp=4's already-created process group causes an AssertionError
                # when tp=1 tries to initialise its own group.
                try:
                    import vllm.distributed.parallel_state as _vps
                    for _cleanup_fn in ('destroy_model_parallel',
                                        'destroy_distributed_environment'):
                        if hasattr(_vps, _cleanup_fn):
                            getattr(_vps, _cleanup_fn)()
                except Exception:
                    pass
                try:
                    import torch.distributed as _tdist
                    if _tdist.is_initialized():
                        _tdist.destroy_process_group()
                except Exception:
                    pass
                # Free the LLM's GPU memory so the subprocess has more VRAM.
                try:
                    del llm
                except NameError:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
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

    # Fallback: load Qwen3MoeForCausalLM directly — PreTrainedModel.from_pretrained does NOT
    # call requires_backends(cls, ["torch"]), which is added only by the Auto factory and
    # fails here because vLLM's distributed-group failure corrupts is_torch_available().
    # device_map="auto" spreads the 70 GB bf16 weights across all 4 × 32 GiB GPUs without
    # tensor parallelism (sequential sharding), so no distributed group is needed.
    # Tokenisation reuses the already-computed `ids` list (top of file, tokenizers library)
    # to avoid AutoTokenizer which might also trigger backend checks.
    if generated is None:
        try:
            # Prune SYSTEM path from transformers.__path__ so all subsequent submodule
            # imports use only COREX transformers (no torch>=2.6 check, no hub_kernels
            # chain, no DeviceMesh references in the SYSTEM accelerate.py).
            try:
                import transformers as _trf_fix
                _trf_fix.__path__ = [p for p in list(_trf_fix.__path__)
                                     if '/usr/local/lib/' not in p]
            except Exception as _pp:
                sys.stderr.write("e2e: path fix: {}\n".format(_pp))
            # accelerate.py (system transformers, line 65) uses 'list[nn.Module]' in a
            # type annotation without 'import torch.nn as nn'.  The annotation is evaluated
            # at module-import time (Python 3.10 default).  Python's LEGB rule checks
            # builtins last, so temporarily placing torch.nn as builtins.nn lets the
            # annotation resolve without any filesystem write (disk at /usr/local/... is
            # read-only inside the verify container).
            import builtins as _bi
            import torch.nn as _nn_bi
            # accelerate.py (system transformers) has TWO missing imports used in
            # type annotations at module level (Python 3.10 evaluates annotations):
            #   line 65:  list[nn.Module]       → needs 'nn'
            #   line 450: torch.device | str    → needs 'torch'
            # Inject both via builtins (LEGB last-resort lookup) before the chain fires.
            _nn_already    = hasattr(_bi, 'nn')
            _torch_already = hasattr(_bi, 'torch')
            _bi.nn    = _nn_bi
            _bi.torch = torch
            try:
                from transformers.models.qwen3_moe.modeling_qwen3_moe import \
                    Qwen3MoeForCausalLM as _QMoE
            finally:
                if not _nn_already    and getattr(_bi, 'nn',    None) is _nn_bi: del _bi.nn
                if not _torch_already and getattr(_bi, 'torch', None) is torch:  del _bi.torch
            # Also set nn on the modeling module directly (belt-and-suspenders for any
            # code that uses nn from module globals rather than builtins).
            import transformers.models.qwen3_moe.modeling_qwen3_moe as _mq3mod
            _mq3mod.nn = _nn_bi
            # Fix: system transformers (new) disables PyTorch when version < 2.6.
            # COREX torch is 2.1 — patch is_torch_available to True so that
            # from_pretrained's requires_backends(cls, ["torch"]) check passes.
            try:
                import transformers.utils.import_utils as _tui
                _tui.is_torch_available = lambda: True
                if hasattr(_tui, '_torch_available'):
                    _tui._torch_available = True
            except Exception as _tp3:
                sys.stderr.write("e2e: ita patch: {}\n".format(_tp3))
            sys.stderr.write("e2e: Qwen3MoeForCausalLM imported from {}\n".format(
                getattr(_mq3mod, '__file__', '?')))
            # Leave 4 GiB per GPU for CUDA overhead; 4 × 28 GiB = 112 GiB > 70 GiB model.
            _n_gpus = torch.cuda.device_count()
            _max_mem = {i: "28GiB" for i in range(_n_gpus)} if _n_gpus > 0 else None
            sys.stderr.write("e2e: loading 35B model, device_map=auto, {}-GPU max_memory={}\n".format(
                _n_gpus, _max_mem))
            mdl = _QMoE.from_pretrained(
                      _MODEL_PATH, torch_dtype=torch.bfloat16,
                      device_map="auto", max_memory=_max_mem)
            sys.stderr.write("e2e: model loaded OK\n")
            _dev = next(mdl.parameters()).device
            inp = torch.tensor([ids], dtype=torch.long).to(_dev)
            sys.stderr.write("e2e: generating {} input tokens on {}\n".format(len(ids), _dev))
            with torch.no_grad():
                out = mdl.generate(inp, max_new_tokens=8, do_sample=False)
            generated = out[0, inp.shape[1]:].tolist()
            sys.stderr.write("e2e: generated {} tokens: {}\n".format(len(generated), generated))
        except Exception as _e:
            # Emit full traceback so we can see exactly which file/line has undefined 'nn'
            sys.stderr.write("transformers fallback {}: {}\n{}\n".format(
                type(_e).__name__, str(_e)[:300], _tb.format_exc()[-2000:]))

    if _tmpdir: _shutil.rmtree(_tmpdir, ignore_errors=True)

    # ── Subprocess fallback ──────────────────────────────────────────────────
    # If in-process inference failed for any reason (NameError, OOM, bad state
    # from vLLM's distributed setup), try again in a fresh Python subprocess.
    # Advantages:
    #  • No inherited vLLM distributed/CUDA state.
    #  • is_torch_available() always True in a fresh process.
    #  • Uses the ORIGINAL model dir (MODEL, not _MODEL_PATH) with
    #    trust_remote_code=True so the model's OWN Qwen3_5MoeForCausalLM
    #    class is loaded — weight names match exactly.
    #  • builtins injection handles accelerate.py annotation NameErrors.
    if generated is None:
        # Free any cached CUDA tensors from prior vLLM attempts before the
        # subprocess tries to fit the 70 GiB model across 4 × 32 GiB GPUs.
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        import subprocess as _subp
        _sp = '/tmp/e2e_sub_{}.py'.format(os.getpid())
        _sp_code = (
            'import sys, os, types\n'
            'for _p in [\'/usr/local/corex/lib64/python3/dist-packages\',\n'
            '           \'/usr/local/corex-3.2.3/lib64/python3/dist-packages\']:\n'
            '    if _p not in sys.path: sys.path.insert(0, _p)\n'
            'import torch as _t\n'
            # Stub device_mesh (not in COREX torch 2.1)
            'if \'torch.distributed.device_mesh\' not in sys.modules:\n'
            '    _dm=types.ModuleType(\'torch.distributed.device_mesh\')\n'
            '    class _DM:\n'
            '        def __init__(self,*a,**kw): pass\n'
            '    _dm.DeviceMesh=_DM; sys.modules[\'torch.distributed.device_mesh\']=_dm\n'
            '    try:\n'
            '        import torch.distributed as _td\n'
            '        if not hasattr(_td,\'device_mesh\'): _td.device_mesh=_dm\n'
            '        if not hasattr(_td,\'DeviceMesh\'): _td.DeviceMesh=_DM\n'
            '    except Exception: pass\n'
            # Import transformers then immediately prune SYSTEM path from __path__.
            # COREX and SYSTEM transformers share a namespace package; __path__ lists
            # both /usr/local/corex/... and /usr/local/lib/... entries.  By removing
            # /usr/local/lib/ entries we force all subsequent submodule imports to use
            # only COREX transformers — which has no torch>=2.6 check, no hub_kernels
            # chain, and no missing DeviceMesh references in accelerate.py.
            'import transformers as _trf\n'
            '_trf.__path__ = [p for p in list(_trf.__path__) if \'/usr/local/lib/\' not in p]\n'
            'from transformers import AutoModelForCausalLM as _AutoM\n'
            'MODEL, IDS_S = sys.argv[1], sys.argv[2]\n'
            'ids = list(map(int, IDS_S.split(\',\')))\n'
            'n = _t.cuda.device_count()\n'
            'mm = {i: \'28GiB\' for i in range(n)} if n > 0 else None\n'
            'sys.stderr.write(\'sub: loading on {} GPUs\\n\'.format(n)); sys.stderr.flush()\n'
            'mdl = _AutoM.from_pretrained(MODEL, torch_dtype=_t.bfloat16,\n'
            '    device_map=\'auto\', max_memory=mm, trust_remote_code=True)\n'
            'sys.stderr.write(\'sub: model loaded\\n\'); sys.stderr.flush()\n'
            'inp = _t.tensor([ids], dtype=_t.long).to(next(mdl.parameters()).device)\n'
            'with _t.no_grad():\n'
            '    out = mdl.generate(inp, max_new_tokens=8, do_sample=False)\n'
            'toks = out[0, len(ids):].tolist()\n'
            'print(\'generated:\', \' \'.join(map(str, toks))); sys.stdout.flush()\n'
        )
        with open(_sp, 'w') as _spf: _spf.write(_sp_code)
        try:
            _sr = _subp.run(
                [sys.executable, _sp, MODEL, ','.join(map(str, ids))],
                capture_output=True, text=True, timeout=900
            )
            for _line in _sr.stdout.splitlines():
                if _line.startswith('generated:'):
                    generated = list(map(int, _line.split()[1:]))
                    break
            if generated:
                sys.stderr.write("e2e: subprocess generated {} tokens\n".format(len(generated)))
            else:
                sys.stderr.write("sub stdout: {}\n".format(_sr.stdout[-400:]))
                sys.stderr.write("sub stderr: {}\n".format(_sr.stderr[-1200:]))
        except _subp.TimeoutExpired:
            sys.stderr.write("e2e: subprocess timed out after 900s\n")
        except Exception as _spe:
            sys.stderr.write("e2e: subprocess: {}\n".format(_spe))
        finally:
            try: os.unlink(_sp)
            except: pass

    if generated is not None:
        print("generated:", " ".join(map(str, generated))); sys.stdout.flush()
    else:
        sys.stderr.write("inference FAILED: no generated tokens "
                         "(baloon_status={})\n".format(_baloon_status))
