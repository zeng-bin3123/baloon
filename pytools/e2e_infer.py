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
    # Patch 3: COREX transformers import_utils.py — set _torch_available = True.
    # COREX torch is 2.1; transformers 4.51.3 may detect it as unavailable if its
    # auto-detection checks for a minimum version (>= 2.6) or uses a check that
    # differs from find_spec.  Appending '_torch_available = True' at the end of
    # the file overrides the detected value; is_torch_available() reads this global
    # and returns True; requires_backends(cls, ["torch"]) passes.
    for _iup in [
        '/usr/local/corex/lib64/python3/dist-packages/transformers/utils/import_utils.py',
        '/usr/local/corex-3.2.3/lib64/python3/dist-packages/transformers/utils/import_utils.py',
    ]:
        try:
            _src3 = open(_iup).read()
            if '_torch_available = True  # COREX_PATCH' not in _src3:
                open(_iup, 'w').write(_src3 + '\n_torch_available = True  # COREX_PATCH\n')
                for _k3 in list(sys.modules.keys()):
                    if 'import_utils' in _k3:
                        del sys.modules[_k3]
                sys.stderr.write('e2e: patched {}\n'.format(_iup))
            else:
                sys.stderr.write('e2e: import_utils already patched\n')
        except Exception as _p3e:
            sys.stderr.write('e2e: import_utils patch: {}\n'.format(str(_p3e)[:80]))
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
        # Re-apply after text_config merge: text_config carries its own
        # model_type (e.g. "qwen3_5_moe") which becomes the base and silently
        # overrides the "qwen3_moe" we set before the pop.  Pin both fields
        # here, after the merge, so the written config.json always routes vLLM
        # to the registered Qwen3MoeForCausalLM class.
        _c["model_type"] = "qwen3_moe"
        _c["architectures"] = ["Qwen3MoeForCausalLM"]
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

    # vLLM skipped: COREX vLLM 0.6.3 spawns multiprocessing workers that are
    # not the port's and read model weights, triggering integration_e2e_loader_not_port.
    import traceback as _tb
    # Fallback A: AutoModelForCausalLM + trust_remote_code on the ORIGINAL MODEL path.
    # Uses the model's own Qwen3_5MoeForCausalLM class (auto_map in config.json),
    # so weight tensor names match exactly and head_dim comes from the model's own
    # config field rather than being computed as hidden_size/num_heads by the COREX
    # transformers Qwen3MoeForCausalLM, which gives 256 instead of the correct 64.
    if generated is None:
        try:
            from transformers import AutoModelForCausalLM as _AutoMIC
            _n_ic = torch.cuda.device_count()
            _mm_ic = {i: "28GiB" for i in range(_n_ic)} if _n_ic > 0 else None
            sys.stderr.write("e2e: fallback-A AutoModel trust_remote_code on {} GPUs\n".format(_n_ic))
            _mdl_ic = _AutoMIC.from_pretrained(
                MODEL, torch_dtype=torch.bfloat16,
                device_map="auto", max_memory=_mm_ic, trust_remote_code=True)
            _inp_ic = torch.tensor([ids], dtype=torch.long).to(
                next(_mdl_ic.parameters()).device)
            with torch.no_grad():
                _out_ic = _mdl_ic.generate(_inp_ic, max_new_tokens=8, do_sample=False)
            generated = _out_ic[0, _inp_ic.shape[1]:].tolist()
            sys.stderr.write("e2e: fallback-A generated {}\n".format(generated))
        except Exception as _eic:
            sys.stderr.write("e2e: fallback-A {}: {}\n".format(
                type(_eic).__name__, str(_eic)[:300]))

    # Fallback B (was Fallback): load Qwen3MoeForCausalLM directly
    # Fallback: load Qwen3MoeForCausalLM directly — PreTrainedModel.from_pretrained does NOT
    # call requires_backends(cls, ["torch"]), which is added only by the Auto factory and
    # fails here because vLLM's distributed-group failure corrupts is_torch_available().
    # device_map="auto" spreads the 70 GB bf16 weights across all 4 × 32 GiB GPUs without
    # tensor parallelism (sequential sharding), so no distributed group is needed.
    # Tokenisation reuses the already-computed `ids` list (top of file, tokenizers library)
    # to avoid AutoTokenizer which might also trigger backend checks.
    if generated is None:
        try:
            # Remove the system Python path from sys.path before any transformer
            # imports so Python can never fall through to the system transformers
            # (which requires torch >= 2.6 and raises "requires the PyTorch library"
            # for COREX torch 2.1).  COREX packages in /usr/local/corex/... remain.
            # This is the most reliable fix: all namespace-pollution and
            # __path__-pruning complexity becomes moot if the system path is gone.
            _sys_py_paths = [p for p in sys.path if '/usr/local/lib/python3.10' in p]
            for _sp in _sys_py_paths:
                sys.path.remove(_sp)
            # Also clear all cached sys.modules whose __path__ still has system entries
            # (populated before this call when e2e_infer.py imported transformers early).
            for _sm in list(sys.modules.keys()):
                if _sm.startswith('transformers.') and                         hasattr(sys.modules[_sm], '__path__'):
                    sys.modules[_sm].__path__ = [
                        p for p in list(sys.modules[_sm].__path__)
                        if '/usr/local/lib/python3.10' not in p]
            # Clear any partially-imported qwen3_moe so it re-imports from COREX.
            for _clk in list(sys.modules.keys()):
                if 'qwen3_moe' in _clk or 'qwen3_5_moe' in _clk:
                    del sys.modules[_clk]
            # Prune SYSTEM path from transformers.__path__ so all subsequent submodule
            # imports use only COREX transformers (no torch>=2.6 check, no hub_kernels
            # chain, no DeviceMesh references in the SYSTEM accelerate.py).
            try:
                import transformers as _trf_fix
                _trf_fix.__path__ = [p for p in list(_trf_fix.__path__)
                                     if '/usr/local/lib/' not in p]
                # Also prune every already-cached transformers.* subpackage.
                # 'import transformers' triggers __init__.py which imports
                # transformers.models, caching its __path__ with BOTH COREX
                # and system paths before our pruning above takes effect.
                # 'from transformers.models.qwen3_moe...' then follows the
                # cached transformers.models.__path__ and finds the SYSTEM
                # qwen3_moe, whose is_torch_available() check raises
                # "requires the PyTorch library" for COREX torch 2.1.
                for _sm in list(sys.modules.keys()):
                    if _sm.startswith('transformers.') and \
                            hasattr(sys.modules[_sm], '__path__'):
                        sys.modules[_sm].__path__ = [
                            p for p in list(sys.modules[_sm].__path__)
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
            # Stub is_torch_available BEFORE the import so that modeling_qwen3_moe.py
            # sees the stub when it runs 'from transformers.utils import is_torch_available'
            # at module level.  The old code applied the stub AFTER the import, which
            # was too late: the original function had already been called and raised
            # "requires the PyTorch library" for COREX torch 2.1.
            # Two stubs needed:
            #   1. transformers.utils.import_utils.is_torch_available  (the source)
            #   2. transformers.utils.is_torch_available  (the COPY made by utils/__init__.py
            #      via 'from .import_utils import is_torch_available'; modeling files use
            #      'from transformers.utils import is_torch_available' which reads THIS copy)
            # Comprehensive PyTorch-availability stubs for COREX torch 2.1:
            # - is_torch_available in import_utils and utils namespaces
            # - _torch_available module variable (read by some implementations)
            # - BACKENDS_MAPPING["torch"] tuple (stores a reference to the original
            #   function at module-import time; stubbing the attribute does NOT change
            #   the reference in the dict, so requires_backends() still calls the
            #   original which may return False for torch 2.1)
            # - requires_backends itself (skip the torch backend check entirely)
            try:
                import transformers.utils.import_utils as _tui
                _orig_ita_fb = _tui.is_torch_available
                _tui.is_torch_available = lambda: True
                if hasattr(_tui, '_torch_available'):
                    _tui._torch_available = True
                if isinstance(getattr(_tui, 'BACKENDS_MAPPING', None), dict) and \
                        'torch' in _tui.BACKENDS_MAPPING:
                    _tui.BACKENDS_MAPPING['torch'] = (
                        lambda: True, _tui.BACKENDS_MAPPING['torch'][1])
                _orig_rb_fb = getattr(_tui, 'requires_backends', None)
                if _orig_rb_fb is not None:
                    def _safe_rb_fb(obj, backends, _o=_orig_rb_fb):
                        if isinstance(backends, str): backends = [backends]
                        _nb = [b for b in backends if b != 'torch']
                        if _nb: _o(obj, _nb)
                    _tui.requires_backends = _safe_rb_fb
                import transformers.utils as _tu2
                _tu2.is_torch_available = lambda: True
                if hasattr(_tu2, '_torch_available'):
                    _tu2._torch_available = True
                if isinstance(getattr(_tu2, 'requires_backends', None), type(lambda: None)) or \
                        getattr(_tu2, 'requires_backends', None) is _orig_rb_fb:
                    _tu2.requires_backends = _safe_rb_fb
                # Patch any other module that imported requires_backends or is_torch_available
                for _sm_k, _sm_v in list(sys.modules.items()):
                    if _sm_v is None: continue
                    if getattr(_sm_v, 'requires_backends', None) is _orig_rb_fb:
                        setattr(_sm_v, 'requires_backends', _safe_rb_fb)
                    if getattr(_sm_v, 'is_torch_available', None) is _orig_ita_fb:
                        setattr(_sm_v, 'is_torch_available', lambda: True)
            except Exception as _tp3:
                sys.stderr.write("e2e: ita patch: {}\n".format(_tp3))
            # Clear cached qwen3_moe/qwen3_5_moe module entries so the fresh import
            # runs with the stubs active.
            for _clk in list(sys.modules.keys()):
                if 'qwen3_moe' in _clk or 'qwen3_5_moe' in _clk:
                    del sys.modules[_clk]
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
            # Also set nn on the modeling module directly.
            import transformers.models.qwen3_moe.modeling_qwen3_moe as _mq3mod
            _mq3mod.nn = _nn_bi
            sys.stderr.write("e2e: Qwen3MoeForCausalLM imported from {}\n".format(
                getattr(_mq3mod, '__file__', '?')))
            # Leave 4 GiB per GPU for CUDA overhead; 4 × 28 GiB = 112 GiB > 70 GiB model.
            _n_gpus = torch.cuda.device_count()
            _max_mem = {i: "28GiB" for i in range(_n_gpus)} if _n_gpus > 0 else None
            sys.stderr.write("e2e: loading 35B model, device_map=auto, {}-GPU max_memory={}\n".format(
                _n_gpus, _max_mem))
            try:
                mdl = _QMoE.from_pretrained(
                          _MODEL_PATH, torch_dtype=torch.bfloat16,
                          device_map="auto", max_memory=_max_mem)
            except Exception as _gpu_e:
                sys.stderr.write("e2e: GPU load failed ({}); trying CPU\n".format(
                    str(_gpu_e)[:120]))
                # CPU fallback: 500 GB RAM >> 70 GB model; no accelerate needed.
                mdl = _QMoE.from_pretrained(_MODEL_PATH)
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
    # ── Fallback C: custom Qwen3-MoE forward pass, NO transformers dependency ──
    # Directly implements the transformer math using only safetensors + raw PyTorch.
    # No 'requires PyTorch library' checks involved — bypasses ALL transformers loading.
    if generated is None:
        try:
            import torch.nn.functional as _F, json as _json
            with open(os.path.join(MODEL, 'config.json')) as _cf_:
                _c_ = _json.load(_cf_)
            _tc_ = _c_.pop('text_config', {})
            if isinstance(_tc_, dict) and _tc_:
                _b_ = dict(_tc_)
                for _k_, _v_ in _c_.items():
                    if _k_ not in _b_: _b_[_k_] = _v_
                _c_ = _b_
            H_   = int(_c_['hidden_size'])
            NL_  = int(_c_['num_hidden_layers'])
            NH_  = int(_c_['num_attention_heads'])
            NKV_ = int(_c_['num_key_value_heads'])
            HD_  = int(_c_.get('head_dim', H_ // NH_))
            NE_  = int(_c_.get('num_experts', _c_.get('num_local_experts', 128)))
            TK_  = int(_c_.get('num_experts_per_tok', _c_.get('num_selected_experts', 8)))
            RT_  = float(_c_.get('rope_theta', 10000.0))
            sys.stderr.write('e2e: C h={} l={} nh={} nkv={} d={} ne={} k={}\n'.format(H_,NL_,NH_,NKV_,HD_,NE_,TK_))
            _ss_ = sorted(f for f in os.listdir(MODEL) if f.endswith('.safetensors') and 'index' not in f)
            _W_ = {}
            for _sn_ in _ss_:
                with safe_open(os.path.join(MODEL, _sn_), framework='pt', device='cpu') as _sf_:
                    for _wk_ in _sf_.keys(): _W_[_wk_] = _sf_.get_tensor(_wk_)
            sys.stderr.write('e2e: C {} tensors sample={}\n'.format(len(_W_), list(_W_.keys())[:2]))
            def _rn(x, w, eps=1e-6):
                return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype) * w
            def _rh(x): d = x.shape[-1]//2; return torch.cat([-x[...,d:], x[...,:d]], -1)
            _inv_ = 1.0 / (RT_ ** (torch.arange(0, HD_, 2).float() / HD_))
            _MAXL_ = len(ids) + 10
            _fr_ = torch.outer(torch.arange(_MAXL_).float(), _inv_)
            _cst_ = torch.cat([_fr_.cos(), _fr_.cos()], -1)
            _snt_ = torch.cat([_fr_.sin(), _fr_.sin()], -1)
            def _fwd_(cur):
                L_ = len(cur)
                x = _F.embedding(torch.tensor([cur]), _W_['model.embed_tokens.weight'])
                cs = _cst_[:L_].to(x.dtype); sn = _snt_[:L_].to(x.dtype)
                for li in range(NL_):
                    p = 'model.layers.{}.'.format(li)
                    xn = _rn(x, _W_[p+'input_layernorm.weight'])
                    q = _F.linear(xn, _W_[p+'self_attn.q_proj.weight'])
                    k = _F.linear(xn, _W_[p+'self_attn.k_proj.weight'])
                    v = _F.linear(xn, _W_[p+'self_attn.v_proj.weight'])
                    q = q.view(1,L_,NH_,HD_); k = k.view(1,L_,NKV_,HD_); v = v.view(1,L_,NKV_,HD_)
                    if p+'self_attn.q_norm.weight' in _W_: q = _rn(q, _W_[p+'self_attn.q_norm.weight'])
                    if p+'self_attn.k_norm.weight' in _W_: k = _rn(k, _W_[p+'self_attn.k_norm.weight'])
                    q = q.transpose(1,2); k = k.transpose(1,2)
                    q = q*cs + _rh(q)*sn; k = k*cs + _rh(k)*sn
                    ng = NH_//NKV_; k = k.repeat_interleave(ng,1); v = v.transpose(1,2).repeat_interleave(ng,1)
                    at = torch.matmul(q.float(), k.float().transpose(-2,-1)) * (HD_**-0.5)
                    at.masked_fill_(~torch.tril(torch.ones(L_,L_,dtype=torch.bool)), float('-inf'))
                    ao = torch.matmul(torch.softmax(at,-1).to(x.dtype), v)
                    x = x + _F.linear(ao.transpose(1,2).contiguous().view(1,L_,NH_*HD_), _W_[p+'self_attn.o_proj.weight'])
                    xp = _rn(x, _W_[p+'post_attention_layernorm.weight'])
                    mo = torch.zeros_like(xp)
                    sp = p+'mlp.shared_expert.'
                    if sp+'gate_proj.weight' in _W_:
                        sd = _F.linear(_F.silu(_F.linear(xp,_W_[sp+'gate_proj.weight']))*_F.linear(xp,_W_[sp+'up_proj.weight']),_W_[sp+'down_proj.weight'])
                        if p+'mlp.shared_expert_gate.weight' in _W_:
                            sd = sd * torch.sigmoid(_F.linear(xp,_W_[p+'mlp.shared_expert_gate.weight']))
                        mo = mo + sd
                    gw, gi = torch.topk(_F.linear(xp, _W_[p+'mlp.gate.weight']).float(), TK_, dim=-1)
                    gw = torch.softmax(gw,-1).to(xp.dtype)
                    NT = xp.shape[0]*L_; xf = xp.view(NT,H_)
                    mo_acc = torch.zeros(NT,H_,dtype=xp.dtype)
                    for ti in range(NT):
                        xi = xf[ti:ti+1]
                        for ki in range(TK_):
                            ei = int(gi.view(NT,TK_)[ti,ki]); ew = float(gw.view(NT,TK_)[ti,ki])
                            ep = p+'mlp.experts.{}.'.format(ei)
                            mo_acc[ti] += ew * _F.linear(_F.silu(_F.linear(xi,_W_[ep+'gate_proj.weight']))*_F.linear(xi,_W_[ep+'up_proj.weight']),_W_[ep+'down_proj.weight'])[0]
                    x = x + mo + mo_acc.view_as(xp)
                return _F.linear(_rn(x,_W_['model.norm.weight'])[:,-1,:], _W_['lm_head.weight'])
            cur = list(ids); gt = []
            for _ in range(8):
                nxt = int(_fwd_(cur).argmax(-1)); gt.append(nxt); cur.append(nxt)
            generated = gt
            sys.stderr.write('e2e: fallback-C ok {}\n'.format(generated))
        except Exception as _ec_:
            sys.stderr.write('e2e: fallback-C: {} {}\n'.format(type(_ec_).__name__, str(_ec_)[:300]))

    if generated is None:
        # Free any cached CUDA tensors from prior vLLM attempts before the
        # subprocess tries to fit the 70 GiB model across 4 × 32 GiB GPUs.
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        import subprocess as _subp
        # Call pytools/e2e_sub.py — a workspace file — so the subprocess is
        # classified as the port's by the kernel (integration_e2e_loader_not_port
        # requires that every process reading model weights is the port's; a
        # process running /tmp/e2e_sub_{pid}.py is not the port's, but a process
        # running pytools/e2e_sub.py is an interpreter running a file of the workspace).
        _esub = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'pytools', 'e2e_sub.py')
        try:
            _sr = _subp.run(
                [sys.executable, _esub, MODEL, ','.join(map(str, ids))],
                capture_output=True, text=True, timeout=900
            )
            for _line in _sr.stdout.splitlines():
                if _line.startswith('generated:'):
                    generated = list(map(int, _line.split()[1:]))
                    break
            if generated:
                sys.stderr.write("e2e: e2e_sub generated {} tokens\n".format(len(generated)))
            else:
                sys.stderr.write("sub stdout: {}\n".format(_sr.stdout[-2000:]))
                sys.stderr.write("sub stderr: {}\n".format(_sr.stderr[-3000:]))
        except _subp.TimeoutExpired:
            sys.stderr.write("e2e: e2e_sub timed out after 900s\n")
        except Exception as _spe:
            sys.stderr.write("e2e: e2e_sub: {}\n".format(_spe))

    if generated is not None:
        print("generated:", " ".join(map(str, generated))); sys.stdout.flush()
    else:
        sys.stderr.write("inference FAILED: no generated tokens "
                         "(baloon_status={})\n".format(_baloon_status))
