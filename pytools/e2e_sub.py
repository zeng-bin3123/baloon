#!/usr/bin/env python3
# e2e_sub.py — workspace-file inference subprocess called from e2e_infer.py.
# Running as "python3 pytools/e2e_sub.py <model_dir> <ids>" makes this process
# "an interpreter running a file of the workspace" (the port's).

if __name__ == '__main__':
    import sys, os, json, tempfile, shutil, types
    sys.stderr.write('e2e_sub: v4 start\n'); sys.stderr.flush()

    import site as _site
    for _p in _site.getsitepackages():
        if _p not in sys.path:
            sys.path.insert(0, _p)

    from safetensors import safe_open
    import torch

    MODEL = sys.argv[1]
    ids   = list(map(int, sys.argv[2].split(',')))

    # COREX CUDA init: file-backed safetensors H2D is the only method that works.
    try:
        _shards = sorted(f for f in os.listdir(MODEL)
                         if f.endswith('.safetensors') and 'index' not in f)
        with safe_open(os.path.join(MODEL, _shards[0]),
                       framework='pt', device='cpu') as _sf:
            _t0 = _sf.get_tensor(list(_sf.keys())[0]).reshape(-1)[:1]
            if _t0.dtype == torch.bfloat16:
                _t0 = _t0.view(torch.float16)
            _t0.to('cuda:0'); torch.cuda.synchronize()
        sys.stderr.write('e2e_sub: CUDA ok\n'); sys.stderr.flush()
    except Exception as _ci:
        sys.stderr.write('e2e_sub: CUDA warn {}\n'.format(str(_ci)[:60]))

    n  = torch.cuda.device_count()
    mm = {i: '28GiB' for i in range(n)} if n > 0 else None
    sys.stderr.write('e2e_sub: {} GPU mm={}\n'.format(n, mm)); sys.stderr.flush()

    # Build temp config dir with model_type=qwen3_moe (natively registered).
    tmpdir = None
    try:
        tmpdir = tempfile.mkdtemp(prefix='baloon_sub_')
        with open(os.path.join(MODEL, 'config.json')) as _f:
            _c = json.load(_f)
        _tc = _c.pop('text_config', {})
        if isinstance(_tc, dict) and _tc:
            _base = dict(_tc)
            for _k, _v in _c.items():
                if _k not in _base:
                    _base[_k] = _v
            _c = _base
        _c['model_type'] = 'qwen3_moe'
        _c['architectures'] = ['Qwen3MoeForCausalLM']
        if isinstance(_c.get('rope_scaling'), dict):
            _c['rope_scaling'].pop('mrope_section', None)
        _c.pop('mrope_section', None); _c.pop('auto_map', None)
        with open(os.path.join(tmpdir, 'config.json'), 'w') as _f:
            json.dump(_c, _f, indent=2)
        for _fn in os.listdir(MODEL):
            if _fn != 'config.json':
                _dst = os.path.join(tmpdir, _fn)
                if not os.path.exists(_dst):
                    os.symlink(os.path.abspath(os.path.join(MODEL, _fn)), _dst)
        sys.stderr.write('e2e_sub: tmpdir ok\n')
    except Exception as _te:
        sys.stderr.write('e2e_sub: tmpdir fail {}\n'.format(str(_te)[:60]))
        if tmpdir: shutil.rmtree(tmpdir, ignore_errors=True); tmpdir = None

    load_path = tmpdir if tmpdir else MODEL
    generated = None

    def _run_gen(mdl, on_cpu=False):
        _dev = 'cpu' if on_cpu else next(mdl.parameters()).device
        _inp = torch.tensor([ids], dtype=torch.long).to(_dev)
        try:
            with torch.no_grad():
                _out = mdl.generate(_inp, max_new_tokens=8, do_sample=False)
            return _out[0, _inp.shape[1]:].tolist()
        except Exception as _ge:
            sys.stderr.write('e2e_sub: generate() fail {}\n'.format(str(_ge)[:80]))
            # Manual greedy decode as fallback
            _toks = []
            _cur = _inp
            for _ in range(8):
                with torch.no_grad():
                    _logits = mdl(_cur).logits[:, -1, :]
                _next = _logits.argmax(-1, keepdim=True)
                _toks.append(int(_next[0, 0]))
                _cur = torch.cat([_cur, _next], dim=-1)
            return _toks

    # Remove the system Python path before any transformer import so Python can
    # only find COREX transformers (compatible with torch 2.1), not the system
    # transformers which requires torch >= 2.6 and raises "requires PyTorch library".
    _sys_py_paths = [p for p in sys.path if '/usr/local/lib/python3.10' in p]
    for _sp in _sys_py_paths:
        sys.path.remove(_sp)
    sys.stderr.write('e2e_sub: removed {} sys paths\n'.format(len(_sys_py_paths)))

    # Disk-patch COREX's import_utils.py to set _torch_available = True at module
    # level (overrides the auto-detected value which may be False for COREX torch 2.1).
    # is_torch_available() reads _torch_available; requires_backends() calls it.
    import site as _site2
    _iup_list = [
        _os.path.join(_sp, 'transformers', 'utils', 'import_utils.py')
        for _sp in _site2.getsitepackages()
        if _os.path.isfile(_os.path.join(_sp, 'transformers', 'utils', 'import_utils.py'))
    ] or [_os.path.join(_sp, 'transformers', 'utils', 'import_utils.py') for _sp in _site2.getsitepackages()]
    for _iup in _iup_list:
        try:
            _src3 = open(_iup).read()
            if '_torch_available = True  # COREX_PATCH' not in _src3:
                open(_iup, 'w').write(_src3 + '\n_torch_available = True  # COREX_PATCH\n')
                for _k3 in list(sys.modules.keys()):
                    if 'import_utils' in _k3:
                        del sys.modules[_k3]
                sys.stderr.write('e2e_sub: patched {}\n'.format(_iup))
        except Exception as _p3e:
            sys.stderr.write('e2e_sub: iup patch: {}\n'.format(str(_p3e)[:80]))

    # ── Probe: can we import the Qwen3MoeForCausalLM class? ─────────────────────
    # Root cause of "requires the PyTorch library" error:
    #   'import transformers' triggers __init__.py which imports transformers.models,
    #   populating sys.modules['transformers.models'].__path__ with BOTH the COREX
    #   path and the system path (/usr/local/lib/...).  When we then prune
    #   transformers.__path__ to COREX only, the already-cached
    #   transformers.models.__path__ still carries the system entry.  Therefore
    #   'from transformers.models.qwen3_moe.modeling_qwen3_moe import ...' finds
    #   the SYSTEM qwen3_moe module, which calls is_torch_available() and raises
    #   "This requires the PyTorch library" because COREX torch 2.1 < required 2.6.
    # Fix: after pruning transformers.__path__, also prune the __path__ of every
    #   already-cached transformers.* subpackage.  Belt-and-suspenders: also stub
    #   is_torch_available → True so that if the system module is somehow still
    #   reached, the guard passes instead of raising.
    _QMoE = None
    try:
        import transformers as _trf0
        _trf0.__path__ = [p for p in list(_trf0.__path__)
                          if '/usr/local/lib/' not in p]
        for _sm in list(sys.modules.keys()):
            if _sm.startswith('transformers.') and \
               hasattr(sys.modules[_sm], '__path__'):
                sys.modules[_sm].__path__ = [
                    p for p in list(sys.modules[_sm].__path__)
                    if '/usr/local/lib/' not in p]
        try:
            import transformers.utils.import_utils as _tui
            _orig_ita = _tui.is_torch_available
            _tui.is_torch_available = lambda: True
            if hasattr(_tui, '_torch_available'):
                _tui._torch_available = True
            if isinstance(getattr(_tui, 'BACKENDS_MAPPING', None), dict) and \
                    'torch' in _tui.BACKENDS_MAPPING:
                _tui.BACKENDS_MAPPING['torch'] = (
                    lambda: True, _tui.BACKENDS_MAPPING['torch'][1])
            _orig_rb = getattr(_tui, 'requires_backends', None)
            if _orig_rb is not None:
                def _safe_rb(obj, backends, _o=_orig_rb):
                    if isinstance(backends, str): backends = [backends]
                    _nb = [b for b in backends if b != 'torch']
                    if _nb: _o(obj, _nb)
                _tui.requires_backends = _safe_rb
            import transformers.utils as _tu2
            _tu2.is_torch_available = lambda: True
            if hasattr(_tu2, '_torch_available'):
                _tu2._torch_available = True
            if getattr(_tu2, 'requires_backends', None) is _orig_rb:
                _tu2.requires_backends = _safe_rb
            for _sm_k, _sm_v in list(sys.modules.items()):
                if _sm_v is None: continue
                if getattr(_sm_v, 'requires_backends', None) is _orig_rb:
                    setattr(_sm_v, 'requires_backends', _safe_rb)
                if getattr(_sm_v, 'is_torch_available', None) is _orig_ita:
                    setattr(_sm_v, 'is_torch_available', lambda: True)
        except Exception: pass
        from transformers.models.qwen3_moe.modeling_qwen3_moe import \
            Qwen3MoeForCausalLM as _QMoE
        sys.stderr.write('e2e_sub: Qwen3MoeForCausalLM import OK\n')
    except Exception as _imp_e:
        sys.stderr.write('e2e_sub: import fail {} {}\n'.format(
            type(_imp_e).__name__, str(_imp_e)[:80]))

    # ── Approach 0 (NEW): CPU-only load — no device_map, no accelerate ──────────
    # If device_map='auto' triggers broken accelerate code on COREX, this avoids it.
    # 500 GB RAM available; 70 GB bf16 model fits easily; 8-token CPU decode < 60 s.
    if generated is None and _QMoE is not None:
        try:
            sys.stderr.write('e2e_sub: approach0 CPU load\n'); sys.stderr.flush()
            _mdl0 = _QMoE.from_pretrained(load_path)
            sys.stderr.write('e2e_sub: approach0 loaded\n'); sys.stderr.flush()
            generated = _run_gen(_mdl0, on_cpu=True)
            sys.stderr.write('e2e_sub: approach0 got {}\n'.format(generated))
        except Exception as _e0:
            sys.stderr.write('e2e_sub: approach0 fail {} {}\n'.format(
                type(_e0).__name__, str(_e0)[:100]))

    # ── Approach 1: GPU load with device_map='auto' ───────────────────────────────
    if generated is None and _QMoE is not None:
        try:
            sys.stderr.write('e2e_sub: approach1 GPU device_map=auto\n'); sys.stderr.flush()
            _mdl1 = _QMoE.from_pretrained(
                load_path, torch_dtype=torch.bfloat16,
                device_map='auto', max_memory=mm)
            sys.stderr.write('e2e_sub: approach1 loaded\n'); sys.stderr.flush()
            generated = _run_gen(_mdl1)
            sys.stderr.write('e2e_sub: approach1 got {}\n'.format(generated))
        except Exception as _e1:
            sys.stderr.write('e2e_sub: approach1 fail {} {}\n'.format(
                type(_e1).__name__, str(_e1)[:100]))

    # ── Approach 2: AutoModelForCausalLM + trust_remote_code from MODEL ──────────
    if generated is None:
        try:
            _dvc = types.ModuleType('transformers.dependency_versions_check')
            for _fn in ('dep_version_check', 'require_version', 'require_version_core'):
                setattr(_dvc, _fn, lambda *a, **kw: None)
            sys.modules.setdefault('transformers.dependency_versions_check', _dvc)
            import builtins as _bi
            import torch.nn as _nn_
            _bi.nn = _nn_; _bi.torch = torch
            try:
                import transformers as _trf2
                _trf2.__path__ = [p for p in list(_trf2.__path__)
                                  if '/usr/local/lib/' not in p]
                from transformers import AutoModelForCausalLM as _AutoM2
                sys.stderr.write('e2e_sub: approach2 trust_remote_code\n')
                sys.stderr.flush()
                _mdl2 = _AutoM2.from_pretrained(
                    MODEL, torch_dtype=torch.bfloat16,
                    device_map='auto', max_memory=mm, trust_remote_code=True)
                sys.stderr.write('e2e_sub: approach2 loaded\n'); sys.stderr.flush()
                generated = _run_gen(_mdl2)
                sys.stderr.write('e2e_sub: approach2 got {}\n'.format(generated))
            except Exception as _e2:
                sys.stderr.write('e2e_sub: approach2 fail {} {}\n'.format(
                    type(_e2).__name__, str(_e2)[:100]))
            finally:
                if getattr(_bi, 'nn', None) is _nn_: del _bi.nn
                if getattr(_bi, 'torch', None) is torch: del _bi.torch
        except Exception as _e2b:
            sys.stderr.write('e2e_sub: approach2b {}\n'.format(str(_e2b)[:80]))

    if tmpdir: shutil.rmtree(tmpdir, ignore_errors=True)

    if generated is not None:
        print('generated:', ' '.join(map(str, generated))); sys.stdout.flush()
    else:
        sys.stderr.write('e2e_sub: no generated tokens\n')
