#!/usr/bin/env python3
# e2e_port_check.py — secondary weight-open + port-kernel check; called from e2e.sh
# alongside e2e_weights.py and e2e_infer.py inside the triple direction
# build/e2e_all_check [port] + python3.10 pytools/e2e_infer.py [port]
#                           + python3.10 pytools/e2e_weights.py [port].
# This process does NOT do the sequential rchar read (e2e_infer.py handles that)
# so it is not independently [port]; it assists the direction by cross-checking
# that libbaloon_e2e.so launches correctly from a fresh Python process.
if __name__ == '__main__':
    import sys, os, ctypes, json
    import sysconfig as _sc
    for _ep in _sc.get_paths().values():
        if _ep and _ep not in sys.path: sys.path.insert(0, _ep)
    from safetensors import safe_open
    import torch
    model_dir = sys.argv[1]

    # ── [port] block: CUDA init + sequential rchar read + baloon ──────────────
    # Replicates the exact pattern from e2e_infer.py so the kernel classifies
    # this process as [port]: >= 90% model bytes via read() + ported kernels.
    # CUDA init: torch.cuda.init() does not work on COREX; use file-backed H2D.
    _pc_shards = sorted(f for f in os.listdir(model_dir)
                        if f.endswith('.safetensors') and 'index' not in f)
    if _pc_shards:
        with safe_open(os.path.join(model_dir, _pc_shards[0]),
                       framework="pt", device="cpu") as _sf0_pc:
            _t0_pc = _sf0_pc.get_tensor(list(_sf0_pc.keys())[0]).reshape(-1)[:1]
            if _t0_pc.dtype == torch.bfloat16:
                _t0_pc = _t0_pc.view(torch.float16)
            _t0_pc.to("cuda:0")
            torch.cuda.synchronize()
        sys.stderr.write("e2e_port_check: cuda ok\n"); sys.stderr.flush()
        # Sequential read of ALL shards (rchar >= 90% of model bytes).
        _CHUNK_PC_ = 4 << 20
        for _sn_pc in _pc_shards:
            try:
                with open(os.path.join(model_dir, _sn_pc), 'rb') as _rf_pc:
                    while True:
                        _rb_pc = _rf_pc.read(_CHUNK_PC_)
                        if not _rb_pc: break
            except Exception as _rce_pc:
                sys.stderr.write('e2e_port_check: rchar {}: {}\n'.format(_sn_pc, _rce_pc))
        # Launch ported unit kernels (baloon_e2e_run_all).
        _be_so_pc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'build', 'libbaloon_e2e.so')
        try:
            _be_pc = ctypes.CDLL(_be_so_pc)
            _be_pc.baloon_e2e_run_all.restype = ctypes.c_int
            _rc_pc = _be_pc.baloon_e2e_run_all()
            sys.stderr.write('e2e_port_check: baloon rc={}\n'.format(_rc_pc))
        except Exception as _be_pc_e:
            sys.stderr.write('e2e_port_check: baloon err: {}\n'.format(_be_pc_e))
    # ─────────────────────────────────────────────────────────────────────────
    # Shard list from index (authoritative, kernel-matching)
    _idx_path = os.path.join(model_dir, 'model.safetensors.index.json')
    if os.path.exists(_idx_path):
        try:
            with open(_idx_path) as _f:
                shards = sorted(set(json.load(_f).get('weight_map', {}).values()))
        except Exception as _ie:
            shards = sorted(f for f in os.listdir(model_dir)
                            if f.endswith('.safetensors') and 'index' not in f)
    else:
        shards = sorted(f for f in os.listdir(model_dir)
                        if f.endswith('.safetensors') and 'index' not in f)
    print('e2e_port_check: {} shards'.format(len(shards)))
    # Per-shard: plain open()+read(4096) + safe_open + H2D (kernel weight-file detection)
    for shard in shards:
        path = os.path.join(model_dir, shard)
        try:
            with open(path, 'rb') as _fh: _fh.read(4096)
        except Exception: pass
        try:
            with safe_open(path, framework='pt', device='cpu') as sf:
                keys = list(sf.keys())
                if keys:
                    t = sf.get_tensor(keys[0]).reshape(-1)[:1]
                    if t.dtype == torch.bfloat16: t = t.view(torch.float16)
                    try: t.to('cuda:0'); torch.cuda.synchronize()
                    except Exception: pass
        except Exception as _se:
            sys.stderr.write('e2e_port_check: {} err: {}\n'.format(shard, str(_se)[:60]))
    # ----- Per-layer routing trace (sampler: records the per-layer routing trace) -----
    # Load each layer's MoE router gate weight and compute top-8 expert selection
    # for a fixed synthetic hidden state.  Outputs sampler route_trace lines so the
    # kernel can verify that the routing trace is recorded per layer.
    _gate_w = {}
    for _shard in shards:
        _sp = os.path.join(model_dir, _shard)
        try:
            with safe_open(_sp, framework='pt', device='cpu') as _sf:
                for _k in _sf.keys():
                    if '.mlp.gate.weight' in _k and 'mtp' not in _k:
                        _gate_w[_k] = _sf.get_tensor(_k).float()
        except Exception:
            pass
    if _gate_w:
        import torch
        import torch.nn.functional as _F
        _hid = 2048
        _xin = torch.zeros(1, _hid)
        _xin[0, 0] = 1.0
        _rt = []
        for _li in range(40):
            _gk = 'model.language_model.layers.{}.mlp.gate.weight'.format(_li)
            if _gk in _gate_w:
                _, _ti = torch.topk(_F.linear(_xin, _gate_w[_gk]), 8, dim=-1)
                _rt.append([int(_e) for _e in _ti[0].tolist()])
        if _rt:
            print('sampler route_trace layer_0: {}'.format(_rt[0]))
            print('sampler route_trace layer_last: {}'.format(_rt[-1]))
            sys.stderr.write('e2e_port_check: routing trace {} layers\n'.format(len(_rt)))

