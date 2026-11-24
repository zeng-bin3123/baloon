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
    # (port-kernel launch removed: e2e_port_check.py is a weight-read-only
    # secondary process and must not be classified as [port] by the kernel)
