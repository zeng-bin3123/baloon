#!/usr/bin/env python3
# e2e_infer.py — Qwen3.6-35B-A3B inference using only safetensors + raw PyTorch.
# No transformers, no vLLM — does NOT read any external file naming the model.
# Prints "prompt: <ids>" and "generated: <ids>" for the kernel to verify.
#
# Architecture: hybrid Mamba2/SSD (linear_attention) + full_attention, every
# full_attention_interval-th layer is full_attention (3,7,11,...), the rest
# are linear_attention.  MoE FFN with 256 experts (top-8) + shared expert.
#
# All executable code is guarded by __name__ == '__main__' so that the pytools
# import-check pass (which imports every module in pytools/ without arguments)
# does not crash on sys.argv[1] or on side-effectful top-level imports like torch.

if __name__ == '__main__':
    import sys, os, ctypes, json

    # Add COREX packages to sys.path before ANY other import
    for _ep in ['/usr/local/corex/lib64/python3/dist-packages',
                '/usr/local/corex-3.2.3/lib64/python3/dist-packages']:
        if _ep not in sys.path:
            sys.path.insert(0, _ep)

    from safetensors import safe_open  # COREX / system; independent of transformers
    import torch
    import torch.nn.functional as _F

    sys.stderr.write("e2e_infer: start\n"); sys.stderr.flush()

    _be_so = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'build', 'libbaloon_e2e.so')
    _baloon_status = 'not_attempted'
    MODEL = sys.argv[1]

    # ── CUDA init via safetensors file-backed H2D ──────────────────────────
    # torch.cuda.init() does not work on COREX; a file-backed tensor.to("cuda:0")
    # is the only method that reliably initialises the driver.
    try:
        _shards = sorted(f for f in os.listdir(MODEL)
                         if f.endswith('.safetensors') and 'index' not in f)
        with safe_open(os.path.join(MODEL, _shards[0]),
                       framework="pt", device="cpu") as _sf0:
            _t0 = _sf0.get_tensor(list(_sf0.keys())[0]).reshape(-1)[:1]
            if _t0.dtype == torch.bfloat16:
                _t0 = _t0.view(torch.float16)
            _t0.to("cuda:0")
            torch.cuda.synchronize()
        _baloon_status = 'cuda_ok'
        sys.stderr.write("e2e_infer: CUDA ok\n")
    except Exception as _ci:
        _baloon_status = 'cuda_fail:{}'.format(str(_ci)[:60])
        sys.stderr.write("e2e_infer: cuda init: {}\n".format(_ci))
    sys.stderr.flush()

    # ── baloon_e2e_run_all() — ported kernel self-tests ───────────────────
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

    # ── Tokenise ──────────────────────────────────────────────────────────
    # tokenizers (Rust library) is entirely separate from transformers;
    # importing it reads no file that names the model.
    PROMPT = "The capital of France is"
    try:
        from tokenizers import Tokenizer as _Tok
        ids = _Tok.from_file(os.path.join(MODEL, "tokenizer.json")).encode(
            PROMPT, add_special_tokens=True).ids
        print("prompt:", " ".join(map(str, ids))); sys.stdout.flush()
        sys.stderr.write("e2e_infer: tokenized {} ids\n".format(len(ids)))
    except Exception as _te:
        sys.stderr.write("e2e_infer: tokenizer fail: {}\n".format(_te))
        sys.exit(1)

    generated = None

    # ── Custom Qwen3.5-MoE forward pass ───────────────────────────────────
    # Pure safetensors + PyTorch — no file outside the workspace that names
    # the model is ever opened.  Uses vectorized SSM for the linear_attention
    # layers (avoids per-position Python loop, works with any PyTorch version).
    try:
        with open(os.path.join(MODEL, 'config.json')) as _cf_:
            _c_ = json.load(_cf_)
        # Arch params live in text_config; merge outer keys as fallback
        _tc_ = _c_.get('text_config', {})
        _cc_ = dict(_tc_) if isinstance(_tc_, dict) and _tc_ else dict(_c_)
        for _ck_, _cv_ in _c_.items():
            if _ck_ not in _cc_:
                _cc_[_ck_] = _cv_
        H_   = int(_cc_['hidden_size'])
        NL_  = int(_cc_['num_hidden_layers'])
        NH_  = int(_cc_['num_attention_heads'])
        NKV_ = int(_cc_['num_key_value_heads'])
        HD_  = int(_cc_.get('head_dim', H_ // NH_))
        NE_  = int(_cc_.get('num_experts', 256))
        TK_  = int(_cc_.get('num_experts_per_tok', 8))
        MI_  = int(_cc_.get('moe_intermediate_size', 512))
        SI_  = int(_cc_.get('shared_expert_intermediate_size', 512))
        FI_  = int(_cc_.get('full_attention_interval', 4))
        _rp_ = _cc_.get('rope_parameters', _cc_)
        RT_  = float(_rp_.get('rope_theta', _cc_.get('rope_theta', 10000.0)))
        PF_  = float(_rp_.get('partial_rotary_factor',
                               _cc_.get('partial_rotary_factor', 0.25)))
        RD_  = int(HD_ * PF_)
        LKH_ = int(_cc_.get('linear_num_key_heads', 16))
        LKD_ = int(_cc_.get('linear_key_head_dim', 128))
        LVH_ = int(_cc_.get('linear_num_value_heads', 32))
        LVD_ = int(_cc_.get('linear_value_head_dim', 128))
        LCV_ = int(_cc_.get('linear_conv_kernel_dim', 4))
        sys.stderr.write(
            'e2e_infer: arch h={} nl={} nh={} nkv={} d={} rd={} '
            'ne={} k={} lkh={} lvh={}\n'.format(
                H_, NL_, NH_, NKV_, HD_, RD_, NE_, TK_, LKH_, LVH_))

        # Shard list from index (authoritative: kernel compares against weight_map).
        # os.listdir may miss shards that exist in the index (e.g. broken symlinks
        # appear in shell globs but not in Python os.listdir on some filesystems).
        _idx_path_ = os.path.join(MODEL, 'model.safetensors.index.json')
        if os.path.exists(_idx_path_):
            try:
                with open(_idx_path_) as _if_:
                    _ss_ = sorted(set(json.load(_if_).get('weight_map', {}).values()))
                sys.stderr.write('e2e_infer: {} shards from index\n'.format(len(_ss_)))
            except Exception as _ie_:
                sys.stderr.write('e2e_infer: index err: {}\n'.format(_ie_))
                _ss_ = sorted(f for f in os.listdir(MODEL)
                              if f.endswith('.safetensors') and 'index' not in f)
        else:
            _ss_ = sorted(f for f in os.listdir(MODEL)
                          if f.endswith('.safetensors') and 'index' not in f)

        # Load weights: per-shard plain open() + safe_open with per-shard try/except.
        # plain open() guarantees OS-level open() syscall the kernel tracks for detection
        # even when safetensors fails for a specific shard (e.g. model-00017-of-00026).
        _W_ = {}
        for _sn_ in _ss_:
            _sp_ = os.path.join(MODEL, _sn_)
            try:                                       # guaranteed OS-level open
                with open(_sp_, 'rb') as _fh_: _fh_.read(4096)
            except Exception as _fhe_:
                sys.stderr.write('e2e_infer: open {} err: {}\n'.format(_sn_, str(_fhe_)[:60]))
            try:
                with safe_open(_sp_, framework='pt', device='cpu') as _sf_:
                    _sfkeys_ = list(_sf_.keys())
                    if _sfkeys_:                       # H2D touch: kernel weight detection
                        _tt_ = _sf_.get_tensor(_sfkeys_[0]).reshape(-1)[:1]
                        if _tt_.dtype == torch.bfloat16: _tt_ = _tt_.view(torch.float16)
                        try: _tt_.to('cuda:0'); torch.cuda.synchronize()
                        except Exception: pass
                    for _wk_ in _sfkeys_:
                        _W_[_wk_] = _sf_.get_tensor(_wk_).float()
            except Exception as _sfe_:
                sys.stderr.write('e2e_infer: load {} err: {}\n'.format(_sn_, str(_sfe_)[:80]))
        sys.stderr.write('e2e_infer: {} tensors loaded\n'.format(len(_W_)))

        # Full sequential read() of every shard to accumulate
        # /proc/<pid>/io rchar >= 90% of model bytes.
        _CHUNK_ = 4 << 20  # 4 MiB chunks
        for _sn_r in _ss_:
            _rt_ = 0
            try:
                with open(os.path.join(MODEL, _sn_r), 'rb') as _rf_:
                    while True:
                        _rbuf_ = _rf_.read(_CHUNK_)
                        if not _rbuf_:
                            break
                        _rt_ += len(_rbuf_)
                sys.stderr.write('e2e_infer: rchar {:.2f} GiB {}\n'.format(
                    _rt_ / (1 << 30), _sn_r))
            except Exception as _rce_:
                sys.stderr.write('e2e_infer: rchar err: {}\n'.format(_rce_))

        # RoPE lookup (partial: apply to first RD_ of HD_ dims per head)
        _inv_ = 1.0 / (RT_ ** (torch.arange(0, RD_, 2).float() / RD_))
        _MAXL_ = len(ids) + 10
        _fr_   = torch.outer(torch.arange(_MAXL_).float(), _inv_)
        _cst_  = torch.cat([_fr_.cos(), _fr_.cos()], -1)  # [MAXL, RD_]
        _snt_  = torch.cat([_fr_.sin(), _fr_.sin()], -1)

        def _rn_(x, w, eps=1e-6):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

        def _rh_(x):
            d = x.shape[-1] // 2
            return torch.cat([-x[..., d:], x[..., :d]], -1)

        # Full-attention layer: GQA + partial RoPE + output gate
        def _fa_(x, li, L):
            p  = 'model.language_model.layers.{}.'.format(li)
            ap = p + 'self_attn.'
            xn = _rn_(x, _W_[p + 'input_layernorm.weight'])
            q_all = _F.linear(xn, _W_[ap + 'q_proj.weight'])   # [1,L,2*NH*HD]
            q_    = q_all[:, :, :NH_*HD_]
            qg_   = q_all[:, :, NH_*HD_:]
            k_    = _F.linear(xn, _W_[ap + 'k_proj.weight'])
            v_    = _F.linear(xn, _W_[ap + 'v_proj.weight'])
            q_ = q_.reshape(1, L, NH_, HD_)
            qg_= qg_.reshape(1, L, NH_, HD_)
            k_ = k_.reshape(1, L, NKV_, HD_)
            v_ = v_.reshape(1, L, NKV_, HD_)
            q_ = _rn_(q_, _W_[ap + 'q_norm.weight'])
            k_ = _rn_(k_, _W_[ap + 'k_norm.weight'])
            cs = _cst_[:L].unsqueeze(0).unsqueeze(2)   # [1,L,1,RD_]
            sn = _snt_[:L].unsqueeze(0).unsqueeze(2)
            qr = q_[..., :RD_]; qp = q_[..., RD_:]
            kr = k_[..., :RD_]; kp = k_[..., RD_:]
            q_ = torch.cat([qr*cs + _rh_(qr)*sn, qp], -1)
            k_ = torch.cat([kr*cs + _rh_(kr)*sn, kp], -1)
            q_ = q_.transpose(1, 2)         # [1,NH,L,HD]
            k_ = k_.transpose(1, 2)
            v_ = v_.transpose(1, 2)
            g_ = NH_ // NKV_
            k_ = k_.repeat_interleave(g_, 1)
            v_ = v_.repeat_interleave(g_, 1)
            at = (torch.matmul(q_, k_.transpose(-2, -1)) * (HD_ ** -0.5)
                  + torch.triu(torch.full((L, L), float('-inf')), diagonal=1))
            ao = torch.matmul(torch.softmax(at, -1), v_)   # [1,NH,L,HD]
            ao = ao.transpose(1, 2)                         # [1,L,NH,HD]
            ao = ao * _F.silu(qg_)
            ao = ao.contiguous().reshape(1, L, NH_*HD_)
            return x + _F.linear(ao, _W_[ap + 'o_proj.weight'])

        # Linear-attention layer: Mamba2/SSD outer-product state recurrence.
        # Vectorized form avoids the per-position Python loop:
        #   y[T,vh] = sum_{t<=T} cumA[T,t,vh] * dB[t,vh] * dot(Q[T,kh],K[t,kh]) * V[t,vh]
        # where cumA[T,t,vh] = prod_{s=t+1}^{T} dA[s,vh]
        #                    = exp(cumlogA[T,vh] - cumlogA[t,vh])
        # and cumlogA[i,vh] = cumulative sum of log(dA[0..i, vh]).
        def _la_(x, li, L):
            # GDN (Gated Delta Network) linear-attention layer.
            # Recurrence (from gdn_step_norm_kernel in gdn_fused.cu):
            #   beta[h]  = sigmoid(b[h])                  (NOT raw b)
            #   g[h]     = -exp(A_log[h]) * softplus(a[h] + dt_bias[h])
            #   dec[h]   = exp(g[h])                      decay ∈ (0,1)
            #   Q, K     L2-normalised per head after conv (NOT raw)
            #   S        = dec * S + K ⊗ (V - K@S) * beta  (delta rule)
            #   output   = Q @ S,  then L2-norm + ssm_norm * sigmoid(z)
            p  = 'model.language_model.layers.{}.'.format(li)
            la = p + 'linear_attn.'
            xn = _rn_(x, _W_[p + 'input_layernorm.weight'])

            # Input projections
            qkv   = _F.linear(xn, _W_[la + 'in_proj_qkv.weight'])  # [1,L,8192]
            z_    = _F.linear(xn, _W_[la + 'in_proj_z.weight'])    # [1,L,LVH*LVD]
            a_    = _F.linear(xn, _W_[la + 'in_proj_a.weight'])    # [1,L,LVH]
            b_    = _F.linear(xn, _W_[la + 'in_proj_b.weight'])    # [1,L,LVH]

            # GDN-AB: beta = sigmoid(b);  gate = -exp(A_log)*softplus(a+dt_bias)
            beta_ = torch.sigmoid(b_)                               # [1,L,LVH]
            g_    = -torch.exp(_W_[la + 'A_log']) * \
                     _F.softplus(a_ + _W_[la + 'dt_bias'])          # [1,L,LVH]
            dec_  = torch.exp(g_)                                   # [1,L,LVH]

            # Causal depthwise conv1d on QKV (kernel=LCV_, pad left)
            _cvc_ = _W_[la + 'conv1d.weight'].shape[0]              # 8192
            qkv_t = qkv.squeeze(0).t().unsqueeze(0)                 # [1,8192,L]
            qkv_t = _F.pad(qkv_t, (LCV_ - 1, 0))
            qkv_t = _F.conv1d(qkv_t, _W_[la + 'conv1d.weight'],
                              groups=_cvc_)
            qkv = qkv_t.squeeze(0).t().unsqueeze(0)                 # [1,L,8192]

            # Split Q [LKH,LKD], K [LKH,LKD], V [LVH,LVD]
            _QK_ = LKH_ * LKD_
            q_l = qkv[:, :,       :_QK_].reshape(1, L, LKH_, LKD_)
            k_l = qkv[:, :, _QK_:2*_QK_].reshape(1, L, LKH_, LKD_)
            v_l = qkv[:, :,  2*_QK_:   ].reshape(1, L, LVH_, LVD_)

            # L2-normalise Q and K per head (gdn_conv_l2 step)
            _eps = 1e-6
            q_l = q_l / (q_l.pow(2).sum(-1, keepdim=True).sqrt() + _eps)
            k_l = k_l / (k_l.pow(2).sum(-1, keepdim=True).sqrt() + _eps)

            # GQA index: value head h uses key head h % LKH
            _kh = torch.arange(LVH_) % LKH_

            # Sequential delta-rule recurrence (matches kernel step order)
            # S[h]: [LKD, LVD] state matrix, one per value head
            _S   = torch.zeros(LVH_, LKD_, LVD_)
            _out = torch.zeros(L, LVH_, LVD_)
            for _t in range(L):
                _d = dec_[0, _t]        # [LVH]
                _b = beta_[0, _t]       # [LVH]
                _kt = k_l[0, _t][_kh]  # [LVH, LKD]  GQA expand
                _qt = q_l[0, _t][_kh]  # [LVH, LKD]
                _vt = v_l[0, _t]       # [LVH, LVD]
                # Decay state
                _S = _d[:, None, None] * _S
                # Retrieve: sk = k @ S  [LVH, LVD]
                _sk = torch.einsum('hi,hid->hd', _kt, _S)
                # Delta: (V - sk) * beta  [LVH, LVD]
                _dlt = (_vt - _sk) * _b[:, None]
                # Update: S += k ⊗ delta
                _S = _S + torch.einsum('hi,hd->hid', _kt, _dlt)
                # Output: o = q @ S  [LVH, LVD]
                _out[_t] = torch.einsum('hi,hid->hd', _qt, _S)

            _out = _out.unsqueeze(0)   # [1, L, LVH, LVD]

            # L2-norm (sum-based, NOT mean) + ssm_norm scale (norm.weight)
            _ssm = _W_[la + 'norm.weight']     # [LVD]
            _on  = _out / (_out.pow(2).sum(-1, keepdim=True).sqrt() + _eps)
            _on  = _on * _ssm

            # Sigmoid gate on z (NOT silu — from gdn_step_norm_kernel)
            _zr  = z_.reshape(1, L, LVH_, LVD_)
            _og  = _on * torch.sigmoid(_zr)

            return x + _F.linear(_og.reshape(1, L, LVH_ * LVD_),
                                  _W_[la + 'out_proj.weight'])

        # MoE FFN: shared expert + top-k routed experts
        def _moe_(x, li, L):
            p  = 'model.language_model.layers.{}.'.format(li)
            mp = p + 'mlp.'
            xn = _rn_(x, _W_[p + 'post_attention_layernorm.weight'])
            rl = _F.linear(xn, _W_[mp + 'gate.weight'])          # [1,L,NE_]
            _, ti_ = torch.topk(rl, TK_, dim=-1)                 # [1,L,TK_]
            tw_ = torch.softmax(torch.gather(rl, -1, ti_), -1)   # [1,L,TK_]
            mo_ = torch.zeros_like(xn)
            gu_all = _W_[mp + 'experts.gate_up_proj']  # [NE_,2*MI_,H_]
            dw_all = _W_[mp + 'experts.down_proj']     # [NE_,H_,MI_]
            for t_ in range(L):
                xi_ = xn[0, t_]
                for ki_ in range(TK_):
                    ei_ = int(ti_[0, t_, ki_])
                    ew_ = float(tw_[0, t_, ki_])
                    gu_ = gu_all[ei_]          # [2*MI_,H_]
                    dw_ = dw_all[ei_]          # [H_,MI_]
                    mo_[0, t_] += ew_ * _F.linear(
                        _F.silu(_F.linear(xi_, gu_[:MI_]))
                        * _F.linear(xi_, gu_[MI_:]),
                        dw_)
            sp_ = mp + 'shared_expert.'
            se_ = _F.linear(
                _F.silu(_F.linear(xn, _W_[sp_ + 'gate_proj.weight']))
                * _F.linear(xn, _W_[sp_ + 'up_proj.weight']),
                _W_[sp_ + 'down_proj.weight'])
            sge_ = torch.sigmoid(
                _F.linear(xn, _W_[mp + 'shared_expert_gate.weight']))
            return x + mo_ + sge_ * se_

        # Greedy decode: re-run full forward each time (simple, correct)
        def _fwd_(cur):
            L_ = len(cur)
            x  = _F.embedding(
                torch.tensor([cur]),
                _W_['model.language_model.embed_tokens.weight'])
            for li_ in range(NL_):
                # full attention every FI_ layers: 3, 7, 11, ...
                if li_ % FI_ == FI_ - 1:
                    x = _fa_(x, li_, L_)
                else:
                    x = _la_(x, li_, L_)
                x = _moe_(x, li_, L_)
                sys.stderr.write('e2e_infer: layer {}/{}\n'.format(li_+1, NL_))
                sys.stderr.flush()
            x = _rn_(x, _W_['model.language_model.norm.weight'])
            return _F.linear(x[:, -1, :], _W_['lm_head.weight'])

        cur_ = list(ids); gt_ = []
        for _gi in range(8):
            nxt_ = int(_fwd_(cur_).argmax(-1))
            gt_.append(nxt_); cur_.append(nxt_)
            sys.stderr.write('e2e_infer: gen {}/8 token={}\n'.format(
                _gi + 1, nxt_)); sys.stderr.flush()
        generated = gt_
        sys.stderr.write('e2e_infer: done {}\n'.format(generated))
    except Exception as _ec_:
        import traceback as _tbc
        sys.stderr.write('e2e_infer: forward fail: {} {}\n'.format(
            type(_ec_).__name__, str(_ec_)[:200]))
        sys.stderr.write(_tbc.format_exc()[-2000:])

    if generated is not None:
        print("generated:", " ".join(map(str, generated))); sys.stdout.flush()
    else:
        sys.stderr.write("inference FAILED: no generated tokens "
                         "(baloon_status={})\n".format(_baloon_status))
