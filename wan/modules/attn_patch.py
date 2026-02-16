import math
import torch
import triton
import triton.language as tl

import torch
import triton
import triton.language as tl

configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
    for BN in [32, 64, 128]\
    for s in [2, 3, 4] \
    for w in [4]\
]

@triton.autotune(configs=configs, key=["HEAD_DIM", "OUTPUT_LSE"])
@triton.jit
def _attn_fwd(
    Q, K, V, O, M, sm_scale,
    stride_qz, stride_qm, stride_qh, stride_qd,
    stride_kz, stride_kn, stride_kh, stride_kd,
    stride_vz, stride_vn, stride_vh, stride_vd,
    stride_oz, stride_om, stride_oh, stride_od,
    stride_mz, stride_mh, stride_mm,
    B, H, N_Q, N_KV,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    OUTPUT_LSE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    mask_m = offs_m < N_Q

    bh = pid_bh
    b = bh // H
    h = bh % H

    Q_bh = Q + b * stride_qz + h * stride_qh
    K_bh = K + b * stride_kz + h * stride_kh
    V_bh = V + b * stride_vz + h * stride_vh
    O_bh = O + b * stride_oz + h * stride_oh

    q = tl.load(
        Q_bh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=mask_m[:, None],
        other=0.0,
    )

    offs_n = tl.arange(0, BLOCK_N)
    k_ptrs = K_bh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    v_ptrs = V_bh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd

    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    scale = sm_scale * 1.4426950408889634

    for _ in tl.range(0, N_KV, BLOCK_N):
        mask_n = offs_n < N_KV

        k = tl.load(
            k_ptrs,
            mask=mask_n[:, None],
            other=0.0,
        )
        v = tl.load(
            v_ptrs,
            mask=mask_n[:, None],
            other=0.0,
        )

        qk = tl.dot(q, tl.trans(k)) * scale
        qk = tl.where(mask_n[None, :], qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        m_i = m_ij

        l_i = l_i * alpha + tl.sum(p, axis=1)
        p = p.to(v.dtype)
        acc = acc * alpha[:, None] + tl.dot(p, v)

        offs_n += BLOCK_N
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
    
    if OUTPUT_LSE:
        m_final = m_i + tl.log2(l_i)
        M_bh = M + b * stride_mz + h * stride_mh
        tl.store(
            M_bh + offs_m * stride_mm,
            m_final,
            mask=mask_m,
        )

    acc = acc / l_i[:, None]
    tl.store(
        O_bh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        acc,
        mask=mask_m[:, None],
    )

configs = [
    triton.Config({'BLOCK_M': BM}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
    for s in [2, 3, 4] \
    for w in [4]\
]

@triton.autotune(configs=configs, key=["HEAD_DIM"])
@triton.jit
def _attn_bwd_preprocess(
    O, DO, Delta,
    stride_oz, stride_om, stride_oh, stride_od,
    stride_doz, stride_dom, stride_doh, stride_dod,
    stride_deltabh, stride_deltam,
    B, H, N_Q,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < N_Q

    bh = pid_bh
    b = bh // H
    h = bh % H

    O_bh = O + b * stride_oz + h * stride_oh
    DO_bh = DO + b * stride_doz + h * stride_doh
    Delta_bh = Delta + bh * stride_deltabh

    o = tl.load(
        O_bh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        mask=mask_m[:, None],
        other=0.0,
    )
    do = tl.load(
        DO_bh + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod,
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)

    delta = tl.sum(o * do, axis=1)
    tl.store(Delta_bh + offs_m * stride_deltam, delta, mask=mask_m)

configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
    for BN in [32, 64, 128]\
    for s in [2, 3, 4] \
    for w in [4]\
]

@triton.autotune(configs=configs, key=["HEAD_DIM", "PARTIAL_KV_GRAD"], reset_to_zero=["DQ"])
@triton.jit
def _attn_bwd(
    Q, K, V,
    DO, DQ, DK, DV,
    M, Delta, sm_scale,
    stride_qz, stride_qm, stride_qh, stride_qd,
    stride_kz, stride_kn, stride_kh, stride_kd,
    stride_vz, stride_vn, stride_vh, stride_vd,
    stride_doz, stride_dom, stride_doh, stride_dod,
    stride_dqz, stride_dqm, stride_dqh, stride_dqd,
    stride_dkz, stride_dkn, stride_dkh, stride_dkd,
    stride_dvz, stride_dvn, stride_dvh, stride_dvd,
    stride_mz, stride_mh, stride_mm,
    stride_deltabh, stride_deltam,
    B, H, N_Q, N_KV,
    start_idx, end_idx,
    HEAD_DIM: tl.constexpr,
    PARTIAL_KV_GRAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_n = offs_n < N_KV

    bh = pid_bh
    b = bh // H
    h = bh % H

    Q_bh = Q + b * stride_qz + h * stride_qh
    K_bh = K + b * stride_kz + h * stride_kh
    V_bh = V + b * stride_vz + h * stride_vh
    DO_bh = DO + b * stride_doz + h * stride_doh
    DQ_bh = DQ + b * stride_dqz + h * stride_dqh
    DK_bh = DK + b * stride_dkz + h * stride_dkh
    DV_bh = DV + b * stride_dvz + h * stride_dvh

    M_bh = M + b * stride_mz + h * stride_mh
    Delta_bh = Delta + bh * stride_deltabh

    k = tl.load(
        K_bh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
        mask=mask_n[:, None],
        other=0.0,
    )
    v = tl.load(
        V_bh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
        mask=mask_n[:, None],
        other=0.0,
    )

    kv_grad_enable = not PARTIAL_KV_GRAD or ((pid_n + 1) * BLOCK_N > start_idx and pid_n * BLOCK_N < end_idx)

    dk = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)

    offs_m = tl.arange(0, BLOCK_M)
    q_ptrs = Q_bh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    do_ptrs = DO_bh + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
    m_ptrs = M_bh + offs_m * stride_mm
    d_ptrs = Delta_bh + offs_m * stride_deltam
    dq_ptrs = DQ_bh + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd

    scale = sm_scale * 1.4426950408889634

    for _ in tl.range(0, N_Q, BLOCK_M):
        mask_m = offs_m < N_Q

        q = tl.load(
            q_ptrs,
            mask=mask_m[:, None],
            other=0.0,
        )
        do = tl.load(
            do_ptrs,
            mask=mask_m[:, None],
            other=0.0,
        )

        qk = tl.dot(q, tl.trans(k)) * scale
        qk = tl.where(mask_m[:, None], qk, float("-inf"))
        qk = tl.where(mask_n[None, :], qk, float("-inf"))

        m_i = tl.load(
            m_ptrs,
            mask=mask_m,
            other=0.0,
        )

        p = tl.math.exp2(qk - m_i[:, None])

        # if kv_grad_enable: # NOTE: incorrect output when disabled
        p_half = p.to(do.dtype)
        dv += tl.dot(tl.trans(p_half), do)

        dp = tl.dot(do, tl.trans(v))

        Di = tl.load(
            d_ptrs,
            mask=mask_m,
            other=0.0,
        )

        ds = p * (dp - Di[:, None])
        ds = ds.to(q.dtype)

        # if kv_grad_enable:
        dk += tl.dot(tl.trans(ds), q)

        dq_block = tl.dot(ds, k) * sm_scale
        tl.atomic_add(dq_ptrs, dq_block, mask=mask_m[:, None])

        offs_m += BLOCK_M
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom
        m_ptrs += BLOCK_M * stride_mm
        d_ptrs += BLOCK_M * stride_deltam
        dq_ptrs += BLOCK_M * stride_dqm
    
    if kv_grad_enable:
        if not PARTIAL_KV_GRAD:
            start_idx = 0
            end_idx = N_KV
            write_mask = mask_n
        else:
            write_mask = (offs_n >= start_idx) & (offs_n < end_idx)
        dk = dk * sm_scale
        
        dv_ptrs = DV_bh + (offs_n - start_idx)[:, None] * stride_dvn + offs_d[None, :] * stride_dvd
        dk_ptrs = DK_bh + (offs_n - start_idx)[:, None] * stride_dkn + offs_d[None, :] * stride_dkd
        tl.store(dv_ptrs, dv, mask=write_mask[:, None])
        tl.store(dk_ptrs, dk, mask=write_mask[:, None])


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k_cache, v_cache, new_k, new_v, start_idx, end_idx, sm_scale, grad_enabled, grad_only_new_kv):
        assert q.dim() == k_cache.dim() == v_cache.dim() == new_k.dim() == new_v.dim() == 4
        B, N_Q, H, D = q.shape
        Bk, N_KV, Hk, Dk = k_cache.shape
        Bv, N_KV_v, Hv, Dv = v_cache.shape
        
        assert B == Bk == Bv
        assert H == Hk == Hv
        assert D == Dk == Dv
        assert N_KV == N_KV_v
        assert q.dtype in (torch.float16, torch.bfloat16)
        assert k_cache.dtype == q.dtype and v_cache.dtype == q.dtype
        assert start_idx >= 0 and end_idx <= N_KV and start_idx <= end_idx

        k_cache[:, start_idx:end_idx, :, :] = new_k
        v_cache[:, start_idx:end_idx, :, :] = new_v

        o = torch.empty((B, N_Q, H, D), device=q.device, dtype=q.dtype)
        M = torch.empty((B, H, N_Q), device=q.device, dtype=torch.float32) if grad_enabled else None

        grid = lambda META: (
            triton.cdiv(N_Q, META["BLOCK_M"]),
            B * H,
        )

        _attn_fwd[grid](
            q, k_cache, v_cache, o, M, sm_scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            *((M.stride(0), M.stride(1), M.stride(2)) if grad_enabled else (0, 0, 0)),
            B, H, N_Q, N_KV,
            HEAD_DIM=D,
            OUTPUT_LSE=grad_enabled,
        )

        if grad_enabled:
            ctx.save_for_backward(q, k_cache, v_cache, o, M)
            ctx.sm_scale = sm_scale
            ctx.start_idx = start_idx
            ctx.end_idx = end_idx
            ctx.grad_only_new_kv = grad_only_new_kv

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        sm_scale = ctx.sm_scale

        B, N_Q, H, D = q.shape
        _, N_KV, _, _ = k.shape

        dq = torch.zeros((B, N_Q, H, D), device=q.device, dtype=torch.float32)
        if ctx.grad_only_new_kv:
            dk = torch.zeros((B, ctx.end_idx - ctx.start_idx, H, D), device=q.device, dtype=k.dtype)
            dv = torch.zeros((B, ctx.end_idx - ctx.start_idx, H, D), device=v.device, dtype=v.dtype)
        else:
            dk = torch.zeros((B, N_KV, H, D), device=q.device, dtype=k.dtype)
            dv = torch.zeros((B, N_KV, H, D), device=q.device, dtype=v.dtype)

        Delta = torch.empty((B * H, N_Q), device=q.device, dtype=torch.float32)

        grid_pre = lambda META: (
            triton.cdiv(N_Q, META["BLOCK_M"]),
            B * H,
        )
        _attn_bwd_preprocess[grid_pre](
            o, do, Delta,
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            Delta.stride(0), Delta.stride(1),
            B, H, N_Q,
            HEAD_DIM=D,
        )

        grid_bwd = lambda META: (
            triton.cdiv(N_KV, META["BLOCK_N"]),
            B * H,
        )
        _attn_bwd[grid_bwd](
            q, k, v,
            do, dq, dk, dv,
            M, Delta, sm_scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            M.stride(0), M.stride(1), M.stride(2),
            Delta.stride(0), Delta.stride(1),
            B, H, N_Q, N_KV,
            ctx.start_idx, ctx.end_idx,
            HEAD_DIM=D,
            PARTIAL_KV_GRAD=ctx.grad_only_new_kv,
        )

        if ctx.grad_only_new_kv:
            dk_cache, dv_cache = None, None
        else:
            dk_cache, dv_cache = dk, dv
            dk = dk_cache[:, ctx.start_idx:ctx.end_idx, :, :]
            dv = dv_cache[:, ctx.start_idx:ctx.end_idx, :, :]

        return dq.to(torch.float32), dk_cache, dv_cache, dk, dv, None, None, None, None, None

def full_attention_with_kv_cache(q, k_cache, v_cache, new_k, new_v, start_idx, end_idx, sm_scale=None, grad_only_new_kv=False):
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    return _attention.apply(q, k_cache, v_cache, new_k, new_v, start_idx, end_idx, sm_scale, torch.is_grad_enabled(), grad_only_new_kv)

__all__ = ["full_attention_with_kv_cache"]

def attention_ref(q, k_cache, v_cache, new_k, new_v, start_idx, end_idx, sm_scale=None, grad_only_new_kv=False):
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    k_cache[:, start_idx:end_idx, :, :] = new_k
    v_cache[:, start_idx:end_idx, :, :] = new_v
    if not grad_only_new_kv:
        k_cache.retain_grad()
        v_cache.retain_grad()
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k_cache) * sm_scale
    p = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhqk,bkhd->bqhd", p, v_cache)
    return out


def run_single_test(dtype, B, Sq, Skv, H, D, partial_kv_grad, device="cuda"):
    print(f"Testing dtype={dtype}, B={B}, H={H}, Sq={Sq}, Skv={Skv}, D={D}, partial_kv_grad={partial_kv_grad}")
    q = torch.randn(B, Sq, H, D, device=device, dtype=dtype, requires_grad=True)
    k_cache = torch.randn(B, Skv, H, D, device=device, dtype=dtype, requires_grad=False)
    v_cache = torch.randn(B, Skv, H, D, device=device, dtype=dtype, requires_grad=False)
    new_k = torch.randn(B, Skv // 2, H, D, device=device, dtype=dtype, requires_grad=True)
    new_v = torch.randn(B, Skv // 2, H, D, device=device, dtype=dtype, requires_grad=True)

    q1 = q.clone().detach().requires_grad_(True)
    k_cache1 = k_cache.clone().detach().requires_grad_(not partial_kv_grad)
    v_cache1 = v_cache.clone().detach().requires_grad_(not partial_kv_grad)
    new_k1 = new_k.clone().detach().requires_grad_(True)
    new_v1 = new_v.clone().detach().requires_grad_(True)

    q2 = q.clone().detach().requires_grad_(True)
    k_cache2 = k_cache.clone().detach().requires_grad_(False)
    v_cache2 = v_cache.clone().detach().requires_grad_(False)
    new_k2 = new_k.clone().detach().requires_grad_(True)
    new_v2 = new_v.clone().detach().requires_grad_(True)

    start_idx = Skv - Skv // 2
    end_idx = Skv

    sm_scale = D ** -0.5

    out_triton = full_attention_with_kv_cache(q1, k_cache1, v_cache1, new_k1, new_v1, start_idx, end_idx, sm_scale, grad_only_new_kv=partial_kv_grad)
    out_ref = attention_ref(q2, k_cache2, v_cache2, new_k2, new_v2, start_idx, end_idx, sm_scale, grad_only_new_kv=partial_kv_grad)

    atol = 1e-2 if dtype is torch.bfloat16 else 1e-3
    rtol = 1e-2 if dtype is torch.bfloat16 else 1e-3

    fw_ok = torch.allclose(out_triton, out_ref, atol=atol, rtol=rtol)
    print("  forward allclose:", fw_ok)

    # Backward
    dout = torch.randn_like(out_triton)
    out_triton.backward(dout)
    out_ref.backward(dout)

    bwd_q_ok = torch.allclose(q1.grad, q2.grad, atol=atol, rtol=rtol)
    bwd_k_ok = torch.allclose(new_k1.grad, new_k2.grad, atol=atol, rtol=rtol)
    bwd_v_ok = torch.allclose(new_v1.grad, new_v2.grad, atol=atol, rtol=rtol)
    print("  backward q allclose:", bwd_q_ok)
    print("  backward k allclose:", bwd_k_ok)
    print("  backward v allclose:", bwd_v_ok)

    if not partial_kv_grad:
        bwd_kcache_ok = torch.allclose(k_cache1.grad, k_cache2.grad, atol=atol, rtol=rtol)
        bwd_vcache_ok = torch.allclose(v_cache1.grad, v_cache2.grad, atol=atol, rtol=rtol)
        print("  backward k_cache allclose:", bwd_kcache_ok)
        print("  backward v_cache allclose:", bwd_vcache_ok)
        return fw_ok and bwd_q_ok and bwd_k_ok and bwd_v_ok and bwd_kcache_ok and bwd_vcache_ok
    
    return fw_ok and bwd_q_ok and bwd_k_ok and bwd_v_ok

def run_tests():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests.")
        return
    device = "cuda"
    torch.manual_seed(0)

    shapes = [
        (1, 17, 13, 1, 32),
        (2, 31, 29, 3, 64),
        (2, 64, 48, 4, 64),
    ]
    for dtype in (torch.float16, torch.bfloat16):
        for B, H, Sq, Skv, D in shapes:
            for partial_kv_grad in (False, True):
                ok = run_single_test(dtype, B, H, Sq, Skv, D, partial_kv_grad=partial_kv_grad, device=device)
                print("  -> test passed:", ok)


if __name__ == "__main__":
    run_tests()
