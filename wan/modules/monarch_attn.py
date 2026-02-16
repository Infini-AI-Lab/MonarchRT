from types import SimpleNamespace
from typing import Optional
from einops import rearrange

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

def _is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def _supports_host_descriptor():
    return _is_cuda() and torch.cuda.get_device_capability()[0] >= 9

assert triton.runtime.driver.active.get_current_target().backend == "cuda"
supports_host_descriptor = _supports_host_descriptor()

@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)

@triton.jit
def _min(a, b):
    return a if a < b else b

@triton.jit
def _max(a, b):
    return a if a > b else b

def _init_al_cl_y_fwd_descs(Z, H, A, F, A_CHUNK, F_CHUNK, HEAD_DIM, block_b1, block_b2, aR, k, v, aL, y):
    ZAF = Z * A_CHUNK * F_CHUNK
    ZFK = Z * F * block_b1
    ZAK = Z * A * block_b1
    hd_stride = H * HEAD_DIM
    jhd_stride = block_b2 * hd_stride
    khd_stride = block_b1 * hd_stride
    jkhd_stride = block_b2 * khd_stride

    descs = SimpleNamespace()
    if supports_host_descriptor:
        descs.aR = TensorDescriptor(
            aR,
            shape=[ZAK, block_b2, H, HEAD_DIM],
            strides=[jhd_stride, hd_stride, HEAD_DIM, 1],
            block_shape=[1, 1, 1, 1]
        )
        descs.k = TensorDescriptor(
            k,
            shape=[ZFK, block_b2, H, HEAD_DIM],
            strides=[jhd_stride, hd_stride, HEAD_DIM, 1],
            block_shape=[1, 1, 1, 1]
        )
        descs.v = TensorDescriptor(
            v,
            shape=[ZFK, block_b2, H, HEAD_DIM],
            strides=[jhd_stride, hd_stride, HEAD_DIM, 1],
            block_shape=[1, 1, 1, 1]
        )
        descs.aL = TensorDescriptor(
            aL,
            shape=[ZAF, block_b2, block_b1, H, HEAD_DIM],
            strides=[jkhd_stride, khd_stride, hd_stride, HEAD_DIM, 1],
            block_shape=[1, 1, 1, 1, 1]
        )
        descs.y = TensorDescriptor(
            y,
            shape=[ZAF, block_b2, block_b1, H, HEAD_DIM],
            strides=[jkhd_stride, khd_stride, hd_stride, HEAD_DIM, 1],
            block_shape=[1, 1, 1, 1, 1]
        )
    else:
        descs.aR = aR
        descs.k = k
        descs.v = v
        descs.aL = aL
        descs.y = y

    return descs

def _al_cl_y_fwd_pre_hook(nargs):
    BLOCK_J = nargs["BLOCK_J"]
    BLOCK_L = nargs["BLOCK_L"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["ar_ptr"], TensorDescriptor):
        return
    nargs["ar_ptr"].block_shape = [1, BLOCK_J, 1, HEAD_DIM]
    nargs["k_ptr"].block_shape = [1, BLOCK_L, 1, HEAD_DIM]
    nargs["v_ptr"].block_shape = [1, BLOCK_L, 1, HEAD_DIM]
    nargs["al_ptr"].block_shape = [1, BLOCK_J, 1, 1, HEAD_DIM]
    nargs["y_ptr"].block_shape = [1, BLOCK_J, 1, 1, HEAD_DIM]

configs = [
    triton.Config({'BLOCK_J': BJ, 'BLOCK_L': BL}, num_stages=s, num_warps=w, pre_hook=_al_cl_y_fwd_pre_hook) \
    for BJ in [16, 32, 64, 128]\
    for BL in [16, 32, 64, 128]\
    for s in [2, 3, 4] \
    for w in [4, 8]\
]

def keep(conf):
    BLOCK_J = conf.kwargs["BLOCK_J"]
    BLOCK_L = conf.kwargs["BLOCK_L"]
    return not (torch.cuda.get_device_capability()[0] == 9 and BLOCK_J * BLOCK_L < 128 * 128
                and conf.num_warps == 8)

@triton.autotune(configs=list(filter(keep, configs)), key=["A_CHUNK", "F_CHUNK", "block_b1", "block_b2", "HEAD_DIM", "OUTPUT_LSE", "CAUSAL_BLOCK_SIZE"], cache_results=True)
@triton.jit
def _al_cl_y_fwd(Z, H, A, F,
                 a_start, f_start,
                 curr_num_a, curr_num_f,
                 ar_ptr,
                 k_ptr, v_ptr,
                 al_ptr, cl_ptr,
                 y_ptr, out_lse_ptr,
                 kv_stride_z,
                 sm_scale_sqrt,
                 A_CHUNK: tl.constexpr,
                 F_CHUNK: tl.constexpr,
                 block_b1: tl.constexpr,
                 block_b2: tl.constexpr,
                 HEAD_DIM: tl.constexpr,
                 OUTPUT_LSE: tl.constexpr,
                 CAUSAL_BLOCK_SIZE: tl.constexpr,
                 BLOCK_J: tl.constexpr,
                 BLOCK_L: tl.constexpr,
                 ):
    start_j = tl.program_id(0) * BLOCK_J
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_afk = tl.program_id(2)
    off_k = off_afk % block_b1
    off_f = (off_afk // block_b1) % curr_num_f + f_start
    off_a = off_afk // (block_b1 * curr_num_f) + a_start

    if CAUSAL_BLOCK_SIZE > 0:
        if (off_a // CAUSAL_BLOCK_SIZE) < (off_f // CAUSAL_BLOCK_SIZE):
            return

    off_zaf = (off_z * A_CHUNK + (off_a - a_start)) * F_CHUNK + (off_f - f_start)
    ZAF = Z * A_CHUNK * F_CHUNK
    off_zfk = off_z * kv_stride_z + off_f * block_b1 + off_k
    ZFK = Z * F * block_b1

    off_zak = (off_z * A + off_a) * block_b1 + off_k
    ZAK = Z * A * block_b1

    hd_stride = H * HEAD_DIM
    jhd_stride = block_b2 * hd_stride

    desc_ar = _maybe_make_tensor_desc(
        ar_ptr,
        shape=[ZAK, block_b2, H, HEAD_DIM],
        strides=[jhd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, BLOCK_J, 1, HEAD_DIM]
    )
    desc_k = _maybe_make_tensor_desc(
        k_ptr,
        shape=[ZFK, block_b2, H, HEAD_DIM],
        strides=[jhd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, BLOCK_L, 1, HEAD_DIM]
    )
    desc_v = _maybe_make_tensor_desc(
        v_ptr,
        shape=[ZFK, block_b2, H, HEAD_DIM],
        strides=[jhd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, BLOCK_L, 1, HEAD_DIM]
    )
    khd_stride = block_b1 * hd_stride
    jkhd_stride = block_b2 * khd_stride
    desc_al = _maybe_make_tensor_desc(
        al_ptr,
        shape=[ZAF, block_b2, block_b1, H, HEAD_DIM],
        strides=[jkhd_stride, khd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, BLOCK_J, 1, 1, HEAD_DIM]
    )
    desc_y = _maybe_make_tensor_desc(
        y_ptr,
        shape=[ZAF, block_b2, block_b1, H, HEAD_DIM],
        strides=[jkhd_stride, khd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, BLOCK_J, 1, 1, HEAD_DIM]
    )
    ZHAF = Z * H * A * F
    off_zhaf = ((off_z * H + off_h) * A_CHUNK + (off_a - a_start)) * F_CHUNK + (off_f - f_start)
    kj_stride = block_b1 * block_b2
    cl_ptrs = cl_ptr + (off_zhaf * kj_stride
                        + off_k) + (start_j + tl.arange(0, BLOCK_J)) * block_b1
    if OUTPUT_LSE:
        desc_lse = tl.make_block_ptr(
            out_lse_ptr,
            shape=[ZHAF, block_b1, block_b2],
            strides=[kj_stride, block_b2, 1],
            block_shape=[1, 1, BLOCK_J],
            order=[2, 1, 0],
            offsets=[0, 0, 0],
        )

    dtype = desc_ar.dtype

    sm_scale_sqrt = sm_scale_sqrt.to(dtype)
    al_acc = tl.zeros([BLOCK_J, HEAD_DIM], dtype=tl.float32)
    y_acc = tl.zeros([BLOCK_J, HEAD_DIM], dtype=tl.float32)
    cl_acc = tl.zeros([BLOCK_J], dtype=tl.float32)
    l_j = tl.zeros([BLOCK_J], dtype=tl.float32)
    m_j = tl.full([BLOCK_J], dtype=tl.float32, value=float("-inf"))

    ar_j = desc_ar.load([off_zak, start_j, off_h, 0]).reshape(BLOCK_J, HEAD_DIM)
    ar_j = ar_j * sm_scale_sqrt
    l_range = tl.arange(0, BLOCK_L)

    for l in tl.static_range(0, block_b2, BLOCK_L):
        l_mask = l_range + l < block_b2

        k_l = desc_k.load([off_zfk, l, off_h, 0]).reshape(BLOCK_L, HEAD_DIM) * sm_scale_sqrt
        v_l = desc_v.load([off_zfk, l, off_h, 0]).reshape(BLOCK_L, HEAD_DIM)
        k_l = tl.where(l_mask[:, None], k_l, 0.0)
        v_l = tl.where(l_mask[:, None], v_l, 0.0)

        br_jl = tl.dot(ar_j, k_l.T) * 1.44269504 # log2(e)
        z_jl = tl.where(l_mask[None, :], br_jl, float("-inf"))

        m_jl = tl.maximum(m_j, tl.max(z_jl, 1))
        z_jl = z_jl - m_jl[:, None]
        p = tl.math.exp2(z_jl)
        l_jl = tl.sum(p, 1)

        alpha = tl.math.exp2(m_j - m_jl)
        al_acc = al_acc * alpha[:, None]
        cl_acc = cl_acc * alpha + tl.sum(br_jl * p, 1)
        y_acc = y_acc * alpha[:, None]

        p = p.to(dtype)
        al_acc = tl.dot(p, k_l, al_acc)
        y_acc = tl.dot(p, v_l, y_acc)

        l_j = l_j * alpha + l_jl
        m_j = m_jl

    al_acc = al_acc / l_j[:, None]
    desc_al.store([off_zaf, start_j, off_k, off_h, 0], al_acc.to(dtype).reshape(1, BLOCK_J, 1, 1, HEAD_DIM))

    y_acc = y_acc / l_j[:, None]
    desc_y.store([off_zaf, start_j, off_k, off_h, 0], y_acc.to(dtype).reshape(1, BLOCK_J, 1, 1, HEAD_DIM))

    lse = m_j + tl.log2(l_j)
    cl_acc = cl_acc / l_j - lse # keep cL in log2 space
    tl.store(cl_ptrs, cl_acc, start_j + tl.arange(0, BLOCK_J) < block_b2)

    if OUTPUT_LSE:
        tl.store(desc_lse.advance([off_zhaf, off_k, start_j]), lse.reshape(1, 1, BLOCK_J), boundary_check=(2,))

@torch.library.custom_op("monarch_rt::al_cl_y_fwd", mutates_args=("aL", "cL", "y", "al_cl_y_lse"))
def al_cl_y_fwd(
    b: int, h: int, a: int, f: int,
    q_frame_start: int, kv_frame_start: int,
    curr_q_frames: int, curr_kv_frames: int,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    aL: torch.Tensor, cL: torch.Tensor, y: torch.Tensor,
    al_cl_y_lse: Optional[torch.Tensor],
    sm_scale_sqrt: float,
    Q_FRAME_CHUNK: int, KV_FRAME_CHUNK: int,
    block_b1: int, block_b2: int, d: int,
    CAUSAL_BLOCK_SIZE: int
) -> None:
    def _al_cl_y_grid(META):
        return (triton.cdiv(block_b2, META["BLOCK_J"]), b * h, curr_q_frames * curr_kv_frames * block_b1)
    al_cl_y_fwd_descs = _init_al_cl_y_fwd_descs(b, h, a, f, Q_FRAME_CHUNK, KV_FRAME_CHUNK, d, block_b1, block_b2, q, k, v, aL, y)
    _al_cl_y_fwd[_al_cl_y_grid](
        b, h, a, f,
        q_frame_start, kv_frame_start,
        curr_q_frames, curr_kv_frames,
        al_cl_y_fwd_descs.aR,
        al_cl_y_fwd_descs.k, al_cl_y_fwd_descs.v,
        al_cl_y_fwd_descs.aL, cL,
        al_cl_y_fwd_descs.y, al_cl_y_lse,
        k.stride(0) // (block_b2 * h * d),
        sm_scale_sqrt,
        Q_FRAME_CHUNK,
        KV_FRAME_CHUNK,
        block_b1,
        block_b2,
        d,
        OUTPUT_LSE=(al_cl_y_lse is not None),
        CAUSAL_BLOCK_SIZE=CAUSAL_BLOCK_SIZE,
    )

@al_cl_y_fwd.register_fake
def _(
    b: int, h: int, a: int, f: int,
    q_frame_start: int, kv_frame_start: int,
    curr_q_frames: int, curr_kv_frames: int,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    aL: torch.Tensor, cL: torch.Tensor, y: torch.Tensor,
    al_cl_y_lse: Optional[torch.Tensor],
    sm_scale_sqrt: float,
    Q_FRAME_CHUNK: int, KV_FRAME_CHUNK: int,
    block_b1: int, block_b2: int, d: int,
    CAUSAL_BLOCK_SIZE: int
) -> None:
    pass


def _init_al_cl_y_bwd_descs(Z, H, A, F, A_CHUNK, F_CHUNK, HEAD_DIM, block_b1, block_b2, q, k, v, grad_q, grad_k, grad_v, grad_aL, grad_y, lse, d):
    HD = H * HEAD_DIM
    KHD = block_b1 * HD
    ZH = Z * H
    jhd_stride = block_b2 * HD
    kjhd_stride = block_b1 * jhd_stride
    fkjhd_stride = F_CHUNK * kjhd_stride
    afkjhd_stride = A_CHUNK * fkjhd_stride
    akjhd_stride = A * kjhd_stride
    lhd_stride = jhd_stride
    klhd_stride = kjhd_stride
    kj_stride = block_b1 * block_b2
    fkj_stride = F_CHUNK * kj_stride
    afkj_stride = A_CHUNK * fkj_stride

    descs = SimpleNamespace()
    if supports_host_descriptor:
        descs.aR = TensorDescriptor(
            q,
            shape=[Z, A, block_b1, block_b2, HD],
            strides=[akjhd_stride, kjhd_stride, jhd_stride, HD, 1],
            block_shape=[1, 1, 1, 1, 1]
        )
        descs.grad_q = TensorDescriptor(
            grad_q,
            shape=[Z, A, block_b1, block_b2, HD],
            strides=[akjhd_stride, kjhd_stride, jhd_stride, HD, 1],
            block_shape=[1, 1, 1, 1, 1]
        )
        descs.k = TensorDescriptor(
            k,
            shape=[Z, F, block_b1, block_b2, HD],
            strides=[k.stride(0), klhd_stride, lhd_stride, HD, 1],
            block_shape=[1, 1, 1, 1, 1]
        )
        descs.grad_k = TensorDescriptor(
            grad_k,
            shape=[Z, grad_k.shape[1], block_b1, block_b2, HD],
            strides=[grad_k.shape[1] * klhd_stride, klhd_stride, lhd_stride, HD, 1],
            block_shape=[1, 1, 1, 1, HEAD_DIM]
        )
        descs.v = TensorDescriptor(
            v,
            shape=[Z, F, block_b1, block_b2, HD],
            strides=[v.stride(0), klhd_stride, lhd_stride, HD, 1],
            block_shape=[1, 1, 1, 1, 1]
        )
        descs.grad_v = TensorDescriptor(
            grad_v,
            shape=[Z, grad_v.shape[1], block_b1, block_b2, HD],
            strides=[grad_v.shape[1] * klhd_stride, klhd_stride, lhd_stride, HD, 1],
            block_shape=[1, 1, 1, 1, 1]
        )
        descs.grad_aL = TensorDescriptor(
            grad_aL,
            shape=[Z, A_CHUNK, F_CHUNK, block_b2, KHD],
            strides=[afkjhd_stride, fkjhd_stride, kjhd_stride, KHD, 1],
            block_shape=[1, 1, 1, 1, 1]
        )
        descs.grad_y = TensorDescriptor(
            grad_y,
            shape=[Z, A_CHUNK, F_CHUNK, block_b2, KHD],
            strides=[afkjhd_stride, fkjhd_stride, kjhd_stride, KHD, 1],
            block_shape=[1, 1, 1, 1, 1]
        )
        # TMA requires 16-byte alignment for leading strides
        descs.lse = TensorDescriptor(
            lse,
            shape=[ZH, A_CHUNK, F_CHUNK, block_b1, block_b2],
            strides=[afkj_stride, fkj_stride, kj_stride, block_b2, 1],
            block_shape=[1, 1, 1, 1, 1]
        ) if block_b2 % 16 == 0 else lse
        descs.d = TensorDescriptor(
            d,
            shape=[ZH, A_CHUNK, F_CHUNK, block_b1, block_b2],
            strides=[afkj_stride, fkj_stride, kj_stride, block_b2, 1],
            block_shape=[1, 1, 1, 1, 1]
        ) if block_b2 % 16 == 0 else d
    else:
        descs.aR = q
        descs.grad_q = grad_q
        descs.k = k
        descs.grad_k = grad_k
        descs.v = v
        descs.grad_v = grad_v
        descs.grad_aL = grad_aL
        descs.grad_y = grad_y
        descs.lse = lse
        descs.d = d

    return descs

_dq_backup = None
_dk_backup = None
_dv_backup = None
def _al_cl_y_bwd_pre_hook(nargs):
    BLOCK_J = nargs["BLOCK_J"]
    BLOCK_L = nargs["BLOCK_L"]
    HEAD_DIM = nargs["HEAD_DIM"]
    global _dq_backup, _dk_backup, _dv_backup

    if not isinstance(nargs["ar_ptr"], TensorDescriptor):
        if _dq_backup is not None:
            nargs["dq_ptr"].copy_(_dq_backup)
            nargs["dk_ptr"].copy_(_dk_backup)
            nargs["dv_ptr"].copy_(_dv_backup)
        return
    if _dq_backup is not None:
        nargs["dq_ptr"].base.copy_(_dq_backup)
        nargs["dk_ptr"].base.copy_(_dk_backup)
        nargs["dv_ptr"].base.copy_(_dv_backup)

    nargs["ar_ptr"].block_shape = [1, 1, 1, BLOCK_J, HEAD_DIM]
    nargs["dq_ptr"].block_shape = [1, 1, 1, BLOCK_J, HEAD_DIM]
    nargs["k_ptr"].block_shape = [1, 1, 1, BLOCK_L, HEAD_DIM]
    nargs["dk_ptr"].block_shape = [1, 1, 1, BLOCK_L, HEAD_DIM]
    nargs["v_ptr"].block_shape = [1, 1, 1, BLOCK_L, HEAD_DIM]
    nargs["dv_ptr"].block_shape = [1, 1, 1, BLOCK_L, HEAD_DIM]
    nargs["dal_ptr"].block_shape = [1, 1, 1, BLOCK_J, HEAD_DIM]
    nargs["dy_ptr"].block_shape = [1, 1, 1, BLOCK_J, HEAD_DIM]
    if isinstance(nargs["lse_ptr"], TensorDescriptor):
        nargs["lse_ptr"].block_shape = [1, 1, 1, 1, BLOCK_J]
    if isinstance(nargs["d_ptr"], TensorDescriptor):
        nargs["d_ptr"].block_shape = [1, 1, 1, 1, BLOCK_J]

configs = [
    triton.Config({'BLOCK_J': BJ, 'BLOCK_L': BL}, num_stages=s, num_warps=w, pre_hook=_al_cl_y_bwd_pre_hook) \
    for BJ in [16, 32]#, 64, 128]\
    for BL in [16, 32]#, 64, 128]\
    for s in [1, 2, 3, 4] \
    for w in [4, 8]\
]

def keep(conf):
    BLOCK_J = conf.kwargs["BLOCK_J"]
    BLOCK_L = conf.kwargs["BLOCK_L"]
    return not (torch.cuda.get_device_capability()[0] == 9 and BLOCK_J * BLOCK_L < 128 * 128
                and conf.num_warps == 8)

_alcly_bwd_evaluated_configs = set()
def _start_alcly_bwd_evaluate(A_CHUNK, F_CHUNK, block_b1, block_b2, HEAD_DIM, IS_FIRST_ITER, PARTIAL_KV_GRAD, CAUSAL_BLOCK_SIZE, dtype):
    global _alcly_bwd_evaluated_configs
    config = (A_CHUNK, F_CHUNK, block_b1, block_b2, HEAD_DIM, IS_FIRST_ITER, PARTIAL_KV_GRAD, CAUSAL_BLOCK_SIZE, dtype)
    if config in _alcly_bwd_evaluated_configs:
        return False
    _alcly_bwd_evaluated_configs.add(config)
    return True

@triton.autotune(configs=list(filter(keep, configs)), key=["A_CHUNK", "F_CHUNK", "block_b1", "block_b2", "HEAD_DIM", "PARTIAL_KV_GRAD", "CAUSAL_BLOCK_SIZE"], cache_results=True)
@triton.jit
def _al_cl_y_bwd(Z, H, A, F,
                 a_start, f_start,
                 curr_num_a, curr_num_f,
                 ar_ptr,
                 k_ptr, v_ptr,
                 dal_ptr, dcl_ptr, dy_ptr,
                 lse_ptr, d_ptr,
                 dq_ptr,
                 dk_ptr, dv_ptr,
                 kv_stride_z,
                 kv_grad_start_f, kv_grad_end_f,
                 sm_scale_sqrt,
                 A_CHUNK: tl.constexpr,
                 F_CHUNK: tl.constexpr,
                 block_b1: tl.constexpr,
                 block_b2: tl.constexpr,
                 HEAD_DIM: tl.constexpr,
                 IS_FIRST_ITER: tl.constexpr,
                 PARTIAL_KV_GRAD: tl.constexpr,
                 CAUSAL_BLOCK_SIZE: tl.constexpr,
                 BLOCK_J: tl.constexpr,
                 BLOCK_L: tl.constexpr,
                 ):
    start_l = tl.program_id(0) * BLOCK_L
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_fk = tl.program_id(2)
    off_f = off_fk // block_b1
    off_f_mod = off_f + f_start
    off_k = off_fk % block_b1

    kv_grad_enable = not PARTIAL_KV_GRAD or (off_f_mod >= kv_grad_start_f and off_f_mod < kv_grad_end_f)

    HD = H * HEAD_DIM
    KHD = block_b1 * HD
    ZH = Z * H

    off_hd = off_h * HEAD_DIM
    off_khd = off_k * HD + off_hd

    jhd_stride = block_b2 * HD
    kjhd_stride = block_b1 * jhd_stride
    fkjhd_stride = F_CHUNK * kjhd_stride
    afkjhd_stride = A_CHUNK * fkjhd_stride
    akjhd_stride = A * kjhd_stride

    desc_ar = _maybe_make_tensor_desc(
        ar_ptr,
        shape=[Z, A, block_b1, block_b2, HD],
        strides=[akjhd_stride, kjhd_stride, jhd_stride, HD, 1],
        block_shape=[1, 1, 1, BLOCK_J, HEAD_DIM]
    )
    desc_dq = _maybe_make_tensor_desc(
        dq_ptr,
        shape=[Z, A, block_b1, block_b2, HD],
        strides=[akjhd_stride, kjhd_stride, jhd_stride, HD, 1],
        block_shape=[1, 1, 1, BLOCK_J, HEAD_DIM]
    )
    lhd_stride = jhd_stride
    klhd_stride = kjhd_stride
    fklhd_stride = (kv_grad_end_f - kv_grad_start_f) * kjhd_stride if PARTIAL_KV_GRAD else F * kjhd_stride
    desc_k = _maybe_make_tensor_desc(
        k_ptr,
        shape=[Z, F, block_b1, block_b2, HD],
        strides=[kv_stride_z, klhd_stride, lhd_stride, HD, 1],
        block_shape=[1, 1, 1, BLOCK_L, HEAD_DIM]
    )
    desc_dk = _maybe_make_tensor_desc(
        dk_ptr,
        shape=[Z, (kv_grad_end_f - kv_grad_start_f) if PARTIAL_KV_GRAD else F, block_b1, block_b2, HD],
        strides=[fklhd_stride, klhd_stride, lhd_stride, HD, 1],
        block_shape=[1, 1, 1, BLOCK_L, HEAD_DIM]
    )
    desc_v = _maybe_make_tensor_desc(
        v_ptr,
        shape=[Z, F, block_b1, block_b2, HD],
        strides=[kv_stride_z, klhd_stride, lhd_stride, HD, 1],
        block_shape=[1, 1, 1, BLOCK_L, HEAD_DIM]
    )
    desc_dv = _maybe_make_tensor_desc(
        dv_ptr,
        shape=[Z, (kv_grad_end_f - kv_grad_start_f) if PARTIAL_KV_GRAD else F, block_b1, block_b2, HD],
        strides=[fklhd_stride, klhd_stride, lhd_stride, HD, 1],
        block_shape=[1, 1, 1, BLOCK_L, HEAD_DIM]
    )
    desc_dal = _maybe_make_tensor_desc(
        dal_ptr,
        shape=[Z, A_CHUNK, F_CHUNK, block_b2, KHD],
        strides=[afkjhd_stride, fkjhd_stride, kjhd_stride, KHD, 1],
        block_shape=[1, 1, 1, BLOCK_J, HEAD_DIM]
    )
    desc_dy = _maybe_make_tensor_desc(
        dy_ptr,
        shape=[Z, A_CHUNK, F_CHUNK, block_b2, KHD],
        strides=[afkjhd_stride, fkjhd_stride, kjhd_stride, KHD, 1],
        block_shape=[1, 1, 1, BLOCK_J, HEAD_DIM]
    )
    kj_stride = block_b1 * block_b2
    fkj_stride = F_CHUNK * kj_stride
    afkj_stride = A_CHUNK * fkj_stride
    cl_off = (off_hz * afkj_stride + off_f * kj_stride + off_k) + (tl.arange(0, BLOCK_J) * block_b1)
    dcl_ptrs = dcl_ptr + cl_off
    desc_lse = tl.make_block_ptr(
        lse_ptr,
        shape=[ZH, A_CHUNK, F_CHUNK, block_b1, block_b2],
        strides=[afkj_stride, fkj_stride, kj_stride, block_b2, 1],
        block_shape=[1, 1, 1, 1, BLOCK_J],
        order=[4, 3, 2, 1, 0],
        offsets=[0, 0, 0, 0, 0],
    ) if not isinstance(lse_ptr, tl.tensor_descriptor) else lse_ptr
    desc_d = tl.make_block_ptr(
        d_ptr,
        shape=[ZH, A_CHUNK, F_CHUNK, block_b1, block_b2],
        strides=[afkj_stride, fkj_stride, kj_stride, block_b2, 1],
        block_shape=[1, 1, 1, 1, BLOCK_J],
        order=[4, 3, 2, 1, 0],
        offsets=[0, 0, 0, 0, 0],
    ) if not isinstance(d_ptr, tl.tensor_descriptor) else d_ptr

    dtype = desc_ar.dtype

    j_range = tl.arange(0, BLOCK_J)
    l_mask = (start_l + tl.arange(0, BLOCK_L)) < block_b2

    dk_acc = tl.zeros([BLOCK_L, HEAD_DIM], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_L, HEAD_DIM], dtype=tl.float32)

    sm_scale_sqrt = sm_scale_sqrt.to(dtype)
    k_l = desc_k.load([off_z, off_f_mod, off_k, start_l, off_hd]).reshape(BLOCK_L, HEAD_DIM) * sm_scale_sqrt
    k_l = tl.where(l_mask[:, None], k_l, 0.0)
    v_l = desc_v.load([off_z, off_f_mod, off_k, start_l, off_hd]).reshape(BLOCK_L, HEAD_DIM)

    if CAUSAL_BLOCK_SIZE > 0:
        a_lim = _max(0, _min(curr_num_a, (off_f_mod // CAUSAL_BLOCK_SIZE) * CAUSAL_BLOCK_SIZE - a_start))
        dcl_ptrs += a_lim * fkj_stride
    else:
        a_lim = 0

    for a in tl.range(a_lim, curr_num_a):
        for j in tl.static_range(0, block_b2, BLOCK_J):
            j_mask = j_range + j < block_b2

            ar_j = desc_ar.load([off_z, a + a_start, off_k, j, off_hd]).reshape(BLOCK_J, HEAD_DIM) * sm_scale_sqrt
            if isinstance(lse_ptr, tl.tensor_descriptor):
                lse_j = desc_lse.load([off_hz, a, off_f, off_k, j]).reshape(BLOCK_J)
            else:
                lse_j = tl.load(desc_lse.advance([off_hz, a, off_f, off_k, j]), boundary_check=(4,)).reshape(BLOCK_J)

            br_jl = tl.dot(ar_j, k_l.T) * 1.44269504
            s_jl = br_jl - lse_j[:, None] # scale br by log2(e) (lse already scaled)
            p = tl.math.exp2(s_jl)

            dcl_j = tl.load(dcl_ptrs + j * block_b1, mask=j_mask)
            dp = dcl_j[:, None] * (1 + s_jl * 0.693147181)

            dy_j = desc_dy.load([off_z, a, off_f, j, off_khd]).reshape(BLOCK_J, HEAD_DIM)
            dp = tl.dot(dy_j, v_l.T, dp)

            dal_j = desc_dal.load([off_z, a, off_f, j, off_khd]).reshape(BLOCK_J, HEAD_DIM)
            dp = tl.dot(dal_j, k_l.T, dp)

            if isinstance(desc_d, tl.tensor_descriptor):
                d_j = desc_d.load([off_hz, a, off_f, off_k, j]).reshape(BLOCK_J)
            else:
                d_j = tl.load(desc_d.advance([off_hz, a, off_f, off_k, j]), boundary_check=(4,)).reshape(BLOCK_J)
            ds_jl = p * (dp - d_j[:, None])

            ds_jl = tl.where(l_mask[None, :], ds_jl.to(dtype), 0.0)
            dar_j = tl.dot(ds_jl, k_l) * sm_scale_sqrt
            desc_dq.atomic_add([off_z, a + a_start, off_k, j, off_hd], dar_j.reshape(1, 1, 1, BLOCK_J, HEAD_DIM))

            if kv_grad_enable:
                ar_j = tl.where(j_mask[:, None], ar_j, 0.0)
                ds_jl = tl.where(j_mask[:, None], ds_jl, 0.0)
                dk_acc = tl.dot(ds_jl.T, ar_j, dk_acc)                

                pT = tl.where(j_mask[:, None], p.to(dtype), 0.0).T
                dal_j = tl.where(j_mask[:, None], dal_j, 0.0)
                dk_acc = tl.dot(pT, dal_j, dk_acc)
                dy_j = tl.where(j_mask[:, None], dy_j, 0.0)
                dv_acc = tl.dot(pT, dy_j, dv_acc)

        dcl_ptrs += fkj_stride

    if kv_grad_enable:
        dk_acc = dk_acc * sm_scale_sqrt
        if not IS_FIRST_ITER:
            dk_acc += desc_dk.load([off_z, off_f_mod - kv_grad_start_f if PARTIAL_KV_GRAD else off_f_mod, off_k, start_l, off_hd]).reshape(BLOCK_L, HEAD_DIM).to(tl.float32)
            dv_acc += desc_dv.load([off_z, off_f_mod - kv_grad_start_f if PARTIAL_KV_GRAD else off_f_mod, off_k, start_l, off_hd]).reshape(BLOCK_L, HEAD_DIM).to(tl.float32)
        desc_dk.store([off_z, off_f_mod - kv_grad_start_f if PARTIAL_KV_GRAD else off_f_mod, off_k, start_l, off_hd], dk_acc.to(dtype).reshape(1, 1, 1, BLOCK_L, HEAD_DIM))
        desc_dv.store([off_z, off_f_mod - kv_grad_start_f if PARTIAL_KV_GRAD else off_f_mod, off_k, start_l, off_hd], dv_acc.to(dtype).reshape(1, 1, 1, BLOCK_L, HEAD_DIM))

@torch.library.custom_op("monarch_rt::al_cl_y_bwd", mutates_args=("grad_q", "grad_k", "grad_v"))
def al_cl_y_bwd(
    b: int, h: int, a: int, f: int,
    q_frame_start: int, kv_frame_start: int,
    curr_q_frames: int, curr_kv_frames: int,
    kv_grad_start_f: int, kv_grad_end_f: int,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    grad_q: torch.Tensor, grad_k: torch.Tensor, grad_v: torch.Tensor,
    grad_aL: torch.Tensor, grad_cL: torch.Tensor, grad_y: torch.Tensor,
    al_cl_y_lse: torch.Tensor, al_cl_y_d: torch.Tensor,
    sm_scale_sqrt: float,
    Q_FRAME_CHUNK: int, KV_FRAME_CHUNK: int,
    block_b1: int, block_b2: int, d: int,
    IS_FIRST_ITER: bool,
    PARTIAL_KV_GRAD: bool,
    CAUSAL_BLOCK_SIZE: int
) -> None:
    def _al_cl_y_bwd_grid(META):
        return (triton.cdiv(block_b2, META["BLOCK_L"]), b * h, curr_kv_frames * block_b1)
    al_cl_y_bwd_descs = _init_al_cl_y_bwd_descs(b, h, a, f, Q_FRAME_CHUNK, KV_FRAME_CHUNK, d, block_b1, block_b2, q, k, v, grad_q, grad_k, grad_v, grad_aL, grad_y, al_cl_y_lse, al_cl_y_d)

    global _dq_backup, _dk_backup, _dv_backup
    if _start_alcly_bwd_evaluate(Q_FRAME_CHUNK, KV_FRAME_CHUNK, block_b1, block_b2, IS_FIRST_ITER, d, PARTIAL_KV_GRAD, CAUSAL_BLOCK_SIZE, q.dtype):
        _dq_backup = grad_q.clone()
        _dk_backup = grad_k.clone()
        _dv_backup = grad_v.clone()

    _al_cl_y_bwd[_al_cl_y_bwd_grid](
        b, h, a, f,
        q_frame_start, kv_frame_start,
        curr_q_frames, curr_kv_frames,
        al_cl_y_bwd_descs.aR,
        al_cl_y_bwd_descs.k, al_cl_y_bwd_descs.v,
        al_cl_y_bwd_descs.grad_aL, grad_cL, al_cl_y_bwd_descs.grad_y,
        al_cl_y_bwd_descs.lse, al_cl_y_bwd_descs.d,
        al_cl_y_bwd_descs.grad_q,
        al_cl_y_bwd_descs.grad_k, al_cl_y_bwd_descs.grad_v,
        k.stride(0),
        kv_grad_start_f, kv_grad_end_f,
        sm_scale_sqrt,
        Q_FRAME_CHUNK,
        KV_FRAME_CHUNK,
        block_b1,
        block_b2,
        d,
        IS_FIRST_ITER=IS_FIRST_ITER,
        PARTIAL_KV_GRAD=PARTIAL_KV_GRAD,
        CAUSAL_BLOCK_SIZE=CAUSAL_BLOCK_SIZE,
    )

    _dq_backup = None
    _dk_backup = None
    _dv_backup = None

@al_cl_y_bwd.register_fake
def _(
    b: int, h: int, a: int, f: int,
    q_frame_start: int, kv_frame_start: int,
    curr_q_frames: int, curr_kv_frames: int,
    kv_grad_start_f: int, kv_grad_end_f: int,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    grad_q: torch.Tensor, grad_k: torch.Tensor, grad_v: torch.Tensor,
    grad_aL: torch.Tensor, grad_cL: torch.Tensor, grad_y: torch.Tensor,
    al_cl_y_lse: torch.Tensor, al_cl_y_d: torch.Tensor,
    sm_scale_sqrt: float,
    Q_FRAME_CHUNK: int, KV_FRAME_CHUNK: int,
    block_b1: int, block_b2: int, d: int,
    IS_FIRST_ITER: bool,
    PARTIAL_KV_GRAD: bool,
    CAUSAL_BLOCK_SIZE: int
) -> None:
    pass


def _init_z_fwd_descs(Z, H, A, F, A_CHUNK, F_CHUNK, HEAD_DIM, block_b1, block_b2, aL, y, q, z, cL):
    ZA = Z * A
    ZA_mod = Z * A_CHUNK
    HD = H * HEAD_DIM
    khd_stride = block_b1 * HD
    jkhd_stride = block_b2 * khd_stride
    fjkhd_stride = F_CHUNK * jkhd_stride
    jhd_stride = block_b2 * HD
    ijhd_stride = jkhd_stride
    ZHA = ZA_mod * H
    jk_stride = block_b2 * block_b1
    fjk_stride = F_CHUNK * jk_stride

    descs = SimpleNamespace()
    if supports_host_descriptor:
        descs.aL = TensorDescriptor(
            aL,
            shape=[ZA_mod, F_CHUNK, block_b2, block_b1, HD],
            strides=[fjkhd_stride, jkhd_stride, khd_stride, HD, 1],
            block_shape=[1, 1, 1, 1, 1]
        )
        descs.y = TensorDescriptor(
            y,
            shape=[ZA_mod, F_CHUNK, block_b2, block_b1, HD],
            strides=[fjkhd_stride, jkhd_stride, khd_stride, HD, 1],
            block_shape=[1, 1, 1, 1, 1]
        )
        descs.q = TensorDescriptor(
            q,
            shape=[ZA, block_b1, block_b2, H, HEAD_DIM],
            strides=[ijhd_stride, jhd_stride, HD, HEAD_DIM, 1],
            block_shape=[1, 1, 1, 1, 1]
        )
        descs.z = TensorDescriptor(
            z,
            shape=[ZA, block_b1, block_b2, H, HEAD_DIM],
            strides=[ijhd_stride, jhd_stride, HD, HEAD_DIM, 1],
            block_shape=[1, 1, 1, 1, 1]
        )
        # TMA requires 16-byte alignment for leading strides
        descs.cL = TensorDescriptor(
            cL,
            shape=[ZHA, F_CHUNK, block_b2, block_b1],
            strides=[fjk_stride, jk_stride, block_b1, 1],
            block_shape=[1, 1, 1, 1]
        ) if (block_b1 * cL.element_size()) % 16 == 0 else cL
    else:
        descs.aL = aL
        descs.y = y
        descs.q = q
        descs.z = z
        descs.cL = cL

    return descs

_z_backup = None
_z_lse_backup = None
def _z_fwd_pre_hook(nargs):
    BLOCK_K = nargs["BLOCK_K"]
    BLOCK_I = nargs["BLOCK_I"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if _z_lse_backup is not None:
        nargs["out_lse_ptr"].copy_(_z_lse_backup)
    if not isinstance(nargs["al_ptr"], TensorDescriptor):
        if _z_backup is not None:
            nargs["z_ptr"].copy_(_z_backup)
        return
    if _z_backup is not None:
        nargs["z_ptr"].base.copy_(_z_backup)
    nargs["al_ptr"].block_shape = [1, 1, 1, BLOCK_K, HEAD_DIM]
    nargs["y_ptr"].block_shape = [1, 1, 1, BLOCK_K, HEAD_DIM]
    nargs["q_ptr"].block_shape = [1, BLOCK_I, 1, 1, HEAD_DIM]
    nargs["z_ptr"].block_shape = [1, BLOCK_I, 1, 1, HEAD_DIM]
    if isinstance(nargs["cl_ptr"], TensorDescriptor):
        nargs["cl_ptr"].block_shape = [1, 1, 1, BLOCK_K]

configs = [
    triton.Config({'BLOCK_I': BI, 'BLOCK_K': BK}, num_stages=s, num_warps=w, pre_hook=_z_fwd_pre_hook) \
    for BI in [16, 32, 64, 128]\
    for BK in [16, 32, 64, 128]\
    for s in [2, 3, 4] \
    for w in [4, 8]\
]

def keep(conf):
    BLOCK_I = conf.kwargs["BLOCK_I"]
    BLOCK_K = conf.kwargs["BLOCK_K"]
    return not (torch.cuda.get_device_capability()[0] == 9 and BLOCK_I * BLOCK_K < 128 * 128
                and conf.num_warps == 8)

_z_fwd_evaluated_configs = set()
def _start_z_fwd_evaluate(IS_FIRST_ITER, A_CHUNK, F_CHUNK, block_b1, block_b2, HEAD_DIM, OUTPUT_FULL_LSE, OUTPUT_PARTIAL_LSE, CAUSAL_BLOCK_SIZE, dtype):
    global _z_fwd_evaluated_configs
    config = (IS_FIRST_ITER, A_CHUNK, F_CHUNK, block_b1, block_b2, HEAD_DIM, OUTPUT_FULL_LSE, OUTPUT_PARTIAL_LSE, CAUSAL_BLOCK_SIZE, dtype)
    if config in _z_fwd_evaluated_configs:
        return False
    _z_fwd_evaluated_configs.add(config)
    return True

@triton.autotune(configs=list(filter(keep, configs)), key=["IS_FIRST_ITER", "A_CHUNK", "F_CHUNK", "block_b1", "block_b2", "HEAD_DIM", "OUTPUT_FULL_LSE", "OUTPUT_PARTIAL_LSE", "CAUSAL_BLOCK_SIZE"], cache_results=True)
@triton.jit
def _z_fwd(Z, H, A, F,
           a_start, f_start,
           curr_num_a, curr_num_f,
           al_ptr, cl_ptr,
           q_ptr, y_ptr,
           z_ptr, out_lse_ptr,
           sm_scale_sqrt,
           IS_FIRST_ITER: tl.constexpr,
           A_CHUNK: tl.constexpr,
           F_CHUNK: tl.constexpr,
           block_b1: tl.constexpr,
           block_b2: tl.constexpr,
           HEAD_DIM: tl.constexpr,
           OUTPUT_FULL_LSE: tl.constexpr,
           OUTPUT_PARTIAL_LSE: tl.constexpr,
           CAUSAL_BLOCK_SIZE: tl.constexpr,
           BLOCK_K: tl.constexpr,
           BLOCK_I: tl.constexpr,
           ):
    start_i = tl.program_id(0) * BLOCK_I
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_hd = off_h * HEAD_DIM
    off_aj = tl.program_id(2)
    off_a = off_aj // block_b2 + a_start
    off_j = off_aj % block_b2
    off_za = off_z * A + off_a
    off_za_mod = off_z * A_CHUNK + (off_a - a_start)
    off_zha_mod = off_hz * A_CHUNK + (off_a - a_start)

    ZA = Z * A
    ZA_mod = Z * A_CHUNK
    HD = H * HEAD_DIM
    khd_stride = block_b1 * HD
    jkhd_stride = block_b2 * khd_stride
    fjkhd_stride = F_CHUNK * jkhd_stride
    desc_al = _maybe_make_tensor_desc(
        al_ptr,
        shape=[ZA_mod, F_CHUNK, block_b2, block_b1, HD],
        strides=[fjkhd_stride, jkhd_stride, khd_stride, HD, 1],
        block_shape=[1, 1, 1, BLOCK_K, HEAD_DIM]
    )
    desc_y = _maybe_make_tensor_desc(
        y_ptr,
        shape=[ZA_mod, F_CHUNK, block_b2, block_b1, HD],
        strides=[fjkhd_stride, jkhd_stride, khd_stride, HD, 1],
        block_shape=[1, 1, 1, BLOCK_K, HEAD_DIM]
    )
    jhd_stride = block_b2 * HD
    ijhd_stride = jkhd_stride
    desc_q = _maybe_make_tensor_desc(
        q_ptr,
        shape=[ZA, block_b1, block_b2, H, HEAD_DIM],
        strides=[ijhd_stride, jhd_stride, HD, HEAD_DIM, 1],
        block_shape=[1, BLOCK_I, 1, 1, HEAD_DIM]
    )
    desc_z = _maybe_make_tensor_desc(
        z_ptr,
        shape=[ZA, block_b1, block_b2, H, HEAD_DIM],
        strides=[ijhd_stride, jhd_stride, HD, HEAD_DIM, 1],
        block_shape=[1, BLOCK_I, 1, 1, HEAD_DIM]
    )
    ZHA = ZA_mod * H
    jk_stride = block_b2 * block_b1
    fjk_stride = F_CHUNK * jk_stride
    desc_cl = tl.make_block_ptr(
        cl_ptr,
        shape=[ZHA, F_CHUNK, block_b2, block_b1],
        strides=[fjk_stride, jk_stride, block_b1, 1],
        block_shape=[1, 1, 1, BLOCK_K],
        order=[3, 2, 1, 0],
        offsets=[0, 0, 0, 0],
    ) if not isinstance(cl_ptr, tl.tensor_descriptor) else cl_ptr
    ji_stride = jk_stride
    if OUTPUT_FULL_LSE or OUTPUT_PARTIAL_LSE or (not IS_FIRST_ITER):
        if OUTPUT_FULL_LSE:
            ZHA = ZA * H
            off_zha = off_hz * A + off_a
        desc_lse = tl.make_block_ptr(
            out_lse_ptr,
            shape=[ZHA, block_b2, block_b1],
            strides=[ji_stride, block_b1, 1],
            block_shape=[1, 1, BLOCK_I],
            order=[2, 1, 0],
            offsets=[0, 0, 0],
        )

    dtype = desc_al.dtype

    if IS_FIRST_ITER:
        z_acc = tl.zeros([BLOCK_I, HEAD_DIM], dtype=tl.float32)
        l_i = tl.zeros([BLOCK_I], dtype=tl.float32)
        m_i = tl.full([BLOCK_I], dtype=tl.float32, value=float("-inf"))
    else:
        z_acc = desc_z.load([off_za, start_i, off_j, off_h, 0]).reshape(BLOCK_I, HEAD_DIM).to(tl.float32)
        l_i = tl.full([BLOCK_I], dtype=tl.float32, value=1.0)
        m_i = tl.load(desc_lse.advance([off_zha if OUTPUT_FULL_LSE else off_zha_mod, off_j, start_i]), boundary_check=(2,)).reshape(BLOCK_I).to(tl.float32)

    sm_scale_sqrt = sm_scale_sqrt.to(dtype)
    q_i = desc_q.load([off_za, start_i, off_j, off_h, 0]).reshape(BLOCK_I, HEAD_DIM)
    q_i = q_i * sm_scale_sqrt
    k_range = tl.arange(0, BLOCK_K)

    if CAUSAL_BLOCK_SIZE > 0:
        f_lim = _min(curr_num_f, _max(0, (off_a // CAUSAL_BLOCK_SIZE + 1) * CAUSAL_BLOCK_SIZE - f_start))
    else:
        f_lim = curr_num_f

    for f in tl.range(0, f_lim):
        for k in tl.static_range(0, block_b1, BLOCK_K):
            k_mask = k_range + k < block_b1
            al_k = desc_al.load([off_za_mod, f, off_j, k, off_hd]).reshape(BLOCK_K, HEAD_DIM)
            if isinstance(desc_cl, tl.tensor_descriptor):
                cl_k = desc_cl.load([off_zha_mod, f, off_j, k]).reshape(BLOCK_K) # already in log2 space
            else:
                cl_k = tl.load(desc_cl.advance([off_zha_mod, f, off_j, k]), boundary_check=(3,)).reshape(BLOCK_K)
            y_k = desc_y.load([off_za_mod, f, off_j, k, off_hd]).reshape(BLOCK_K, HEAD_DIM)
            y_k = tl.where(k_mask[:, None], y_k, 0.0)

            bl_ik = tl.dot(q_i, al_k.T) * 1.44269504 # log2(e)
            z_ik = tl.where(k_mask[None, :], bl_ik - cl_k[None, :], float("-inf"))

            m_ik = tl.maximum(m_i, tl.max(z_ik, 1))
            z_ik = z_ik - m_ik[:, None]
            p = tl.math.exp2(z_ik)
            l_ik = tl.sum(p, 1)

            alpha = tl.math.exp2(m_i - m_ik)
            z_acc = z_acc * alpha[:, None]

            p = p.to(dtype)
            z_acc = tl.dot(p, y_k, z_acc)

            l_i = l_i * alpha + l_ik
            m_i = m_ik

    z_acc = z_acc / l_i[:, None]
    desc_z.store([off_za, start_i, off_j, off_h, 0], z_acc.to(dtype).reshape(1, BLOCK_I, 1, 1, HEAD_DIM))

    if OUTPUT_FULL_LSE or OUTPUT_PARTIAL_LSE:
        lse = m_i + tl.math.log2(l_i)
        tl.store(desc_lse.advance([off_zha if OUTPUT_FULL_LSE else off_zha_mod, off_j, start_i]), lse.reshape(1, 1, BLOCK_I), boundary_check=(2,))

@torch.library.custom_op("monarch_rt::z_fwd", mutates_args=("z", "z_lse"))
def z_fwd(
    b: int, h: int, a: int, f: int,
    q_frame_start: int, kv_frame_start: int,
    curr_q_frames: int, curr_kv_frames: int,
    aL: torch.Tensor, cL: torch.Tensor,
    q: torch.Tensor, y: torch.Tensor,
    z: torch.Tensor,
    z_lse: Optional[torch.Tensor],
    sm_scale_sqrt: float,
    Q_FRAME_CHUNK: int, KV_FRAME_CHUNK: int,
    block_b1: int, block_b2: int, d: int,
    IS_FIRST_ITER: bool,
    OUTPUT_FULL_LSE: bool,
    OUTPUT_PARTIAL_LSE: bool,
    CAUSAL_BLOCK_SIZE: int
) -> None:
    def _z_grid(META):
        return (triton.cdiv(block_b1, META["BLOCK_I"]), b * h, curr_q_frames * block_b2)
    z_fwd_descs = _init_z_fwd_descs(b, h, a, f, Q_FRAME_CHUNK, KV_FRAME_CHUNK, d, block_b1, block_b2, aL, y, q, z, cL)
    global _z_backup, _z_lse_backup
    if _start_z_fwd_evaluate(IS_FIRST_ITER, Q_FRAME_CHUNK, KV_FRAME_CHUNK, block_b1, block_b2, d, OUTPUT_FULL_LSE, OUTPUT_PARTIAL_LSE, CAUSAL_BLOCK_SIZE, q.dtype):
        _z_backup = z.clone()
    _z_fwd[_z_grid](
        b, h, a, f,
        q_frame_start, kv_frame_start,
        curr_q_frames, curr_kv_frames,
        z_fwd_descs.aL, z_fwd_descs.cL,
        z_fwd_descs.q, z_fwd_descs.y,
        z_fwd_descs.z, z_lse,
        sm_scale_sqrt,
        IS_FIRST_ITER=IS_FIRST_ITER,
        A_CHUNK=Q_FRAME_CHUNK,
        F_CHUNK=KV_FRAME_CHUNK,
        block_b1=block_b1,
        block_b2=block_b2,
        HEAD_DIM=d,
        OUTPUT_FULL_LSE=OUTPUT_FULL_LSE,
        OUTPUT_PARTIAL_LSE=OUTPUT_PARTIAL_LSE,
        CAUSAL_BLOCK_SIZE=CAUSAL_BLOCK_SIZE,
    )
    _z_backup = None
    _z_lse_backup = None

@z_fwd.register_fake
def _(
    b: int, h: int, a: int, f: int,
    q_frame_start: int, kv_frame_start: int,
    curr_q_frames: int, curr_kv_frames: int,
    aL: torch.Tensor, cL: torch.Tensor,
    q: torch.Tensor, y: torch.Tensor,
    z: torch.Tensor,
    z_lse: Optional[torch.Tensor],
    sm_scale_sqrt: float,
    IS_FIRST_ITER: bool,
    Q_FRAME_CHUNK: int, KV_FRAME_CHUNK: int,
    block_b1: int, block_b2: int, d: int,
    OUTPUT_FULL_LSE: bool,
    OUTPUT_PARTIAL_LSE: bool,
    CAUSAL_BLOCK_SIZE: int
) -> None:
    pass


configs = [
    triton.Config({'BLOCK_Q': BQ, 'BLOCK_HEAD_DIM': BHD}, num_stages=s, num_warps=w) \
    for BQ in [16, 32, 64, 128]\
    for BHD in [16, 32, 64, 128]\
    for s in [2, 3, 4] \
    for w in [4, 8]\
]

@triton.autotune(configs=configs, key=["HEAD_DIM", "block_b1", "block_b2"], cache_results=True)
@triton.jit
def _z_bwd_preprocess(A, H,
                      z_ptr, dz_ptr, d_ptr,
                      block_b1: tl.constexpr,
                      block_b2: tl.constexpr,
                      HEAD_DIM: tl.constexpr,
                      BLOCK_Q: tl.constexpr,
                      BLOCK_HEAD_DIM: tl.constexpr
                      ):
    off_z = tl.program_id(0)
    start_q = tl.program_id(1) * BLOCK_Q
    off_h = tl.program_id(2)

    q_range = tl.arange(0, BLOCK_Q) + start_q
    i_idxs = q_range % block_b1 # make i contiguous for the d store
    j_idxs = (q_range // block_b1) % block_b2
    a_idxs = q_range // (block_b2 * block_b1)
    q_mask = a_idxs < A

    z_ptr_offset = (((((off_z * A) + a_idxs) * block_b1 + i_idxs) * block_b2 + j_idxs) * H + off_h) * HEAD_DIM
    z_ptrs = z_ptr + z_ptr_offset
    dz_ptrs = dz_ptr + z_ptr_offset
    d_range = tl.arange(0, BLOCK_HEAD_DIM)

    d_ptrs = d_ptr + (((off_z * H + off_h) * A + a_idxs) * block_b2 + j_idxs) * block_b1 + i_idxs

    acc = tl.zeros([BLOCK_Q], dtype=tl.float32)
    tl.static_assert(HEAD_DIM % BLOCK_HEAD_DIM == 0, "HEAD_DIM must be multiple of BLOCK_HEAD_DIM")
    for _ in tl.static_range(0, HEAD_DIM, BLOCK_HEAD_DIM):
        z_i = tl.load(z_ptrs[:, None] + d_range[None, :], mask=q_mask[:, None])
        dz_i = tl.load(dz_ptrs[:, None] + d_range[None, :], mask=q_mask[:, None])
        acc += (z_i * dz_i).sum(1)
        d_range += BLOCK_HEAD_DIM
    tl.store(d_ptrs, acc, mask=q_mask)

@torch.library.custom_op("monarch_rt::z_bwd_preprocess", mutates_args=("z_d",))
def z_bwd_preprocess(
    b: int, h: int, a: int,
    z: torch.Tensor, grad_z: torch.Tensor,
    z_d: torch.Tensor,
    block_b1: int, block_b2: int, d: int,
) -> None:
    def grid(META):
        return (b, triton.cdiv(a * block_b1 * block_b2, META["BLOCK_Q"]), h)
    _z_bwd_preprocess[grid](
        a, h,
        z, grad_z, z_d,
        block_b1,
        block_b2,
        d,
    )

@z_bwd_preprocess.register_fake
def _(
    b: int, h: int, a: int,
    z: torch.Tensor, grad_z: torch.Tensor,
    z_d: torch.Tensor,
    block_b1: int, block_b2: int, d: int,
) -> None:
    pass


def _init_z_bwd_descs(Z, H, A, F, A_CHUNK, F_CHUNK, HEAD_DIM, block_b1, block_b2, aL, grad_aL, y, grad_y, q, grad_q, grad_z, lse, d, cL, grad_cL):
    AFJ = A_CHUNK * F_CHUNK * block_b2
    ZAFJ = Z * AFJ
    hd_stride = H * HEAD_DIM
    khd_stride = block_b1 * hd_stride
    ZA = Z * A
    jhd_stride = block_b2 * hd_stride
    ijhd_stride = block_b1 * jhd_stride
    ZHAJ = Z * H * A * block_b2
    ZHAFJ = ZAFJ * H

    descs = SimpleNamespace()
    if supports_host_descriptor:
        descs.aL = TensorDescriptor(
            aL,
            shape=[ZAFJ, block_b1, H, HEAD_DIM],
            strides=[khd_stride, hd_stride, HEAD_DIM, 1],
            block_shape=[1, 1, 1, 1]
        )
        descs.grad_aL = TensorDescriptor(
            grad_aL,
            shape=[ZAFJ, block_b1, H, HEAD_DIM],
            strides=[khd_stride, hd_stride, HEAD_DIM, 1],
            block_shape=[1, 1, 1, 1]
        )
        descs.y = TensorDescriptor(
            y,
            shape=[ZAFJ, block_b1, H, HEAD_DIM],
            strides=[khd_stride, hd_stride, HEAD_DIM, 1],
            block_shape=[1, 1, 1, 1]
        )
        descs.grad_y = TensorDescriptor(
            grad_y,
            shape=[ZAFJ, block_b1, H, HEAD_DIM],
            strides=[khd_stride, hd_stride, HEAD_DIM, 1],
            block_shape=[1, 1, 1, 1]
        )
        descs.q = TensorDescriptor(
            q,
            shape=[ZA, block_b1, block_b2, H, HEAD_DIM],
            strides=[ijhd_stride, jhd_stride, hd_stride, HEAD_DIM, 1],
            block_shape=[1, 1, 1, 1, 1]
        )
        descs.grad_q = TensorDescriptor(
            grad_q,
            shape=[ZA, block_b1, block_b2, H, HEAD_DIM],
            strides=[ijhd_stride, jhd_stride, hd_stride, HEAD_DIM, 1],
            block_shape=[1, 1, 1, 1, 1]
        )
        descs.grad_z = TensorDescriptor(
            grad_z,
            shape=[ZA, block_b1, block_b2, H, HEAD_DIM],
            strides=[ijhd_stride, jhd_stride, hd_stride, HEAD_DIM, 1],
            block_shape=[1, 1, 1, 1, 1]
        )
        # TMA requires 16-byte alignment for leading strides
        descs.lse = TensorDescriptor(
            lse,
            shape=[ZHAJ, block_b1],
            strides=[block_b1, 1],
            block_shape=[1, 1]
        ) if (block_b1 * lse.element_size()) % 16 == 0 else lse
        descs.d = TensorDescriptor(
            d,
            shape=[ZHAJ, block_b1],
            strides=[block_b1, 1],
            block_shape=[1, 1]
        ) if (block_b1 * d.element_size()) % 16 == 0 else d
        descs.cL = TensorDescriptor(
            cL,
            shape=[ZHAFJ, block_b1],
            strides=[block_b1, 1],
            block_shape=[1, 1]
        ) if (block_b1 * cL.element_size()) % 16 == 0 else cL
        descs.grad_cL = TensorDescriptor(
            grad_cL,
            shape=[ZHAFJ, block_b1],
            strides=[block_b1, 1],
            block_shape=[1, 1]
        ) if (block_b1 * grad_cL.element_size()) % 16 == 0 else grad_cL
    else:
        descs.aL = aL
        descs.grad_aL = grad_aL
        descs.y = y
        descs.grad_y = grad_y
        descs.q = q
        descs.grad_q = grad_q
        descs.grad_z = grad_z
        descs.lse = lse
        descs.d = d
        descs.cL = cL
        descs.grad_cL = grad_cL

    return descs

def _z_bwd_pre_hook(nargs):
    BLOCK_K = nargs["BLOCK_K"]
    BLOCK_I = nargs["BLOCK_I"]
    HEAD_DIM = nargs["HEAD_DIM"]
    global _dq_backup
    if not isinstance(nargs["al_ptr"], TensorDescriptor):
        if _dq_backup is not None:
            nargs["dq_ptr"].copy_(_dq_backup)
        return
    if _dq_backup is not None:
        nargs["dq_ptr"].base.copy_(_dq_backup)
    nargs["al_ptr"].block_shape = [1, BLOCK_K, 1, HEAD_DIM]
    nargs["dal_ptr"].block_shape = [1, BLOCK_K, 1, HEAD_DIM]
    nargs["y_ptr"].block_shape = [1, BLOCK_K, 1, HEAD_DIM]
    nargs["dy_ptr"].block_shape = [1, BLOCK_K, 1, HEAD_DIM]
    nargs["q_ptr"].block_shape = [1, BLOCK_I, 1, 1, HEAD_DIM]
    nargs["dq_ptr"].block_shape = [1, BLOCK_I, 1, 1, HEAD_DIM]
    nargs["dz_ptr"].block_shape = [1, BLOCK_I, 1, 1, HEAD_DIM]
    if isinstance(nargs["lse_ptr"], TensorDescriptor):
        nargs["lse_ptr"].block_shape = [1, BLOCK_I]
    if isinstance(nargs["d_ptr"], TensorDescriptor):
        nargs["d_ptr"].block_shape = [1, BLOCK_I]
    if isinstance(nargs["cl_ptr"], TensorDescriptor):
        nargs["cl_ptr"].block_shape = [1, BLOCK_K]
    if isinstance(nargs["dcl_ptr"], TensorDescriptor):
        nargs["dcl_ptr"].block_shape = [1, BLOCK_K]

configs = [
    triton.Config({'BLOCK_I': BI, 'BLOCK_K': BK}, num_stages=s, num_warps=w, pre_hook=_z_bwd_pre_hook) \
    for BI in [64, 128]\
    for BK in [32, 64, 128]\
    for s in [2, 3, 4] \
    for w in [4, 8]\
]

def keep(conf):
    BLOCK_I = conf.kwargs["BLOCK_I"]
    BLOCK_K = conf.kwargs["BLOCK_K"]
    return not (torch.cuda.get_device_capability()[0] == 9 and BLOCK_I * BLOCK_K < 128 * 128
                and conf.num_warps == 8)

_z_bwd_evaluated_configs = set()
def _start_z_bwd_evaluate(A_CHUNK, F_CHUNK, block_b1, block_b2, HEAD_DIM, CAUSAL_BLOCK_SIZE, dtype):
    global _z_bwd_evaluated_configs
    config = (A_CHUNK, F_CHUNK, block_b1, block_b2, HEAD_DIM, CAUSAL_BLOCK_SIZE, dtype)
    if config in _z_bwd_evaluated_configs:
        return False
    _z_bwd_evaluated_configs.add(config)
    return True

@triton.autotune(configs=list(filter(keep, configs)), key=["A_CHUNK", "F_CHUNK", "block_b1", "block_b2", "HEAD_DIM", "CAUSAL_BLOCK_SIZE"], cache_results=True)
@triton.jit
def _z_bwd(Z, H, A, F,
           a_start, f_start,
           curr_num_a, curr_num_f,
           al_ptr, cl_ptr,
           q_ptr, y_ptr,
           dz_ptr, lse_ptr, d_ptr,
           dal_ptr, dcl_ptr,
           dq_ptr, dy_ptr,
           out_d_ptr,
           sm_scale_sqrt,
           A_CHUNK: tl.constexpr,
           F_CHUNK: tl.constexpr,
           block_b1: tl.constexpr,
           block_b2: tl.constexpr,
           HEAD_DIM: tl.constexpr,
           CAUSAL_BLOCK_SIZE: tl.constexpr,
           BLOCK_K: tl.constexpr,
           BLOCK_I: tl.constexpr,
           ):
    start_k = tl.program_id(0) * BLOCK_K
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_afj = tl.program_id(2)
    off_j = off_afj % block_b2
    off_f = (off_afj // block_b2) % curr_num_f
    off_a = off_afj // (block_b2 * curr_num_f)
    off_za = off_z * A + (off_a + a_start)

    if CAUSAL_BLOCK_SIZE > 0:
        if ((off_a + a_start) // CAUSAL_BLOCK_SIZE) < ((off_f + f_start) // CAUSAL_BLOCK_SIZE):
            return

    AFJ = A_CHUNK * F_CHUNK * block_b2
    off_afj = (off_a * F_CHUNK + off_f) * block_b2 + off_j
    off_zafj = off_z * AFJ + off_afj
    off_zhaj = (off_hz * A + (off_a + a_start)) * block_b2 + off_j
    off_zhafj = off_hz * AFJ + off_afj

    ZAFJ = Z * AFJ
    hd_stride = H * HEAD_DIM
    khd_stride = block_b1 * hd_stride
    desc_al = _maybe_make_tensor_desc(
        al_ptr,
        shape=[ZAFJ, block_b1, H, HEAD_DIM],
        strides=[khd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, BLOCK_K, 1, HEAD_DIM]
    )
    desc_dal = _maybe_make_tensor_desc(
        dal_ptr,
        shape=[ZAFJ, block_b1, H, HEAD_DIM],
        strides=[khd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, BLOCK_K, 1, HEAD_DIM]
    )
    desc_y = _maybe_make_tensor_desc(
        y_ptr,
        shape=[ZAFJ, block_b1, H, HEAD_DIM],
        strides=[khd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, BLOCK_K, 1, HEAD_DIM]
    )
    desc_dy = _maybe_make_tensor_desc(
        dy_ptr,
        shape=[ZAFJ, block_b1, H, HEAD_DIM],
        strides=[khd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, BLOCK_K, 1, HEAD_DIM]
    )

    ZA = Z * A
    jhd_stride = block_b2 * hd_stride
    ijhd_stride = block_b1 * jhd_stride
    desc_q = _maybe_make_tensor_desc(
        q_ptr,
        shape=[ZA, block_b1, block_b2, H, HEAD_DIM],
        strides=[ijhd_stride, jhd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, BLOCK_I, 1, 1, HEAD_DIM]
    )
    desc_dq = _maybe_make_tensor_desc(
        dq_ptr,
        shape=[ZA, block_b1, block_b2, H, HEAD_DIM],
        strides=[ijhd_stride, jhd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, BLOCK_I, 1, 1, HEAD_DIM]
    )
    desc_dz = _maybe_make_tensor_desc(
        dz_ptr,
        shape=[ZA, block_b1, block_b2, H, HEAD_DIM],
        strides=[ijhd_stride, jhd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, BLOCK_I, 1, 1, HEAD_DIM]
    )
    ZHAJ = Z * H * A * block_b2
    desc_lse = tl.make_block_ptr(
        lse_ptr,
        shape=[ZHAJ, block_b1],
        strides=[block_b1, 1],
        block_shape=[1, BLOCK_I],
        order=[1, 0],
        offsets=[0, 0],
    ) if not isinstance(lse_ptr, tl.tensor_descriptor) else lse_ptr
    desc_d = tl.make_block_ptr(
        d_ptr,
        shape=[ZHAJ, block_b1],
        strides=[block_b1, 1],
        block_shape=[1, BLOCK_I],
        order=[1, 0],
        offsets=[0, 0],
    ) if not isinstance(d_ptr, tl.tensor_descriptor) else d_ptr
    ZHAFJ = ZAFJ * H
    desc_cl = tl.make_block_ptr(
        cl_ptr,
        shape=[ZHAFJ, block_b1],
        strides=[block_b1, 1],
        block_shape=[1, BLOCK_K],
        order=[1, 0],
        offsets=[0, 0],
    ) if not isinstance(cl_ptr, tl.tensor_descriptor) else cl_ptr
    desc_dcl = tl.make_block_ptr(
        dcl_ptr,
        shape=[ZHAFJ, block_b1],
        strides=[block_b1, 1],
        block_shape=[1, BLOCK_K],
        order=[1, 0],
        offsets=[0, 0],
    ) if not isinstance(dcl_ptr, tl.tensor_descriptor) else dcl_ptr

    dtype = desc_al.dtype

    dal_k_acc = tl.zeros([BLOCK_K, HEAD_DIM], dtype=tl.float32)
    dy_k_acc = tl.zeros([BLOCK_K, HEAD_DIM], dtype=tl.float32)
    dcl_k_acc = tl.zeros([BLOCK_K], dtype=tl.float32)

    al_k = desc_al.load([off_zafj, start_k, off_h, 0]).reshape(BLOCK_K, HEAD_DIM)
    y_kt = desc_y.load([off_zafj, start_k, off_h, 0]).reshape(BLOCK_K, HEAD_DIM).T
    if isinstance(desc_cl, tl.tensor_descriptor):
        cl_k = desc_cl.load([off_zhafj, start_k]).reshape(BLOCK_K) # already in log2 space
    else:
        cl_k = tl.load(desc_cl.advance([off_zhafj, start_k]), boundary_check=(1,)).reshape(BLOCK_K)
    k_mask = (start_k + tl.arange(0, BLOCK_K)) < block_b1
    al_k = tl.where(k_mask[:, None], al_k, 0.0)

    i_range = tl.arange(0, BLOCK_I)

    sm_scale_sqrt = sm_scale_sqrt.to(dtype)

    for i in tl.static_range(0, block_b1, BLOCK_I):
        i_mask = i_range + i < block_b1
        q_i = desc_q.load([off_za, i, off_j, off_h, 0]).reshape(BLOCK_I, HEAD_DIM) * sm_scale_sqrt
        dz_i = desc_dz.load([off_za, i, off_j, off_h, 0]).reshape(BLOCK_I, HEAD_DIM)
        dz_i = tl.where(i_mask[:, None], dz_i, 0.0)
        if isinstance(desc_lse, tl.tensor_descriptor):
            lse_i = desc_lse.load([off_zhaj, i]).reshape(BLOCK_I)
        else:
            lse_i = tl.load(desc_lse.advance([off_zhaj, i]), boundary_check=(1,)).reshape(BLOCK_I)   
        if isinstance(desc_d, tl.tensor_descriptor):
            d_i = desc_d.load([off_zhaj, i]).reshape(BLOCK_I)
            d_i = tl.where(i_mask, d_i, 0.0)
        else:
            d_i = tl.load(desc_d.advance([off_zhaj, i]), boundary_check=(1,), padding_option="zero").reshape(BLOCK_I)

        bl_ik = tl.dot(q_i, al_k.T) * 1.44269504 # log2(e)
        s_ik = bl_ik - cl_k[None, :] - lse_i[:, None]
        s_ik = tl.where(i_mask[:, None], s_ik, float("-inf"))
        p = tl.math.exp2(s_ik)

        dp = tl.dot(dz_i, y_kt)
        ds_ik = p * (dp - d_i[:, None])
        dcl_k_acc -= ds_ik.sum(0)

        p = p.to(dtype)
        dy_k_acc = tl.dot(p.T, dz_i, dy_k_acc)

        ds_ik = ds_ik.to(dtype)
        ds_ik = tl.where(k_mask[None, :], ds_ik, 0.0)
        dq_i = tl.dot(ds_ik, al_k) * sm_scale_sqrt
        desc_dq.atomic_add([off_za, i, off_j, off_h, 0], dq_i.reshape(1, BLOCK_I, 1, 1, HEAD_DIM))

        q_i = tl.where(i_mask[:, None], q_i, 0.0)
        dal_k_acc = tl.dot(ds_ik.T, q_i, dal_k_acc)

    out_d_ptrs = out_d_ptr + ((((off_z * H + off_h) * A_CHUNK + off_a) * F_CHUNK + off_f) * block_b1 + start_k + tl.arange(0, BLOCK_K)) * block_b2 + off_j
    d = (dal_k_acc * al_k).sum(1) + (dy_k_acc * y_kt.T).sum(1) + dcl_k_acc * (1 + cl_k * 0.693147181)
    tl.store(out_d_ptrs, d, mask=k_mask)

    desc_dal.store([off_zafj, start_k, off_h, 0], dal_k_acc.reshape(1, BLOCK_K, 1, HEAD_DIM).to(dtype))
    desc_dy.store([off_zafj, start_k, off_h, 0], dy_k_acc.reshape(1, BLOCK_K, 1, HEAD_DIM).to(dtype))
    if isinstance(desc_dcl, tl.tensor_descriptor):
        desc_dcl.store([off_zhafj, start_k], dcl_k_acc.reshape(1, BLOCK_K))
    else:
        tl.store(desc_dcl.advance([off_zhafj, start_k]), dcl_k_acc.reshape(1, BLOCK_K), boundary_check=(1,))

@torch.library.custom_op("monarch_rt::z_bwd", mutates_args=("grad_aL", "grad_cL", "grad_q", "grad_y", "al_cl_y_d"))
def z_bwd(
    b: int, h: int, a: int, f: int,
    q_frame_start: int, kv_frame_start: int,
    curr_q_frames: int, curr_kv_frames: int,
    aL: torch.Tensor, cL: torch.Tensor,
    q: torch.Tensor, y: torch.Tensor,
    grad_aL: torch.Tensor, grad_cL: torch.Tensor,
    grad_q: torch.Tensor, grad_y: torch.Tensor,
    grad_z: torch.Tensor, z_lse: torch.Tensor,
    z_d: torch.Tensor, al_cl_y_d: torch.Tensor,
    sm_scale_sqrt: float,
    Q_FRAME_CHUNK: int, KV_FRAME_CHUNK: int,
    block_b1: int, block_b2: int, d: int,
    CAUSAL_BLOCK_SIZE: int
) -> None:
    def _z_bwd_grid(META):
        return (triton.cdiv(block_b1, META["BLOCK_K"]), b * h, curr_q_frames * curr_kv_frames * block_b2)
    z_bwd_descs = _init_z_bwd_descs(b, h, a, f, Q_FRAME_CHUNK, KV_FRAME_CHUNK, d, block_b1, block_b2, aL, grad_aL, y, grad_y, q, grad_q, grad_z, z_lse, z_d, cL, grad_cL)
    
    global _dq_backup
    if _start_z_bwd_evaluate(Q_FRAME_CHUNK, KV_FRAME_CHUNK, block_b1, block_b2, d, CAUSAL_BLOCK_SIZE, q.dtype):
        _dq_backup = grad_q.clone()
    _z_bwd[_z_bwd_grid](
        b, h, a, f,
        q_frame_start, kv_frame_start,
        curr_q_frames, curr_kv_frames,
        z_bwd_descs.aL, z_bwd_descs.cL,
        z_bwd_descs.q, z_bwd_descs.y,
        z_bwd_descs.grad_z, z_bwd_descs.lse, z_bwd_descs.d,
        z_bwd_descs.grad_aL, z_bwd_descs.grad_cL,
        z_bwd_descs.grad_q, z_bwd_descs.grad_y,
        al_cl_y_d,
        sm_scale_sqrt,
        Q_FRAME_CHUNK,
        KV_FRAME_CHUNK,
        block_b1,
        block_b2,
        d,
        CAUSAL_BLOCK_SIZE=CAUSAL_BLOCK_SIZE,
    )
    _dq_backup = None

@z_bwd.register_fake
def _(
    b: int, h: int, a: int, f: int,
    q_frame_start: int, kv_frame_start: int,
    curr_q_frames: int, curr_kv_frames: int,
    aL: torch.Tensor, cL: torch.Tensor,
    q: torch.Tensor, y: torch.Tensor,
    grad_aL: torch.Tensor, grad_cL: torch.Tensor,
    grad_q: torch.Tensor, grad_y: torch.Tensor,
    grad_z: torch.Tensor, z_lse: torch.Tensor,
    z_d: torch.Tensor, al_cl_y_d: torch.Tensor,
    sm_scale_sqrt: float,
    Q_FRAME_CHUNK: int, KV_FRAME_CHUNK: int,
    block_b1: int, block_b2: int, d: int,
    CAUSAL_BLOCK_SIZE: int
) -> None:
    pass


class _attention_with_cache(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k_cache, v_cache, new_k, new_v, start_idx, end_idx, sm_scale, grad_enabled, grad_only_new_kv):
        b, a, block_b1, block_b2, h, d = q.shape
        
        k_cache[:, start_idx:end_idx, :, :] = new_k
        v_cache[:, start_idx:end_idx, :, :] = new_v

        k_cache = k_cache.view(b, -1, block_b1, block_b2, h, d)
        v_cache = v_cache.view(b, -1, block_b1, block_b2, h, d)
        f = k_cache.shape[1]

        sm_scale_sqrt = sm_scale ** 0.5

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")
        triton.set_allocator(alloc_fn)

        # TODO: probably make this tunable/dependent on memory constraints (temporary hardcoded sizes to use less memory but have roughly same speed as non-streaming implementation)
        Q_FRAME_CHUNK = 1
        KV_FRAME_CHUNK = f

        aL = torch.empty((b, Q_FRAME_CHUNK, KV_FRAME_CHUNK, block_b2, block_b1, h, d), device=q.device, dtype=q.dtype) # (b, a, f, j, k, h, d)
        cL = torch.empty((b, h, Q_FRAME_CHUNK, KV_FRAME_CHUNK, block_b2, block_b1), device=q.device, dtype=torch.float32)  # (b, h, a, f, j, k)
        y = torch.empty((b, Q_FRAME_CHUNK, KV_FRAME_CHUNK, block_b2, block_b1, h, d), device=q.device, dtype=q.dtype) # (b, a, f, j, k, h, d)
        z = torch.empty((b, a, block_b1, block_b2, h, d), device=q.device, dtype=q.dtype) # (b, a, i, j, h, d)
        if grad_enabled:
            z_lse = torch.empty((b, h, a, block_b2, block_b1), device=q.device, dtype=torch.float32) # (b, h, a, j, i)
        elif KV_FRAME_CHUNK < f:
            z_lse = torch.empty((b, h, Q_FRAME_CHUNK, block_b2, block_b1), device=q.device, dtype=torch.float32) # (b, h, a, j, i)
        else:
            z_lse = None
        
        assert k_cache.stride() == v_cache.stride(), "current implementation assumes same strides for k and v"
        for q_frame_start in range(0, a, Q_FRAME_CHUNK):
            curr_q_frames = min(Q_FRAME_CHUNK, a - q_frame_start)
            for kv_frame_start in range(0, f, KV_FRAME_CHUNK):
                curr_kv_frames = min(KV_FRAME_CHUNK, f - kv_frame_start)
                al_cl_y_fwd(
                    b, h, a, f,
                    q_frame_start, kv_frame_start,
                    curr_q_frames, curr_kv_frames,
                    q, k_cache, v_cache,
                    aL, cL, y, None,
                    sm_scale_sqrt,
                    Q_FRAME_CHUNK, KV_FRAME_CHUNK,
                    block_b1, block_b2, d,
                    CAUSAL_BLOCK_SIZE=0,
                )
                z_fwd(
                    b, h, a, f,
                    q_frame_start, kv_frame_start,
                    curr_q_frames, curr_kv_frames,
                    aL, cL,
                    q, y,
                    z, z_lse,
                    sm_scale_sqrt,
                    Q_FRAME_CHUNK, KV_FRAME_CHUNK,
                    block_b1, block_b2, d,
                    IS_FIRST_ITER=(kv_frame_start == 0),
                    OUTPUT_FULL_LSE=grad_enabled,
                    OUTPUT_PARTIAL_LSE=(not grad_enabled and kv_frame_start + KV_FRAME_CHUNK < f),
                    CAUSAL_BLOCK_SIZE=0,
                )
        
        if grad_enabled:
            ctx.save_for_backward(q, k_cache, v_cache, z, z_lse)
            ctx.sm_scale_sqrt = sm_scale_sqrt
            ctx.start_idx = start_idx
            ctx.end_idx = end_idx
            ctx.grad_only_new_kv = grad_only_new_kv

        return z

    @staticmethod
    def backward(ctx, grad_z):
        q, k, v, z, z_lse = ctx.saved_tensors
        sm_scale_sqrt = ctx.sm_scale_sqrt

        b, a, block_b1, block_b2, h, d = z.shape
        f = k.shape[1]

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")
        triton.set_allocator(alloc_fn)

        z_d = torch.empty_like(z_lse)
        z_bwd_preprocess(
            b, h, a,
            z, grad_z, z_d,
            block_b1,
            block_b2,
            d,
        )

        Q_FRAME_CHUNK = 1
        KV_FRAME_CHUNK = f

        aL = torch.empty((b, Q_FRAME_CHUNK, KV_FRAME_CHUNK, block_b2, block_b1, h, d), device=q.device, dtype=q.dtype) # (b, a, f, j, k, h, d)
        cL = torch.empty((b, h, Q_FRAME_CHUNK, KV_FRAME_CHUNK, block_b2, block_b1), device=q.device, dtype=torch.float32)  # (b, h, a, f, j, k)
        y = torch.empty((b, Q_FRAME_CHUNK, KV_FRAME_CHUNK, block_b2, block_b1, h, d), device=q.device, dtype=q.dtype) # (b, a, f, j, k, h, d)
        al_cl_y_lse = torch.empty((b, h, Q_FRAME_CHUNK, KV_FRAME_CHUNK, block_b1, block_b2), device=q.device, dtype=torch.float32)

        grad_aL = torch.empty_like(aL)
        grad_cL = torch.empty_like(cL)
        grad_q = torch.zeros_like(q, dtype=torch.float32) # float32 for atomic add
        grad_y = torch.empty_like(y)
        al_cl_y_d = torch.empty_like(al_cl_y_lse)

        if ctx.grad_only_new_kv:
            assert ctx.start_idx % (block_b1 * block_b2) == 0 and ctx.end_idx % (block_b1 * block_b2) == 0, "implementation currently doesn't support partial kv gradient for misaligned indices"
            kv_start_frame = ctx.start_idx // (block_b1 * block_b2)
            kv_end_frame = ctx.end_idx // (block_b1 * block_b2)
            grad_k = torch.zeros((b, kv_end_frame - kv_start_frame, block_b1, block_b2, h, d), device=k.device, dtype=k.dtype)
            grad_v = torch.zeros((b, kv_end_frame - kv_start_frame, block_b1, block_b2, h, d), device=v.device, dtype=v.dtype)
        else:
            kv_start_frame = 0
            kv_end_frame = f
            grad_k = torch.empty_like(k)
            grad_v = torch.empty_like(v)

        for q_frame_start in range(0, a, Q_FRAME_CHUNK):
            curr_q_frames = min(Q_FRAME_CHUNK, a - q_frame_start)
            for kv_frame_start in range(0, f, KV_FRAME_CHUNK):
                curr_kv_frames = min(KV_FRAME_CHUNK, f - kv_frame_start)                

                al_cl_y_fwd(
                    b, h, a, f,
                    q_frame_start, kv_frame_start,
                    curr_q_frames, curr_kv_frames,
                    q, k, v,
                    aL, cL, y,
                    al_cl_y_lse,
                    sm_scale_sqrt,
                    Q_FRAME_CHUNK, KV_FRAME_CHUNK,
                    block_b1, block_b2, d,
                    CAUSAL_BLOCK_SIZE=0,
                )

                z_bwd(
                    b, h, a, f,
                    q_frame_start, kv_frame_start,
                    curr_q_frames, curr_kv_frames,
                    aL, cL,
                    q, y,
                    grad_aL, grad_cL,
                    grad_q, grad_y,
                    grad_z, z_lse,
                    z_d, al_cl_y_d,
                    sm_scale_sqrt,
                    Q_FRAME_CHUNK, KV_FRAME_CHUNK,
                    block_b1, block_b2, d,
                    CAUSAL_BLOCK_SIZE=0,
                )

                al_cl_y_bwd(
                    b, h, a, f,
                    q_frame_start, kv_frame_start,
                    curr_q_frames, curr_kv_frames,
                    kv_start_frame, kv_end_frame,
                    q, k, v,
                    grad_q, grad_k, grad_v,
                    grad_aL, grad_cL, grad_y,
                    al_cl_y_lse, al_cl_y_d,
                    sm_scale_sqrt,
                    Q_FRAME_CHUNK, KV_FRAME_CHUNK,
                    block_b1, block_b2, d,
                    IS_FIRST_ITER=(q_frame_start == 0),
                    PARTIAL_KV_GRAD=ctx.grad_only_new_kv,
                    CAUSAL_BLOCK_SIZE=0,
                )

        grad_k = grad_k.view(b, -1, h, d)
        grad_v = grad_v.view(b, -1, h, d)
        if ctx.grad_only_new_kv:
            grad_k_cache, grad_v_cache = None, None
        else:
            grad_k_cache, grad_v_cache = grad_k, grad_v
            grad_k = grad_k_cache[:, ctx.start_idx:ctx.end_idx, :, :]
            grad_v = grad_v_cache[:, ctx.start_idx:ctx.end_idx, :, :]

        return grad_q.to(q.dtype), grad_k_cache, grad_v_cache, grad_k, grad_v, None, None, None, None, None

class _attention_no_cache(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale, causal_block_size, grad_enabled):
        b, a, block_b1, block_b2, h, d = q.shape
        f = k.shape[1]

        sm_scale_sqrt = sm_scale ** 0.5

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")
        triton.set_allocator(alloc_fn)

        Q_FRAME_CHUNK = 1
        KV_FRAME_CHUNK = f

        aL = torch.empty((b, Q_FRAME_CHUNK, KV_FRAME_CHUNK, block_b2, block_b1, h, d), device=q.device, dtype=q.dtype) # (b, a, f, j, k, h, d)
        cL = torch.empty((b, h, Q_FRAME_CHUNK, KV_FRAME_CHUNK, block_b2, block_b1), device=q.device, dtype=torch.float32)  # (b, h, a, f, j, k)
        y = torch.empty((b, Q_FRAME_CHUNK, KV_FRAME_CHUNK, block_b2, block_b1, h, d), device=q.device, dtype=q.dtype) # (b, a, f, j, k, h, d)
        z = torch.empty((b, a, block_b1, block_b2, h, d), device=q.device, dtype=q.dtype) # (b, a, i, j, h, d)
        if grad_enabled:
            z_lse = torch.empty((b, h, a, block_b2, block_b1), device=q.device, dtype=torch.float32) # (b, h, a, j, i)
        elif KV_FRAME_CHUNK < f:
            z_lse = torch.empty((b, h, Q_FRAME_CHUNK, block_b2, block_b1), device=q.device, dtype=torch.float32) # (b, h, a, j, i)
        else:
            z_lse = None

        assert k.stride() == v.stride(), "current implementation assumes same strides for k and v"
        for q_frame_start in range(0, a, Q_FRAME_CHUNK):
            curr_q_frames = min(Q_FRAME_CHUNK, a - q_frame_start)
            for kv_frame_start in range(0, f, KV_FRAME_CHUNK):
                curr_kv_frames = min(KV_FRAME_CHUNK, f - kv_frame_start)
                al_cl_y_fwd(
                    b, h, a, f,
                    q_frame_start, kv_frame_start,
                    curr_q_frames, curr_kv_frames,
                    q, k, v,
                    aL, cL, y, None,
                    sm_scale_sqrt,
                    Q_FRAME_CHUNK, KV_FRAME_CHUNK,
                    block_b1, block_b2, d,
                    CAUSAL_BLOCK_SIZE=causal_block_size,
                )
                z_fwd(
                    b, h, a, f,
                    q_frame_start, kv_frame_start,
                    curr_q_frames, curr_kv_frames,
                    aL, cL,
                    q, y,
                    z, z_lse,
                    sm_scale_sqrt,
                    Q_FRAME_CHUNK, KV_FRAME_CHUNK,
                    block_b1, block_b2, d,
                    IS_FIRST_ITER=(kv_frame_start == 0),
                    OUTPUT_FULL_LSE=grad_enabled,
                    OUTPUT_PARTIAL_LSE=(not grad_enabled and kv_frame_start + KV_FRAME_CHUNK < f),
                    CAUSAL_BLOCK_SIZE=causal_block_size,
                )
        
        if grad_enabled:
            ctx.save_for_backward(q, k, v, z, z_lse)
            ctx.sm_scale_sqrt = sm_scale_sqrt
            ctx.causal_block_size = causal_block_size

        return z

    @staticmethod
    def backward(ctx, grad_z):
        q, k, v, z, z_lse = ctx.saved_tensors
        sm_scale_sqrt = ctx.sm_scale_sqrt
        causal_block_size = ctx.causal_block_size

        b, a, block_b1, block_b2, h, d = z.shape
        f = k.shape[1]

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")
        triton.set_allocator(alloc_fn)

        z_d = torch.empty_like(z_lse)
        z_bwd_preprocess(
            b, h, a,
            z, grad_z, z_d,
            block_b1,
            block_b2,
            d,
        )

        Q_FRAME_CHUNK = 1
        KV_FRAME_CHUNK = f

        aL = torch.empty((b, Q_FRAME_CHUNK, KV_FRAME_CHUNK, block_b2, block_b1, h, d), device=q.device, dtype=q.dtype) # (b, a, f, j, k, h, d)
        cL = torch.empty((b, h, Q_FRAME_CHUNK, KV_FRAME_CHUNK, block_b2, block_b1), device=q.device, dtype=torch.float32)  # (b, h, a, f, j, k)
        y = torch.empty((b, Q_FRAME_CHUNK, KV_FRAME_CHUNK, block_b2, block_b1, h, d), device=q.device, dtype=q.dtype) # (b, a, f, j, k, h, d)
        al_cl_y_lse = torch.empty((b, h, Q_FRAME_CHUNK, KV_FRAME_CHUNK, block_b1, block_b2), device=q.device, dtype=torch.float32)

        grad_aL = torch.empty_like(aL)
        grad_cL = torch.empty_like(cL)
        grad_q = torch.zeros_like(q, dtype=torch.float32) # float32 for atomic add
        grad_y = torch.empty_like(y)
        al_cl_y_d = torch.empty_like(al_cl_y_lse)

        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)

        for q_frame_start in range(0, a, Q_FRAME_CHUNK):
            curr_q_frames = min(Q_FRAME_CHUNK, a - q_frame_start)
            for kv_frame_start in range(0, f, KV_FRAME_CHUNK):
                curr_kv_frames = min(KV_FRAME_CHUNK, f - kv_frame_start)

                al_cl_y_fwd(
                    b, h, a, f,
                    q_frame_start, kv_frame_start,
                    curr_q_frames, curr_kv_frames,
                    q, k, v,
                    aL, cL, y,
                    al_cl_y_lse,
                    sm_scale_sqrt,
                    Q_FRAME_CHUNK, KV_FRAME_CHUNK,
                    block_b1, block_b2, d,
                    CAUSAL_BLOCK_SIZE=causal_block_size
                )

                z_bwd(
                    b, h, a, f,
                    q_frame_start, kv_frame_start,
                    curr_q_frames, curr_kv_frames,
                    aL, cL,
                    q, y,
                    grad_aL, grad_cL,
                    grad_q, grad_y,
                    grad_z, z_lse,
                    z_d, al_cl_y_d,
                    sm_scale_sqrt,
                    Q_FRAME_CHUNK, KV_FRAME_CHUNK,
                    block_b1, block_b2, d,
                    CAUSAL_BLOCK_SIZE=causal_block_size,
                )

                al_cl_y_bwd(
                    b, h, a, f,
                    q_frame_start, kv_frame_start,
                    curr_q_frames, curr_kv_frames,
                    0, f,
                    q, k, v,
                    grad_q, grad_k, grad_v,
                    grad_aL, grad_cL, grad_y,
                    al_cl_y_lse, al_cl_y_d,
                    sm_scale_sqrt,
                    Q_FRAME_CHUNK, KV_FRAME_CHUNK,
                    block_b1, block_b2, d,
                    IS_FIRST_ITER=(q_frame_start == 0),
                    PARTIAL_KV_GRAD=False,
                    CAUSAL_BLOCK_SIZE=causal_block_size,
                )

        return grad_q.to(q.dtype), grad_k, grad_v, None, None, None

def _get_rearrange_fns(x, f_tied, h_reduce, w_reduce, h, w):
    b, _, nh, d = x.shape
    def rearrange_fn(x):
        x = x.view(b, -1, f_tied, h_reduce, h // h_reduce, w_reduce, w // w_reduce, nh, d)
        return rearrange(x, 'b a f c i e j h d -> b (a c e) (f i) j h d')
    def return_fn(x):
        return rearrange(x, 'b (a c e) (f i) j h d -> b (a f c i e j) h d', c=h_reduce, e=w_reduce, f=f_tied)
    return rearrange_fn, return_fn

def monarch_attn_slow(Q, K, V, sm_scale, num_iters=1):
    b, a, i, j, h, _ = Q.shape
    block_b1, block_b2 = i, j
    k, l = block_b1, block_b2
    f = K.shape[-5]

    sm_scale_sqrt = sm_scale ** 0.5
    Q = Q * sm_scale_sqrt
    K = K * sm_scale_sqrt

    aR = Q.clone().unsqueeze(-5).expand(-1, -1, f, -1, -1, -1, -1) # (b, a, f, k, j, h, d)
    cR = torch.ones((b, h, a, f, k, j, 1), device=Q.device, dtype=Q.dtype) # (b, h, a, f, k, j, 1)

    for _ in range(num_iters - 1):
        bR = torch.einsum("bafkjhd,bfklhd->bhafkjl", aR, K)
        z = bR.to(torch.float32) * (1.0 / (cR + 1e-6)).clamp_max(1e4)
        z = z - z.amax(dim=-1, keepdim=True)
        R = torch.softmax(z, dim=-1).to(Q.dtype)
        aL = torch.einsum("bhafkjl,bfklhd->bafjkhd", R, K)
        logz = torch.logsumexp(z, dim=-1, keepdim=True) # (b, h, a, f, k, j, 1)
        cL = (R * (z - logz)).sum(dim=-1, keepdim=True).transpose(-2, -3) # (b, h, a, f, j, k, 1)

        bL = torch.einsum("bafjkhd,baijhd->bhafjki", aL, Q)
        L = rearrange(bL - cL, "b h a f j k i -> b h a j i (f k)")
        L = torch.softmax(L, dim=-1).to(Q.dtype)
        L = rearrange(L, "b h a j i (f k) -> b h a f j k i", f=f, k=k)

        aR = torch.einsum("bhafjki,baijhd->bafkjhd", L, Q) 
        cR = L.sum(dim=-1, dtype=torch.float32).unsqueeze(-1).transpose(-2, -3) # (b, h, a, f, k, j, 1)

    bR = torch.einsum("bafkjhd,bfklhd->bhafkjl", aR, K)
    z = bR.to(torch.float32) * (1.0 / (cR + 1e-6)).clamp_max(1e4)
    z = z - z.amax(dim=-1, keepdim=True)
    R = torch.softmax(z, dim=-1).to(Q.dtype)
    aL = torch.einsum("bhafkjl,bfklhd->bafjkhd", R, K)
    logz = torch.logsumexp(z, dim=-1, keepdim=True) # (b, h, a, f, k, j, 1)
    cL = (R * (z - logz)).sum(dim=-1, keepdim=True).transpose(-2, -3) # (b, h, a, f, j, k, 1)
    Y = torch.einsum("bhafkjl,bfklhd->bafjkhd", R, V)

    bL = torch.einsum("bafjkhd,baijhd->bhafjki", aL, Q)
    L = rearrange(bL - cL, "b h a f j k i -> b h a j i (f k)")
    L = torch.softmax(L, dim=-1).to(Q.dtype)
    L = rearrange(L, "b h a j i (f k) -> b h a f j k i", f=f, k=k)

    out = torch.einsum("bhafjki,bafjkhd->baijhd", L, Y)

    return out

def monarch_attn(q, k, v, f_tied, h_reduce, w_reduce, h, w, sm_scale=None, block_causal_size=None, num_iters=1):
    b, qs, nh, d = q.shape
    ks = k.shape[1]
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    rearrange_fn, return_fn = _get_rearrange_fns(q, f_tied, h_reduce, w_reduce, h, w)
    q = rearrange_fn(q).contiguous()
    k = rearrange_fn(k).contiguous()
    v = rearrange_fn(v).contiguous()

    causal_block_size = 0
    if block_causal_size is not None:
        assert qs % block_causal_size == 0, f"block_causal_size ({block_causal_size}) must divide sequence length ({qs})"
        assert ks == qs, "currently only support causal attention with kv length equal to q length"
        assert block_causal_size % (q.shape[2] * q.shape[3]) == 0, "block_causal_size must align with Monarch block sizes"
        causal_block_size = block_causal_size // (q.shape[2] * q.shape[3])
        assert num_iters == 1, "causal currently only supported for num_iters=1"
    
    if num_iters == 1:
        z = _attention_no_cache.apply(q, k, v, sm_scale, causal_block_size, torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad))
    else:
        z = monarch_attn_slow(q, k, v, sm_scale, num_iters=num_iters)
    z = return_fn(z)
    return z

def monarch_attn_with_kv_cache(q, k_cache, v_cache, new_k, new_v, start_idx, end_idx, f_tied, h_reduce, w_reduce, h, w, sm_scale=None, num_iters=1):
    b, _, nh, d = q.shape
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    rearrange_fn, return_fn = _get_rearrange_fns(q, f_tied, h_reduce, w_reduce, h, w)
    q = rearrange_fn(q).contiguous()
    new_k = rearrange_fn(new_k).reshape(b, -1, nh, d)
    new_v = rearrange_fn(new_v).reshape(b, -1, nh, d)
    if num_iters == 1:
        grad_enabled = torch.is_grad_enabled() and (q.requires_grad or k_cache.requires_grad or v_cache.requires_grad or new_k.requires_grad or new_v.requires_grad)
        grad_only_new_kv = grad_enabled and not (k_cache.requires_grad or v_cache.requires_grad) # only new KV gets gradients in Self-Forcing training
        z = _attention_with_cache.apply(q, k_cache, v_cache, new_k, new_v, start_idx, end_idx, sm_scale, grad_enabled, grad_only_new_kv)
    else:
        k_cache[:, start_idx:end_idx, :, :] = new_k
        v_cache[:, start_idx:end_idx, :, :] = new_v
        block_b1, block_b2 = q.shape[2], q.shape[3]
        k_cache = k_cache.view(b, -1, block_b1, block_b2, nh, d)
        v_cache = v_cache.view(b, -1, block_b1, block_b2, nh, d)
        z = monarch_attn_slow(q, k_cache, v_cache, sm_scale, num_iters=num_iters)
    z = return_fn(z)
    return z

__all__ = ["monarch_attn", "monarch_attn_with_kv_cache"]


def monarch_attn_ref_op(Q, K, V, sm_scale):
    b, a, i, j, h, _ = Q.shape
    block_b1, block_b2 = i, j
    f = K.size(-5)

    sm_scale_sqrt = sm_scale ** 0.5
    Q = Q * sm_scale_sqrt
    K = K * sm_scale_sqrt

    L = torch.eye(block_b1, device=Q.device, dtype=Q.dtype).view(1, 1, 1, 1, 1, block_b1, block_b1).expand(b, h, a, f, block_b2, block_b1, block_b1) # (b, h, a, f, j, k, i)

    aR = torch.einsum("bhafjki,baijhd->bafkjhd", L, Q)
    bR = torch.einsum("bafkjhd,bfklhd->bhafkjl", aR, K)

    cR = L.sum(dim=-1, dtype=torch.float32).unsqueeze(-1).transpose(-2, -3) # (b, h, a, f, k, j, 1)
    z = bR.to(torch.float32) * (1.0 / (cR))
    z = z - z.amax(dim=-1, keepdim=True)
    R = torch.softmax(z, dim=-1).to(Q.dtype)

    aL = torch.einsum("bhafkjl,bfklhd->bafjkhd", R, K)
    bL = torch.einsum("bafjkhd,baijhd->bhafjki", aL, Q)
    logz = torch.logsumexp(z, dim=-1, keepdim=True) # (b, h, a, f, k, j, 1)
    cL = (R * (z - logz)).sum(dim=-1, keepdim=True).transpose(-2, -3) # (b, h, a, f, j, k, 1)
    L = rearrange(bL - cL, "b h a f j k i -> b h a j i (f k)")
    L = torch.softmax(L, dim=-1).to(Q.dtype)
    L = rearrange(L, "b h a j i (f k) -> b h a f j k i", f=f, k=block_b1)

    out = torch.einsum("bhafkjl,bfklhd->bafjkhd", R, V)
    out = torch.einsum("bhafjki,bafjkhd->baijhd", L, out)
    return out

def monarch_attn_causal_ref(q, k, v, f_tied, h_reduce, w_reduce, h, w, block_causal_size, sm_scale=None):
    b, qs, nh, d = q.shape
    ks = k.shape[1]
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    rearrange_fn, return_fn = _get_rearrange_fns(q, f_tied, h_reduce, w_reduce, h, w)
    q = rearrange_fn(q).contiguous()
    k = rearrange_fn(k).contiguous()
    v = rearrange_fn(v).contiguous()

    causal_block_size = 0
    if block_causal_size is not None:
        assert qs % block_causal_size == 0, f"block_causal_size ({block_causal_size}) must divide sequence length ({qs})"
        assert ks == qs, "currently only support causal attention with kv length equal to q length"
        assert block_causal_size % (q.shape[2] * q.shape[3]) == 0, "block_causal_size must align with Monarch block sizes"
        causal_block_size = block_causal_size // (q.shape[2] * q.shape[3])

    outs = []
    for i in range(0, q.shape[1], causal_block_size):
        curr_q = q[:, i:i+causal_block_size, :, :, :, :]
        curr_k = k[:, :i+causal_block_size, :, :, :, :]
        curr_v = v[:, :i+causal_block_size, :, :, :, :]
        out = monarch_attn_ref_op(curr_q, curr_k, curr_v, sm_scale)
        outs.append(out)
    out = torch.cat(outs, dim=1)
    out = return_fn(out)
    return out

def monarch_attn_with_kv_cache_ref(q, k_cache, v_cache, new_k, new_v, start_idx, end_idx, f_tied, h_reduce, w_reduce, h, w, sm_scale=None, grad_only_new_kv=False):
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    rearrange_fn, return_fn = _get_rearrange_fns(q, f_tied, h_reduce, w_reduce, h, w)
    q = rearrange_fn(q)
    b, _, block_b1, block_b2, h, d = q.shape

    new_k = rearrange_fn(new_k).reshape(b, -1, h, d)
    new_v = rearrange_fn(new_v).reshape(b, -1, h, d)
    k_cache[:, start_idx:end_idx, :, :] = new_k
    v_cache[:, start_idx:end_idx, :, :] = new_v
    if not grad_only_new_kv:
        k_cache.retain_grad()
        v_cache.retain_grad()
    k_cache = k_cache.view(b, -1, block_b1, block_b2, h, d)
    v_cache = v_cache.view(b, -1, block_b1, block_b2, h, d)

    out = monarch_attn_ref_op(q, k_cache, v_cache, sm_scale)
    return return_fn(out)

def run_kv_cache_test(dtype, B, A, F, block_b1, block_b2, H, D, new_f, partial_kv_grad, device="cuda"):
    print(f"Testing dtype={dtype}, B={B}, A={A}, F={F}, block_b1={block_b1}, block_b2={block_b2}, H={H}, D={D}, new_f={new_f}, partial_kv_grad={partial_kv_grad}")
    q = torch.randn(B, A * block_b1 * block_b2, H, D, device=device, dtype=dtype, requires_grad=True)
    k_cache = torch.randn(B, F * block_b1 * block_b2, H, D, device=device, dtype=dtype, requires_grad=False)
    v_cache = torch.randn(B, F * block_b1 * block_b2, H, D, device=device, dtype=dtype, requires_grad=False)
    new_k = torch.randn(B, new_f * block_b1 * block_b2, H, D, device=device, dtype=dtype, requires_grad=True)
    new_v = torch.randn(B, new_f * block_b1 * block_b2, H, D, device=device, dtype=dtype, requires_grad=True)

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

    start_idx = k_cache.shape[1] - new_k.shape[1]
    end_idx = k_cache.shape[1]

    sm_scale = D ** -0.5

    out_triton = monarch_attn_with_kv_cache(q1, k_cache1, v_cache1, new_k1, new_v1, start_idx, end_idx, 1, 1, 1, block_b1, block_b2, sm_scale)
    out_ref = monarch_attn_with_kv_cache_ref(q2, k_cache2, v_cache2, new_k2, new_v2, start_idx, end_idx, 1, 1, 1, block_b1, block_b2, sm_scale)

    atol = 1e-2 if dtype is torch.bfloat16 else 1e-3
    rtol = 1e-2 if dtype is torch.bfloat16 else 1e-3

    fw_ok = torch.allclose(out_triton, out_ref, atol=atol, rtol=rtol)
    print("  forward allclose:", fw_ok)

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

def run_causal_test(dtype, B, F, block_b1, block_b2, H, D, device="cuda"):
    print(f"Testing dtype={dtype}, B={B}, F={F}, block_b1={block_b1}, block_b2={block_b2}, H={H}, D={D}")
    q = torch.randn(B, F * block_b1 * block_b2, H, D, device=device, dtype=dtype)
    k = torch.randn(B, F * block_b1 * block_b2, H, D, device=device, dtype=dtype)
    v = torch.randn(B, F * block_b1 * block_b2, H, D, device=device, dtype=dtype)

    q1 = q.clone().detach().requires_grad_(True)
    k1 = k.clone().detach().requires_grad_(True)
    v1 = v.clone().detach().requires_grad_(True)
    
    q2 = q.clone().detach().requires_grad_(True)
    k2 = k.clone().detach().requires_grad_(True)
    v2 = v.clone().detach().requires_grad_(True)

    sm_scale = D ** -0.5

    out_triton = monarch_attn_causal_ref(q1, k1, v1, 1, 1, 1, block_b1, block_b2, block_b1 * block_b2, sm_scale)
    out_ref = monarch_attn(q2, k2, v2, 1, 1, 1, block_b1, block_b2, sm_scale, block_b1 * block_b2)

    atol = 2e-2 if dtype is torch.bfloat16 else 2e-3
    rtol = 2e-2 if dtype is torch.bfloat16 else 2e-3

    fw_ok = torch.allclose(out_triton, out_ref, atol=atol, rtol=rtol)
    print("  forward allclose:", fw_ok)

    dout = torch.randn_like(out_triton)
    out_triton.backward(dout)
    out_ref.backward(dout)

    bwd_q_ok = torch.allclose(q1.grad, q2.grad, atol=atol, rtol=rtol)
    bwd_k_ok = torch.allclose(k1.grad, k2.grad, atol=atol, rtol=rtol)
    bwd_v_ok = torch.allclose(v1.grad, v2.grad, atol=atol, rtol=rtol)
    print("  backward q allclose:", bwd_q_ok)
    print("  backward k allclose:", bwd_k_ok)
    print("  backward v allclose:", bwd_v_ok)

    return fw_ok and bwd_q_ok and bwd_k_ok and bwd_v_ok

def run_tests():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests.")
        return
    device = "cuda"
    torch.manual_seed(0)

    print("Running KV cache tests...")

    shapes = [
        (1, 3, 21, 30, 52, 12, 128, 3),
        (2, 3, 21, 30, 52, 12, 128, 3),
    ]
    for dtype in (torch.float16, torch.bfloat16):
        for B, A, F, block_b1, block_b2, H, D, new_f in shapes:
            for partial_kv_grad in (False, True):
                ok = run_kv_cache_test(dtype, B, A, F, block_b1, block_b2, H, D, new_f, partial_kv_grad=partial_kv_grad, device=device)
                print("  -> test passed:", ok)

    print("Running causal tests...")

    shapes = [
        (1, 21, 30, 52, 12, 128),
        (2, 21, 30, 52, 12, 128),
    ]
    for dtype in (torch.float16, torch.bfloat16):
        for B, F, block_b1, block_b2, H, D in shapes:
            ok = run_causal_test(dtype, B, F, block_b1, block_b2, H, D, device=device)
            print("  -> test passed:", ok)


if __name__ == "__main__":
    run_tests()
