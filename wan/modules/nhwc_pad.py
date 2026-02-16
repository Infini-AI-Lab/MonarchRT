import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 2048}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 4096}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK": 8192}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 8192}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK": 16384}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 16384}, num_warps=16, num_stages=1),
    ],
    key=["H", "W", "C", "C_out", "has_cache"],
    cache_results=True,
)
@triton.jit
def _cat_pad_ndhwc_kernel(
    x_ptr,              # [B, Dx, H, W, C]
    cache_ptr,          # [B, Dc, H, W, C] or dummy
    y_ptr,              # [B, D_out, H_out, W_out, C_out]
    # shapes
    B: tl.constexpr,
    Dx: tl.constexpr,
    Dc: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    D_out: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    C_out: tl.constexpr,
    # padding (PyTorch 5D order)
    pad_c_l: tl.constexpr, pad_c_r: tl.constexpr,
    pad_w_l: tl.constexpr, pad_w_r: tl.constexpr,
    pad_h_l: tl.constexpr, pad_h_r: tl.constexpr,
    pad_d_l: tl.constexpr, pad_d_r: tl.constexpr,
    has_cache: tl.constexpr,
    # strides (elements) NDHWC
    xB: tl.constexpr, xD: tl.constexpr, xH: tl.constexpr, xW: tl.constexpr, xC: tl.constexpr,
    cB, cD, cH, cW, cC,
    yB: tl.constexpr, yD: tl.constexpr, yH: tl.constexpr, yW: tl.constexpr, yC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    total = B * D_out * H_out * W_out * C_out
    inb = offs < total

    # unravel linear index -> (b, d, h, w, c) in output
    c_out = offs % C_out
    t = offs // C_out
    w_out = t % W_out
    t = t // W_out
    h_out = t % H_out
    t = t // H_out
    d_out = t % D_out
    b = t // D_out

    # inside padded-valid region?
    inside_c = (c_out >= pad_c_l) & (c_out < pad_c_l + C)
    inside_w = (w_out >= pad_w_l) & (w_out < pad_w_l + W)
    inside_h = (h_out >= pad_h_l) & (h_out < pad_h_l + H)
    inside_d = (d_out >= pad_d_l) & (d_out < pad_d_l + Dc + Dx)
    inside = inb & inside_c & inside_w & inside_h & inside_d

    # map to input coords
    c_in = c_out - pad_c_l
    w_in = w_out - pad_w_l
    h_in = h_out - pad_h_l
    d_cat = d_out - pad_d_l              # in concatenated tensor [Dc+Dx]
    from_cache = has_cache & (d_cat < Dc)
    d_in_x = d_cat - Dc                  # valid only when not from_cache

    b = b.to(tl.int64)
    c_in = c_in.to(tl.int64)
    w_in = w_in.to(tl.int64)
    h_in = h_in.to(tl.int64)
    d_cat = d_cat.to(tl.int64)
    d_in_x = d_in_x.to(tl.int64)

    # load from cache where from_cache & inside
    cache_mask = inside & from_cache
    off_c = b * cB + d_cat * cD + h_in * cH + w_in * cW + c_in * cC
    ptr_c = cache_ptr + off_c
    v_cache = tl.load(ptr_c, mask=cache_mask, other=0.0).to(tl.float32)

    # load from x where (~from_cache) & inside
    x_mask = inside & (~from_cache)
    ptr_x = x_ptr + b * xB + d_in_x * xD + h_in * xH + w_in * xW + c_in * xC
    v_x = tl.load(ptr_x, mask=x_mask, other=0.0).to(tl.float32)

    out = v_cache + v_x  # mutually exclusive masks

    # store
    ptr_y = y_ptr + b * yB + d_out * yD + h_out * yH + w_out * yW + c_out * yC
    tl.store(ptr_y, out.to(y_ptr.dtype.element_ty), mask=inb)

@torch.library.custom_op("monarch_rt::cat_pad_ndhwc", mutates_args=("y",))
def cat_pad_ndhwc_kernel(
    x_: torch.Tensor, cache_: torch.Tensor, y: torch.Tensor,
    B: int, Dx: int, Dc: int, H: int, W: int, C: int,
    D_out: int, H_out: int, W_out: int, C_out: int,
    pad_c_l: int, pad_c_r: int,
    pad_w_l: int, pad_w_r: int,
    pad_h_l: int, pad_h_r: int,
    pad_d_l_eff: int, pad_d_r: int,
    has_cache: bool,
    xB: int, xD: int, xH: int, xW: int, xC: int,
    cB: int, cD: int, cH: int, cW: int, cC: int,
    yB: int, yD: int, yH: int, yW: int, yC: int,
) -> None:
    grid = lambda meta: (triton.cdiv(B * D_out * H_out * W_out * C_out, meta["BLOCK"]),)
    _cat_pad_ndhwc_kernel[grid](
        x_, cache_, y,
        B=B, Dx=Dx, Dc=Dc, H=H, W=W, C=C,
        D_out=D_out, H_out=H_out, W_out=W_out, C_out=C_out,
        pad_c_l=pad_c_l, pad_c_r=pad_c_r,
        pad_w_l=pad_w_l, pad_w_r=pad_w_r,
        pad_h_l=pad_h_l, pad_h_r=pad_h_r,
        pad_d_l=pad_d_l_eff, pad_d_r=pad_d_r,
        has_cache=has_cache,
        xB=xB, xD=xD, xH=xH, xW=xW, xC=xC,
        cB=cB, cD=cD, cH=cH, cW=cW, cC=cC,
        yB=yB, yD=yD, yH=yH, yW=yW, yC=yC,
    )

@cat_pad_ndhwc_kernel.register_fake
def _(
    x_: torch.Tensor, cache_: torch.Tensor, y: torch.Tensor,
    B: int, Dx: int, Dc: int, H: int, W: int, C: int,
    D_out: int, H_out: int, W_out: int, C_out: int,
    pad_c_l: int, pad_c_r: int,
    pad_w_l: int, pad_w_r: int,
    pad_h_l: int, pad_h_r: int,
    pad_d_l_eff: int, pad_d_r: int,
    has_cache: bool,
    xB: int, xD: int, xH: int, xW: int, xC: int,
    cB: int, cD: int, cH: int, cW: int, cC: int,
    yB: int, yD: int, yH: int, yW: int, yC: int,
) -> None:
    pass

def cat_pad_ndhwc_fused(x: torch.Tensor, cache_x: torch.Tensor | None, padding):
    """
    x: [B,D,H,W,C]
    cache_x: [B,Dc,H,W,C] or None (can be non-contiguous)
    padding: 8 ints (pad_C_l, pad_C_r, pad_W_l, pad_W_r, pad_H_l, pad_H_r, pad_D_l, pad_D_r), all >=0
    """
    assert x.is_cuda and x.dim() == 5
    assert len(padding) == 8
    pad_c_l, pad_c_r, pad_w_l, pad_w_r, pad_h_l, pad_h_r, pad_d_l, pad_d_r = map(int, padding)
    assert min(padding) >= 0

    x_ = x.contiguous()
    B, Dx, H, W, C = x_.shape

    has_cache = (cache_x is not None) and (pad_d_l > 0)
    if has_cache:
        cache_ = cache_x if cache_x.device == x_.device else cache_x.to(device=x_.device)
        assert cache_.is_cuda
        assert (
            cache_.shape[0] == B and cache_.shape[2] == H and cache_.shape[3] == W and cache_.shape[4] == C
        ), f"cache shape {cache_.shape}, stride {cache_.stride()}"
        Dc = cache_.shape[1]
        pad_d_l_eff = pad_d_l - Dc
        if pad_d_l_eff < 0:
            pad_d_l_eff = 0
    else:
        cache_ = x_.new_empty((1, 1, 1, 1, 1))
        Dc = 0
        pad_d_l_eff = pad_d_l

    D_out = pad_d_l_eff + (Dc + Dx) + pad_d_r
    H_out = pad_h_l + H + pad_h_r
    W_out = pad_w_l + W + pad_w_r
    C_out = pad_c_l + C + pad_c_r

    y = torch.empty((B, D_out, H_out, W_out, C_out), device=x_.device, dtype=x_.dtype)

    xB, xD, xH, xW, xC = x_.stride()
    cB, cD, cH, cW, cC = cache_.stride()
    yB, yD, yH, yW, yC = y.stride()

    cat_pad_ndhwc_kernel(
        x_, cache_, y,
        B=B, Dx=Dx, Dc=Dc, H=H, W=W, C=C,
        D_out=D_out, H_out=H_out, W_out=W_out, C_out=C_out,
        pad_c_l=pad_c_l, pad_c_r=pad_c_r,
        pad_w_l=pad_w_l, pad_w_r=pad_w_r,
        pad_h_l=pad_h_l, pad_h_r=pad_h_r,
        pad_d_l_eff=pad_d_l_eff, pad_d_r=pad_d_r,
        has_cache=has_cache,
        xB=xB, xD=xD, xH=xH, xW=xW, xC=xC,
        cB=cB, cD=cD, cH=cH, cW=cW, cC=cC,
        yB=yB, yD=yD, yH=yH, yW=yW, yC=yC,
    )
    return y
