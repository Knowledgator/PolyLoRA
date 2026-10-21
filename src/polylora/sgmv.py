"""
Segmented Grouped Matrix-Vector (SGMV) Triton kernels for LoRA forward pass.

Layout convention (packed only):
    input   : (T, K)  token rows; segment b holds rows [seg_starts[b], seg_starts[b] + seq_lens[b])
    weights : (num_adapters, K, N)
    output  : (T, N), only segment rows are written

Segments use adapter weight_indices[b]. Rows outside every segment are not read
or written, so a padded (B, L, K) batch is the packed case with
seg_starts = arange(B) * L over its (B * L, K) view (see `padded_segments`).

LoRA forward decomposes into two such matmuls per layer:
    A:  x @ A_w           where A_w is (K=in_dim, N=rank*stack_num)         "down-project"
    B:  (x @ A_w) @ B_w   where B_w is (K=rank,   N=out_dim), with scale+add  "up-project"

LoRA-A uses a direct `(tile, segment)` grid to avoid per-program prefix-search
scheduling overhead on its small rank/output tiles. LoRA-B keeps the compact
flat schedule because its wide output axis makes wasted programs more expensive.

Rank-variation axis:
    LoRA-A: rank lives on the OUTPUT axis (N).  Per-segment N is clamped to
            rank*stack_num, so smaller-rank segments produce fewer N-tiles.
    LoRA-B: rank lives on the REDUCTION axis (K). Per-segment K-loop is shortened;
            tile count along (M, N) is unaffected by rank.
"""

import torch
import triton
import triton.language as tl


def padded_segments(batch_size: int, seq_len: int, device) -> torch.Tensor:
    """seg_starts of a padded (B, L, ...) batch viewed as (B * L, ...)."""
    return torch.arange(batch_size, device=device, dtype=torch.int32) * seq_len


def _build_flat_schedule(
    seq_lens: torch.Tensor,        # (B,)
    n_eff_per_batch: torch.Tensor, # (B,) effective N per segment (after rank clamp)
    block_m: int,
    block_n: int,
):
    """
    Returns (cum_tiles, num_n_tiles, total_tiles).

    cum_tiles    : (B + 1,) int32, prefix sum of tiles_per_segment
    num_n_tiles  : (B,)     int32, cdiv(n_eff_per_batch, BLOCK_N)
    total_tiles  : int,            cum_tiles[-1] (host-side via .item())

    Segments with seq_len == 0 or n_eff == 0 contribute zero tiles.
    """
    device = seq_lens.device
    num_m_tiles = (seq_lens + block_m - 1) // block_m
    num_n_tiles = (n_eff_per_batch + block_n - 1) // block_n

    tiles_per_batch = (num_m_tiles * num_n_tiles).to(torch.int32)
    # Defensive: zero-out segments that should produce no work.
    no_work = (seq_lens == 0) | (n_eff_per_batch == 0)
    tiles_per_batch = torch.where(no_work, torch.zeros_like(tiles_per_batch), tiles_per_batch)

    B = seq_lens.shape[0]
    cum_tiles = torch.zeros(B + 1, dtype=torch.int32, device=device)
    torch.cumsum(tiles_per_batch, dim=0, out=cum_tiles[1:])
    total_tiles = int(cum_tiles[-1].item())

    return cum_tiles, num_n_tiles.to(torch.int32), total_tiles


@triton.jit
def _sgmv_lora_a_kernel(
    # Pointers
    input_ptr,        # (T, K)
    weight_ptr,       # (num_adapters, K, N_max)   N_max = max_rank * stack_num
    output_ptr,       # (T, N_max)
    # Dimensions
    K,                # input_dim (reduction)
    stack_num,
    # Strides
    input_stride_t, input_stride_k,
    weight_stride_b, weight_stride_k, weight_stride_n,
    output_stride_t, output_stride_n,
    # Per-segment metadata
    seq_lens_ptr,            # (B,)
    seg_starts_ptr,          # (B,)
    weight_indices_ptr,      # (B,)
    lora_ranks_ptr,          # (num_adapters,)
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_N_TILES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    seg_id = tl.program_id(axis=1)

    pid_m = pid // NUM_N_TILES
    pid_n = pid - pid_m * NUM_N_TILES

    seq_len = tl.load(seq_lens_ptr + seg_id)
    w_index = tl.load(weight_indices_ptr + seg_id)
    rank = tl.load(lora_ranks_ptr + w_index)
    N_eff = rank * stack_num
    if seq_len == 0 or pid_m * BLOCK_M >= seq_len or pid_n * BLOCK_N >= N_eff:
        return

    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offset = tl.arange(0, BLOCK_K)

    row0 = tl.load(seg_starts_ptr + seg_id).to(tl.int64)
    x_base = input_ptr + row0 * input_stride_t
    w_base = weight_ptr + w_index * weight_stride_b
    o_base = output_ptr + row0 * output_stride_t

    x_ptrs = x_base + (m_offset[:, None] * input_stride_t + k_offset[None, :] * input_stride_k)
    w_ptrs = w_base + (k_offset[:, None] * weight_stride_k + n_offset[None, :] * weight_stride_n)

    m_mask = m_offset[:, None] < seq_len
    n_mask = n_offset[None, :] < N_eff

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        x_tile = tl.load(x_ptrs, mask=m_mask & (k_offset[None, :] < k_remaining), other=0.0)
        w_tile = tl.load(w_ptrs, mask=(k_offset[:, None] < k_remaining) & n_mask, other=0.0)
        acc += tl.dot(x_tile, w_tile)
        x_ptrs += BLOCK_K * input_stride_k
        w_ptrs += BLOCK_K * weight_stride_k

    acc = acc.to(output_ptr.dtype.element_ty)
    o_ptrs = o_base + (m_offset[:, None] * output_stride_t + n_offset[None, :] * output_stride_n)
    tl.store(o_ptrs, acc, mask=m_mask & n_mask)


def _check_segments(x, seq_lens, seg_starts, weight_indices):
    assert x.dim() == 2, "input must be packed (T, K)"
    assert x.is_contiguous()
    assert seq_lens.dim() == 1 and seq_lens.shape == seg_starts.shape == weight_indices.shape


def sgmv_lora_a_fwd(
    x: torch.Tensor,               # (T, K)
    weights: torch.Tensor,         # (num_adapters, K, N_max)
    seq_lens: torch.Tensor,        # (B,)
    seg_starts: torch.Tensor,      # (B,) first row of each segment
    weight_indices: torch.Tensor,  # (B,)
    lora_ranks: torch.Tensor,      # (num_adapters,)
    stack_num: int = 1,
    max_seq_len: int | None = None,  # upper bound on seq_lens; only sizes the grid
) -> torch.Tensor:
    _check_segments(x, seq_lens, seg_starts, weight_indices)
    assert weights.is_contiguous() and weights.dim() == 3

    T, K = x.shape
    _, Kw, N_max = weights.shape
    assert Kw == K, f"K mismatch: x={K}, weights={Kw}"
    B = seq_lens.shape[0]
    M = T if max_seq_len is None else min(int(max_seq_len), T)

    output = torch.empty((T, N_max), device=x.device, dtype=x.dtype)

    BLOCK_M, BLOCK_N, BLOCK_K = 16, 16, 256
    if B == 0 or M == 0 or N_max == 0:
        return output

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N_max, BLOCK_N), B)
    _sgmv_lora_a_kernel[grid](
        x, weights, output,
        K, stack_num,
        x.stride(0), x.stride(1),
        weights.stride(0), weights.stride(1), weights.stride(2),
        output.stride(0), output.stride(1),
        seq_lens, seg_starts, weight_indices, lora_ranks,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_N_TILES=triton.cdiv(N_max, BLOCK_N),
    )
    return output


@triton.jit
def _sgmv_lora_b_kernel(
    # Pointers
    input_ptr,        # (T, K_max)
    weight_ptr,       # (num_adapters, K_max, N)
    output_ptr,       # (T, N) -- fused add target
    # Dimensions
    K_max,            # upper bound on rank
    N,                # output_dim
    # Strides
    input_stride_t, input_stride_k,
    weight_stride_b, weight_stride_k, weight_stride_n,
    output_stride_t, output_stride_n,
    # Per-segment metadata
    seq_lens_ptr,            # (B,)
    seg_starts_ptr,          # (B,)
    weight_indices_ptr,      # (B,)
    lora_ranks_ptr,          # (num_adapters,)
    scalings_ptr,            # (num_adapters,) fp32
    # Flat schedule
    cum_tiles_ptr,           # (B + 1,)
    num_n_tiles_ptr,         # (B,)  = cdiv(N, BLOCK_N), uniform but kept for symmetry
    B,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    lo = 0
    hi = B
    while lo < hi:
        mid = (lo + hi) // 2
        upper = tl.load(cum_tiles_ptr + mid + 1)
        if upper <= pid:
            lo = mid + 1
        else:
            hi = mid
    seg_id = lo

    base = tl.load(cum_tiles_ptr + seg_id)
    local_pid = pid - base

    num_pid_n = tl.load(num_n_tiles_ptr + seg_id)
    pid_m = local_pid // num_pid_n
    pid_n = local_pid % num_pid_n

    seq_len = tl.load(seq_lens_ptr + seg_id)
    w_index = tl.load(weight_indices_ptr + seg_id)
    rank = tl.load(lora_ranks_ptr + w_index)
    scaling = tl.load(scalings_ptr + w_index)

    # Per-adapter reduction length (rank is on the K axis here)
    K_eff = tl.minimum(K_max, rank)

    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offset = tl.arange(0, BLOCK_K)

    row0 = tl.load(seg_starts_ptr + seg_id).to(tl.int64)
    x_base = input_ptr + row0 * input_stride_t
    w_base = weight_ptr + w_index * weight_stride_b
    o_base = output_ptr + row0 * output_stride_t

    x_ptrs = x_base + (m_offset[:, None] * input_stride_t + k_offset[None, :] * input_stride_k)
    w_ptrs = w_base + (k_offset[:, None] * weight_stride_k + n_offset[None, :] * weight_stride_n)

    m_mask = m_offset[:, None] < seq_len
    n_mask = n_offset[None, :] < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K_eff, BLOCK_K)):
        k_remaining = K_eff - k * BLOCK_K
        x_tile = tl.load(x_ptrs, mask=m_mask & (k_offset[None, :] < k_remaining), other=0.0)
        w_tile = tl.load(w_ptrs, mask=(k_offset[:, None] < k_remaining) & n_mask, other=0.0)
        acc += tl.dot(x_tile, w_tile)
        x_ptrs += BLOCK_K * input_stride_k
        w_ptrs += BLOCK_K * weight_stride_k

    acc *= scaling
    o_ptrs = o_base + (m_offset[:, None] * output_stride_t + n_offset[None, :] * output_stride_n)
    o_mask = m_mask & n_mask
    prev = tl.load(o_ptrs, mask=o_mask, other=0.0)
    acc = acc.to(output_ptr.dtype.element_ty) + prev
    tl.store(o_ptrs, acc, mask=o_mask)


def sgmv_lora_b_fwd(
    x: torch.Tensor,               # (T, K_max)
    weights: torch.Tensor,         # (num_adapters, K_max, N)
    seq_lens: torch.Tensor,        # (B,)
    seg_starts: torch.Tensor,      # (B,) first row of each segment
    weight_indices: torch.Tensor,  # (B,)
    lora_ranks: torch.Tensor,      # (num_adapters,)
    scalings: torch.Tensor,        # (num_adapters,) fp32
    base_output: torch.Tensor | None = None,  # (T, N), optional fused add target, updated in place
) -> torch.Tensor:
    _check_segments(x, seq_lens, seg_starts, weight_indices)
    assert weights.is_contiguous() and weights.dim() == 3

    T, K_max = x.shape
    _, Kw, N = weights.shape
    assert Kw == K_max, f"K mismatch: x={K_max}, weights={Kw}"
    B = seq_lens.shape[0]

    if base_output is None:
        output = torch.zeros((T, N), device=x.device, dtype=x.dtype)
    else:
        assert base_output.shape == (T, N) and base_output.is_contiguous()
        output = base_output

    BLOCK_M, BLOCK_N, BLOCK_K = 16, 256, 16

    # For LoRA-B the (M, N) tile grid does not depend on rank — only on whether
    # the segment has any work at all. Use rank to gate empty segments.
    ranks_per_batch = lora_ranks[weight_indices]  # (B,)
    n_eff = torch.where(ranks_per_batch > 0, torch.full_like(ranks_per_batch, N), torch.zeros_like(ranks_per_batch))

    cum_tiles, num_n_tiles, total_tiles = _build_flat_schedule(seq_lens, n_eff, BLOCK_M, BLOCK_N)
    if total_tiles == 0:
        return output

    _sgmv_lora_b_kernel[(total_tiles,)](
        x, weights, output,
        K_max, N,
        x.stride(0), x.stride(1),
        weights.stride(0), weights.stride(1), weights.stride(2),
        output.stride(0), output.stride(1),
        seq_lens, seg_starts, weight_indices, lora_ranks, scalings,
        cum_tiles, num_n_tiles,
        B,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return output
