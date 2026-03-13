"""
RoPE (Rotary Position Embedding) for toy-wm.

For this model: d_head=16, n_heads=20, so QK tensors are (seq, 320).
RoPE is: out[i] = cos[pos,i]*x[i] + sin[pos,i]*x_perm[i]
where x_perm swaps even/odd pairs with negation within each d_head block:
  x_perm[..., 2k]   = -x[..., 2k+1]
  x_perm[..., 2k+1] =  x[..., 2k]

Strategy: precompute cos_table and sin_table as (max_seq, 320) tensors
on device at init time. The permutation is baked into a perm_sign table:
  perm_sign[..., 2k]   = -1
  perm_sign[..., 2k+1] = +1

Then RoPE is: out = cos * x + sin * perm_sign * x_shuffled
where x_shuffled swaps adjacent pairs (no sign).

But element-wise shuffle within a tile is tricky in TT-Lang.

ALTERNATIVE APPROACH: Precompute two tables on host:
  cos_table: standard cos values at each position
  sin_perm_table: sin values with the perm_sign already baked in,
                  AND with indices already swapped.

Then: out = cos_table * x + sin_perm_table * x_adjacent_swap

Since we can't easily do adjacent-swap in TT-Lang elementwise ops,
we use a different decomposition:

For each pair (x[2k], x[2k+1]):
  out[2k]   = cos[2k]*x[2k]   - sin[2k]*x[2k+1]
  out[2k+1] = cos[2k+1]*x[2k+1] + sin[2k+1]*x[2k]

Note cos[2k] == cos[2k+1] and sin[2k] == sin[2k+1] for standard RoPE.

We precompute on host:
  cos_table[pos, i] = cos(pos * theta_i)  -- standard, same for even/odd pairs
  neg_sin_table[pos, 2k] = -sin(pos * theta_k)   -- for x[2k+1] term
  neg_sin_table[pos, 2k+1] = +sin(pos * theta_k)  -- for x[2k] term

Then if we had x_swapped where adjacent pairs are swapped:
  out = cos_table * x + neg_sin_table * x_swapped

Since we can't easily swap in TT-Lang, let's try a different approach:
just apply RoPE via ttnn ops (ttnn multiply, ttnn add) since it's purely
elementwise and the tables handle the permutation logic.

SIMPLEST APPROACH: Apply RoPE on host during weight prep, precomputing
for each possible position. But we want everything on device.

PRAGMATIC APPROACH: Use ttnn.multiply and ttnn.add for RoPE since it's
purely elementwise. The cos/sin tables are precomputed on host and sent
to device. We just need two ttnn multiplies and one ttnn add.

For now, let's test with a mul+add pattern using our existing kernels.
The "swap" is handled by creating two weight tensors on host where the
sin term already accounts for the swap and sign.
"""

import torch
import math


TILE = 32


def precompute_rope_tables(d_head, n_heads, max_seq, C=5000):
    """
    Precompute cos and sin_perm tables for RoPE.

    Returns cos_table, sin_perm_table both of shape (max_seq, n_heads * d_head).

    cos_table[pos, i] = cos(pos * theta_{i % d_head})
    sin_perm_table is set up so that:
      rope(x) = cos_table * x + sin_perm_table * x_swapped
    where x_swapped[2k] = x[2k+1], x_swapped[2k+1] = x[2k]

    We bake the sign into sin_perm_table:
      sin_perm_table[pos, 2k]   = -sin(pos * theta_k)
      sin_perm_table[pos, 2k+1] = +sin(pos * theta_k)
    """
    d_model = n_heads * d_head

    # Theta frequencies for one head
    thetas = torch.exp(-math.log(C) * torch.arange(0, d_head, 2).float() / d_head)
    # Expand to pairs: [theta0, theta0, theta1, theta1, ...]
    thetas_paired = thetas.repeat_interleave(2)
    # Tile across all heads
    thetas_full = thetas_paired.repeat(n_heads)  # (d_model,)

    positions = torch.arange(max_seq).float()  # (max_seq,)
    angles = positions.unsqueeze(1) * thetas_full.unsqueeze(0)  # (max_seq, d_model)

    cos_table = torch.cos(angles)

    sin_vals = torch.sin(angles)
    # Bake sign into sin: even indices get -sin, odd get +sin
    sign = torch.ones(d_model)
    sign[0::2] = -1.0
    sin_perm_table = sin_vals * sign.unsqueeze(0)

    return cos_table.to(torch.bfloat16), sin_perm_table.to(torch.bfloat16)


def precompute_swap_indices(d_head, n_heads):
    """
    Create a permutation that swaps adjacent pairs within each head.
    x_swapped[2k] = x[2k+1], x_swapped[2k+1] = x[2k]

    Returns permutation indices for a single row of (n_heads * d_head,).
    """
    d_model = n_heads * d_head
    perm = torch.arange(d_model)
    for h in range(n_heads):
        base = h * d_head
        for k in range(0, d_head, 2):
            perm[base + k] = base + k + 1
            perm[base + k + 1] = base + k
    return perm


def apply_rope_host(x, cos_table, sin_perm_table, d_head, n_heads, offset=0):
    """
    Apply RoPE on host tensors for reference/testing.
    x: (seq, d_model) where d_model = n_heads * d_head
    """
    seq = x.shape[0]
    d_model = n_heads * d_head
    perm = precompute_swap_indices(d_head, n_heads)
    x_swapped = x[:, perm]
    cos_slice = cos_table[offset:offset+seq, :d_model]
    sin_slice = sin_perm_table[offset:offset+seq, :d_model]
    return cos_slice * x + sin_slice * x_swapped


if __name__ == "__main__":
    import ttnn
    import ttl

    # Inline the kernels we need (can't import since file runs in isolation)
    @ttl.kernel(grid="auto")
    def mul_kernel(a, b, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        row_tiles = a.shape[0] // TILE
        col_tiles = a.shape[1] // TILE
        total_tiles = row_tiles * col_tiles
        tiles_per_core = -(-total_tiles // grid_cols)
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total_tiles:
                    with a_dfb.wait() as av, b_dfb.wait() as bv, out_dfb.reserve() as o:
                        o.store(av * bv)
        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total_tiles:
                    row = t // col_tiles
                    col = t % col_tiles
                    with a_dfb.reserve() as blk:
                        tx = ttl.copy(a[row, col], blk); tx.wait()
                    with b_dfb.reserve() as blk:
                        tx = ttl.copy(b[row, col], blk); tx.wait()
        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total_tiles:
                    row = t // col_tiles
                    col = t % col_tiles
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[row, col]); tx.wait()

    @ttl.kernel(grid="auto")
    def add_kernel(a, b, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        row_tiles = a.shape[0] // TILE
        col_tiles = a.shape[1] // TILE
        total_tiles = row_tiles * col_tiles
        tiles_per_core = -(-total_tiles // grid_cols)
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total_tiles:
                    with a_dfb.wait() as av, b_dfb.wait() as bv, out_dfb.reserve() as o:
                        o.store(av + bv)
        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total_tiles:
                    row = t // col_tiles
                    col = t % col_tiles
                    with a_dfb.reserve() as blk:
                        tx = ttl.copy(a[row, col], blk); tx.wait()
                    with b_dfb.reserve() as blk:
                        tx = ttl.copy(b[row, col], blk); tx.wait()
        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total_tiles:
                    row = t // col_tiles
                    col = t % col_tiles
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[row, col]); tx.wait()

    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    D_HEAD = 16
    N_HEADS = 20
    D_MODEL = D_HEAD * N_HEADS  # 320
    MAX_SEQ = 2048
    SEQ = 64

    def to_tt(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def zeros_tt(shape):
        return to_tt(torch.zeros(shape, dtype=torch.bfloat16))

    # Precompute tables
    cos_table, sin_perm_table = precompute_rope_tables(D_HEAD, N_HEADS, MAX_SEQ, C=5000)

    # Create swap permutation
    perm = precompute_swap_indices(D_HEAD, N_HEADS)

    # Test input
    x_torch = torch.randn(SEQ, D_MODEL, dtype=torch.bfloat16) * 0.1

    # Reference
    ref = apply_rope_host(x_torch.float(), cos_table.float(), sin_perm_table.float(),
                          D_HEAD, N_HEADS, offset=0).to(torch.bfloat16)

    # On device: rope(x) = cos * x + sin_perm * x_swapped
    # x_swapped is created on host (it's a permutation, hard to do in TT-Lang)
    # but we only need to send it once per position
    x_swapped = x_torch[:, perm]

    cos_slice = cos_table[:SEQ, :D_MODEL]
    sin_slice = sin_perm_table[:SEQ, :D_MODEL]

    x_tt = to_tt(x_torch)
    x_swap_tt = to_tt(x_swapped)
    cos_tt = to_tt(cos_slice)
    sin_tt = to_tt(sin_slice)

    # cos * x
    cos_x_tt = zeros_tt((SEQ, D_MODEL))
    mul_kernel(cos_tt, x_tt, cos_x_tt)

    # sin_perm * x_swapped
    sin_x_tt = zeros_tt((SEQ, D_MODEL))
    mul_kernel(sin_tt, x_swap_tt, sin_x_tt)

    # result = cos*x + sin*x_swapped
    out_tt = zeros_tt((SEQ, D_MODEL))
    add_kernel(cos_x_tt, sin_x_tt, out_tt)

    result = ttnn.to_torch(out_tt)
    max_diff = (result.float() - ref.float()).abs().max().item()
    mean_diff = (result.float() - ref.float()).abs().mean().item()
    print(f"RoPE test (d_head={D_HEAD}, n_heads={N_HEADS}, seq={SEQ}):")
    print(f"  Max diff:  {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  ref[0,:8]:    {ref[0,:8].tolist()}")
    print(f"  result[0,:8]: {result[0,:8].tolist()}")
    print(f"  PASS: {max_diff < 0.1}")

    ttnn.close_device(device)
