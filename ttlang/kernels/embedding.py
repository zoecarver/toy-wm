"""
Embedding kernels for toy-wm.

NumericEncoding: sinusoidal positional encoding for timesteps.
  Precompute the full table on host, send to device once.
  At runtime, index into it. Since the index is a scalar known on host
  (the diffusion timestep), we can just slice the table on host and send
  the right row. No need for a gather kernel.

ActionEmbedding: learned lookup table (4 actions -> d_model=320).
  Same approach: table on device, index known on host, send the right row.

For the actual model flow, conditioning = time_emb_mixer(time_emb(ts)) + action_emb(actions)
where time_emb is sinusoidal and time_emb_mixer is a linear layer.

Since the conditioning is small (n_frames x d_model), and the index is
known on host, we do the lookup on host and send the result to device.
The linear mixer runs on device.

This file provides the precomputation utilities and a test.
"""

import torch
import math

TILE = 32


def precompute_sinusoidal_table(dim, n_max, C=10000):
    """
    Precompute sinusoidal encoding table of shape (n_max, dim).
    Matches NumericEncoding from the model.
    """
    args = torch.exp(-math.log(C) * torch.arange(0, dim, 2).float() / dim)
    args = torch.arange(n_max).float().unsqueeze(1) * args.unsqueeze(0)
    pe = torch.empty(n_max, dim)
    pe[:, 0::2] = torch.sin(args)
    pe[:, 1::2] = torch.cos(args)
    return pe.to(torch.bfloat16)


def lookup_embedding(table, indices):
    """
    Simple host-side lookup. table: (N, D), indices: (B,) or (B, T).
    Returns: gathered rows from table.
    """
    return table[indices]


if __name__ == "__main__":
    import ttnn
    import ttl

    # Inline linear kernel for testing the mixer
    def make_linear_kernel(k_chunk):
        @ttl.kernel(grid="auto")
        def linear_kernel(x, w, out):
            grid_cols, _ = ttl.grid_size(dims=2)
            m_tiles = x.shape[0] // TILE
            n_tiles = w.shape[1] // TILE
            total_out = m_tiles * n_tiles
            tiles_per_core = -(-total_out // grid_cols)
            x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, k_chunk), buffer_factor=2)
            w_dfb = ttl.make_dataflow_buffer_like(w, shape=(k_chunk, 1), buffer_factor=2)
            out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
            @ttl.compute()
            def compute():
                core_x, _ = ttl.core(dims=2)
                for local_t in range(tiles_per_core):
                    idx = core_x * tiles_per_core + local_t
                    if idx < total_out:
                        with x_dfb.wait() as xv, w_dfb.wait() as wv, out_dfb.reserve() as o:
                            o.store(xv @ wv)
            @ttl.datamovement()
            def dm_read():
                core_x, _ = ttl.core(dims=2)
                for local_t in range(tiles_per_core):
                    idx = core_x * tiles_per_core + local_t
                    if idx < total_out:
                        row = idx // n_tiles
                        col = idx % n_tiles
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[row, 0:k_chunk], blk); tx.wait()
                        with w_dfb.reserve() as blk:
                            tx = ttl.copy(w[0:k_chunk, col], blk); tx.wait()
            @ttl.datamovement()
            def dm_write():
                core_x, _ = ttl.core(dims=2)
                for local_t in range(tiles_per_core):
                    idx = core_x * tiles_per_core + local_t
                    if idx < total_out:
                        row = idx // n_tiles
                        col = idx % n_tiles
                        with out_dfb.wait() as blk:
                            tx = ttl.copy(blk, out[row, col]); tx.wait()
        return linear_kernel

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

    linear_k10 = make_linear_kernel(10)

    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    D_MODEL = 320
    T_MAX = 1000

    def to_tt(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    def zeros_tt(shape):
        return to_tt(torch.zeros(shape, dtype=torch.bfloat16))

    # Test 1: Sinusoidal encoding table matches PyTorch NumericEncoding
    print("--- Sinusoidal encoding test ---")
    table = precompute_sinusoidal_table(D_MODEL, T_MAX, C=10000)

    # Compare against PyTorch NumericEncoding
    # NumericEncoding uses C=1e4 by default
    args_ref = torch.exp(-math.log(10000) * torch.arange(0, D_MODEL, 2).float() / D_MODEL)
    args_ref = torch.arange(T_MAX).float().unsqueeze(1) * args_ref.unsqueeze(0)
    pe_ref = torch.empty(T_MAX, D_MODEL)
    pe_ref[:, 0::2] = torch.sin(args_ref)
    pe_ref[:, 1::2] = torch.cos(args_ref)

    max_diff = (table.float() - pe_ref).abs().max().item()
    print(f"  Table vs ref max diff: {max_diff:.6f}  PASS: {max_diff < 0.01}")

    # Test 2: Lookup + mixer linear on device
    # Simulate: cond = time_emb_mixer(time_emb(ts)) + action_emb(actions)
    print("\n--- Conditioning pipeline test ---")
    # Random mixer weights
    mixer_w = torch.randn(D_MODEL, D_MODEL, dtype=torch.bfloat16) * 0.01
    mixer_b = torch.randn(D_MODEL, dtype=torch.bfloat16) * 0.01
    action_table = torch.randn(4, D_MODEL, dtype=torch.bfloat16) * 0.1

    # Simulate for a batch of timesteps and actions
    ts = torch.tensor([500], dtype=torch.long)  # single frame
    actions = torch.tensor([2], dtype=torch.long)

    # Host: lookup time embedding
    time_emb = table[ts]  # (1, 320)
    # Pad to tile-aligned rows (need at least 32 rows)
    time_emb_padded = torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16)
    time_emb_padded[:time_emb.shape[0]] = time_emb

    # Host: lookup action embedding
    action_emb = action_table[actions]  # (1, 320)
    action_emb_padded = torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16)
    action_emb_padded[:action_emb.shape[0]] = action_emb

    # Device: mixer linear
    mixer_b_padded = torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16)
    mixer_b_padded[0] = mixer_b

    mixer_out = zeros_tt((TILE, D_MODEL))
    linear_k10(to_tt(time_emb_padded), to_tt(mixer_w), mixer_out)
    mixer_biased = zeros_tt((TILE, D_MODEL))
    add_kernel(mixer_out, to_tt(mixer_b_padded), mixer_biased)

    # Device: add action embedding
    cond_tt = zeros_tt((TILE, D_MODEL))
    add_kernel(mixer_biased, to_tt(action_emb_padded), cond_tt)

    # PyTorch reference
    ref_mixer = (time_emb_padded.float() @ mixer_w.float() + mixer_b_padded.float())
    ref_cond = (ref_mixer + action_emb_padded.float()).to(torch.bfloat16)

    result = ttnn.to_torch(cond_tt)
    max_diff = (result.float() - ref_cond.float()).abs().max().item()
    print(f"  Conditioning max diff: {max_diff:.6f}  PASS: {max_diff < 0.5}")
    print(f"  ref[0,:5]:    {ref_cond[0,:5].tolist()}")
    print(f"  result[0,:5]: {result[0,:5].tolist()}")

    ttnn.close_device(device)
