"""
Linear projection kernel for toy-wm.

out = x @ w

x:    (M_tiles, K_tiles)
w:    (K_tiles, N_tiles)
out:  (M_tiles, N_tiles)

Bias is applied separately via add_kernel to keep DFB count low.

Distributes output tiles (M*N) across cores. Each output tile at (m, n)
is computed by loading x[m, 0:k_chunk] and w[0:k_chunk, n] and doing matmul.
"""

import ttl

TILE = 32


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


# d_model=320 -> K=10 tiles
linear_k10 = make_linear_kernel(10)

# d_mid=1280 -> K=40 tiles
linear_k40 = make_linear_kernel(40)


if __name__ == "__main__":
    import torch
    import ttnn

    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    def to_tt(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Test 1: (32, 320) @ (320, 1280) - single row tile
    print("--- Linear (32,320) @ (320,1280) test ---")
    x1 = torch.randn(32, 320, dtype=torch.bfloat16) * 0.1
    w1 = torch.randn(320, 1280, dtype=torch.bfloat16) * 0.01
    ref1 = (x1.float() @ w1.float()).to(torch.bfloat16)

    out1 = to_tt(torch.zeros(32, 1280, dtype=torch.bfloat16))
    linear_k10(to_tt(x1), to_tt(w1), out1)
    r1 = ttnn.to_torch(out1)
    d1 = (r1.float() - ref1.float()).abs().max().item()
    print(f"  Max diff: {d1:.6f}  PASS: {d1 < 1.0}")

    # Test 2: (64, 320) @ (320, 320) - multi-row
    print("\n--- Linear (64,320) @ (320,320) multi-row test ---")
    x2 = torch.randn(64, 320, dtype=torch.bfloat16) * 0.1
    w2 = torch.randn(320, 320, dtype=torch.bfloat16) * 0.01
    ref2 = (x2.float() @ w2.float()).to(torch.bfloat16)

    out2 = to_tt(torch.zeros(64, 320, dtype=torch.bfloat16))
    linear_k10(to_tt(x2), to_tt(w2), out2)
    r2 = ttnn.to_torch(out2)
    d2 = (r2.float() - ref2.float()).abs().max().item()
    print(f"  Max diff: {d2:.6f}  PASS: {d2 < 1.0}")

    # Test 3: (32, 1280) @ (1280, 320) - large K
    print("\n--- Linear (32,1280) @ (1280,320) large-K test ---")
    x3 = torch.randn(32, 1280, dtype=torch.bfloat16) * 0.1
    w3 = torch.randn(1280, 320, dtype=torch.bfloat16) * 0.01
    ref3 = (x3.float() @ w3.float()).to(torch.bfloat16)

    out3 = to_tt(torch.zeros(32, 320, dtype=torch.bfloat16))
    linear_k40(to_tt(x3), to_tt(w3), out3)
    r3 = ttnn.to_torch(out3)
    d3 = (r3.float() - ref3.float()).abs().max().item()
    print(f"  Max diff: {d3:.6f}  PASS: {d3 < 1.0}")

    ttnn.close_device(device)
