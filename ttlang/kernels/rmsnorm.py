"""
RMSNorm kernel for toy-wm.

out = x / rms(x)   (weight multiply done separately via mul_kernel)

where rms(x) = sqrt(mean(x^2) + eps)

Follows nanochat rmsnorm pattern exactly. Weight is applied as a separate
mul_kernel call after this kernel.
"""

import ttl

TILE = 32


def make_rmsnorm_kernel(dim_tiles):

    @ttl.kernel(grid="auto")
    def rmsnorm_kernel(x, scaler, mean_scale, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        seq_tiles = x.shape[0] // TILE
        tiles_per_core = -(-seq_tiles // grid_cols)

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        ms_dfb = ttl.make_dataflow_buffer_like(mean_scale, shape=(1, 1), buffer_factor=1)
        sq_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        bcast_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        rsq_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            with sc_dfb.wait() as sc, ms_dfb.wait() as ms:
                for local_t in range(tiles_per_core):
                    tile_idx = core_x * tiles_per_core + local_t
                    if tile_idx < seq_tiles:
                        # Pass 1: sum of squares across dim tiles
                        with x_dfb.wait() as x0:
                            with sq_dfb.reserve() as sq:
                                sq.store(x0 * x0)
                        with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                            r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                        with red_dfb.wait() as rv, acc_dfb.reserve() as acc:
                            acc.store(rv)
                        for j in range(dim_tiles - 1):
                            with x_dfb.wait() as xj:
                                with sq_dfb.reserve() as sq:
                                    sq.store(xj * xj)
                            with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                                r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                            with red_dfb.wait() as rv, acc_dfb.wait() as av, acc_dfb.reserve() as new_acc:
                                new_acc.store(av + rv)

                        # broadcast, scale by 1/N, add eps, rsqrt
                        with acc_dfb.wait() as total, bcast_dfb.reserve() as bc:
                            bc.store(ttl.math.broadcast(total, dims=[1]))
                        with bcast_dfb.wait() as bv, red_dfb.reserve() as scaled:
                            scaled.store(bv * ms + ttl.math.fill(bv, 1e-5))
                        with red_dfb.wait() as msq, rsq_dfb.reserve() as rsq:
                            rsq.store(ttl.math.rsqrt(msq))

                        # Pass 2: x * rsqrt(mean(x^2) + eps)
                        with rsq_dfb.wait() as rsqv:
                            for j in range(dim_tiles):
                                with x_dfb.wait() as xj, out_dfb.reserve() as o:
                                    o.store(xj * rsqv)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            with sc_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0, 0], blk); tx.wait()
            with ms_dfb.reserve() as blk:
                tx = ttl.copy(mean_scale[0, 0], blk); tx.wait()
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    for j in range(dim_tiles):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[tile_idx, j], blk); tx.wait()
                    for j in range(dim_tiles):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[tile_idx, j], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    for j in range(dim_tiles):
                        with out_dfb.wait() as blk:
                            tx = ttl.copy(blk, out[tile_idx, j]); tx.wait()

    return rmsnorm_kernel


rmsnorm_d320 = make_rmsnorm_kernel(10)   # d_model = 320
rmsnorm_d1280 = make_rmsnorm_kernel(40)  # d_mid = 1280


if __name__ == "__main__":
    import torch
    import ttnn

    device = ttnn.open_device(device_id=0)

    DIM = 320
    SEQ = 64
    eps = 1e-5

    torch.manual_seed(42)
    x_torch = torch.randn(SEQ, DIM, dtype=torch.bfloat16)

    # PyTorch reference: normalized only (no weight)
    rms = (x_torch.float() ** 2).mean(dim=-1, keepdim=True)
    ref = (x_torch.float() / (rms + eps).sqrt()).to(torch.bfloat16)

    def to_tt(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    x_tt = to_tt(x_torch)
    out_tt = to_tt(torch.zeros(SEQ, DIM, dtype=torch.bfloat16))
    scaler_tt = to_tt(torch.ones(TILE, TILE, dtype=torch.bfloat16))
    mean_scale_tt = to_tt(torch.full((TILE, TILE), 1.0 / DIM, dtype=torch.bfloat16))

    rmsnorm_d320(x_tt, scaler_tt, mean_scale_tt, out_tt)

    result = ttnn.to_torch(out_tt)
    max_diff = (result.float() - ref.float()).abs().max().item()
    mean_diff = (result.float() - ref.float()).abs().mean().item()
    print(f"RMSNorm d=320 test (no weight):")
    print(f"  Max diff:  {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  ref[0,:5]:    {ref[0,:5].tolist()}")
    print(f"  result[0,:5]: {result[0,:5].tolist()}")
    print(f"  PASS: {max_diff < 0.5}")

    ttnn.close_device(device)
