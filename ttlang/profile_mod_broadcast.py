"""Profile mod_broadcast kernel in isolation."""
import torch
import ttnn
import ttl

TILE = 32
D_MODEL = 320
SEQ_PADDED = 96
MOD_D_TILES = D_MODEL // TILE  # 10
MOD_GRAN = 5
MOD_COL_BLOCKS = MOD_D_TILES // MOD_GRAN  # 2

def make_mod_broadcast_kernel(col_offset_tiles):
    @ttl.kernel(grid="auto")
    def mod_broadcast(src, bias, scaler, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        seq_tiles = SEQ_PADDED // TILE
        total_blocks = seq_tiles * MOD_COL_BLOCKS
        tiles_per_core = -(-total_blocks // grid_cols)
        src_dfb = ttl.make_dataflow_buffer_like(src, shape=(1, MOD_GRAN), buffer_factor=2)
        bias_dfb = ttl.make_dataflow_buffer_like(bias, shape=(1, MOD_GRAN), buffer_factor=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, MOD_GRAN), buffer_factor=2)
        bcast_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, MOD_GRAN), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, MOD_GRAN), buffer_factor=2)
        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            with sc_dfb.wait() as sc:
                for local_t in range(tiles_per_core):
                    t = core_x * tiles_per_core + local_t
                    if t < total_blocks:
                        with src_dfb.wait() as sv:
                            with red_dfb.reserve() as rd:
                                rd.store(ttl.math.reduce_sum(sv, sc, dims=[0]))
                        with red_dfb.wait() as rdv, bcast_dfb.reserve() as bc:
                            bc.store(ttl.math.broadcast(rdv, dims=[0]))
                        with bcast_dfb.wait() as bcv, bias_dfb.wait() as bv, out_dfb.reserve() as o:
                            o.store(bcv + bv)
        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            with sc_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0, 0], blk); tx.wait()
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total_blocks:
                    cb = t % MOD_COL_BLOCKS
                    row = t // MOD_COL_BLOCKS
                    sc_start = col_offset_tiles + cb * MOD_GRAN
                    with src_dfb.reserve() as blk:
                        tx = ttl.copy(src[0, sc_start:sc_start + MOD_GRAN], blk); tx.wait()
                    with bias_dfb.reserve() as blk:
                        tx = ttl.copy(bias[row, cb * MOD_GRAN:(cb + 1) * MOD_GRAN], blk); tx.wait()
        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total_blocks:
                    cb = t % MOD_COL_BLOCKS
                    row = t // MOD_COL_BLOCKS
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[row, cb * MOD_GRAN:(cb + 1) * MOD_GRAN]); tx.wait()
    return mod_broadcast

broadcast_mu1 = make_mod_broadcast_kernel(0)

if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    def to_tt(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    scaler_tt = to_tt(torch.ones(TILE, TILE, dtype=torch.bfloat16))

    # Source: (TILE, 1920) with data only in row 0
    mod_data = torch.zeros(TILE, 1920, dtype=torch.bfloat16)
    mod_data[0] = torch.randn(1920, dtype=torch.bfloat16)
    mod_tt = to_tt(mod_data)

    # Bias: (SEQ_PADDED, D_MODEL)
    bias_data = torch.randn(SEQ_PADDED, D_MODEL, dtype=torch.bfloat16) * 0.1
    bias_tt = to_tt(bias_data)

    # Output
    out_tt = to_tt(torch.zeros(SEQ_PADDED, D_MODEL, dtype=torch.bfloat16))

    # Run once
    broadcast_mu1(mod_tt, bias_tt, scaler_tt, out_tt)

    # Verify
    result = ttnn.to_torch(out_tt)
    expected_row = mod_data[0, :D_MODEL]
    max_diff = 0.0
    for r in range(SEQ_PADDED):
        diff = (result[r].float() - (expected_row.float() + bias_data[r].float())).abs().max().item()
        max_diff = max(max_diff, diff)
    print(f"Max diff: {max_diff}")
    print("PASS" if max_diff < 0.05 else "FAIL")

    ttnn.close_device(device)
