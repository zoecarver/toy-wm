"""
Modulation chunk + broadcast kernel.

Reads a column range from row 0 of the linear output (TILE, 1920),
broadcasts row 0 to all rows via reduce_sum + broadcast (works because
rows 1-31 are zero in the linear output), writes to (SEQ_PADDED, D_MODEL).

Bias is added separately after broadcast via add_kernel.
"""

import torch
import ttnn
import ttl

TILE = 32
D_MODEL = 320
D_TILES = D_MODEL // TILE  # 10
SEQ_PADDED = 96
SEQ_TILES = SEQ_PADDED // TILE  # 3
TOKS_PER_FRAME = 65


def make_mod_broadcast_kernel(col_offset_tiles):
    """Read D_TILES columns at col_offset_tiles from row 0 of src (TILE, 1920).
    Broadcast row 0 to all 32 rows within each tile, write to (SEQ_PADDED, D_MODEL)."""

    @ttl.kernel(grid="auto")
    def mod_broadcast(src, scaler, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        total_tiles = SEQ_TILES * D_TILES  # 3 * 10 = 30
        tiles_per_core = -(-total_tiles // grid_cols)

        src_dfb = ttl.make_dataflow_buffer_like(src, shape=(1, 1), buffer_factor=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        bcast_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            with sc_dfb.wait() as sc:
                for local_t in range(tiles_per_core):
                    t = core_x * tiles_per_core + local_t
                    if t < total_tiles:
                        with src_dfb.wait() as sv:
                            # reduce rows: sum over dim 0 (rows 1-31 are zero, so sum = row 0)
                            with red_dfb.reserve() as rd:
                                rd.store(ttl.math.reduce_sum(sv, sc, dims=[0]))
                        # broadcast row 0 back to all 32 rows
                        with red_dfb.wait() as rdv, bcast_dfb.reserve() as bc:
                            bc.store(ttl.math.broadcast(rdv, dims=[0]))
                        with bcast_dfb.wait() as bcv, out_dfb.reserve() as o:
                            o.store(bcv)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            with sc_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0, 0], blk); tx.wait()
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total_tiles:
                    col = t % D_TILES
                    with src_dfb.reserve() as blk:
                        tx = ttl.copy(src[0, col_offset_tiles + col], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total_tiles:
                    row = t // D_TILES
                    col = t % D_TILES
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[row, col]); tx.wait()

    return mod_broadcast


broadcast_mu1 = make_mod_broadcast_kernel(0)
broadcast_sigma1 = make_mod_broadcast_kernel(10)
broadcast_c1 = make_mod_broadcast_kernel(20)
broadcast_mu2 = make_mod_broadcast_kernel(30)
broadcast_sigma2 = make_mod_broadcast_kernel(40)
broadcast_c2 = make_mod_broadcast_kernel(50)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    def to_tt(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    scaler_tt = to_tt(torch.ones(TILE, TILE, dtype=torch.bfloat16))

    # Simulate LINEAR output: only row 0 has data, rest zeros (matches linear kernel behavior)
    mod_data = torch.zeros(TILE, 1920, dtype=torch.bfloat16)
    mod_data[0] = torch.randn(1920, dtype=torch.bfloat16)
    mod_tt = to_tt(mod_data)

    # Test mu1: cols 0:320
    out_tt = to_tt(torch.zeros(SEQ_PADDED, D_MODEL, dtype=torch.bfloat16))
    broadcast_mu1(mod_tt, scaler_tt, out_tt)
    result = ttnn.to_torch(out_tt)

    expected_row = mod_data[0, :D_MODEL]
    print(f"Source row[0, :8]: {expected_row[:8].tolist()}")
    print(f"Out row 0[:8]:     {result[0, :8].tolist()}")
    print(f"Out row 1[:8]:     {result[1, :8].tolist()}")
    print(f"Out row 31[:8]:    {result[31, :8].tolist()}")
    print(f"Out row 32[:8]:    {result[32, :8].tolist()}")
    print(f"Out row 64[:8]:    {result[64, :8].tolist()}")

    max_diff = 0.0
    for r in range(SEQ_PADDED):
        diff = (result[r].float() - expected_row.float()).abs().max().item()
        max_diff = max(max_diff, diff)
    print(f"Max diff across all {SEQ_PADDED} rows: {max_diff}")

    # Test c2: cols 1600:1920
    out2_tt = to_tt(torch.zeros(SEQ_PADDED, D_MODEL, dtype=torch.bfloat16))
    broadcast_c2(mod_tt, scaler_tt, out2_tt)
    result2 = ttnn.to_torch(out2_tt)
    expected_row2 = mod_data[0, 1600:1920]
    max_diff2 = 0.0
    for r in range(SEQ_PADDED):
        diff = (result2[r].float() - expected_row2.float()).abs().max().item()
        max_diff2 = max(max_diff2, diff)
    print(f"c2 max diff: {max_diff2}")

    ok = max_diff < 0.01 and max_diff2 < 0.01
    print("PASS" if ok else "FAIL")

    ttnn.close_device(device)
