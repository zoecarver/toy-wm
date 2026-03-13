"""
AdaLN modulation kernel for toy-wm.

out = x * (1 + scale) + shift

In the model, shift and scale come from the conditioning vector and are
per-frame (broadcast across tokens within a frame). For simplicity in v1,
we pre-expand shift/scale to match x shape on the host side before sending
to device. This kernel just does the elementwise math.

Also includes gated residual: out = residual + x * gate
Same broadcast consideration -- gate is pre-expanded.
"""

import ttl

TILE = 32


@ttl.kernel(grid="auto")
def adaln_modulate_kernel(x, shift, scale, out):
    """out = x * (1 + scale) + shift. All tensors same shape."""
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = x.shape[0] // TILE
    col_tiles = x.shape[1] // TILE
    total_tiles = row_tiles * col_tiles
    tiles_per_core = -(-total_tiles // grid_cols)

    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    sh_dfb = ttl.make_dataflow_buffer_like(shift, shape=(1, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scale, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                with x_dfb.wait() as xv, sh_dfb.wait() as shv, sc_dfb.wait() as scv, out_dfb.reserve() as o:
                    o.store(xv * (scv + ttl.math.fill(scv, 1.0)) + shv)

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with x_dfb.reserve() as blk:
                    tx = ttl.copy(x[row, col], blk); tx.wait()
                with sh_dfb.reserve() as blk:
                    tx = ttl.copy(shift[row, col], blk); tx.wait()
                with sc_dfb.reserve() as blk:
                    tx = ttl.copy(scale[row, col], blk); tx.wait()

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
def gated_residual_kernel(residual, x, gate, out):
    """out = residual + x * gate. All tensors same shape."""
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = residual.shape[0] // TILE
    col_tiles = residual.shape[1] // TILE
    total_tiles = row_tiles * col_tiles
    tiles_per_core = -(-total_tiles // grid_cols)

    res_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, 1), buffer_factor=2)
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    g_dfb = ttl.make_dataflow_buffer_like(gate, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                with res_dfb.wait() as rv, x_dfb.wait() as xv, g_dfb.wait() as gv, out_dfb.reserve() as o:
                    o.store(rv + xv * gv)

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with res_dfb.reserve() as blk:
                    tx = ttl.copy(residual[row, col], blk); tx.wait()
                with x_dfb.reserve() as blk:
                    tx = ttl.copy(x[row, col], blk); tx.wait()
                with g_dfb.reserve() as blk:
                    tx = ttl.copy(gate[row, col], blk); tx.wait()

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


if __name__ == "__main__":
    import torch
    import ttnn

    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    ROWS, COLS = 64, 320

    def to_tt(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    x = torch.randn(ROWS, COLS, dtype=torch.bfloat16)
    shift = torch.randn(ROWS, COLS, dtype=torch.bfloat16) * 0.1
    scale = torch.randn(ROWS, COLS, dtype=torch.bfloat16) * 0.1
    gate = torch.randn(ROWS, COLS, dtype=torch.bfloat16) * 0.5
    residual = torch.randn(ROWS, COLS, dtype=torch.bfloat16)

    # Test AdaLN modulate
    print("--- AdaLN modulate test ---")
    ref_mod = (x.float() * (scale.float() + 1.0) + shift.float()).to(torch.bfloat16)
    out_mod = to_tt(torch.zeros(ROWS, COLS, dtype=torch.bfloat16))
    adaln_modulate_kernel(to_tt(x), to_tt(shift), to_tt(scale), out_mod)
    r_mod = ttnn.to_torch(out_mod)
    d_mod = (r_mod.float() - ref_mod.float()).abs().max().item()
    print(f"  Max diff: {d_mod:.6f}  PASS: {d_mod < 0.1}")

    # Test gated residual
    print("--- Gated residual test ---")
    ref_gr = (residual.float() + x.float() * gate.float()).to(torch.bfloat16)
    out_gr = to_tt(torch.zeros(ROWS, COLS, dtype=torch.bfloat16))
    gated_residual_kernel(to_tt(residual), to_tt(x), to_tt(gate), out_gr)
    r_gr = ttnn.to_torch(out_gr)
    d_gr = (r_gr.float() - ref_gr.float()).abs().max().item()
    print(f"  Max diff: {d_gr:.6f}  PASS: {d_gr < 0.1}")

    ttnn.close_device(device)
