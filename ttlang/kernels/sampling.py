"""
Sampling kernels for toy-wm diffusion inference.

euler_step_kernel: z_next = z + dt * v_pred
cfg_blend_kernel:  out = v_uncond + cfg * (v_cond - v_uncond)

dt and cfg are scalar constants, pre-filled into full tiles.
"""

import ttl

TILE = 32


@ttl.kernel(grid="auto")
def euler_step_kernel(z, v_pred, dt_tile, out):
    """out = z + dt * v_pred. dt_tile is a full tile of the dt scalar."""
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = z.shape[0] // TILE
    col_tiles = z.shape[1] // TILE
    total_tiles = row_tiles * col_tiles
    tiles_per_core = -(-total_tiles // grid_cols)

    z_dfb = ttl.make_dataflow_buffer_like(z, shape=(1, 1), buffer_factor=2)
    v_dfb = ttl.make_dataflow_buffer_like(v_pred, shape=(1, 1), buffer_factor=2)
    dt_dfb = ttl.make_dataflow_buffer_like(dt_tile, shape=(1, 1), buffer_factor=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        with dt_dfb.wait() as dt:
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total_tiles:
                    with z_dfb.wait() as zv, v_dfb.wait() as vv, out_dfb.reserve() as o:
                        o.store(zv + dt * vv)

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        with dt_dfb.reserve() as blk:
            tx = ttl.copy(dt_tile[0, 0], blk); tx.wait()
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with z_dfb.reserve() as blk:
                    tx = ttl.copy(z[row, col], blk); tx.wait()
                with v_dfb.reserve() as blk:
                    tx = ttl.copy(v_pred[row, col], blk); tx.wait()

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
def cfg_blend_kernel(v_cond, v_uncond, cfg_tile, out):
    """out = v_uncond + cfg * (v_cond - v_uncond). cfg_tile is full tile of cfg scalar."""
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = v_cond.shape[0] // TILE
    col_tiles = v_cond.shape[1] // TILE
    total_tiles = row_tiles * col_tiles
    tiles_per_core = -(-total_tiles // grid_cols)

    vc_dfb = ttl.make_dataflow_buffer_like(v_cond, shape=(1, 1), buffer_factor=2)
    vu_dfb = ttl.make_dataflow_buffer_like(v_uncond, shape=(1, 1), buffer_factor=2)
    cfg_dfb = ttl.make_dataflow_buffer_like(cfg_tile, shape=(1, 1), buffer_factor=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        with cfg_dfb.wait() as cfg:
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total_tiles:
                    with vc_dfb.wait() as vc, vu_dfb.wait() as vu, out_dfb.reserve() as o:
                        o.store(vu + cfg * (vc - vu))

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        with cfg_dfb.reserve() as blk:
            tx = ttl.copy(cfg_tile[0, 0], blk); tx.wait()
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with vc_dfb.reserve() as blk:
                    tx = ttl.copy(v_cond[row, col], blk); tx.wait()
                with vu_dfb.reserve() as blk:
                    tx = ttl.copy(v_uncond[row, col], blk); tx.wait()

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

    z = torch.randn(ROWS, COLS, dtype=torch.bfloat16)
    v = torch.randn(ROWS, COLS, dtype=torch.bfloat16) * 0.1
    dt_val = 0.1
    cfg_val = 2.0

    dt_tile = torch.full((TILE, TILE), dt_val, dtype=torch.bfloat16)
    cfg_tile = torch.full((TILE, TILE), cfg_val, dtype=torch.bfloat16)

    # Test Euler step
    print("--- Euler step test ---")
    ref_euler = (z.float() + dt_val * v.float()).to(torch.bfloat16)
    out_euler = to_tt(torch.zeros(ROWS, COLS, dtype=torch.bfloat16))
    euler_step_kernel(to_tt(z), to_tt(v), to_tt(dt_tile), out_euler)
    r_euler = ttnn.to_torch(out_euler)
    d_euler = (r_euler.float() - ref_euler.float()).abs().max().item()
    print(f"  Max diff: {d_euler:.6f}  PASS: {d_euler < 0.1}")

    # Test CFG blend
    print("--- CFG blend test ---")
    v_cond = torch.randn(ROWS, COLS, dtype=torch.bfloat16) * 0.1
    v_uncond = torch.randn(ROWS, COLS, dtype=torch.bfloat16) * 0.1
    ref_cfg = (v_uncond.float() + cfg_val * (v_cond.float() - v_uncond.float())).to(torch.bfloat16)
    out_cfg = to_tt(torch.zeros(ROWS, COLS, dtype=torch.bfloat16))
    cfg_blend_kernel(to_tt(v_cond), to_tt(v_uncond), to_tt(cfg_tile), out_cfg)
    r_cfg = ttnn.to_torch(out_cfg)
    d_cfg = (r_cfg.float() - ref_cfg.float()).abs().max().item()
    print(f"  Max diff: {d_cfg:.6f}  PASS: {d_cfg < 0.1}")

    ttnn.close_device(device)
