"""
Elementwise kernels for toy-wm.

All kernels stream over tiles with grid="auto". They operate on 2D
tensors of shape (row_tiles * TILE, col_tiles * TILE).

  silu_kernel:  out = x * sigmoid(x)
  mul_kernel:   out = a * b
  add_kernel:   out = a + b
"""

import ttl

TILE = 32


@ttl.kernel(grid="auto")
def silu_kernel(x, out):
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = x.shape[0] // TILE
    col_tiles = x.shape[1] // TILE
    total_tiles = row_tiles * col_tiles
    tiles_per_core = -(-total_tiles // grid_cols)

    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                with x_dfb.wait() as xv, out_dfb.reserve() as o:
                    o.store(xv * ttl.math.sigmoid(xv))

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


if __name__ == "__main__":
    import torch
    import ttnn

    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    ROWS, COLS = 64, 320

    a_torch = torch.randn(ROWS, COLS, dtype=torch.bfloat16)
    b_torch = torch.randn(ROWS, COLS, dtype=torch.bfloat16)

    def to_tt(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def zeros_tt(shape):
        return to_tt(torch.zeros(shape, dtype=torch.bfloat16))

    a_tt = to_tt(a_torch)
    b_tt = to_tt(b_torch)

    # Test Add
    print("--- Add test ---")
    out_add = zeros_tt((ROWS, COLS))
    add_kernel(a_tt, b_tt, out_add)
    ref_add = (a_torch.float() + b_torch.float()).to(torch.bfloat16)
    result_add = ttnn.to_torch(out_add)
    max_diff = (result_add.float() - ref_add.float()).abs().max().item()
    print(f"  max_diff={max_diff:.6f}  PASS={max_diff < 0.01}")

    # Test Mul
    print("--- Mul test ---")
    out_mul = zeros_tt((ROWS, COLS))
    mul_kernel(a_tt, b_tt, out_mul)
    ref_mul = (a_torch.float() * b_torch.float()).to(torch.bfloat16)
    result_mul = ttnn.to_torch(out_mul)
    max_diff = (result_mul.float() - ref_mul.float()).abs().max().item()
    print(f"  max_diff={max_diff:.6f}  PASS={max_diff < 0.01}")

    # Test SiLU
    print("--- SiLU test ---")
    out_silu = zeros_tt((ROWS, COLS))
    silu_kernel(a_tt, out_silu)
    ref_silu = (a_torch.float() * torch.sigmoid(a_torch.float())).to(torch.bfloat16)
    result_silu = ttnn.to_torch(out_silu)
    max_diff = (result_silu.float() - ref_silu.float()).abs().max().item()
    print(f"  max_diff={max_diff:.6f}  PASS={max_diff < 0.1}")

    ttnn.close_device(device)
