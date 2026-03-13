"""
Test GEGLU MLP by composing linear, silu, mul kernels.

GEGLU forward:
  gate_out = silu(x @ W_gate + b_gate)
  up_out   = x @ W_up + b_up
  mid      = up_out * gate_out
  out      = mid @ W_down + b_down

For this model: d_model=320, d_mid=1280
  W_up:   (320, 1280), b_up:   (1, 1280)
  W_gate: (320, 1280), b_gate: (1, 1280)
  W_down: (1280, 320), b_down: (1, 320)

Tests the composition of multiple kernel calls with tensors staying on device.
"""

import torch
import ttnn
import sys
import os

# We need to import the kernels - they're defined as top-level scripts
# so we import via the module's __main__ test pattern. Instead, we just
# inline the kernel definitions here for the integration test.
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


linear_k10 = make_linear_kernel(10)
linear_k40 = make_linear_kernel(40)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    D_MODEL = 320
    D_MID = 1280
    SEQ = 64  # 2 tile rows

    def to_tt(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def zeros_tt(shape):
        return to_tt(torch.zeros(shape, dtype=torch.bfloat16))

    # Random weights (small scale for numerical stability)
    x_torch = torch.randn(SEQ, D_MODEL, dtype=torch.bfloat16) * 0.1
    w_up = torch.randn(D_MODEL, D_MID, dtype=torch.bfloat16) * 0.01
    b_up = torch.randn(SEQ, D_MID, dtype=torch.bfloat16) * 0.01
    w_gate = torch.randn(D_MODEL, D_MID, dtype=torch.bfloat16) * 0.01
    b_gate = torch.randn(SEQ, D_MID, dtype=torch.bfloat16) * 0.01
    w_down = torch.randn(D_MID, D_MODEL, dtype=torch.bfloat16) * 0.01
    b_down = torch.randn(SEQ, D_MODEL, dtype=torch.bfloat16) * 0.01

    # PyTorch reference
    up_out = x_torch.float() @ w_up.float() + b_up.float()
    gate_out = (x_torch.float() @ w_gate.float() + b_gate.float())
    gate_out = gate_out * torch.sigmoid(gate_out)  # SiLU
    mid = up_out * gate_out
    ref = (mid @ w_down.float() + b_down.float()).to(torch.bfloat16)

    # TT-Lang: compose kernels, all tensors on device
    x_tt = to_tt(x_torch)
    w_up_tt = to_tt(w_up)
    b_up_tt = to_tt(b_up)
    w_gate_tt = to_tt(w_gate)
    b_gate_tt = to_tt(b_gate)
    w_down_tt = to_tt(w_down)
    b_down_tt = to_tt(b_down)

    # Step 1: up_proj = x @ W_up
    up_tt = zeros_tt((SEQ, D_MID))
    linear_k10(x_tt, w_up_tt, up_tt)
    # Step 1b: up_proj += b_up
    up_biased_tt = zeros_tt((SEQ, D_MID))
    add_kernel(up_tt, b_up_tt, up_biased_tt)

    # Step 2: gate_proj = x @ W_gate
    gate_tt = zeros_tt((SEQ, D_MID))
    linear_k10(x_tt, w_gate_tt, gate_tt)
    # Step 2b: gate_proj += b_gate
    gate_biased_tt = zeros_tt((SEQ, D_MID))
    add_kernel(gate_tt, b_gate_tt, gate_biased_tt)

    # Step 3: gate_activated = silu(gate_proj)
    gate_act_tt = zeros_tt((SEQ, D_MID))
    silu_kernel(gate_biased_tt, gate_act_tt)

    # Step 4: mid = up_proj * gate_activated
    mid_tt = zeros_tt((SEQ, D_MID))
    mul_kernel(up_biased_tt, gate_act_tt, mid_tt)

    # Step 5: out = mid @ W_down
    out_tt = zeros_tt((SEQ, D_MODEL))
    linear_k40(mid_tt, w_down_tt, out_tt)
    # Step 5b: out += b_down
    out_biased_tt = zeros_tt((SEQ, D_MODEL))
    add_kernel(out_tt, b_down_tt, out_biased_tt)

    result = ttnn.to_torch(out_biased_tt)
    max_diff = (result.float() - ref.float()).abs().max().item()
    mean_diff = (result.float() - ref.float()).abs().mean().item()
    print(f"GEGLU integration test (SEQ={SEQ}, d_model={D_MODEL}, d_mid={D_MID}):")
    print(f"  Max diff:  {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  ref[0,:5]:    {ref[0,:5].tolist()}")
    print(f"  result[0,:5]: {result[0,:5].tolist()}")
    print(f"  PASS: {max_diff < 2.0}")

    ttnn.close_device(device)
