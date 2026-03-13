"""
Optimized diffusion sampling on TT hardware.

v2: Pre-cache all weights on device at startup. Eliminate per-forward
host-device weight transfers. Vectorize expand_per_frame.
"""

import torch
import torch.nn.functional as F
import ttnn
import ttl
import math
import time

TILE = 32
D_MODEL = 320
D_MID = 1280
N_HEADS = 20
D_HEAD = 16
D_HEAD_PAD = 32  # D_HEAD padded to tile size
N_BLOCKS = 8
PATCH_SIZE = 3
HEIGHT = 24
WIDTH = 24
TOKS_PER_FRAME = (HEIGHT // PATCH_SIZE) * (WIDTH // PATCH_SIZE) + 1  # 65
T_MAX = 1000
SEQ = TOKS_PER_FRAME  # 65
SEQ_PADDED = ((SEQ + TILE - 1) // TILE) * TILE  # 96
N_WINDOW = 30
MAX_KV_SEQ = (N_WINDOW - 1) * TOKS_PER_FRAME + SEQ  # 1950
MAX_KV_PADDED = ((MAX_KV_SEQ + TILE - 1) // TILE) * TILE  # 1952
HEAD_BATCH = N_HEADS * SEQ_PADDED  # 1920 = 60 tiles

# ============================================================
# TT-Lang Kernels (same as before)
# ============================================================

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

ELEM_GRAN = 5  # column tiles per block for elementwise kernels

@ttl.kernel(grid="auto")
def add_kernel(a, b, out):
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = a.shape[0] // TILE
    col_blocks = a.shape[1] // TILE // ELEM_GRAN
    total = row_tiles * col_blocks
    tiles_per_core = -(-total // grid_cols)
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, ELEM_GRAN), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, ELEM_GRAN), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, ELEM_GRAN), buffer_factor=2)
    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                with a_dfb.wait() as av, b_dfb.wait() as bv, out_dfb.reserve() as o:
                    o.store(av + bv)
    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                row = t // col_blocks
                cb = t % col_blocks
                sc = cb * ELEM_GRAN
                with a_dfb.reserve() as blk:
                    tx = ttl.copy(a[row, sc:sc + ELEM_GRAN], blk); tx.wait()
                with b_dfb.reserve() as blk:
                    tx = ttl.copy(b[row, sc:sc + ELEM_GRAN], blk); tx.wait()
    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                row = t // col_blocks
                cb = t % col_blocks
                sc = cb * ELEM_GRAN
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[row, sc:sc + ELEM_GRAN]); tx.wait()

@ttl.kernel(grid="auto")
def mul_kernel(a, b, out):
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = a.shape[0] // TILE
    col_blocks = a.shape[1] // TILE // ELEM_GRAN
    total = row_tiles * col_blocks
    tiles_per_core = -(-total // grid_cols)
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, ELEM_GRAN), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, ELEM_GRAN), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, ELEM_GRAN), buffer_factor=2)
    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                with a_dfb.wait() as av, b_dfb.wait() as bv, out_dfb.reserve() as o:
                    o.store(av * bv)
    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                row = t // col_blocks
                cb = t % col_blocks
                sc = cb * ELEM_GRAN
                with a_dfb.reserve() as blk:
                    tx = ttl.copy(a[row, sc:sc + ELEM_GRAN], blk); tx.wait()
                with b_dfb.reserve() as blk:
                    tx = ttl.copy(b[row, sc:sc + ELEM_GRAN], blk); tx.wait()
    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                row = t // col_blocks
                cb = t % col_blocks
                sc = cb * ELEM_GRAN
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[row, sc:sc + ELEM_GRAN]); tx.wait()

@ttl.kernel(grid="auto")
def silu_kernel(x, out):
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = x.shape[0] // TILE
    col_blocks = x.shape[1] // TILE // ELEM_GRAN
    total = row_tiles * col_blocks
    tiles_per_core = -(-total // grid_cols)
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, ELEM_GRAN), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, ELEM_GRAN), buffer_factor=2)
    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                with x_dfb.wait() as xv, out_dfb.reserve() as o:
                    o.store(xv * ttl.math.sigmoid(xv))
    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                row = t // col_blocks
                cb = t % col_blocks
                sc = cb * ELEM_GRAN
                with x_dfb.reserve() as blk:
                    tx = ttl.copy(x[row, sc:sc + ELEM_GRAN], blk); tx.wait()
    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                row = t // col_blocks
                cb = t % col_blocks
                sc = cb * ELEM_GRAN
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[row, sc:sc + ELEM_GRAN]); tx.wait()

@ttl.kernel(grid="auto")
def adaln_modulate_kernel(x, shift, scale, out):
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = x.shape[0] // TILE
    col_blocks = x.shape[1] // TILE // ELEM_GRAN
    total = row_tiles * col_blocks
    tiles_per_core = -(-total // grid_cols)
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, ELEM_GRAN), buffer_factor=2)
    sh_dfb = ttl.make_dataflow_buffer_like(shift, shape=(1, ELEM_GRAN), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scale, shape=(1, ELEM_GRAN), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, ELEM_GRAN), buffer_factor=2)
    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                with x_dfb.wait() as xv, sh_dfb.wait() as shv, sc_dfb.wait() as scv, out_dfb.reserve() as o:
                    o.store(xv * (scv + ttl.math.fill(scv, 1.0)) + shv)
    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                row = t // col_blocks
                cb = t % col_blocks
                sc = cb * ELEM_GRAN
                with x_dfb.reserve() as blk:
                    tx = ttl.copy(x[row, sc:sc + ELEM_GRAN], blk); tx.wait()
                with sh_dfb.reserve() as blk:
                    tx = ttl.copy(shift[row, sc:sc + ELEM_GRAN], blk); tx.wait()
                with sc_dfb.reserve() as blk:
                    tx = ttl.copy(scale[row, sc:sc + ELEM_GRAN], blk); tx.wait()
    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                row = t // col_blocks
                cb = t % col_blocks
                sc = cb * ELEM_GRAN
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[row, sc:sc + ELEM_GRAN]); tx.wait()

@ttl.kernel(grid="auto")
def gated_residual_kernel(residual, x, gate, out):
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = residual.shape[0] // TILE
    col_blocks = residual.shape[1] // TILE // ELEM_GRAN
    total = row_tiles * col_blocks
    tiles_per_core = -(-total // grid_cols)
    res_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, ELEM_GRAN), buffer_factor=2)
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, ELEM_GRAN), buffer_factor=2)
    g_dfb = ttl.make_dataflow_buffer_like(gate, shape=(1, ELEM_GRAN), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, ELEM_GRAN), buffer_factor=2)
    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                with res_dfb.wait() as rv, x_dfb.wait() as xv, g_dfb.wait() as gv, out_dfb.reserve() as o:
                    o.store(rv + xv * gv)
    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                row = t // col_blocks
                cb = t % col_blocks
                sc = cb * ELEM_GRAN
                with res_dfb.reserve() as blk:
                    tx = ttl.copy(residual[row, sc:sc + ELEM_GRAN], blk); tx.wait()
                with x_dfb.reserve() as blk:
                    tx = ttl.copy(x[row, sc:sc + ELEM_GRAN], blk); tx.wait()
                with g_dfb.reserve() as blk:
                    tx = ttl.copy(gate[row, sc:sc + ELEM_GRAN], blk); tx.wait()
    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total:
                row = t // col_blocks
                cb = t % col_blocks
                sc = cb * ELEM_GRAN
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[row, sc:sc + ELEM_GRAN]); tx.wait()

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
                        with acc_dfb.wait() as total, bcast_dfb.reserve() as bc:
                            bc.store(ttl.math.broadcast(total, dims=[1]))
                        with bcast_dfb.wait() as bv, red_dfb.reserve() as scaled:
                            scaled.store(bv * ms + ttl.math.fill(bv, 1e-5))
                        with red_dfb.wait() as msq, rsq_dfb.reserve() as rsq:
                            rsq.store(ttl.math.rsqrt(msq))
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

def make_fused_norm_mod_kernel(dim_tiles):
    """Fused RMSNorm + weight multiply + AdaLN modulate in one kernel.
    Replaces: rmsnorm → mul(norm_w) → adaln_modulate(mu, sigma)."""
    @ttl.kernel(grid="auto")
    def fused_norm_mod(x, norm_w, mu, sigma, scaler, mean_scale, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        seq_tiles = x.shape[0] // TILE
        tiles_per_core = -(-seq_tiles // grid_cols)
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        nw_dfb = ttl.make_dataflow_buffer_like(norm_w, shape=(1, 1), buffer_factor=2)
        mu_dfb = ttl.make_dataflow_buffer_like(mu, shape=(1, 1), buffer_factor=2)
        sig_dfb = ttl.make_dataflow_buffer_like(sigma, shape=(1, 1), buffer_factor=2)
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
                        # Pass 1: sum of squares for RMSNorm
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
                        with acc_dfb.wait() as total, bcast_dfb.reserve() as bc:
                            bc.store(ttl.math.broadcast(total, dims=[1]))
                        with bcast_dfb.wait() as bv, red_dfb.reserve() as scaled:
                            scaled.store(bv * ms + ttl.math.fill(bv, 1e-5))
                        with red_dfb.wait() as msq, rsq_dfb.reserve() as rsq:
                            rsq.store(ttl.math.rsqrt(msq))
                        # Pass 2: normalize * weight * (1+sigma) + mu
                        with rsq_dfb.wait() as rsqv:
                            for j in range(dim_tiles):
                                with x_dfb.wait() as xj, nw_dfb.wait() as nw, mu_dfb.wait() as muv, sig_dfb.wait() as sigv, out_dfb.reserve() as o:
                                    normed = xj * rsqv * nw
                                    o.store(normed * (sigv + ttl.math.fill(sigv, 1.0)) + muv)
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
                    # Pass 1: x tiles for RMS
                    for j in range(dim_tiles):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[tile_idx, j], blk); tx.wait()
                    # Pass 2: x + norm_w + mu + sigma
                    for j in range(dim_tiles):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[tile_idx, j], blk); tx.wait()
                        with nw_dfb.reserve() as blk:
                            tx = ttl.copy(norm_w[tile_idx, j], blk); tx.wait()
                        with mu_dfb.reserve() as blk:
                            tx = ttl.copy(mu[tile_idx, j], blk); tx.wait()
                        with sig_dfb.reserve() as blk:
                            tx = ttl.copy(sigma[tile_idx, j], blk); tx.wait()
        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    for j in range(dim_tiles):
                        with out_dfb.wait() as blk:
                            tx = ttl.copy(blk, out[tile_idx, j]); tx.wait()
    return fused_norm_mod

def make_fused_linear_bias_kernel(k_chunk):
    """Fused linear + bias add in one kernel. Eliminates intermediate DRAM write."""
    @ttl.kernel(grid="auto")
    def linear_bias_kernel(x, w, bias, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        m_tiles = x.shape[0] // TILE
        n_tiles = w.shape[1] // TILE
        total_out = m_tiles * n_tiles
        tiles_per_core = -(-total_out // grid_cols)
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, k_chunk), buffer_factor=2)
        w_dfb = ttl.make_dataflow_buffer_like(w, shape=(k_chunk, 1), buffer_factor=2)
        b_dfb = ttl.make_dataflow_buffer_like(bias, shape=(1, 1), buffer_factor=2)
        mm_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                idx = core_x * tiles_per_core + local_t
                if idx < total_out:
                    with x_dfb.wait() as xv, w_dfb.wait() as wv, mm_dfb.reserve() as mm:
                        mm.store(xv @ wv)
                    with mm_dfb.wait() as mmv, b_dfb.wait() as bv, out_dfb.reserve() as o:
                        o.store(mmv + bv)
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
                    with b_dfb.reserve() as blk:
                        tx = ttl.copy(bias[row, col], blk); tx.wait()
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
    return linear_bias_kernel

def make_fused_norm_rope_kernel(qkv_col_offset):
    """Fused RMSNorm + weight mul + RoPE in one kernel.
    Reads head tiles directly from qkv_out at col_offset, writes to head-batched output.
    Eliminates all ttnn reshape/permute/slice ops for Q or K."""
    SEQ_TILES = SEQ_PADDED // TILE  # 3

    @ttl.kernel(grid="auto")
    def fused_norm_rope(qkv, norm_w, sin_tab, cos_tab, perm, scaler, mean_scale, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        total_tiles = N_HEADS * SEQ_TILES  # 60
        tiles_per_core = -(-total_tiles // grid_cols)

        x_dfb = ttl.make_dataflow_buffer_like(qkv, shape=(1, 1), buffer_factor=2)
        nw_dfb = ttl.make_dataflow_buffer_like(norm_w, shape=(1, 1), buffer_factor=2)
        sin_dfb = ttl.make_dataflow_buffer_like(sin_tab, shape=(1, 1), buffer_factor=2)
        cos_dfb = ttl.make_dataflow_buffer_like(cos_tab, shape=(1, 1), buffer_factor=2)
        p_dfb = ttl.make_dataflow_buffer_like(perm, shape=(1, 1), buffer_factor=1)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        ms_dfb = ttl.make_dataflow_buffer_like(mean_scale, shape=(1, 1), buffer_factor=1)

        sq_dfb = ttl.make_dataflow_buffer_like(qkv, shape=(1, 1), buffer_factor=2)
        red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        bcast_dfb = ttl.make_dataflow_buffer_like(qkv, shape=(1, 1), buffer_factor=2)
        rsq_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        pm_dfb = ttl.make_dataflow_buffer_like(qkv, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            with sc_dfb.wait() as sc, ms_dfb.wait() as ms, p_dfb.wait() as p:
                for local_t in range(tiles_per_core):
                    t = core_x * tiles_per_core + local_t
                    if t < total_tiles:
                        # RMSNorm pass 1: x^2 → reduce → rsqrt (dim_tiles=1)
                        with x_dfb.wait() as x0:
                            with sq_dfb.reserve() as sq:
                                sq.store(x0 * x0)
                        with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                            r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                        with red_dfb.wait() as rv, bcast_dfb.reserve() as bc:
                            bc.store(ttl.math.broadcast(rv, dims=[1]))
                        with bcast_dfb.wait() as bv, red_dfb.reserve() as scaled:
                            scaled.store(bv * ms + ttl.math.fill(bv, 1e-5))
                        with red_dfb.wait() as msq, rsq_dfb.reserve() as rsq:
                            rsq.store(ttl.math.rsqrt(msq))
                        # Pass 2: normalize + weight → matmul P for rope perm
                        with rsq_dfb.wait() as rsqv:
                            with x_dfb.wait() as x1, nw_dfb.wait() as nw, sq_dfb.reserve() as normed:
                                normed.store(x1 * rsqv * nw)
                            with sq_dfb.wait() as nv, pm_dfb.reserve() as pm:
                                pm.store(nv @ p)
                            # Pass 3: recompute normed, combine with rope
                            with x_dfb.wait() as x2, nw_dfb.wait() as nw2, pm_dfb.wait() as pv, sin_dfb.wait() as sv, cos_dfb.wait() as cv, out_dfb.reserve() as o:
                                o.store(cv * (x2 * rsqv * nw2) + sv * pv)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            with sc_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0, 0], blk); tx.wait()
            with ms_dfb.reserve() as blk:
                tx = ttl.copy(mean_scale[0, 0], blk); tx.wait()
            with p_dfb.reserve() as blk:
                tx = ttl.copy(perm[0, 0], blk); tx.wait()
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total_tiles:
                    head = t // SEQ_TILES
                    seq_r = t % SEQ_TILES
                    qkv_col = qkv_col_offset + head
                    out_row = head * SEQ_TILES + seq_r
                    # x read 1 (rmsnorm pass 1)
                    with x_dfb.reserve() as blk:
                        tx = ttl.copy(qkv[seq_r, qkv_col], blk); tx.wait()
                    # x read 2 + norm_w (normalize + matmul)
                    with x_dfb.reserve() as blk:
                        tx = ttl.copy(qkv[seq_r, qkv_col], blk); tx.wait()
                    with nw_dfb.reserve() as blk:
                        tx = ttl.copy(norm_w[out_row, 0], blk); tx.wait()
                    # x read 3 + norm_w + sin + cos (recompute normed + final)
                    with x_dfb.reserve() as blk:
                        tx = ttl.copy(qkv[seq_r, qkv_col], blk); tx.wait()
                    with nw_dfb.reserve() as blk:
                        tx = ttl.copy(norm_w[out_row, 0], blk); tx.wait()
                    with sin_dfb.reserve() as blk:
                        tx = ttl.copy(sin_tab[out_row, 0], blk); tx.wait()
                    with cos_dfb.reserve() as blk:
                        tx = ttl.copy(cos_tab[out_row, 0], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total_tiles:
                    head = t // SEQ_TILES
                    seq_r = t % SEQ_TILES
                    out_row = head * SEQ_TILES + seq_r
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[out_row, 0]); tx.wait()

    return fused_norm_rope

fused_q_norm_rope = make_fused_norm_rope_kernel(0)           # Q cols 0-19
fused_k_norm_rope = make_fused_norm_rope_kernel(N_HEADS)     # K cols 20-39

rmsnorm_d320 = make_rmsnorm_kernel(D_MODEL // TILE)
rmsnorm_d1 = make_rmsnorm_kernel(1)  # for QK-norm over D_HEAD_PAD=32 (1 tile)
fused_norm_mod_d320 = make_fused_norm_mod_kernel(D_MODEL // TILE)
linear_k10 = make_linear_kernel(10)
linear_k40 = make_linear_kernel(40)
linear_bias_k10 = make_fused_linear_bias_kernel(10)
linear_bias_k20 = make_fused_linear_bias_kernel(20)  # for O proj with padded input (640/32=20)
linear_bias_k40 = make_fused_linear_bias_kernel(40)

# Fused modulation broadcast: all 6 params (mu1,sig1,c1,mu2,sig2,c2) in one kernel.
# 6 separate kernels → 1 kernel, saving 5 launch overheads per block.
N_MODS = 6
MOD_D_TILES = D_MODEL // TILE  # 10
MOD_GRAN = 5
MOD_COL_BLOCKS = MOD_D_TILES // MOD_GRAN  # 2
MOD_SEQ_TILES = SEQ_PADDED // TILE  # 3
MOD_BLOCKS_PER_MOD = MOD_SEQ_TILES * MOD_COL_BLOCKS  # 6

@ttl.kernel(grid="auto")
def mod_broadcast_all(src, b1, b2, b3, b4, b5, b6, scaler, o1, o2, o3, o4, o5, o6):
    """Broadcast all 6 modulation params from linear output row 0 + bias in one kernel."""
    grid_cols, _ = ttl.grid_size(dims=2)
    total_blocks = N_MODS * MOD_BLOCKS_PER_MOD  # 6 * 6 = 36
    tiles_per_core = -(-total_blocks // grid_cols)
    src_dfb = ttl.make_dataflow_buffer_like(src, shape=(1, MOD_GRAN), buffer_factor=2)
    bias_dfb = ttl.make_dataflow_buffer_like(b1, shape=(1, MOD_GRAN), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
    red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, MOD_GRAN), buffer_factor=2)
    bcast_dfb = ttl.make_dataflow_buffer_like(o1, shape=(1, MOD_GRAN), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(o1, shape=(1, MOD_GRAN), buffer_factor=2)
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
                mod_idx = t // MOD_BLOCKS_PER_MOD
                rem = t % MOD_BLOCKS_PER_MOD
                row = rem // MOD_COL_BLOCKS
                cb = rem % MOD_COL_BLOCKS
                src_col = mod_idx * MOD_D_TILES + cb * MOD_GRAN
                bc = cb * MOD_GRAN
                ec = bc + MOD_GRAN
                with src_dfb.reserve() as blk:
                    tx = ttl.copy(src[0, src_col:src_col + MOD_GRAN], blk); tx.wait()
                # Select bias tensor by mod_idx
                if mod_idx == 0:
                    with bias_dfb.reserve() as blk:
                        tx = ttl.copy(b1[row, bc:ec], blk); tx.wait()
                if mod_idx == 1:
                    with bias_dfb.reserve() as blk:
                        tx = ttl.copy(b2[row, bc:ec], blk); tx.wait()
                if mod_idx == 2:
                    with bias_dfb.reserve() as blk:
                        tx = ttl.copy(b3[row, bc:ec], blk); tx.wait()
                if mod_idx == 3:
                    with bias_dfb.reserve() as blk:
                        tx = ttl.copy(b4[row, bc:ec], blk); tx.wait()
                if mod_idx == 4:
                    with bias_dfb.reserve() as blk:
                        tx = ttl.copy(b5[row, bc:ec], blk); tx.wait()
                if mod_idx == 5:
                    with bias_dfb.reserve() as blk:
                        tx = ttl.copy(b6[row, bc:ec], blk); tx.wait()
    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_blocks:
                mod_idx = t // MOD_BLOCKS_PER_MOD
                rem = t % MOD_BLOCKS_PER_MOD
                row = rem // MOD_COL_BLOCKS
                cb = rem % MOD_COL_BLOCKS
                bc = cb * MOD_GRAN
                ec = bc + MOD_GRAN
                if mod_idx == 0:
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, o1[row, bc:ec]); tx.wait()
                if mod_idx == 1:
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, o2[row, bc:ec]); tx.wait()
                if mod_idx == 2:
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, o3[row, bc:ec]); tx.wait()
                if mod_idx == 3:
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, o4[row, bc:ec]); tx.wait()
                if mod_idx == 4:
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, o5[row, bc:ec]); tx.wait()
                if mod_idx == 5:
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, o6[row, bc:ec]); tx.wait()

# ============================================================
# Host helpers
# ============================================================

def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def zeros_tt(shape, device):
    return to_tt(torch.zeros(shape, dtype=torch.bfloat16), device)

def expand_bias(bias, seq_len):
    dim = bias.shape[0]
    seq_padded = ((seq_len + TILE - 1) // TILE) * TILE
    return bias.unsqueeze(0).expand(seq_padded, -1).contiguous().to(torch.bfloat16)

def expand_per_frame(vec, toks, seq_padded):
    """Broadcast (1, D) to (seq_padded, D) by repeating for each token in frame."""
    D = vec.shape[-1]
    out = torch.zeros(seq_padded, D, dtype=torch.bfloat16)
    out[:toks] = vec[0]
    return out

def patch_forward(frame, state):
    x = F.conv2d(frame.float(), state["patch.init_conv_seq.0.weight"].float(),
                 state["patch.init_conv_seq.0.bias"].float(), padding=2)
    x = F.silu(x)
    x = F.group_norm(x, 32, state["patch.init_conv_seq.2.weight"].float(),
                     state["patch.init_conv_seq.2.bias"].float())
    x = F.conv2d(x, state["patch.init_conv_seq.3.weight"].float(),
                 state["patch.init_conv_seq.3.bias"].float(), padding=2)
    x = F.silu(x)
    x = F.group_norm(x, 32, state["patch.init_conv_seq.5.weight"].float(),
                     state["patch.init_conv_seq.5.bias"].float())
    ps = PATCH_SIZE
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//ps, ps, W//ps, ps)
    x = x.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)
    x = (x @ state["patch.x_embedder.weight"].float().T + state["patch.x_embedder.bias"].float())
    return x.to(torch.bfloat16)

def unpatch_forward(x, state):
    x = (x.float() @ state["unpatch.unpatch.weight"].float().T + state["unpatch.unpatch.bias"].float())
    B, seq, d = x.shape
    c, p = 3, PATCH_SIZE
    h, w = HEIGHT // p, WIDTH // p
    x = x.reshape(B, h, w, p, p, c)
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(B, c, HEIGHT, WIDTH).to(torch.bfloat16)

def rmsnorm_host(x, w, eps=1e-6):
    rms = (x.float() ** 2).mean(dim=-1, keepdim=True)
    return ((x.float() / (rms + eps).sqrt()) * w.float()).to(x.dtype)

def apply_rope(x, sins, coss):
    x_perm = torch.empty_like(x)
    even = torch.arange(0, x.shape[-1], 2)
    odd = torch.arange(1, x.shape[-1], 2)
    x_perm[:, :, :, even] = -x[:, :, :, odd]
    x_perm[:, :, :, odd] = x[:, :, :, even]
    return (coss * x.float() + sins * x_perm.float()).to(x.dtype)

def build_rope_tables(state, offset, tt_device):
    """Build sin/cos tables in head-batched 2D format for device-side RoPE.
    Returns dict of device tensors: {block_idx: (sin_tt, cos_tt)}.
    Each is (HEAD_BATCH, D_HEAD_PAD) = (1920, 32)."""
    tables = {}
    for i in range(N_BLOCKS):
        p = f"blocks.{i}"
        sins = state[f"{p}.selfattn.rope.sins"]  # (1, max_pos, N_HEADS, D_HEAD)
        coss = state[f"{p}.selfattn.rope.coss"]
        # Build (HEAD_BATCH, D_HEAD_PAD) = (N_HEADS * SEQ_PADDED, 32)
        sin_2d = torch.zeros(HEAD_BATCH, D_HEAD_PAD, dtype=torch.bfloat16)
        cos_2d = torch.zeros(HEAD_BATCH, D_HEAD_PAD, dtype=torch.bfloat16)
        n_rope_heads = sins.shape[2]  # 1 if shared, N_HEADS if per-head
        for h in range(N_HEADS):
            rh = min(h, n_rope_heads - 1)
            for s in range(SEQ):
                pos = offset + s
                sin_2d[h * SEQ_PADDED + s, :D_HEAD] = sins[0, pos, rh, :]
                cos_2d[h * SEQ_PADDED + s, :D_HEAD] = coss[0, pos, rh, :]
        tables[i] = (to_tt(sin_2d, tt_device), to_tt(cos_2d, tt_device))
    return tables


# ============================================================
# Weight preloading: send everything to device ONCE
# ============================================================

def preload_weights(state, tt_device):
    """Pre-load all model weights and biases to TT device DRAM.
    Called once at startup. Returns dict of device tensors."""
    t0 = time.time()
    dev = {}

    # Conditioning
    dev['mixer_w'] = to_tt(state["time_emb_mixer.weight"].T.contiguous(), tt_device)
    dev['mixer_bias'] = to_tt(expand_bias(state["time_emb_mixer.bias"], TILE), tt_device)

    for i in range(N_BLOCKS):
        p = f"blocks.{i}"

        # Modulation weight (bias split into 6 chunks, broadcast to SEQ_PADDED)
        dev[f'{p}.mod_w'] = to_tt(state[f"{p}.modulation.1.weight"].T.contiguous(), tt_device)
        mod_bias = state[f"{p}.modulation.1.bias"]
        for ci, name in enumerate(['mu1', 'sigma1', 'c1', 'mu2', 'sigma2', 'c2']):
            chunk = mod_bias[ci*D_MODEL:(ci+1)*D_MODEL]
            dev[f'{p}.mod_bias_{name}'] = to_tt(
                chunk.unsqueeze(0).expand(SEQ_PADDED, -1).contiguous(), tt_device)

        # Norm weights (expanded to seq dim)
        dev[f'{p}.norm1_w'] = to_tt(
            state[f"{p}.norm1.w"].unsqueeze(0).expand(SEQ_PADDED, -1).contiguous(), tt_device)
        dev[f'{p}.norm2_w'] = to_tt(
            state[f"{p}.norm2.w"].unsqueeze(0).expand(SEQ_PADDED, -1).contiguous(), tt_device)

        # QKV with D_HEAD padded to D_HEAD_PAD=32 (tile-aligned per head)
        qkv_w_orig = state[f"{p}.selfattn.QKV.weight"].T.contiguous()  # (D_MODEL, 3*N_HEADS*D_HEAD)
        qkv_w_pad = torch.zeros(D_MODEL, 3 * N_HEADS * D_HEAD_PAD, dtype=torch.bfloat16)
        for h in range(3 * N_HEADS):
            qkv_w_pad[:, h*D_HEAD_PAD:h*D_HEAD_PAD+D_HEAD] = qkv_w_orig[:, h*D_HEAD:h*D_HEAD+D_HEAD]
        dev[f'{p}.qkv_w'] = to_tt(qkv_w_pad, tt_device)
        qkv_bias_orig = state[f"{p}.selfattn.QKV.bias"]
        qkv_bias_pad = torch.zeros(3 * N_HEADS * D_HEAD_PAD, dtype=torch.bfloat16)
        for h in range(3 * N_HEADS):
            qkv_bias_pad[h*D_HEAD_PAD:h*D_HEAD_PAD+D_HEAD] = qkv_bias_orig[h*D_HEAD:h*D_HEAD+D_HEAD]
        dev[f'{p}.qkv_bias'] = to_tt(expand_bias(qkv_bias_pad, SEQ_PADDED), tt_device)

        # O projection with padded input: (N_HEADS*D_HEAD_PAD, D_MODEL) = (640, 320)
        o_w_orig = state[f"{p}.selfattn.O.weight"].T.contiguous()  # (D_MODEL, D_MODEL) = (320, 320)
        o_w_pad = torch.zeros(N_HEADS * D_HEAD_PAD, D_MODEL, dtype=torch.bfloat16)
        for h in range(N_HEADS):
            o_w_pad[h*D_HEAD_PAD:h*D_HEAD_PAD+D_HEAD, :] = o_w_orig[h*D_HEAD:h*D_HEAD+D_HEAD, :]
        dev[f'{p}.o_w'] = to_tt(o_w_pad, tt_device)
        dev[f'{p}.o_bias'] = to_tt(expand_bias(state[f"{p}.selfattn.O.bias"], SEQ_PADDED), tt_device)

        # QK-norm weights: (D_HEAD,) padded to (D_HEAD_PAD,), expanded to (HEAD_BATCH, D_HEAD_PAD)
        for qk, name in [('lnq', 'lnq_w'), ('lnk', 'lnk_w')]:
            w = state[f"{p}.selfattn.{qk}.w"]  # (D_HEAD,)
            w_pad = torch.zeros(D_HEAD_PAD, dtype=torch.bfloat16)
            w_pad[:D_HEAD] = w
            dev[f'{p}.{name}'] = to_tt(
                w_pad.unsqueeze(0).expand(HEAD_BATCH, -1).contiguous(), tt_device)

        # GEGLU
        dev[f'{p}.geglu_up_w'] = to_tt(state[f"{p}.geglu.up_proj.weight"].T.contiguous(), tt_device)
        dev[f'{p}.geglu_up_bias'] = to_tt(expand_bias(state[f"{p}.geglu.up_proj.bias"], SEQ_PADDED), tt_device)
        dev[f'{p}.geglu_gate_w'] = to_tt(state[f"{p}.geglu.up_gate.weight"].T.contiguous(), tt_device)
        dev[f'{p}.geglu_gate_bias'] = to_tt(expand_bias(state[f"{p}.geglu.up_gate.bias"], SEQ_PADDED), tt_device)
        dev[f'{p}.geglu_down_w'] = to_tt(state[f"{p}.geglu.down.weight"].T.contiguous(), tt_device)
        dev[f'{p}.geglu_down_bias'] = to_tt(expand_bias(state[f"{p}.geglu.down.bias"], SEQ_PADDED), tt_device)

    # Final modulation + norm
    dev['final_mod_w'] = to_tt(state["modulation.1.weight"].T.contiguous(), tt_device)
    dev['final_mod_bias'] = to_tt(expand_bias(state["modulation.1.bias"], TILE), tt_device)
    dev['final_norm_w'] = to_tt(
        state["norm.w"].unsqueeze(0).expand(SEQ_PADDED, -1).contiguous(), tt_device)

    # RoPE permutation matrix: (32, 32) that does even/odd swap with negation
    # P[2i+1, 2i] = -1 (put -x[odd] at even position)
    # P[2i, 2i+1] = 1 (put x[even] at odd position)
    P = torch.zeros(D_HEAD_PAD, D_HEAD_PAD, dtype=torch.bfloat16)
    for i in range(D_HEAD // 2):
        P[2*i+1, 2*i] = -1.0
        P[2*i, 2*i+1] = 1.0
    dev['rope_perm'] = to_tt(P, tt_device)

    elapsed = time.time() - t0
    print(f"Preloaded {len(dev)} tensors to device in {elapsed:.1f}s")
    return dev


def prealloc_scratch(tt_device):
    """Pre-allocate reusable scratch tensors. Called once at startup."""
    t0 = time.time()
    s = {}
    # Double-buffer for residual stream (z alternates between a/b across blocks)
    s['z_a'] = zeros_tt((SEQ_PADDED, D_MODEL), tt_device)
    s['z_b'] = zeros_tt((SEQ_PADDED, D_MODEL), tt_device)
    # (SEQ_PADDED, D_MODEL) scratch - need 3 for norm+mul+adaln chain
    s['d320_a'] = zeros_tt((SEQ_PADDED, D_MODEL), tt_device)
    s['d320_b'] = zeros_tt((SEQ_PADDED, D_MODEL), tt_device)
    s['d320_c'] = zeros_tt((SEQ_PADDED, D_MODEL), tt_device)
    # QKV with padded heads: (SEQ_PADDED, 3*N_HEADS*D_HEAD_PAD) = (96, 1920)
    s['qkv_out'] = zeros_tt((SEQ_PADDED, 3 * N_HEADS * D_HEAD_PAD), tt_device)
    # Head-batched format: (N_HEADS*SEQ_PADDED, D_HEAD_PAD) = (1920, 32)
    s['hb_a'] = zeros_tt((HEAD_BATCH, D_HEAD_PAD), tt_device)
    s['hb_b'] = zeros_tt((HEAD_BATCH, D_HEAD_PAD), tt_device)
    s['hb_c'] = zeros_tt((HEAD_BATCH, D_HEAD_PAD), tt_device)
    # (SEQ_PADDED, D_MID) scratch for GEGLU - need 3 (u_b + g_a alive for mul)
    s['d1280_a'] = zeros_tt((SEQ_PADDED, D_MID), tt_device)
    s['d1280_b'] = zeros_tt((SEQ_PADDED, D_MID), tt_device)
    s['d1280_c'] = zeros_tt((SEQ_PADDED, D_MID), tt_device)
    # (TILE, 1920) for modulation
    s['mod_a'] = zeros_tt((TILE, 1920), tt_device)
    s['mod_b'] = zeros_tt((TILE, 1920), tt_device)
    # (TILE, D_MODEL) for conditioning
    s['cond_a'] = zeros_tt((TILE, D_MODEL), tt_device)
    s['cond_b'] = zeros_tt((TILE, D_MODEL), tt_device)
    s['cond_silu'] = zeros_tt((TILE, D_MODEL), tt_device)
    # (TILE, 640) for final modulation
    s['fm_a'] = zeros_tt((TILE, 640), tt_device)
    s['fm_b'] = zeros_tt((TILE, 640), tt_device)
    # Modulation broadcast buffers (SEQ_PADDED, D_MODEL) × 6
    s['mu1'] = zeros_tt((SEQ_PADDED, D_MODEL), tt_device)
    s['sigma1'] = zeros_tt((SEQ_PADDED, D_MODEL), tt_device)
    s['c1'] = zeros_tt((SEQ_PADDED, D_MODEL), tt_device)
    s['mu2'] = zeros_tt((SEQ_PADDED, D_MODEL), tt_device)
    s['sigma2'] = zeros_tt((SEQ_PADDED, D_MODEL), tt_device)
    s['c2'] = zeros_tt((SEQ_PADDED, D_MODEL), tt_device)
    # Final modulation broadcast
    s['mu_f'] = zeros_tt((SEQ_PADDED, D_MODEL), tt_device)
    s['sig_f'] = zeros_tt((SEQ_PADDED, D_MODEL), tt_device)
    # KV cache managed dynamically via ttnn.concat (not pre-allocated)
    elapsed = time.time() - t0
    print(f"Pre-allocated {len(s)} scratch tensors in {elapsed:.1f}s")
    return s


# ============================================================
# Forward pass with pre-cached weights + scratch reuse
# ============================================================

def dit_forward(z_frame, action_idx, timestep_float, state, dev, scr, tt_device, scaler_tt, mean_scale_tt,
                mean_scale_16_tt, rope_tables, kv_cache=None, frame_idx=0):
    timers = {}
    def tnow(): return time.time()

    t0 = tnow()
    # Conditioning
    ts_scaled = int(timestep_float * (T_MAX - 1))
    action_emb = state["action_emb.weight"][action_idx]
    time_pe = state["time_emb.pe"][ts_scaled]

    cond_padded = torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16)
    cond_padded[0] = time_pe
    linear_k10(to_tt(cond_padded, tt_device), dev['mixer_w'], scr['cond_a'])
    action_padded = torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16)
    action_padded[0] = action_emb
    add_kernel(scr['cond_a'], dev['mixer_bias'], scr['cond_b'])
    add_kernel(scr['cond_b'], to_tt(action_padded, tt_device), scr['cond_a'])
    silu_kernel(scr['cond_a'], scr['cond_silu'])
    cond_host = ttnn.to_torch(scr['cond_a'])
    cond_vec = cond_host[0:1]
    timers['conditioning'] = tnow() - t0

    t0 = tnow()
    patched = patch_forward(z_frame, state)
    reg = state["registers"].unsqueeze(0)
    patched = torch.cat([patched, reg], dim=1)
    z_2d = torch.zeros(SEQ_PADDED, D_MODEL, dtype=torch.bfloat16)
    z_2d[:SEQ] = patched.squeeze(0)

    # z_cur starts as fresh tensor from patch, then alternates with z_next
    z_cur = to_tt(z_2d, tt_device)
    z_next = scr['z_b']
    timers['patch'] = tnow() - t0

    new_kv = []

    block_timers = {k: 0.0 for k in [
        'modulation', 'norm1_mod', 'qkv_proj',
        'host_attn', 'sdpa', 'o_proj', 'gated_res1',
        'norm2_mod', 'geglu', 'gated_res2'
    ]}

    for block_idx in range(N_BLOCKS):
        p = f"blocks.{block_idx}"

        # Modulation: linear + fused 6-in-1 broadcast (7 calls → 2)
        t0 = tnow()
        linear_k10(scr['cond_silu'], dev[f'{p}.mod_w'], scr['mod_a'])
        mod_broadcast_all(scr['mod_a'],
            dev[f'{p}.mod_bias_mu1'], dev[f'{p}.mod_bias_sigma1'], dev[f'{p}.mod_bias_c1'],
            dev[f'{p}.mod_bias_mu2'], dev[f'{p}.mod_bias_sigma2'], dev[f'{p}.mod_bias_c2'],
            scaler_tt,
            scr['mu1'], scr['sigma1'], scr['c1'], scr['mu2'], scr['sigma2'], scr['c2'])
        block_timers['modulation'] += tnow() - t0

        # Fused RMSNorm1 + weight mul + adaln modulate (3 kernels → 1)
        t0 = tnow()
        fused_norm_mod_d320(z_cur, dev[f'{p}.norm1_w'], scr['mu1'], scr['sigma1'],
                            scaler_tt, mean_scale_tt, scr['d320_a'])
        block_timers['norm1_mod'] += tnow() - t0

        # QKV projection with padded heads → (SEQ_PADDED, 1920) on device
        t0 = tnow()
        linear_bias_k10(scr['d320_a'], dev[f'{p}.qkv_w'], dev[f'{p}.qkv_bias'], scr['qkv_out'])
        block_timers['qkv_proj'] += tnow() - t0

        # Device-side attention: fused QK-norm+RoPE → KV cache → SDPA
        t0 = tnow()

        # Q: fused rmsnorm + weight mul + RoPE (reads directly from qkv_out cols 0-19)
        fused_q_norm_rope(scr['qkv_out'], dev[f'{p}.lnq_w'],
                          rope_tables[block_idx][0], rope_tables[block_idx][1],
                          dev['rope_perm'], scaler_tt, mean_scale_16_tt, scr['hb_c'])
        q_sdpa = ttnn.reshape(scr['hb_c'], [1, N_HEADS, SEQ_PADDED, D_HEAD_PAD])
        t_q = tnow() - t0

        # K: fused rmsnorm + weight mul + RoPE (reads from qkv_out cols 20-39)
        t1 = tnow()
        fused_k_norm_rope(scr['qkv_out'], dev[f'{p}.lnk_w'],
                          rope_tables[block_idx][0], rope_tables[block_idx][1],
                          dev['rope_perm'], scaler_tt, mean_scale_16_tt, scr['hb_b'])
        k_new_sdpa = ttnn.reshape(scr['hb_b'], [1, N_HEADS, SEQ_PADDED, D_HEAD_PAD])
        t_k = tnow() - t1

        # V: just reshape from qkv_out cols 40-59 → (1, 20, 96, 32)
        t1 = tnow()
        v_flat = ttnn.slice(scr['qkv_out'],
                            [0, 2 * N_HEADS * D_HEAD_PAD],
                            [SEQ_PADDED, 3 * N_HEADS * D_HEAD_PAD])
        v_3d = ttnn.reshape(v_flat, [SEQ_PADDED, N_HEADS, D_HEAD_PAD])
        v_hm = ttnn.permute(v_3d, [1, 0, 2])
        v_sdpa = ttnn.reshape(v_hm, [1, N_HEADS, SEQ_PADDED, D_HEAD_PAD])
        t_v = tnow() - t1

        # KV cache: concat with previous frames' K/V (device tensors)
        t1 = tnow()
        if kv_cache is not None and kv_cache[block_idx] is not None:
            k_sdpa = ttnn.concat([kv_cache[block_idx]['k'], k_new_sdpa], dim=2)
            v_full = ttnn.concat([kv_cache[block_idx]['v'], v_sdpa], dim=2)
        else:
            k_sdpa = k_new_sdpa
            v_full = v_sdpa
        new_kv.append({'k': k_new_sdpa, 'v': v_sdpa})
        t_kv_cache = tnow() - t1

        # SDPA on device
        t1 = tnow()
        attn_out_tt = ttnn.transformer.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_full, is_causal=False, scale=1.0)
        t_sdpa = tnow() - t1

        # Reshape output: (1, 20, 96, 32) → (96, 640) for O projection
        t1 = tnow()
        attn_perm = ttnn.permute(attn_out_tt, [0, 2, 1, 3])
        attn_2d = ttnn.reshape(attn_perm, [SEQ_PADDED, N_HEADS * D_HEAD_PAD])
        t_out_reshape = tnow() - t1

        block_timers['host_attn'] += tnow() - t0
        if block_idx == 0:
            attn_sub = {'q_norm_rope': t_q, 'k_norm_rope': t_k, 'v_reshape': t_v,
                        'kv_cache': t_kv_cache, 'sdpa': t_sdpa, 'out_reshape': t_out_reshape}

        # O projection with padded weight: (96, 640) @ (640, 320) → (96, 320)
        t0 = tnow()
        linear_bias_k20(attn_2d, dev[f'{p}.o_w'], dev[f'{p}.o_bias'], scr['d320_b'])
        block_timers['o_proj'] += tnow() - t0

        # Gated residual 1: z_next = z_cur + o_biased * c1
        t0 = tnow()
        gated_residual_kernel(z_cur, scr['d320_b'], scr['c1'], z_next)
        z_cur, z_next = z_next, z_cur  # swap
        block_timers['gated_res1'] += tnow() - t0

        # Fused RMSNorm2 + weight mul + adaln modulate (3 kernels → 1)
        t0 = tnow()
        fused_norm_mod_d320(z_cur, dev[f'{p}.norm2_w'], scr['mu2'], scr['sigma2'],
                            scaler_tt, mean_scale_tt, scr['d320_a'])
        block_timers['norm2_mod'] += tnow() - t0

        t0 = tnow()
        # Fused GEGLU: up + gate with bias, silu, mul, down with bias
        linear_bias_k10(scr['d320_a'], dev[f'{p}.geglu_up_w'], dev[f'{p}.geglu_up_bias'], scr['d1280_b'])
        linear_bias_k10(scr['d320_a'], dev[f'{p}.geglu_gate_w'], dev[f'{p}.geglu_gate_bias'], scr['d1280_c'])
        silu_kernel(scr['d1280_c'], scr['d1280_a'])
        mul_kernel(scr['d1280_b'], scr['d1280_a'], scr['d1280_c'])
        linear_bias_k40(scr['d1280_c'], dev[f'{p}.geglu_down_w'], dev[f'{p}.geglu_down_bias'], scr['d320_b'])
        block_timers['geglu'] += tnow() - t0

        # Gated residual 2
        t0 = tnow()
        gated_residual_kernel(z_cur, scr['d320_b'], scr['c2'], z_next)
        z_cur, z_next = z_next, z_cur
        block_timers['gated_res2'] += tnow() - t0

    # Final modulation + norm
    t0 = tnow()
    cs2 = (cond_vec.float() * torch.sigmoid(cond_vec.float())).to(torch.bfloat16)
    cs2p = torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16); cs2p[0] = cs2[0]
    linear_k10(to_tt(cs2p, tt_device), dev['final_mod_w'], scr['fm_a'])
    add_kernel(scr['fm_a'], dev['final_mod_bias'], scr['fm_b'])
    fm_h = ttnn.to_torch(scr['fm_b'])
    mu_f, sigma_f = fm_h[0, :640].reshape(2, D_MODEL).chunk(2, dim=0)

    rmsnorm_d320(z_cur, scaler_tt, mean_scale_tt, scr['d320_a'])
    mul_kernel(scr['d320_a'], dev['final_norm_w'], scr['d320_b'])

    mu_fe = to_tt(expand_per_frame(mu_f, TOKS_PER_FRAME, SEQ_PADDED), tt_device)
    sig_fe = to_tt(expand_per_frame(sigma_f, TOKS_PER_FRAME, SEQ_PADDED), tt_device)
    adaln_modulate_kernel(scr['d320_b'], mu_fe, sig_fe, scr['d320_a'])
    timers['final_mod'] = tnow() - t0

    t0 = tnow()
    z_h = ttnn.to_torch(scr['d320_a'])[:SEQ]
    z_no_reg = z_h[:SEQ-1].unsqueeze(0)
    out = unpatch_forward(z_no_reg, state)
    timers['unpatch'] = tnow() - t0

    total = sum(timers.values()) + sum(block_timers.values())
    print(f"  dit_forward total: {total*1000:.0f}ms")
    print(f"    conditioning: {timers['conditioning']*1000:.1f}ms")
    print(f"    patch:        {timers['patch']*1000:.1f}ms")
    print(f"    final_mod:    {timers['final_mod']*1000:.1f}ms")
    print(f"    unpatch:      {timers['unpatch']*1000:.1f}ms")
    print(f"    --- 8 blocks ({sum(block_timers.values())*1000:.0f}ms) ---")
    for k, v in sorted(block_timers.items(), key=lambda x: -x[1]):
        pct = v / total * 100 if total > 0 else 0
        print(f"    {k:25s}: {v*1000:7.1f}ms ({pct:4.1f}%)")
    if attn_sub:
        print(f"    --- attn breakdown (block 0, 1 iter) ---")
        for k, v in attn_sub.items():
            print(f"      {k:20s}: {v*1000:6.2f}ms")

    return out, new_kv


def sample_frame(z_noise, action_idx, n_steps, cfg, state, dev, scr, tt_device, scaler_tt, mean_scale_tt,
                 mean_scale_16_tt, kv_cache=None, frame_idx=0):
    ts = 1 - torch.linspace(0, 1, n_steps + 1)
    ts = 3 * ts / (2 * ts + 1)

    # Build RoPE sin/cos tables for this frame's position offset (constant across denoise steps)
    kv_offset = 0
    if kv_cache is not None and kv_cache[0] is not None:
        kv_offset = kv_cache[0]['k'].shape[2]  # dim 2 is seq in (1, N_HEADS, seq, D_HEAD_PAD)
    rope_tables = build_rope_tables(state, kv_offset, tt_device)

    z = z_noise.clone()
    new_kv = None
    for i in range(n_steps):
        t_val = ts[i].item()
        dt = (ts[i] - ts[i+1]).item()

        v_cond, new_kv = dit_forward(
            z, action_idx, t_val, state, dev, scr, tt_device, scaler_tt, mean_scale_tt,
            mean_scale_16_tt, rope_tables, kv_cache=kv_cache, frame_idx=frame_idx)

        if cfg > 0 and cfg != 1.0:
            v_uncond, _ = dit_forward(
                z, 0, t_val, state, dev, scr, tt_device, scaler_tt, mean_scale_tt,
                mean_scale_16_tt, rope_tables, kv_cache=kv_cache, frame_idx=frame_idx)
            v_pred = v_uncond.float() + cfg * (v_cond.float() - v_uncond.float())
        else:
            v_pred = v_cond.float()

        z = (z.float() + dt * v_pred).to(torch.bfloat16)

    return z.clamp(-1, 1), new_kv


def extend_kv_cache(kv_cache, new_kv, n_window):
    """Extend KV cache with new K/V (device tensors). Concat along seq dim (dim=2)."""
    max_cached_frames = n_window - 1
    max_cached_seq = max_cached_frames * TOKS_PER_FRAME

    if kv_cache is None:
        return new_kv

    updated = []
    for layer_idx in range(N_BLOCKS):
        old_k = kv_cache[layer_idx]['k']
        old_v = kv_cache[layer_idx]['v']
        cur_k = new_kv[layer_idx]['k']
        cur_v = new_kv[layer_idx]['v']

        full_k = ttnn.concat([old_k, cur_k], dim=2)
        full_v = ttnn.concat([old_v, cur_v], dim=2)

        # Trim to max window
        total_seq = full_k.shape[2]
        if total_seq > max_cached_seq:
            trim_start = total_seq - max_cached_seq
            full_k = ttnn.slice(full_k, [0, 0, trim_start, 0],
                                [1, N_HEADS, total_seq, D_HEAD_PAD])
            full_v = ttnn.slice(full_v, [0, 0, trim_start, 0],
                                [1, N_HEADS, total_seq, D_HEAD_PAD])

        updated.append({'k': full_k, 'v': full_v})

    return updated


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    tt_device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    print("Loading weights...")
    ckpt = torch.load("/tmp/model.pt", map_location="cpu", weights_only=False)
    state = {k.replace("_orig_mod.", ""): v.to(torch.bfloat16) for k, v in ckpt.items()}

    scaler_tt = to_tt(torch.ones(TILE, TILE, dtype=torch.bfloat16), tt_device)
    mean_scale_tt = to_tt(torch.full((TILE, TILE), 1.0/D_MODEL, dtype=torch.bfloat16), tt_device)
    mean_scale_16_tt = to_tt(torch.full((TILE, TILE), 1.0/D_HEAD, dtype=torch.bfloat16), tt_device)

    # Pre-load all weights and scratch buffers to device
    dev = preload_weights(state, tt_device)
    scr = prealloc_scratch(tt_device)

    N_STEPS = 8
    CFG = 1.0
    N_FRAMES_GEN = 3
    N_WINDOW = 30
    actions = [2] * N_FRAMES_GEN

    frames = []
    kv_cache = None
    t_total = time.time()

    for fidx in range(N_FRAMES_GEN):
        action = actions[fidx]
        cached_frames = 0 if kv_cache is None else kv_cache[0]['k'].shape[2] // TOKS_PER_FRAME
        print(f"Frame {fidx+1}/{N_FRAMES_GEN} (action={action}, cached={cached_frames} frames)...")

        noise = torch.randn(1, 3, HEIGHT, WIDTH, dtype=torch.bfloat16)
        t0 = time.time()

        frame, new_kv = sample_frame(
            noise, action, N_STEPS, CFG, state, dev, scr, tt_device, scaler_tt, mean_scale_tt,
            mean_scale_16_tt, kv_cache=kv_cache, frame_idx=fidx)

        kv_cache = extend_kv_cache(kv_cache, new_kv, N_WINDOW)

        elapsed = time.time() - t0
        print(f"  FRAME DONE in {elapsed:.1f}s, range=[{frame.min().item():.2f}, {frame.max().item():.2f}]")

        img = ((frame[0].float() + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
        frames.append(img)

    total_time = time.time() - t_total
    print(f"\nAll {N_FRAMES_GEN} frames in {total_time:.1f}s ({total_time/N_FRAMES_GEN:.1f}s/frame)")

    ttnn.close_device(tt_device)
