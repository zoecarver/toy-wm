"""
Diffusion sampling on TT hardware.

Generates Pong frames by running the CausalDiT forward pass through
an Euler sampler with classifier-free guidance (CFG).

For each frame:
  1. Start with noise z ~ N(0, 1)
  2. For each denoise step:
     a. Run forward(z, action, t) -> v_cond
     b. Run forward(z, null_action=0, t) -> v_uncond
     c. CFG blend: v = v_uncond + cfg * (v_cond - v_uncond)
     d. Euler step: z = z + dt * v
  3. Output: denoised frame

No KV cache for v1 (single frame, no autoregressive context).
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
N_BLOCKS = 8
PATCH_SIZE = 3
HEIGHT = 24
WIDTH = 24
TOKS_PER_FRAME = (HEIGHT // PATCH_SIZE) * (WIDTH // PATCH_SIZE) + 1  # 65
T_MAX = 1000
SEQ = TOKS_PER_FRAME  # 65
SEQ_PADDED = ((SEQ + TILE - 1) // TILE) * TILE  # 96

# ============================================================
# TT-Lang Kernels (same as forward.py)
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
def adaln_modulate_kernel(x, shift, scale, out):
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

rmsnorm_d320 = make_rmsnorm_kernel(D_MODEL // TILE)
linear_k10 = make_linear_kernel(10)
linear_k40 = make_linear_kernel(40)

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
    out = torch.zeros(seq_padded, dim, dtype=torch.bfloat16)
    for i in range(seq_len):
        out[i] = bias
    return out

def expand_per_frame(vec, toks, n_frames, seq_padded):
    D = vec.shape[-1]
    out = torch.zeros(seq_padded, D, dtype=torch.bfloat16)
    for f in range(n_frames):
        for t in range(toks):
            idx = f * toks + t
            if idx < seq_padded:
                out[idx] = vec[f]
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


def dit_forward(z_frame, action_idx, timestep_float, state, tt_device, scaler_tt, mean_scale_tt,
                kv_cache=None, frame_idx=0):
    """
    Single forward pass of CausalDiT with KV cache support.

    kv_cache: list of N_BLOCKS dicts, each with 'k' and 'v' tensors
              of shape (1, N_HEADS, cached_seq, D_HEAD) post QK-norm and RoPE.
              None = no cache (first frame).
    frame_idx: which frame we're generating (for RoPE offset).

    Returns: (output_frame, new_kv) where new_kv is the current frame's
             K/V post-norm/rope for each layer (to be appended to cache).
    """
    N_FRAMES = 1
    rope_offset = frame_idx * TOKS_PER_FRAME

    # Conditioning
    ts_scaled = int(timestep_float * (T_MAX - 1))
    action_emb = state["action_emb.weight"][action_idx]
    time_pe = state["time_emb.pe"][ts_scaled]

    cond_padded = torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16)
    cond_padded[0] = time_pe
    mixer_w = to_tt(state["time_emb_mixer.weight"].T.contiguous(), tt_device)
    mixer_out = zeros_tt((TILE, D_MODEL), tt_device)
    linear_k10(to_tt(cond_padded, tt_device), mixer_w, mixer_out)
    combined = torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16)
    combined[0] = state["time_emb_mixer.bias"] + action_emb
    cond_tt = zeros_tt((TILE, D_MODEL), tt_device)
    add_kernel(mixer_out, to_tt(combined, tt_device), cond_tt)
    cond_host = ttnn.to_torch(cond_tt)
    cond_vec = cond_host[0:1]

    # Patch
    patched = patch_forward(z_frame, state)
    reg = state["registers"].unsqueeze(0)
    patched = torch.cat([patched, reg], dim=1)

    z_2d = torch.zeros(SEQ_PADDED, D_MODEL, dtype=torch.bfloat16)
    z_2d[:SEQ] = patched.squeeze(0)
    z_tt = to_tt(z_2d, tt_device)

    # Collect new K/V for cache update
    new_kv = []

    # Transformer blocks
    for block_idx in range(N_BLOCKS):
        prefix = f"blocks.{block_idx}"

        # Modulation
        mod_w = to_tt(state[f"{prefix}.modulation.1.weight"].T.contiguous(), tt_device)
        cond_silu = (cond_vec.float() * torch.sigmoid(cond_vec.float())).to(torch.bfloat16)
        cond_silu_padded = torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16)
        cond_silu_padded[0] = cond_silu[0]
        mod_out = zeros_tt((TILE, 1920), tt_device)
        linear_k10(to_tt(cond_silu_padded, tt_device), mod_w, mod_out)
        mod_host = ttnn.to_torch(mod_out)
        mod_host[0] = mod_host[0] + state[f"{prefix}.modulation.1.bias"]
        chunks = mod_host[0, :D_MODEL*6].reshape(6, D_MODEL)
        mu1, sigma1, c1, mu2, sigma2, c2 = [chunks[i] for i in range(6)]

        mu1_e = expand_per_frame(mu1.unsqueeze(0), TOKS_PER_FRAME, N_FRAMES, SEQ_PADDED)
        sigma1_e = expand_per_frame(sigma1.unsqueeze(0), TOKS_PER_FRAME, N_FRAMES, SEQ_PADDED)
        c1_e = expand_per_frame(c1.unsqueeze(0), TOKS_PER_FRAME, N_FRAMES, SEQ_PADDED)
        mu2_e = expand_per_frame(mu2.unsqueeze(0), TOKS_PER_FRAME, N_FRAMES, SEQ_PADDED)
        sigma2_e = expand_per_frame(sigma2.unsqueeze(0), TOKS_PER_FRAME, N_FRAMES, SEQ_PADDED)
        c2_e = expand_per_frame(c2.unsqueeze(0), TOKS_PER_FRAME, N_FRAMES, SEQ_PADDED)

        # RMSNorm1 + weight + modulate
        norm1_out = zeros_tt((SEQ_PADDED, D_MODEL), tt_device)
        rmsnorm_d320(z_tt, scaler_tt, mean_scale_tt, norm1_out)
        nw = state[f"{prefix}.norm1.w"].unsqueeze(0).expand(SEQ_PADDED, -1).contiguous()
        norm1_w = zeros_tt((SEQ_PADDED, D_MODEL), tt_device)
        mul_kernel(norm1_out, to_tt(nw, tt_device), norm1_w)
        z_mod = zeros_tt((SEQ_PADDED, D_MODEL), tt_device)
        adaln_modulate_kernel(norm1_w, to_tt(mu1_e, tt_device), to_tt(sigma1_e, tt_device), z_mod)

        # QKV projection
        qkv_w = to_tt(state[f"{prefix}.selfattn.QKV.weight"].T.contiguous(), tt_device)
        qkv_out = zeros_tt((SEQ_PADDED, 960), tt_device)
        linear_k10(z_mod, qkv_w, qkv_out)
        qkv_b_e = expand_bias(state[f"{prefix}.selfattn.QKV.bias"], SEQ_PADDED)
        qkv_biased = zeros_tt((SEQ_PADDED, 960), tt_device)
        add_kernel(qkv_out, to_tt(qkv_b_e, tt_device), qkv_biased)

        qkv_h = ttnn.to_torch(qkv_biased)[:SEQ]
        q_h, k_h, v_h = qkv_h.chunk(3, dim=-1)
        q_heads = q_h.reshape(1, SEQ, N_HEADS, D_HEAD)
        k_heads = k_h.reshape(1, SEQ, N_HEADS, D_HEAD)
        v_heads = v_h.reshape(1, SEQ, N_HEADS, D_HEAD)

        # Cache stores raw (pre-norm, pre-rope) K/V, matching original model.
        # Save current frame's raw K/V for cache.
        k_new_raw = k_heads  # (1, SEQ, N_HEADS, D_HEAD)
        v_new_raw = v_heads
        new_kv.append({'k': k_new_raw, 'v': v_new_raw})

        # Concatenate cached raw K/V with current frame's raw K/V
        if kv_cache is not None and kv_cache[block_idx] is not None:
            cached_k_raw = kv_cache[block_idx]['k']  # (1, cached_seq, N_HEADS, D_HEAD)
            cached_v_raw = kv_cache[block_idx]['v']
            k_all = torch.cat([cached_k_raw, k_heads], dim=1)  # (1, total_seq, N_HEADS, D_HEAD)
            v_all = torch.cat([cached_v_raw, v_heads], dim=1)
            offset = cached_k_raw.shape[1]
        else:
            k_all = k_heads
            v_all = v_heads
            offset = 0

        # Apply QK-norm to Q and full K (matching original: norm on concatenated K)
        q_n = rmsnorm_host(q_heads, state[f"{prefix}.selfattn.lnq.w"])
        k_n = rmsnorm_host(k_all, state[f"{prefix}.selfattn.lnk.w"])

        # Apply RoPE: Q gets offset, K gets full positions from 0
        sins = state[f"{prefix}.selfattn.rope.sins"]
        coss = state[f"{prefix}.selfattn.rope.coss"]
        total_kv_seq = k_all.shape[1]
        q_r = apply_rope(q_n, sins[:, offset:offset+SEQ, :, :], coss[:, offset:offset+SEQ, :, :])
        k_r = apply_rope(k_n, sins[:, :total_kv_seq, :, :], coss[:, :total_kv_seq, :, :])

        # Pad for SDPA
        kv_padded = ((total_kv_seq + TILE - 1) // TILE) * TILE

        q_s = F.pad(F.pad(q_r.permute(0,2,1,3), (0,16)), (0,0,0,SEQ_PADDED-SEQ))
        k_s = F.pad(F.pad(k_r.permute(0,2,1,3), (0,16)), (0,0,0,kv_padded-total_kv_seq))
        v_s = F.pad(F.pad(v_all.permute(0,2,1,3), (0,16)), (0,0,0,kv_padded-total_kv_seq))

        # scale=1.0 matches the original model which uses QK-norm instead of 1/sqrt(d) scaling
        attn_out_tt = ttnn.transformer.scaled_dot_product_attention(
            to_tt(q_s, tt_device), to_tt(k_s, tt_device), to_tt(v_s, tt_device),
            is_causal=False, scale=1.0)
        attn_h = ttnn.to_torch(attn_out_tt)[:, :, :SEQ, :D_HEAD]
        attn_2d = attn_h.permute(0,2,1,3).reshape(SEQ, D_MODEL)
        attn_p = torch.zeros(SEQ_PADDED, D_MODEL, dtype=torch.bfloat16)
        attn_p[:SEQ] = attn_2d

        # O projection
        o_w = to_tt(state[f"{prefix}.selfattn.O.weight"].T.contiguous(), tt_device)
        o_out = zeros_tt((SEQ_PADDED, D_MODEL), tt_device)
        linear_k10(to_tt(attn_p, tt_device), o_w, o_out)
        o_b_e = expand_bias(state[f"{prefix}.selfattn.O.bias"], SEQ_PADDED)
        o_biased = zeros_tt((SEQ_PADDED, D_MODEL), tt_device)
        add_kernel(o_out, to_tt(o_b_e, tt_device), o_biased)

        z_new = zeros_tt((SEQ_PADDED, D_MODEL), tt_device)
        gated_residual_kernel(z_tt, o_biased, to_tt(c1_e, tt_device), z_new)
        z_tt = z_new

        # RMSNorm2 + GEGLU
        n2_out = zeros_tt((SEQ_PADDED, D_MODEL), tt_device)
        rmsnorm_d320(z_tt, scaler_tt, mean_scale_tt, n2_out)
        nw2 = state[f"{prefix}.norm2.w"].unsqueeze(0).expand(SEQ_PADDED, -1).contiguous()
        n2_w = zeros_tt((SEQ_PADDED, D_MODEL), tt_device)
        mul_kernel(n2_out, to_tt(nw2, tt_device), n2_w)
        z_m2 = zeros_tt((SEQ_PADDED, D_MODEL), tt_device)
        adaln_modulate_kernel(n2_w, to_tt(mu2_e, tt_device), to_tt(sigma2_e, tt_device), z_m2)

        uw = to_tt(state[f"{prefix}.geglu.up_proj.weight"].T.contiguous(), tt_device)
        gw = to_tt(state[f"{prefix}.geglu.up_gate.weight"].T.contiguous(), tt_device)
        dw = to_tt(state[f"{prefix}.geglu.down.weight"].T.contiguous(), tt_device)

        u_o = zeros_tt((SEQ_PADDED, D_MID), tt_device)
        linear_k10(z_m2, uw, u_o)
        u_b = zeros_tt((SEQ_PADDED, D_MID), tt_device)
        add_kernel(u_o, to_tt(expand_bias(state[f"{prefix}.geglu.up_proj.bias"], SEQ_PADDED), tt_device), u_b)

        g_o = zeros_tt((SEQ_PADDED, D_MID), tt_device)
        linear_k10(z_m2, gw, g_o)
        g_b = zeros_tt((SEQ_PADDED, D_MID), tt_device)
        add_kernel(g_o, to_tt(expand_bias(state[f"{prefix}.geglu.up_gate.bias"], SEQ_PADDED), tt_device), g_b)
        g_a = zeros_tt((SEQ_PADDED, D_MID), tt_device)
        silu_kernel(g_b, g_a)

        mid = zeros_tt((SEQ_PADDED, D_MID), tt_device)
        mul_kernel(u_b, g_a, mid)
        d_o = zeros_tt((SEQ_PADDED, D_MODEL), tt_device)
        linear_k40(mid, dw, d_o)
        d_b = zeros_tt((SEQ_PADDED, D_MODEL), tt_device)
        add_kernel(d_o, to_tt(expand_bias(state[f"{prefix}.geglu.down.bias"], SEQ_PADDED), tt_device), d_b)

        z_new2 = zeros_tt((SEQ_PADDED, D_MODEL), tt_device)
        gated_residual_kernel(z_tt, d_b, to_tt(c2_e, tt_device), z_new2)
        z_tt = z_new2

    # Final modulation + norm
    fm_w = to_tt(state["modulation.1.weight"].T.contiguous(), tt_device)
    cs2 = (cond_vec.float() * torch.sigmoid(cond_vec.float())).to(torch.bfloat16)
    cs2p = torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16); cs2p[0] = cs2[0]
    fm_out = zeros_tt((TILE, 640), tt_device)
    linear_k10(to_tt(cs2p, tt_device), fm_w, fm_out)
    fm_h = ttnn.to_torch(fm_out)
    fm_h[0] = fm_h[0] + state["modulation.1.bias"]
    mu_f, sigma_f = fm_h[0, :640].reshape(2, D_MODEL).chunk(2, dim=0)

    fn_out = zeros_tt((SEQ_PADDED, D_MODEL), tt_device)
    rmsnorm_d320(z_tt, scaler_tt, mean_scale_tt, fn_out)
    fnw = state["norm.w"].unsqueeze(0).expand(SEQ_PADDED, -1).contiguous()
    fn_w = zeros_tt((SEQ_PADDED, D_MODEL), tt_device)
    mul_kernel(fn_out, to_tt(fnw, tt_device), fn_w)

    mu_fe = expand_per_frame(mu_f, TOKS_PER_FRAME, 1, SEQ_PADDED)
    sig_fe = expand_per_frame(sigma_f, TOKS_PER_FRAME, 1, SEQ_PADDED)
    z_final = zeros_tt((SEQ_PADDED, D_MODEL), tt_device)
    adaln_modulate_kernel(fn_w, to_tt(mu_fe, tt_device), to_tt(sig_fe, tt_device), z_final)

    z_h = ttnn.to_torch(z_final)[:SEQ]
    z_no_reg = z_h[:SEQ-1].unsqueeze(0)
    return unpatch_forward(z_no_reg, state), new_kv


def sample_frame(z_noise, action_idx, n_steps, cfg, state, tt_device, scaler_tt, mean_scale_tt,
                 kv_cache=None, frame_idx=0):
    """
    Denoise a single frame from noise using Euler sampling with CFG.
    z_noise: (1, 3, 24, 24) initial noise
    kv_cache: cached K/V from previous frames (or None)
    frame_idx: current frame index (for RoPE offset)
    Returns: (denoised_frame, new_kv_from_last_step)
    """
    ts = 1 - torch.linspace(0, 1, n_steps + 1)
    ts = 3 * ts / (2 * ts + 1)

    z = z_noise.clone()
    new_kv = None
    for i in range(n_steps):
        t_val = ts[i].item()
        dt = (ts[i] - ts[i+1]).item()

        v_cond, new_kv = dit_forward(
            z, action_idx, t_val, state, tt_device, scaler_tt, mean_scale_tt,
            kv_cache=kv_cache, frame_idx=frame_idx)

        if cfg > 0:
            v_uncond, _ = dit_forward(
                z, 0, t_val, state, tt_device, scaler_tt, mean_scale_tt,
                kv_cache=kv_cache, frame_idx=frame_idx)
            v_pred = v_uncond.float() + cfg * (v_cond.float() - v_uncond.float())
        else:
            v_pred = v_cond.float()

        z = (z.float() + dt * v_pred).to(torch.bfloat16)

    return z.clamp(-1, 1), new_kv


def extend_kv_cache(kv_cache, new_kv, n_window):
    """
    Append new frame's raw K/V to the cache, maintaining a rolling window.
    kv_cache: list of N_BLOCKS dicts with 'k','v' each (1, cached_seq, N_HEADS, D_HEAD)
    new_kv: list of N_BLOCKS dicts with 'k','v' each (1, SEQ, N_HEADS, D_HEAD)
    n_window: max frames to keep in cache (cache holds n_window-1=29 past frames)
    """
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

        full_k = torch.cat([old_k, cur_k], dim=1)  # concat on seq dim
        full_v = torch.cat([old_v, cur_v], dim=1)

        # Trim to max window (drop oldest frames)
        if full_k.shape[1] > max_cached_seq:
            full_k = full_k[:, -max_cached_seq:]
            full_v = full_v[:, -max_cached_seq:]

        updated.append({'k': full_k, 'v': full_v})

    return updated


# ============================================================
# Main: Generate 30 frames as MP4 with KV cache!
# ============================================================

if __name__ == "__main__":
    tt_device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    print("Loading weights...")
    ckpt = torch.load("/tmp/model.pt", map_location="cpu", weights_only=False)
    state = {k.replace("_orig_mod.", ""): v.to(torch.bfloat16) for k, v in ckpt.items()}

    scaler_tt = to_tt(torch.ones(TILE, TILE, dtype=torch.bfloat16), tt_device)
    mean_scale_tt = to_tt(torch.full((TILE, TILE), 1.0/D_MODEL, dtype=torch.bfloat16), tt_device)

    # Sampling parameters
    # cfg=1.0: pure conditional (v_pred = v_cond), no uncond path needed.
    # The original batches cond+uncond; with cfg=1.0 cond-only cache is self-consistent.
    N_STEPS = 8
    CFG = 1.0
    N_FRAMES_GEN = 30
    N_WINDOW = 30
    FPS = 10

    # Actions: 0=unconditional, 1=don't move, 2=up, 3=down (for cyan paddle)
    actions = [2] * 30  # up the whole time

    frames = []
    kv_cache = None
    t_total = time.time()

    for fidx in range(N_FRAMES_GEN):
        action = actions[fidx % len(actions)]
        cached_frames = 0 if kv_cache is None else kv_cache[0]['k'].shape[1] // TOKS_PER_FRAME
        print(f"Frame {fidx+1}/{N_FRAMES_GEN} (action={action}, cached={cached_frames} frames)...")

        noise = torch.randn(1, 3, HEIGHT, WIDTH, dtype=torch.bfloat16)
        t0 = time.time()

        frame, new_kv = sample_frame(
            noise, action, N_STEPS, CFG, state, tt_device, scaler_tt, mean_scale_tt,
            kv_cache=kv_cache, frame_idx=fidx)

        # Update cache with K/V from the last denoise step (conditional path)
        kv_cache = extend_kv_cache(kv_cache, new_kv, N_WINDOW)

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s, range=[{frame.min().item():.2f}, {frame.max().item():.2f}]")

        img = ((frame[0].float() + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
        frames.append(img)

    total_time = time.time() - t_total
    print(f"\nAll {N_FRAMES_GEN} frames generated in {total_time:.1f}s ({total_time/N_FRAMES_GEN:.1f}s/frame)")

    # Save individual PNGs (upscaled)
    import numpy as np
    from PIL import Image
    import subprocess

    frames_np = [f.permute(1, 2, 0).numpy() for f in frames]
    for i, f in enumerate(frames_np):
        img = Image.fromarray(f).resize((240, 240), Image.NEAREST)
        img.save(f"/tmp/pong_frame_{i:03d}.png")
    print(f"Saved {N_FRAMES_GEN} PNGs to /tmp/pong_frame_*.png")

    # Create MP4 via ffmpeg
    try:
        subprocess.run([
            'ffmpeg', '-y', '-framerate', str(FPS),
            '-i', '/tmp/pong_frame_%03d.png',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '/tmp/pong_tt.mp4'
        ], check=True, capture_output=True)
        print("Saved /tmp/pong_tt.mp4")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"ffmpeg failed: {e}")

    ttnn.close_device(tt_device)
