"""
Full CausalDiT forward pass on TT hardware.

Loads real weights, runs one forward pass on a single frame,
compares against PyTorch reference.

Host-side: Patch (conv2d+groupnorm+patchify), UnPatch, embedding lookups,
           QK-norm (d_head=16), head reshape, SDPA padding.
Device-side: All linear projections, RMSNorm, AdaLN, GEGLU, gated residuals,
             SDPA via ttnn.
"""

import torch
import torch.nn.functional as F
import ttnn
import ttl
import math

TILE = 32
D_MODEL = 320
D_MID = 1280
N_HEADS = 20
D_HEAD = 16
N_BLOCKS = 8
PATCH_SIZE = 3
HEIGHT = 24
WIDTH = 24
TOKS_PER_FRAME = (HEIGHT // PATCH_SIZE) * (WIDTH // PATCH_SIZE) + 1  # 64 + 1 register = 65
T_MAX = 1000

# ============================================================
# TT-Lang Kernels
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

DIM_TILES_320 = D_MODEL // TILE

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

rmsnorm_d320 = make_rmsnorm_kernel(DIM_TILES_320)
linear_k10 = make_linear_kernel(10)
linear_k40 = make_linear_kernel(40)

# ============================================================
# Host-side helpers
# ============================================================

def patch_forward(frame, state):
    """Run Patch module on host. frame: (B, 3, 24, 24) -> (B, 64, 320)"""
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
    """Run UnPatch on host. x: (B, 64, 320) -> (B, 3, 24, 24)"""
    x = (x.float() @ state["unpatch.unpatch.weight"].float().T + state["unpatch.unpatch.bias"].float())
    B, seq, d = x.shape
    c = 3
    p = PATCH_SIZE
    h = HEIGHT // p
    w = WIDTH // p
    x = x.reshape(B, h, w, p, p, c)
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(B, c, HEIGHT, WIDTH).to(torch.bfloat16)

def rmsnorm_host(x, w, eps=1e-6):
    rms = (x.float() ** 2).mean(dim=-1, keepdim=True)
    return ((x.float() / (rms + eps).sqrt()) * w.float()).to(x.dtype)

def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def zeros_tt(shape, device):
    return to_tt(torch.zeros(shape, dtype=torch.bfloat16), device)

def expand_bias(bias, seq_len):
    """Expand 1D bias to (seq_padded, dim) for tile alignment."""
    dim = bias.shape[0]
    seq_padded = ((seq_len + TILE - 1) // TILE) * TILE
    out = torch.zeros(seq_padded, dim, dtype=torch.bfloat16)
    for i in range(seq_len):
        out[i] = bias
    return out

def expand_per_frame(vec, toks_per_frame, n_frames, seq_padded):
    """Expand per-frame vector (n_frames, D) to (seq_padded, D) by repeating within frames."""
    D = vec.shape[-1]
    out = torch.zeros(seq_padded, D, dtype=torch.bfloat16)
    for f in range(n_frames):
        for t in range(toks_per_frame):
            idx = f * toks_per_frame + t
            if idx < seq_padded:
                out[idx] = vec[f]
    return out


# ============================================================
# Main: Single-frame forward pass
# ============================================================

if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    # Load weights
    ckpt = torch.load("/tmp/model.pt", map_location="cpu", weights_only=False)
    state = {k.replace("_orig_mod.", ""): v.to(torch.bfloat16) for k, v in ckpt.items()}

    # Constants
    scaler_tt = to_tt(torch.ones(TILE, TILE, dtype=torch.bfloat16), device)
    mean_scale_tt = to_tt(torch.full((TILE, TILE), 1.0 / D_MODEL, dtype=torch.bfloat16), device)

    # Input: single frame + action + timestep
    frame = torch.randn(1, 1, 3, HEIGHT, WIDTH, dtype=torch.bfloat16)  # (B=1, dur=1, C, H, W)
    action = torch.tensor([[2]], dtype=torch.long)  # (B=1, dur=1)
    ts = torch.tensor([[0.5]], dtype=torch.float)  # (B=1, dur=1)

    N_FRAMES = 1
    ts_scaled = (ts * (T_MAX - 1)).long()

    # ---- Step 1: Embeddings (host) ----
    print("Step 1: Embeddings...")
    action_emb = state["action_emb.weight"][action[0, 0]]  # (320,)
    time_pe = state["time_emb.pe"][ts_scaled[0, 0]]  # (320,)

    # Pad conditioning to tile-aligned
    cond_padded = torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16)
    cond_padded[0] = time_pe

    # Mixer linear on device
    mixer_w = to_tt(state["time_emb_mixer.weight"].T.contiguous(), device)
    mixer_out = zeros_tt((TILE, D_MODEL), device)
    linear_k10(to_tt(cond_padded, device), mixer_w, mixer_out)

    # Add mixer bias + action emb
    mixer_b_padded = torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16)
    mixer_b_padded[0] = state["time_emb_mixer.bias"]
    action_padded = torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16)
    action_padded[0] = action_emb
    combined = torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16)
    combined[0] = state["time_emb_mixer.bias"] + action_emb

    cond_tt = zeros_tt((TILE, D_MODEL), device)
    add_kernel(mixer_out, to_tt(combined, device), cond_tt)
    cond_host = ttnn.to_torch(cond_tt)  # (32, 320), row 0 is the conditioning
    cond_vec = cond_host[0:1]  # (1, 320) - the actual conditioning for 1 frame

    # ---- Step 2: Patch (host) ----
    print("Step 2: Patch...")
    patched = patch_forward(frame[:, 0], state)  # (1, 64, 320)

    # Add register token -> (1, 65, 320)
    reg = state["registers"].unsqueeze(0)  # (1, 1, 320)
    patched = torch.cat([patched, reg], dim=1)  # -> (1, 65, 320)

    SEQ = TOKS_PER_FRAME * N_FRAMES  # 65
    SEQ_PADDED = ((SEQ + TILE - 1) // TILE) * TILE  # 96

    # Flatten to 2D and pad to tile-aligned
    z_2d = torch.zeros(SEQ_PADDED, D_MODEL, dtype=torch.bfloat16)
    z_2d[:SEQ] = patched.squeeze(0)

    z_tt = to_tt(z_2d, device)

    # ---- Step 3: Transformer blocks ----
    for block_idx in range(N_BLOCKS):
        print(f"Step 3.{block_idx}: Block {block_idx}...")
        prefix = f"blocks.{block_idx}"

        # 3a: Modulation: silu(cond) @ W_mod + b_mod -> 6 chunks
        mod_w = to_tt(state[f"{prefix}.modulation.1.weight"].T.contiguous(), device)
        # silu(cond) on host (cond is just 1 row, tiny)
        cond_silu = (cond_vec.float() * torch.sigmoid(cond_vec.float())).to(torch.bfloat16)
        cond_silu_padded = torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16)
        cond_silu_padded[0] = cond_silu[0]

        mod_out = zeros_tt((TILE, 1920), device)
        linear_k10(to_tt(cond_silu_padded, device), mod_w, mod_out)
        mod_host = ttnn.to_torch(mod_out)
        mod_host[0] = mod_host[0] + state[f"{prefix}.modulation.1.bias"]

        # Split into 6 chunks of 320
        mu1, sigma1, c1, mu2, sigma2, c2 = mod_host[0, :D_MODEL*6].reshape(6, D_MODEL).chunk(6, dim=0)
        mu1, sigma1, c1 = mu1.squeeze(0), sigma1.squeeze(0), c1.squeeze(0)
        mu2, sigma2, c2 = mu2.squeeze(0), sigma2.squeeze(0), c2.squeeze(0)

        # Expand per-frame modulation to full sequence
        mu1_exp = expand_per_frame(mu1.unsqueeze(0), TOKS_PER_FRAME, N_FRAMES, SEQ_PADDED)
        sigma1_exp = expand_per_frame(sigma1.unsqueeze(0), TOKS_PER_FRAME, N_FRAMES, SEQ_PADDED)
        c1_exp = expand_per_frame(c1.unsqueeze(0), TOKS_PER_FRAME, N_FRAMES, SEQ_PADDED)
        mu2_exp = expand_per_frame(mu2.unsqueeze(0), TOKS_PER_FRAME, N_FRAMES, SEQ_PADDED)
        sigma2_exp = expand_per_frame(sigma2.unsqueeze(0), TOKS_PER_FRAME, N_FRAMES, SEQ_PADDED)
        c2_exp = expand_per_frame(c2.unsqueeze(0), TOKS_PER_FRAME, N_FRAMES, SEQ_PADDED)

        # 3b: RMSNorm1 + weight
        norm1_out = zeros_tt((SEQ_PADDED, D_MODEL), device)
        rmsnorm_d320(z_tt, scaler_tt, mean_scale_tt, norm1_out)
        norm1_w = state[f"{prefix}.norm1.w"]
        norm1_w_exp = norm1_w.unsqueeze(0).expand(SEQ_PADDED, -1).contiguous()
        norm1_weighted = zeros_tt((SEQ_PADDED, D_MODEL), device)
        mul_kernel(norm1_out, to_tt(norm1_w_exp, device), norm1_weighted)

        # 3c: AdaLN modulate
        z_mod = zeros_tt((SEQ_PADDED, D_MODEL), device)
        adaln_modulate_kernel(norm1_weighted, to_tt(mu1_exp, device), to_tt(sigma1_exp, device), z_mod)

        # 3d: Attention
        # QKV projection
        qkv_w = to_tt(state[f"{prefix}.selfattn.QKV.weight"].T.contiguous(), device)
        qkv_out = zeros_tt((SEQ_PADDED, 960), device)
        linear_k10(z_mod, qkv_w, qkv_out)
        qkv_b_exp = expand_bias(state[f"{prefix}.selfattn.QKV.bias"], SEQ_PADDED)
        qkv_biased = zeros_tt((SEQ_PADDED, 960), device)
        add_kernel(qkv_out, to_tt(qkv_b_exp, device), qkv_biased)

        # Read back for head reshaping + QK-norm + SDPA
        qkv_host = ttnn.to_torch(qkv_biased)[:SEQ]  # (65, 960) unpadded
        q_h, k_h, v_h = qkv_host.chunk(3, dim=-1)  # each (65, 320)

        q_heads = q_h.reshape(1, SEQ, N_HEADS, D_HEAD)
        k_heads = k_h.reshape(1, SEQ, N_HEADS, D_HEAD)
        v_heads = v_h.reshape(1, SEQ, N_HEADS, D_HEAD)

        # QK-norm
        lnq_w = state[f"{prefix}.selfattn.lnq.w"]
        lnk_w = state[f"{prefix}.selfattn.lnk.w"]
        q_normed = rmsnorm_host(q_heads, lnq_w)
        k_normed = rmsnorm_host(k_heads, lnk_w)

        # RoPE from checkpoint
        rope_sins = state[f"{prefix}.selfattn.rope.sins"]  # (1, 1950, 1, 16)
        rope_coss = state[f"{prefix}.selfattn.rope.coss"]
        sins = rope_sins[:, :SEQ, :, :]  # (1, 65, 1, 16)
        coss = rope_coss[:, :SEQ, :, :]

        def apply_rope(x, sins, coss):
            x_perm = torch.empty_like(x)
            even = torch.arange(0, x.shape[-1], 2)
            odd = torch.arange(1, x.shape[-1], 2)
            x_perm[:, :, :, even] = -x[:, :, :, odd]
            x_perm[:, :, :, odd] = x[:, :, :, even]
            return (coss * x.float() + sins * x_perm.float()).to(x.dtype)

        q_roped = apply_rope(q_normed, sins, coss)
        k_roped = apply_rope(k_normed, sins, coss)

        # Pad seq to tile-aligned for SDPA (65 -> 96)
        def pad_seq(t, target):
            if t.shape[2] == target:
                return t
            return F.pad(t, (0, 0, 0, target - t.shape[2]))

        q_sdpa = pad_seq(q_roped.permute(0, 2, 1, 3), SEQ_PADDED)  # (1, 20, 96, 16)
        k_sdpa = pad_seq(k_roped.permute(0, 2, 1, 3), SEQ_PADDED)
        v_sdpa = pad_seq(v_heads.permute(0, 2, 1, 3), SEQ_PADDED)

        # Pad d_head to 32
        q_pad = F.pad(q_sdpa, (0, 16))
        k_pad = F.pad(k_sdpa, (0, 16))
        v_pad = F.pad(v_sdpa, (0, 16))

        # SDPA on device
        q_tt = to_tt(q_pad, device)
        k_tt = to_tt(k_pad, device)
        v_tt = to_tt(v_pad, device)

        attn_out_tt = ttnn.transformer.scaled_dot_product_attention(
            q_tt, k_tt, v_tt, is_causal=False
        )

        attn_host = ttnn.to_torch(attn_out_tt)[:, :, :SEQ, :D_HEAD]  # (1, 20, 65, 16)
        attn_2d = attn_host.permute(0, 2, 1, 3).reshape(SEQ, D_MODEL)  # (65, 320)

        # Pad back and send to device for O projection
        attn_padded = torch.zeros(SEQ_PADDED, D_MODEL, dtype=torch.bfloat16)
        attn_padded[:SEQ] = attn_2d
        attn_tt = to_tt(attn_padded, device)

        # O projection
        o_w = to_tt(state[f"{prefix}.selfattn.O.weight"].T.contiguous(), device)
        o_out = zeros_tt((SEQ_PADDED, D_MODEL), device)
        linear_k10(attn_tt, o_w, o_out)
        o_b_exp = expand_bias(state[f"{prefix}.selfattn.O.bias"], SEQ_PADDED)
        o_biased = zeros_tt((SEQ_PADDED, D_MODEL), device)
        add_kernel(o_out, to_tt(o_b_exp, device), o_biased)

        # 3e: Gated residual
        z_new = zeros_tt((SEQ_PADDED, D_MODEL), device)
        gated_residual_kernel(z_tt, o_biased, to_tt(c1_exp, device), z_new)
        z_tt = z_new

        # 3f: RMSNorm2 + weight + modulate
        norm2_out = zeros_tt((SEQ_PADDED, D_MODEL), device)
        rmsnorm_d320(z_tt, scaler_tt, mean_scale_tt, norm2_out)
        norm2_w = state[f"{prefix}.norm2.w"]
        norm2_w_exp = norm2_w.unsqueeze(0).expand(SEQ_PADDED, -1).contiguous()
        norm2_weighted = zeros_tt((SEQ_PADDED, D_MODEL), device)
        mul_kernel(norm2_out, to_tt(norm2_w_exp, device), norm2_weighted)
        z_mod2 = zeros_tt((SEQ_PADDED, D_MODEL), device)
        adaln_modulate_kernel(norm2_weighted, to_tt(mu2_exp, device), to_tt(sigma2_exp, device), z_mod2)

        # 3g: GEGLU MLP
        geglu_up_w = to_tt(state[f"{prefix}.geglu.up_proj.weight"].T.contiguous(), device)
        geglu_gate_w = to_tt(state[f"{prefix}.geglu.up_gate.weight"].T.contiguous(), device)
        geglu_down_w = to_tt(state[f"{prefix}.geglu.down.weight"].T.contiguous(), device)

        up_out = zeros_tt((SEQ_PADDED, D_MID), device)
        linear_k10(z_mod2, geglu_up_w, up_out)
        up_b_exp = expand_bias(state[f"{prefix}.geglu.up_proj.bias"], SEQ_PADDED)
        up_biased = zeros_tt((SEQ_PADDED, D_MID), device)
        add_kernel(up_out, to_tt(up_b_exp, device), up_biased)

        gate_out = zeros_tt((SEQ_PADDED, D_MID), device)
        linear_k10(z_mod2, geglu_gate_w, gate_out)
        gate_b_exp = expand_bias(state[f"{prefix}.geglu.up_gate.bias"], SEQ_PADDED)
        gate_biased = zeros_tt((SEQ_PADDED, D_MID), device)
        add_kernel(gate_out, to_tt(gate_b_exp, device), gate_biased)
        gate_act = zeros_tt((SEQ_PADDED, D_MID), device)
        silu_kernel(gate_biased, gate_act)

        mid = zeros_tt((SEQ_PADDED, D_MID), device)
        mul_kernel(up_biased, gate_act, mid)

        down_out = zeros_tt((SEQ_PADDED, D_MODEL), device)
        linear_k40(mid, geglu_down_w, down_out)
        down_b_exp = expand_bias(state[f"{prefix}.geglu.down.bias"], SEQ_PADDED)
        mlp_biased = zeros_tt((SEQ_PADDED, D_MODEL), device)
        add_kernel(down_out, to_tt(down_b_exp, device), mlp_biased)

        # 3h: Gated residual
        z_new2 = zeros_tt((SEQ_PADDED, D_MODEL), device)
        gated_residual_kernel(z_tt, mlp_biased, to_tt(c2_exp, device), z_new2)
        z_tt = z_new2

    # ---- Step 4: Final modulation + norm ----
    print("Step 4: Final norm + modulation...")
    # Final modulation: silu(cond) @ W + b -> 2 chunks (mu, sigma)
    final_mod_w = to_tt(state["modulation.1.weight"].T.contiguous(), device)
    cond_silu_padded2 = torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16)
    cond_silu2 = (cond_vec.float() * torch.sigmoid(cond_vec.float())).to(torch.bfloat16)
    cond_silu_padded2[0] = cond_silu2[0]

    final_mod_out = zeros_tt((TILE, 640), device)
    linear_k10(to_tt(cond_silu_padded2, device), final_mod_w, final_mod_out)
    final_mod_host = ttnn.to_torch(final_mod_out)
    final_mod_host[0] = final_mod_host[0] + state["modulation.1.bias"]
    mu_final, sigma_final = final_mod_host[0, :640].reshape(2, D_MODEL).chunk(2, dim=0)
    mu_final = mu_final.squeeze(0)
    sigma_final = sigma_final.squeeze(0)

    # Final RMSNorm + weight
    final_norm_out = zeros_tt((SEQ_PADDED, D_MODEL), device)
    rmsnorm_d320(z_tt, scaler_tt, mean_scale_tt, final_norm_out)
    fnorm_w = state["norm.w"]
    fnorm_w_exp = fnorm_w.unsqueeze(0).expand(SEQ_PADDED, -1).contiguous()
    fnorm_weighted = zeros_tt((SEQ_PADDED, D_MODEL), device)
    mul_kernel(final_norm_out, to_tt(fnorm_w_exp, device), fnorm_weighted)

    # Final modulate
    mu_f_exp = expand_per_frame(mu_final.unsqueeze(0), TOKS_PER_FRAME, N_FRAMES, SEQ_PADDED)
    sigma_f_exp = expand_per_frame(sigma_final.unsqueeze(0), TOKS_PER_FRAME, N_FRAMES, SEQ_PADDED)
    z_final = zeros_tt((SEQ_PADDED, D_MODEL), device)
    adaln_modulate_kernel(fnorm_weighted, to_tt(mu_f_exp, device), to_tt(sigma_f_exp, device), z_final)

    # Read back for unpatch
    z_host = ttnn.to_torch(z_final)[:SEQ]  # (65, 320)

    # ---- Step 5: UnPatch (host) ----
    print("Step 5: UnPatch...")
    # Remove register token
    z_no_reg = z_host[:SEQ-1].unsqueeze(0)  # (1, 64, 320)
    output = unpatch_forward(z_no_reg, state)  # (1, 3, 24, 24)

    print(f"\nForward pass complete!")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"  Output mean: {output.mean().item():.4f}")
    print(f"  Output[0,0,0,:5]: {output[0,0,0,:5].tolist()}")

    ttnn.close_device(device)
