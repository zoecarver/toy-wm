"""
Weight loading and preparation for toy-wm on TT hardware.

Loads the PyTorch checkpoint, reshapes weights to tile-aligned 2D tensors,
and converts to ttnn device tensors. This is the ONLY host->device transfer
for weights (done once at startup).

Weight naming convention from checkpoint: _orig_mod.{path}
We strip the _orig_mod. prefix.
"""

import torch
import ttnn

TILE = 32
D_MODEL = 320
D_MID = 1280
N_HEADS = 20
D_HEAD = 16
N_BLOCKS = 8


def pad_to_tile(dim):
    return ((dim + TILE - 1) // TILE) * TILE


def pad_2d(t, target_rows=None, target_cols=None):
    """Pad a 2D tensor to tile-aligned dimensions."""
    rows, cols = t.shape
    tr = target_rows or pad_to_tile(rows)
    tc = target_cols or pad_to_tile(cols)
    if tr == rows and tc == cols:
        return t
    out = torch.zeros(tr, tc, dtype=t.dtype)
    out[:rows, :cols] = t
    return out


def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def load_weights(ckpt_path, device):
    """
    Load checkpoint and prepare all weights as device tensors.
    Returns a nested dict matching the model structure.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt

    # Strip _orig_mod. prefix
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    w = {}

    # ============================================================
    # Global weights
    # ============================================================

    # Action embedding: (4, 320) -> pad to (32, 320)
    w["action_emb"] = state["action_emb.weight"].to(torch.bfloat16)  # keep on host for lookup

    # Time embedding: precomputed sinusoidal table (1000, 320) - already tile-aligned
    w["time_emb_pe"] = state["time_emb.pe"].to(torch.bfloat16)  # keep on host for lookup

    # Time embedding mixer: linear (320, 320)
    w["time_emb_mixer_w"] = to_tt(state["time_emb_mixer.weight"].T.contiguous().to(torch.bfloat16), device)
    mixer_b = state["time_emb_mixer.bias"].to(torch.bfloat16)
    w["time_emb_mixer_b"] = mixer_b  # host, will be expanded per-use

    # Register token: (1, 320) -> keep on host
    w["registers"] = state["registers"].to(torch.bfloat16)

    # Final modulation: silu -> linear (320, 640) with bias
    # 640 = 2 * d_model (mu, sigma for final norm)
    w["final_mod_w"] = to_tt(state["modulation.1.weight"].T.contiguous().to(torch.bfloat16), device)
    w["final_mod_b"] = state["modulation.1.bias"].to(torch.bfloat16)  # host

    # Final norm
    w["final_norm_w"] = state["norm.w"].to(torch.bfloat16)  # host, will be expanded

    # RoPE sin/cos from checkpoint: (1, 1950, 1, 16) per block
    # All blocks share the same rope (it's positional), so just grab block 0
    rope_sins = state["rope_seq.sins"].to(torch.bfloat16)  # (1, 1950, 1, 16)
    rope_coss = state["rope_seq.coss"].to(torch.bfloat16)
    w["rope_sins"] = rope_sins
    w["rope_coss"] = rope_coss

    # Patch module
    w["patch_conv1_w"] = state["patch.init_conv_seq.0.weight"].to(torch.bfloat16)  # (160, 3, 5, 5)
    w["patch_conv1_b"] = state["patch.init_conv_seq.0.bias"].to(torch.bfloat16)    # (160,)
    w["patch_gn1_w"] = state["patch.init_conv_seq.2.weight"].to(torch.bfloat16)    # (160,) GroupNorm weight
    w["patch_gn1_b"] = state["patch.init_conv_seq.2.bias"].to(torch.bfloat16)      # (160,) GroupNorm bias
    w["patch_conv2_w"] = state["patch.init_conv_seq.3.weight"].to(torch.bfloat16)  # (160, 160, 5, 5)
    w["patch_conv2_b"] = state["patch.init_conv_seq.3.bias"].to(torch.bfloat16)
    w["patch_gn2_w"] = state["patch.init_conv_seq.5.weight"].to(torch.bfloat16)
    w["patch_gn2_b"] = state["patch.init_conv_seq.5.bias"].to(torch.bfloat16)
    # x_embedder: linear (1440, 320) - 1440 = patch_size^2 * 160 = 9 * 160
    w["patch_embed_w"] = to_tt(state["patch.x_embedder.weight"].T.contiguous().to(torch.bfloat16), device)
    w["patch_embed_b"] = state["patch.x_embedder.bias"].to(torch.bfloat16)

    # Unpatch: linear (320, 27) - 27 = 3 * patch_size^2 = 3 * 9
    w["unpatch_w"] = state["unpatch.unpatch.weight"].T.contiguous().to(torch.bfloat16)  # (320, 27)
    w["unpatch_b"] = state["unpatch.unpatch.bias"].to(torch.bfloat16)  # (27,)

    # ============================================================
    # Per-block weights
    # ============================================================
    w["blocks"] = []
    for i in range(N_BLOCKS):
        b = {}
        prefix = f"blocks.{i}"

        # Norms
        b["norm1_w"] = state[f"{prefix}.norm1.w"].to(torch.bfloat16)  # (320,)
        b["norm2_w"] = state[f"{prefix}.norm2.w"].to(torch.bfloat16)

        # Attention
        # QKV: (960, 320) -> transpose to (320, 960) for our linear kernel
        b["qkv_w"] = to_tt(state[f"{prefix}.selfattn.QKV.weight"].T.contiguous().to(torch.bfloat16), device)
        b["qkv_b"] = state[f"{prefix}.selfattn.QKV.bias"].to(torch.bfloat16)  # (960,)

        # O: (320, 320)
        b["o_w"] = to_tt(state[f"{prefix}.selfattn.O.weight"].T.contiguous().to(torch.bfloat16), device)
        b["o_b"] = state[f"{prefix}.selfattn.O.bias"].to(torch.bfloat16)  # (320,)

        # QK-norm weights
        b["lnq_w"] = state[f"{prefix}.selfattn.lnq.w"].to(torch.bfloat16)  # (16,)
        b["lnk_w"] = state[f"{prefix}.selfattn.lnk.w"].to(torch.bfloat16)  # (16,)

        # Modulation: silu -> linear (320, 1920) with bias
        # 1920 = 6 * 320 (mu1, sigma1, c1, mu2, sigma2, c2)
        b["mod_w"] = to_tt(state[f"{prefix}.modulation.1.weight"].T.contiguous().to(torch.bfloat16), device)
        b["mod_b"] = state[f"{prefix}.modulation.1.bias"].to(torch.bfloat16)  # (1920,)

        # GEGLU
        b["geglu_up_w"] = to_tt(state[f"{prefix}.geglu.up_proj.weight"].T.contiguous().to(torch.bfloat16), device)
        b["geglu_up_b"] = state[f"{prefix}.geglu.up_proj.bias"].to(torch.bfloat16)  # (1280,)
        b["geglu_gate_w"] = to_tt(state[f"{prefix}.geglu.up_gate.weight"].T.contiguous().to(torch.bfloat16), device)
        b["geglu_gate_b"] = state[f"{prefix}.geglu.up_gate.bias"].to(torch.bfloat16)  # (1280,)
        b["geglu_down_w"] = to_tt(state[f"{prefix}.geglu.down.weight"].T.contiguous().to(torch.bfloat16), device)
        b["geglu_down_b"] = state[f"{prefix}.geglu.down.bias"].to(torch.bfloat16)  # (320,)

        w["blocks"].append(b)

    # ============================================================
    # Precomputed constants for RMSNorm
    # ============================================================
    w["scaler"] = to_tt(torch.ones(TILE, TILE, dtype=torch.bfloat16), device)
    w["mean_scale_320"] = to_tt(torch.full((TILE, TILE), 1.0 / D_MODEL, dtype=torch.bfloat16), device)

    return w


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    print("Loading weights...")
    w = load_weights("/tmp/model.pt", device)

    print(f"Loaded {N_BLOCKS} blocks")
    print(f"Action embedding shape: {w['action_emb'].shape}")
    print(f"Time embedding table shape: {w['time_emb_pe'].shape}")
    print(f"RoPE sins shape: {w['rope_sins'].shape}")
    print(f"Register shape: {w['registers'].shape}")

    # Verify a block's weights
    b0 = w["blocks"][0]
    print(f"\nBlock 0:")
    print(f"  QKV weight (device tensor): {b0['qkv_w']}")
    print(f"  QKV bias: {b0['qkv_b'].shape}")
    print(f"  Modulation weight (device): {b0['mod_w']}")
    print(f"  GEGLU up weight (device): {b0['geglu_up_w']}")

    print("\nWeight loading successful!")

    ttnn.close_device(device)
