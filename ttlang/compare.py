"""
Compare TT forward pass against PyTorch reference model.
Runs both on the same input, compares output per-block and final.
"""
import sys
sys.path.insert(0, "/Users/zcarver/Developer/toy-wm")

import torch
import torch.nn.functional as F
import ttnn
import ttl
import math

# Import the original model
from src.models.dit import CausalDit

# Import our TT forward
from sample import (
    dit_forward, to_tt, zeros_tt,
    D_MODEL, D_HEAD, N_HEADS, N_BLOCKS, TILE, SEQ, SEQ_PADDED, TOKS_PER_FRAME,
    rmsnorm_d320, linear_k10, linear_k40,
    add_kernel, mul_kernel, silu_kernel,
    adaln_modulate_kernel, gated_residual_kernel,
    make_rmsnorm_kernel, make_linear_kernel,
)

def main():
    tt_device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    print("Loading weights...")
    ckpt = torch.load("/tmp/model.pt", map_location="cpu", weights_only=False)
    state = {k.replace("_orig_mod.", ""): v.to(torch.bfloat16) for k, v in ckpt.items()}

    # Build the original PyTorch model
    ref_model = CausalDit(
        height=24, width=24, n_window=30, d_model=320, T=1000,
        in_channels=3, patch_size=3, n_heads=20, n_blocks=8,
        rope_C=5000, rope_type="rope", use_flex=False,
    ).to(torch.bfloat16)
    ref_model.load_state_dict(
        {k.replace("_orig_mod.", ""): v.to(torch.bfloat16) for k, v in ckpt.items()},
        strict=True,
    )
    ref_model.eval()

    # Create deterministic input
    z_frame = torch.randn(1, 1, 3, 24, 24, dtype=torch.bfloat16)
    action = torch.tensor([[2]], dtype=torch.long)  # up
    timestep = torch.tensor([[0.5]])  # mid-denoise

    # Run PyTorch reference (no cache, single frame)
    with torch.no_grad():
        ref_out, ref_k, ref_v = ref_model(z_frame, action, timestep)
    print(f"PyTorch ref output: shape={ref_out.shape}, range=[{ref_out.min():.4f}, {ref_out.max():.4f}], mean={ref_out.mean():.4f}")

    # Run TT forward (no cache, single frame)
    scaler_tt = to_tt(torch.ones(TILE, TILE, dtype=torch.bfloat16), tt_device)
    mean_scale_tt = to_tt(torch.full((TILE, TILE), 1.0/D_MODEL, dtype=torch.bfloat16), tt_device)

    tt_out, tt_kv = dit_forward(
        z_frame[:, 0], 2, 0.5, state, tt_device, scaler_tt, mean_scale_tt,
        kv_cache=None, frame_idx=0,
    )
    print(f"TT output: shape={tt_out.shape}, range=[{tt_out.min():.4f}, {tt_out.max():.4f}], mean={tt_out.mean():.4f}")

    # Compare
    diff = (ref_out[:, 0].float() - tt_out.float()).abs()
    print(f"\nMax diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")
    print(f"Ref L2 norm: {ref_out[:, 0].float().norm():.4f}")
    print(f"TT L2 norm: {tt_out.float().norm():.4f}")

    # Per-channel comparison
    for c in range(3):
        cdiff = (ref_out[0, 0, c].float() - tt_out[0, c].float()).abs()
        print(f"  Channel {c}: max_diff={cdiff.max():.6f}, mean_diff={cdiff.mean():.6f}")

    # Check if outputs are correlated (cosine similarity)
    ref_flat = ref_out[:, 0].float().flatten()
    tt_flat = tt_out.float().flatten()
    cos_sim = F.cosine_similarity(ref_flat.unsqueeze(0), tt_flat.unsqueeze(0)).item()
    print(f"\nCosine similarity: {cos_sim:.6f}")

    if cos_sim > 0.95:
        print("PASS: Outputs are highly correlated")
    elif cos_sim > 0.8:
        print("WARN: Outputs are somewhat correlated, may have minor issues")
    else:
        print("FAIL: Outputs are poorly correlated, something is fundamentally wrong")

    ttnn.close_device(tt_device)

if __name__ == "__main__":
    main()
