"""
Test the Patch module pipeline using ttnn conv2d.

Patch pipeline:
  1. Conv2d(3, 160, 5x5, stride=1, pad=2) + SiLU
  2. GroupNorm(32, 160)
  3. Conv2d(160, 160, 5x5, stride=1, pad=2) + SiLU
  4. GroupNorm(32, 160)
  5. Patchify: (B, 160, 24, 24) -> (B, 64, 1440)  [8x8 patches of 3x3x160]
  6. x_embedder linear: (B, 64, 1440) -> (B, 64, 320)

First let's check what ttnn conv2d API looks like and if it works.
If ttnn conv doesn't work well, fallback: run patch on host (it's tiny).
"""

import torch
import torch.nn.functional as F
import ttnn

TILE = 32


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    # Test: can we do conv2d via ttnn?
    print("--- Testing ttnn conv2d ---")

    B, C_in, H, W = 1, 3, 24, 24
    C_out = 160
    K = 5

    x_torch = torch.randn(B, C_in, H, W, dtype=torch.bfloat16)
    w_torch = torch.randn(C_out, C_in, K, K, dtype=torch.bfloat16) * 0.01
    b_torch = torch.randn(C_out, dtype=torch.bfloat16) * 0.01

    # PyTorch reference
    ref = F.conv2d(x_torch.float(), w_torch.float(), b_torch.float(), padding=2)
    ref = ref.to(torch.bfloat16)
    print(f"  PyTorch conv2d output shape: {ref.shape}")

    # Try ttnn conv2d
    # ttnn.conv2d expects NHWC format typically
    try:
        # Convert to NHWC
        x_nhwc = x_torch.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)

        x_tt = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                               device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        print(f"  Input tensor on device: {x_tt.shape}")

        result_tt = ttnn.conv2d(
            input_tensor=x_tt,
            weight_tensor=ttnn.from_torch(w_torch, dtype=ttnn.bfloat16),
            bias_tensor=ttnn.from_torch(b_torch.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16),
            device=device,
            in_channels=C_in,
            out_channels=C_out,
            batch_size=B,
            input_height=H,
            input_width=W,
            kernel_size=(K, K),
            stride=(1, 1),
            padding=(2, 2),
        )

        print(f"  ttnn conv2d succeeded! Output: {result_tt}")
        out_host = ttnn.to_torch(result_tt)
        print(f"  Output shape: {out_host.shape}")

    except Exception as e:
        print(f"  ttnn conv2d failed: {e}")
        print("  Will use host-side conv2d as fallback for Patch module")
        print("  (Patch is tiny: 24x24 input, runs once per frame)")

    # Regardless of ttnn conv2d, test the full patch pipeline on host
    # to verify we understand the shapes
    print("\n--- Full Patch pipeline (host reference) ---")

    # Load real weights
    ckpt = torch.load("/tmp/model.pt", map_location="cpu", weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}

    conv1_w = state["patch.init_conv_seq.0.weight"].to(torch.bfloat16)
    conv1_b = state["patch.init_conv_seq.0.bias"].to(torch.bfloat16)
    gn1_w = state["patch.init_conv_seq.2.weight"].to(torch.bfloat16)
    gn1_b = state["patch.init_conv_seq.2.bias"].to(torch.bfloat16)
    conv2_w = state["patch.init_conv_seq.3.weight"].to(torch.bfloat16)
    conv2_b = state["patch.init_conv_seq.3.bias"].to(torch.bfloat16)
    gn2_w = state["patch.init_conv_seq.5.weight"].to(torch.bfloat16)
    gn2_b = state["patch.init_conv_seq.5.bias"].to(torch.bfloat16)
    embed_w = state["patch.x_embedder.weight"].to(torch.bfloat16)
    embed_b = state["patch.x_embedder.bias"].to(torch.bfloat16)

    # Random input frame
    frame = torch.randn(1, 3, 24, 24, dtype=torch.bfloat16)

    # Conv1 + SiLU
    x = F.conv2d(frame.float(), conv1_w.float(), conv1_b.float(), padding=2).to(torch.bfloat16)
    x = F.silu(x.float()).to(torch.bfloat16)
    print(f"  After conv1+silu: {x.shape}")  # (1, 160, 24, 24)

    # GroupNorm1
    x = F.group_norm(x.float(), 32, gn1_w.float(), gn1_b.float()).to(torch.bfloat16)
    print(f"  After groupnorm1: {x.shape}")

    # Conv2 + SiLU
    x = F.conv2d(x.float(), conv2_w.float(), conv2_b.float(), padding=2).to(torch.bfloat16)
    x = F.silu(x.float()).to(torch.bfloat16)
    print(f"  After conv2+silu: {x.shape}")  # (1, 160, 24, 24)

    # GroupNorm2
    x = F.group_norm(x.float(), 32, gn2_w.float(), gn2_b.float()).to(torch.bfloat16)
    print(f"  After groupnorm2: {x.shape}")

    # Patchify: (B, C, H, W) -> (B, n_patches, patch_dim)
    # patch_size=3, so 8x8=64 patches, each 3x3x160=1440
    ps = 3
    B_, C_, H_, W_ = x.shape
    x_patches = x.reshape(B_, C_, H_//ps, ps, W_//ps, ps)
    x_patches = x_patches.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)
    print(f"  After patchify: {x_patches.shape}")  # (1, 64, 1440)

    # x_embedder: linear (1440, 320)
    x_embed = (x_patches.float() @ embed_w.float().T + embed_b.float()).to(torch.bfloat16)
    print(f"  After x_embedder: {x_embed.shape}")  # (1, 64, 320)

    print(f"\n  Final patch output: {x_embed.shape}")
    print(f"  Values[0,0,:5]: {x_embed[0,0,:5].tolist()}")

    ttnn.close_device(device)
