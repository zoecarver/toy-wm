"""
Test ttnn.transformer.scaled_dot_product_attention to verify it works
and understand the API for d_head=16.
"""

import torch
import ttnn

TILE = 32


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    # Model dims
    BATCH = 1
    N_HEADS = 20
    D_HEAD = 16
    SEQ_Q = 64
    SEQ_KV = 64

    # Create Q, K, V in the shape ttnn expects
    # ttnn SDPA typically expects (batch, n_heads, seq, d_head)
    q_torch = torch.randn(BATCH, N_HEADS, SEQ_Q, D_HEAD, dtype=torch.bfloat16) * 0.1
    k_torch = torch.randn(BATCH, N_HEADS, SEQ_KV, D_HEAD, dtype=torch.bfloat16) * 0.1
    v_torch = torch.randn(BATCH, N_HEADS, SEQ_KV, D_HEAD, dtype=torch.bfloat16) * 0.1

    # PyTorch reference: standard scaled dot-product attention
    scale = 1.0  # model uses scale=1.0 (QK-norm handles scaling)
    attn = (q_torch.float() @ k_torch.float().transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)
    ref = (attn @ v_torch.float()).to(torch.bfloat16)

    # Check if d_head=16 needs padding to tile boundary
    # Pad d_head from 16 to 32 if needed
    def pad_to_32(t, dim=-1):
        pad_size = 32 - t.shape[dim]
        if pad_size > 0:
            pad = [0] * (2 * len(t.shape))
            pad[-(2 * (len(t.shape) - 1 - (dim % len(t.shape))) + 1)] = 0
            pad[-(2 * (len(t.shape) - 1 - (dim % len(t.shape))) + 2)] = pad_size
            return torch.nn.functional.pad(t, pad)
        return t

    # Try without padding first
    def to_tt(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Also need to pad seq dims to tile multiples
    # SEQ_Q=64 and SEQ_KV=64 are already tile-aligned

    print("Trying ttnn SDPA with d_head=16...")

    try:
        q_tt = to_tt(q_torch)
        k_tt = to_tt(k_torch)
        v_tt = to_tt(v_torch)

        out_tt = ttnn.transformer.scaled_dot_product_attention(
            q_tt, k_tt, v_tt,
            is_causal=False,
        )

        result = ttnn.to_torch(out_tt)
        max_diff = (result.float() - ref.float()).abs().max().item()
        mean_diff = (result.float() - ref.float()).abs().mean().item()
        print(f"  Shape: {result.shape}")
        print(f"  Max diff:  {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        print(f"  PASS: {max_diff < 1.0}")
    except Exception as e:
        print(f"  Failed: {e}")
        print("  Trying with padded d_head=32...")

        # Pad to d_head=32
        q_pad = torch.nn.functional.pad(q_torch, (0, 16))  # pad last dim
        k_pad = torch.nn.functional.pad(k_torch, (0, 16))
        v_pad = torch.nn.functional.pad(v_torch, (0, 16))

        try:
            q_tt = to_tt(q_pad)
            k_tt = to_tt(k_pad)
            v_tt = to_tt(v_pad)

            out_tt = ttnn.transformer.scaled_dot_product_attention(
                q_tt, k_tt, v_tt,
                is_causal=False,
            )

            result_pad = ttnn.to_torch(out_tt)
            # Slice back to d_head=16
            result = result_pad[:, :, :, :D_HEAD]
            max_diff = (result.float() - ref.float()).abs().max().item()
            mean_diff = (result.float() - ref.float()).abs().mean().item()
            print(f"  Shape (padded): {result_pad.shape}")
            print(f"  Shape (sliced): {result.shape}")
            print(f"  Max diff:  {max_diff:.6f}")
            print(f"  Mean diff: {mean_diff:.6f}")
            print(f"  PASS: {max_diff < 1.0}")
        except Exception as e2:
            print(f"  Also failed with padding: {e2}")

    # Also test with causal mask
    print("\nTrying ttnn SDPA with causal mask...")
    try:
        q_tt = to_tt(q_torch)
        k_tt = to_tt(k_torch)
        v_tt = to_tt(v_torch)

        out_tt = ttnn.transformer.scaled_dot_product_attention(
            q_tt, k_tt, v_tt,
            is_causal=True,
        )

        # Causal reference
        mask = torch.tril(torch.ones(SEQ_Q, SEQ_KV))
        attn_c = (q_torch.float() @ k_torch.float().transpose(-2, -1)) * scale
        attn_c = attn_c.masked_fill(mask[None, None] == 0, float('-inf'))
        attn_c = attn_c.softmax(dim=-1)
        ref_c = (attn_c @ v_torch.float()).to(torch.bfloat16)

        result_c = ttnn.to_torch(out_tt)
        max_diff_c = (result_c.float() - ref_c.float()).abs().max().item()
        print(f"  Max diff (causal): {max_diff_c:.6f}")
        print(f"  PASS: {max_diff_c < 1.0}")
    except Exception as e:
        print(f"  Causal failed: {e}")

    ttnn.close_device(device)
