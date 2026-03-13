"""
Test ttnn SDPA scale parameter. The toy-wm model uses scale=1.0
because QK-norm handles scaling. Default SDPA uses 1/sqrt(d_head).
"""
import torch
import torch.nn.functional as F
import ttnn

if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    B, NH, SEQ, D = 1, 20, 64, 32
    q = torch.randn(B, NH, SEQ, D, dtype=torch.bfloat16) * 0.1
    k = torch.randn(B, NH, SEQ, D, dtype=torch.bfloat16) * 0.1
    v = torch.randn(B, NH, SEQ, D, dtype=torch.bfloat16) * 0.1

    def to_tt(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Reference with scale=1.0
    attn = (q.float() @ k.float().transpose(-2, -1)) * 1.0
    ref_s1 = (attn.softmax(dim=-1) @ v.float()).to(torch.bfloat16)

    # Reference with default scale = 1/sqrt(32)
    ref_default = F.scaled_dot_product_attention(q.float(), k.float(), v.float()).to(torch.bfloat16)

    # ttnn default
    out_default = ttnn.to_torch(ttnn.transformer.scaled_dot_product_attention(
        to_tt(q), to_tt(k), to_tt(v), is_causal=False))

    d_vs_s1 = (out_default.float() - ref_s1.float()).abs().max().item()
    d_vs_def = (out_default.float() - ref_default.float()).abs().max().item()
    print(f"ttnn default vs scale=1.0 ref: max_diff={d_vs_s1:.6f}")
    print(f"ttnn default vs 1/sqrt(d) ref: max_diff={d_vs_def:.6f}")

    # Try passing scale parameter
    try:
        out_s1 = ttnn.to_torch(ttnn.transformer.scaled_dot_product_attention(
            to_tt(q), to_tt(k), to_tt(v), is_causal=False, scale=1.0))
        d_s1 = (out_s1.float() - ref_s1.float()).abs().max().item()
        print(f"ttnn scale=1.0 vs scale=1.0 ref: max_diff={d_s1:.6f}")
    except Exception as e:
        print(f"scale=1.0 failed: {e}")

    # What about pre-scaling Q?
    # If we multiply Q by sqrt(d_head) before SDPA, it cancels the 1/sqrt(d) scaling
    import math
    prescale = math.sqrt(D)
    q_prescaled = (q.float() * prescale).to(torch.bfloat16)
    out_prescaled = ttnn.to_torch(ttnn.transformer.scaled_dot_product_attention(
        to_tt(q_prescaled), to_tt(k), to_tt(v), is_causal=False))
    d_pre = (out_prescaled.float() - ref_s1.float()).abs().max().item()
    print(f"Pre-scaled Q (Q*sqrt(d)) vs scale=1.0 ref: max_diff={d_pre:.6f}")

    ttnn.close_device(device)
