"""
Test if ttnn SDPA supports attention masks or if we can use
negative padding in K to mask out padding positions.
"""
import torch
import torch.nn.functional as F
import ttnn

TILE = 32

if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    # Simulate: Q has 65 tokens (padded to 96), KV has 130 tokens (padded to 160)
    # The 31 padding tokens in Q and 30 in KV should not affect the result
    B, NH, D = 1, 20, 32
    SEQ_Q = 65
    SEQ_KV = 130
    Q_PAD = 96
    KV_PAD = 160

    q = torch.randn(B, NH, SEQ_Q, D, dtype=torch.bfloat16) * 0.1
    k = torch.randn(B, NH, SEQ_KV, D, dtype=torch.bfloat16) * 0.1
    v = torch.randn(B, NH, SEQ_KV, D, dtype=torch.bfloat16) * 0.1

    # Reference: no padding
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float()).to(torch.bfloat16)

    # Test 1: zero padding (current approach)
    q_zpad = F.pad(q, (0, 0, 0, Q_PAD - SEQ_Q))
    k_zpad = F.pad(k, (0, 0, 0, KV_PAD - SEQ_KV))
    v_zpad = F.pad(v, (0, 0, 0, KV_PAD - SEQ_KV))

    def to_tt(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    out_zpad = ttnn.to_torch(ttnn.transformer.scaled_dot_product_attention(
        to_tt(q_zpad), to_tt(k_zpad), to_tt(v_zpad), is_causal=False))
    out_zpad = out_zpad[:, :, :SEQ_Q, :]

    d1 = (out_zpad.float() - ref.float()).abs().max().item()
    print(f"Zero-pad vs no-pad max diff: {d1:.6f}")

    # Test 2: fill K padding with large negative to get ~0 attention after softmax
    k_negpad = F.pad(k, (0, 0, 0, KV_PAD - SEQ_KV), value=0)
    k_negpad[:, :, SEQ_KV:, :] = -100.0  # large negative -> Q@K^T will be very negative -> softmax ~0
    v_negpad = F.pad(v, (0, 0, 0, KV_PAD - SEQ_KV))  # V padding doesn't matter if attn weight ~0

    out_negpad = ttnn.to_torch(ttnn.transformer.scaled_dot_product_attention(
        to_tt(q_zpad), to_tt(k_negpad), to_tt(v_negpad), is_causal=False))
    out_negpad = out_negpad[:, :, :SEQ_Q, :]

    d2 = (out_negpad.float() - ref.float()).abs().max().item()
    print(f"Neg-pad vs no-pad max diff: {d2:.6f}")

    print(f"\nref[0,0,0,:5]: {ref[0,0,0,:5].tolist()}")
    print(f"zpad[0,0,0,:5]: {out_zpad[0,0,0,:5].tolist()}")
    print(f"negpad[0,0,0,:5]: {out_negpad[0,0,0,:5].tolist()}")

    ttnn.close_device(device)
