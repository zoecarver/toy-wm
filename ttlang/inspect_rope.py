"""Inspect stored RoPE tables to determine the formula."""
import torch

ckpt = torch.load("/tmp/model.pt", map_location="cpu", weights_only=False)
state = {k.replace("_orig_mod.", ""): v.to(torch.bfloat16) for k, v in ckpt.items()}

sins = state["blocks.0.selfattn.rope.sins"]
coss = state["blocks.0.selfattn.rope.coss"]
print("Shape:", sins.shape)
print("n_heads:", sins.shape[2])

D_HEAD = 16

# Print a few positions
for pos in [0, 1, 2, 10]:
    print(f"\nPos {pos}:")
    print(f"  sin: {sins[0, pos, 0, :].float().tolist()}")
    print(f"  cos: {coss[0, pos, 0, :].float().tolist()}")

# Try to infer frequencies from position 1
# sin(1 * freq_k) = sins[0, 1, 0, k]
sin1 = sins[0, 1, 0, :].float()
cos1 = coss[0, 1, 0, :].float()
angles_at_1 = torch.atan2(sin1, cos1)
print(f"\nInferred angles at pos 1: {angles_at_1.tolist()}")

# Check if pairs share the same angle (standard RoPE interleaving)
print("\nAngle pairs (even, odd):")
for i in range(0, D_HEAD, 2):
    print(f"  [{i},{i+1}]: {angles_at_1[i].item():.6f}, {angles_at_1[i+1].item():.6f}")

# Try standard RoPE formula: inv_freq = 1/(10000^(2k/d))
inv_freq_std = 1.0 / (10000.0 ** (torch.arange(0, D_HEAD, 2).float() / D_HEAD))
print(f"\nStandard inv_freq: {inv_freq_std.tolist()}")
print(f"Inferred inv_freq (from even indices): {angles_at_1[0::2].tolist()}")

# Try repeat_interleave vs repeat
sins_ri = torch.sin(torch.tensor([1.0]).unsqueeze(1) * inv_freq_std.unsqueeze(0)).repeat_interleave(2, dim=1)
sins_rp = torch.sin(torch.tensor([1.0]).unsqueeze(1) * inv_freq_std.unsqueeze(0))
sins_rp = sins_rp.repeat(1, 2)  # [s0,s1,...,s7,s0,s1,...,s7]

print(f"\nStored sin[1]:        {sin1.tolist()}")
print(f"repeat_interleave:    {sins_ri[0].tolist()}")
print(f"repeat (cat halves):  {sins_rp[0].tolist()}")

# Try different theta values
for theta in [10000, 1000, 100, 500, 5000]:
    inv_f = 1.0 / (theta ** (torch.arange(0, D_HEAD, 2).float() / D_HEAD))
    s = torch.sin(inv_f)
    err = (s.repeat_interleave(2) - sin1).abs().max().item()
    err2 = (torch.cat([s, s]) - sin1).abs().max().item()
    print(f"theta={theta:6d}: repeat_interleave max_err={err:.4f}, cat max_err={err2:.4f}")

# Check if all heads share the same table
if sins.shape[2] > 1:
    diff = (sins[0, :, 0, :] - sins[0, :, 1, :]).abs().max().item()
    print(f"\nHead 0 vs head 1 max diff: {diff}")
