"""Compare a single dit_forward between v1 and v2."""
import torch
import ttnn
import time

TILE = 32
D_MODEL = 320
D_HEAD = 16
HEIGHT = 24
WIDTH = 24

device = ttnn.open_device(device_id=0)
torch.manual_seed(42)

print("Loading weights...")
ckpt = torch.load("/tmp/model.pt", map_location="cpu", weights_only=False)
state = {k.replace("_orig_mod.", ""): v.to(torch.bfloat16) for k, v in ckpt.items()}

def to_tt(t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

scaler_tt = to_tt(torch.ones(TILE, TILE, dtype=torch.bfloat16))
mean_scale_tt = to_tt(torch.full((TILE, TILE), 1.0/D_MODEL, dtype=torch.bfloat16))

# Same input
torch.manual_seed(123)
z = torch.randn(1, 3, HEIGHT, WIDTH, dtype=torch.bfloat16)
action_idx = 2
t_val = 0.75  # some timestep

# V1 single forward
import sample as v1
print("\n=== V1 dit_forward ===")
out_v1, _ = v1.dit_forward(z.clone(), action_idx, t_val, state, device, scaler_tt, mean_scale_tt)
print(f"V1 shape: {out_v1.shape}")
print(f"V1 range: [{out_v1.min().item():.4f}, {out_v1.max().item():.4f}]")
print(f"V1 mean:  {out_v1.float().mean().item():.4f}")
print(f"V1 std:   {out_v1.float().std().item():.4f}")
print(f"V1[0,0,:4,:4]:\n{out_v1[0,0,:4,:4]}")

# V2 single forward
import sample_v2 as v2
print("\n=== V2 dit_forward ===")
mean_scale_16_tt = to_tt(torch.full((TILE, TILE), 1.0/D_HEAD, dtype=torch.bfloat16))
dev = v2.preload_weights(state, device)
scr = v2.prealloc_scratch(device)

# Build rope tables and pre-cache conditioning (same as sample_frame does)
rope_tables = v2.build_rope_tables(state, 0, device)
action_padded = torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16)
action_padded[0] = state["action_emb.weight"][action_idx]
action_tt = to_tt(action_padded)
ts_scaled = int(t_val * (1000 - 1))
time_pe = state["time_emb.pe"][ts_scaled]
cond_padded = torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16)
cond_padded[0] = time_pe
time_pe_tt = to_tt(cond_padded)

out_v2, _ = v2.dit_forward(z.clone(), time_pe_tt, action_tt, state, dev, scr, device,
                            scaler_tt, mean_scale_tt, mean_scale_16_tt, rope_tables)
print(f"V2 shape: {out_v2.shape}")
print(f"V2 range: [{out_v2.min().item():.4f}, {out_v2.max().item():.4f}]")
print(f"V2 mean:  {out_v2.float().mean().item():.4f}")
print(f"V2 std:   {out_v2.float().std().item():.4f}")
print(f"V2[0,0,:4,:4]:\n{out_v2[0,0,:4,:4]}")

# Compare
diff = (out_v1.float() - out_v2.float()).abs()
print(f"\n=== DIFF ===")
print(f"Max diff:  {diff.max().item():.4f}")
print(f"Mean diff: {diff.mean().item():.4f}")
print(f"Match: {'YES' if diff.max().item() < 0.5 else 'NO'}")

ttnn.close_device(device)
