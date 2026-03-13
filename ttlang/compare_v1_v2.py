"""Compare v1 and v2 sample_frame outputs on the same input."""
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

# V1
import sample as v1
print("\n=== V1 ===")
torch.manual_seed(123)
noise = torch.randn(1, 3, HEIGHT, WIDTH, dtype=torch.bfloat16)
t0 = time.time()
frame_v1, _ = v1.sample_frame(noise.clone(), 2, 8, 1.0, state, device, scaler_tt, mean_scale_tt)
print(f"V1 time: {time.time()-t0:.1f}s")
print(f"V1 range: [{frame_v1.min().item():.4f}, {frame_v1.max().item():.4f}]")
print(f"V1 mean:  {frame_v1.float().mean().item():.4f}")
print(f"V1 std:   {frame_v1.float().std().item():.4f}")
print(f"V1[0,0,:4,:4]:\n{frame_v1[0,0,:4,:4]}")

# V2
import sample_v2 as v2
print("\n=== V2 ===")
mean_scale_16_tt = to_tt(torch.full((TILE, TILE), 1.0/D_HEAD, dtype=torch.bfloat16))
dev = v2.preload_weights(state, device)
scr = v2.prealloc_scratch(device)
torch.manual_seed(123)
noise2 = torch.randn(1, 3, HEIGHT, WIDTH, dtype=torch.bfloat16)
t0 = time.time()
frame_v2, _ = v2.sample_frame(noise2.clone(), 2, 8, 1.0, state, dev, scr, device, scaler_tt, mean_scale_tt, mean_scale_16_tt)
print(f"V2 time: {time.time()-t0:.1f}s")
print(f"V2 range: [{frame_v2.min().item():.4f}, {frame_v2.max().item():.4f}]")
print(f"V2 mean:  {frame_v2.float().mean().item():.4f}")
print(f"V2 std:   {frame_v2.float().std().item():.4f}")
print(f"V2[0,0,:4,:4]:\n{frame_v2[0,0,:4,:4]}")

# Compare
diff = (frame_v1.float() - frame_v2.float()).abs()
print(f"\n=== DIFF ===")
print(f"Max diff:  {diff.max().item():.4f}")
print(f"Mean diff: {diff.mean().item():.4f}")
print(f"Match: {'YES' if diff.max().item() < 0.1 else 'NO'}")

ttnn.close_device(device)
