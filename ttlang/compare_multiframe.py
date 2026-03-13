"""Compare v1 and v2 across 2 frames to verify RoPE offset fix."""
import torch
import torch.nn.functional as F
import ttnn
import time

TILE = 32
D_MODEL = 320
D_HEAD = 16
D_HEAD_PAD = 32
N_HEADS = 20
HEIGHT = 24
WIDTH = 24
N_BLOCKS = 8
TOKS_PER_FRAME = 65
SEQ = 65
SEQ_PADDED = 96

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
mean_scale_16_tt = to_tt(torch.full((TILE, TILE), 1.0/D_HEAD, dtype=torch.bfloat16))

import sample as v1
import sample_v2 as v2

dev = v2.preload_weights(state, device)
scr = v2.prealloc_scratch(device)
v2.extend_rope_tables(state)

N_STEPS = 8
CFG = 1.0

v1_kv = None
v2_kv = None

for fidx in range(3):
    torch.manual_seed(100 + fidx)
    noise = torch.randn(1, 3, HEIGHT, WIDTH, dtype=torch.bfloat16)

    print(f"\n=== Frame {fidx} ===")

    # V1
    v1_cached = 0 if v1_kv is None else v1_kv[0]['k'].shape[1]
    print(f"V1 cache: {v1_cached} tokens")
    frame_v1, nkv1 = v1.sample_frame(
        noise.clone(), 2, N_STEPS, CFG, state, device, scaler_tt, mean_scale_tt,
        kv_cache=v1_kv, frame_idx=fidx)
    v1_kv = v1.extend_kv_cache(v1_kv, nkv1, 30)

    # V2
    v2_cached = 0 if v2_kv is None else v2_kv[0]['k'].shape[2] // SEQ_PADDED
    print(f"V2 cache: {v2_cached} frames")
    frame_v2, nkv2 = v2.sample_frame(
        noise.clone(), 2, N_STEPS, CFG, state, dev, scr, device, scaler_tt, mean_scale_tt,
        mean_scale_16_tt, device_kv_cache=v2_kv, frame_idx=fidx)
    v2_kv = v2.trim_kv_cache(nkv2, 30)

    diff = (frame_v1.float() - frame_v2.float()).abs()
    print(f"Max diff: {diff.max().item():.4f}  Mean diff: {diff.mean().item():.4f}")
    print(f"V1 mean: {frame_v1.float().mean().item():.6f}  V2 mean: {frame_v2.float().mean().item():.6f}")

print(f"\nFinal V1 cache: {v1_kv[0]['k'].shape[1]} tokens")
print(f"Final V2 cache: {v2_kv[0]['k'].shape[2]} padded tokens")

ttnn.close_device(device)
