"""Step-by-step comparison of v1 vs v2 dit_forward block 0."""
import torch
import torch.nn.functional as F
import ttnn
import time

TILE = 32
D_MODEL = 320
D_HEAD = 16
D_MID = 1280
HEIGHT = 24
WIDTH = 24
N_HEADS = 20
SEQ_PADDED = 96
SEQ = 65
TOKS_PER_FRAME = 65

device = ttnn.open_device(device_id=0)
torch.manual_seed(42)

print("Loading weights...")
ckpt = torch.load("/tmp/model.pt", map_location="cpu", weights_only=False)
state = {k.replace("_orig_mod.", ""): v.to(torch.bfloat16) for k, v in ckpt.items()}

def to_tt(t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def tt2t(t):
    return ttnn.to_torch(t)

def stat(name, t):
    if hasattr(t, 'shape') and not isinstance(t, ttnn.Tensor):
        f = t.float()
    else:
        f = tt2t(t).float()
    print(f"  {name:30s}: mean={f.mean().item():10.4f}  std={f.std().item():10.4f}  max={f.abs().max().item():10.4f}  shape={list(f.shape)}")

def compare(name, t1, t2):
    if isinstance(t1, ttnn.Tensor):
        t1 = tt2t(t1)
    if isinstance(t2, ttnn.Tensor):
        t2 = tt2t(t2)
    f1, f2 = t1.float(), t2.float()
    # Handle different shapes by trimming to common
    if f1.shape != f2.shape:
        print(f"  {name:30s}: SHAPE MISMATCH v1={list(f1.shape)} v2={list(f2.shape)}")
        return
    diff = (f1 - f2).abs()
    print(f"  {name:30s}: max_diff={diff.max().item():10.4f}  mean_diff={diff.mean().item():10.4f}  {'OK' if diff.max().item() < 0.1 else 'DIVERGED'}")

scaler_tt = to_tt(torch.ones(TILE, TILE, dtype=torch.bfloat16))
mean_scale_tt = to_tt(torch.full((TILE, TILE), 1.0/D_MODEL, dtype=torch.bfloat16))

# Same input
torch.manual_seed(123)
z_frame = torch.randn(1, 3, HEIGHT, WIDTH, dtype=torch.bfloat16)
action_idx = 2
t_val = 0.75

import sample as v1
import sample_v2 as v2

# ===== CONDITIONING =====
print("\n=== CONDITIONING ===")
# V1 conditioning
ts_scaled = int(t_val * (1000 - 1))
time_pe = state["time_emb.pe"][ts_scaled]
action_emb = state["action_emb.weight"][action_idx]

cond_padded = torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16)
cond_padded[0] = time_pe
mixer_w = to_tt(state["time_emb_mixer.weight"].T.contiguous())
mixer_out_v1 = v1.zeros_tt((TILE, D_MODEL), device)
v1.linear_k10(to_tt(cond_padded), mixer_w, mixer_out_v1)
combined = torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16)
combined[0] = state["time_emb_mixer.bias"] + action_emb
cond_v1 = v1.zeros_tt((TILE, D_MODEL), device)
v1.add_kernel(mixer_out_v1, to_tt(combined), cond_v1)
cond_host_v1 = tt2t(cond_v1)
cond_vec_v1 = cond_host_v1[0:1]
stat("v1 cond_vec", cond_vec_v1)

# V2 conditioning
mean_scale_16_tt = to_tt(torch.full((TILE, TILE), 1.0/D_HEAD, dtype=torch.bfloat16))
dev = v2.preload_weights(state, device)
scr = v2.prealloc_scratch(device)

time_pe_tt = to_tt(cond_padded)  # same cond_padded
action_padded = torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16)
action_padded[0] = action_emb
action_tt = to_tt(action_padded)

v2.linear_k10(time_pe_tt, dev['mixer_w'], scr['cond_a'])
v2.add_kernel(scr['cond_a'], dev['mixer_bias'], scr['cond_b'])
v2.add_kernel(scr['cond_b'], action_tt, scr['cond_a'])
stat("v2 cond (pre-silu)", scr['cond_a'])
compare("cond (pre-silu)", cond_v1, scr['cond_a'])

# V1 does silu on host for modulation, V2 does silu on device
v2.silu_kernel(scr['cond_a'], scr['cond_silu'])

# ===== PATCH =====
print("\n=== PATCH ===")
patched = v1.patch_forward(z_frame, state)
reg = state["registers"].unsqueeze(0)
patched = torch.cat([patched, reg], dim=1)
z_2d = torch.zeros(SEQ_PADDED, D_MODEL, dtype=torch.bfloat16)
z_2d[:SEQ] = patched.squeeze(0)
z_tt_v1 = to_tt(z_2d)
z_tt_v2 = to_tt(z_2d)  # same patched input
stat("z_patched", z_2d)

# ===== BLOCK 0: MODULATION =====
print("\n=== BLOCK 0: MODULATION ===")
prefix = "blocks.0"

# V1 modulation
cond_silu_v1 = (cond_vec_v1.float() * torch.sigmoid(cond_vec_v1.float())).to(torch.bfloat16)
cond_silu_padded = torch.zeros(TILE, D_MODEL, dtype=torch.bfloat16)
cond_silu_padded[0] = cond_silu_v1[0]
mod_w = to_tt(state[f"{prefix}.modulation.1.weight"].T.contiguous())
mod_out_v1 = v1.zeros_tt((TILE, 1920), device)
v1.linear_k10(to_tt(cond_silu_padded), mod_w, mod_out_v1)
mod_host_v1 = tt2t(mod_out_v1)
mod_host_v1[0] = mod_host_v1[0] + state[f"{prefix}.modulation.1.bias"]
chunks_v1 = mod_host_v1[0, :D_MODEL*6].reshape(6, D_MODEL)
mu1_v1 = chunks_v1[0]
sigma1_v1 = chunks_v1[1]
stat("v1 mu1", mu1_v1)
stat("v1 sigma1", sigma1_v1)

# V2 modulation
# First check silu output matches
cond_silu_v2_h = tt2t(scr['cond_silu'])
stat("v1 cond_silu row0", cond_silu_v1)
stat("v2 cond_silu row0", cond_silu_v2_h[0:1])
compare("cond_silu row0", cond_silu_v1, cond_silu_v2_h[0:1])

v2.linear_k10(scr['cond_silu'], dev[f'{prefix}.mod_w'], scr['mod_a'])
# Compare mod linear output before broadcast
mod_v2_h = tt2t(scr['mod_a'])
stat("v1 mod_linear row0", mod_host_v1[0:1, :320])
stat("v2 mod_linear row0", mod_v2_h[0:1, :320])
compare("mod_linear row0 [:320]", mod_host_v1[0:1, :320], mod_v2_h[0:1, :320])
v2.mod_broadcast_all(scr['mod_a'],
    dev[f'{prefix}.mod_bias_mu1'], dev[f'{prefix}.mod_bias_sigma1'], dev[f'{prefix}.mod_bias_c1'],
    dev[f'{prefix}.mod_bias_mu2'], dev[f'{prefix}.mod_bias_sigma2'], dev[f'{prefix}.mod_bias_c2'],
    scaler_tt,
    scr['mu1'], scr['sigma1'], scr['c1'], scr['mu2'], scr['sigma2'], scr['c2'])

# Check bias tensors
bias_v1 = state[f"{prefix}.modulation.1.bias"][:D_MODEL]  # mu1 bias
bias_v2_h = tt2t(dev[f'{prefix}.mod_bias_mu1'])
stat("v1 mu1_bias", bias_v1)
stat("v2 mu1_bias[0]", bias_v2_h[0:1])
compare("mu1_bias row0", bias_v1.unsqueeze(0), bias_v2_h[0:1])

# Check what broadcast kernel produces
mu1_v2_h = tt2t(scr['mu1'])
sigma1_v2_h = tt2t(scr['sigma1'])
stat("v2 mu1[0]", mu1_v2_h[0:1])
stat("v2 sigma1[0]", sigma1_v2_h[0:1])

# Expected mu1 = linear_row0 + bias
expected_mu1 = tt2t(scr['mod_a'])[0, :D_MODEL] + bias_v1
stat("expected mu1", expected_mu1)
compare("v2 mu1[0] vs expected", expected_mu1.unsqueeze(0), mu1_v2_h[0:1])

# Compare mu1 row 0
print("  mu1 row comparison (v1 vs v2):")
compare("mu1 row0", mu1_v1.unsqueeze(0), mu1_v2_h[0:1])

# ===== BLOCK 0: NORM1 + MOD =====
print("\n=== BLOCK 0: NORM1_MOD ===")
# V1
norm1_out_v1 = v1.zeros_tt((SEQ_PADDED, D_MODEL), device)
v1.rmsnorm_d320(z_tt_v1, scaler_tt, mean_scale_tt, norm1_out_v1)
stat("v1 rmsnorm1", norm1_out_v1)

nw = state[f"{prefix}.norm1.w"].unsqueeze(0).expand(SEQ_PADDED, -1).contiguous()
norm1_w_v1 = v1.zeros_tt((SEQ_PADDED, D_MODEL), device)
v1.mul_kernel(norm1_out_v1, to_tt(nw), norm1_w_v1)

mu1_e_v1 = v1.expand_per_frame(mu1_v1.unsqueeze(0), TOKS_PER_FRAME, 1, SEQ_PADDED)
sigma1_e_v1 = v1.expand_per_frame(sigma1_v1.unsqueeze(0), TOKS_PER_FRAME, 1, SEQ_PADDED)
z_mod_v1 = v1.zeros_tt((SEQ_PADDED, D_MODEL), device)
v1.adaln_modulate_kernel(norm1_w_v1, to_tt(mu1_e_v1), to_tt(sigma1_e_v1), z_mod_v1)
stat("v1 z_mod (norm1+mod)", z_mod_v1)

# V2
v2.fused_norm_mod_d320(z_tt_v2, dev[f'{prefix}.norm1_w'], scr['mu1'], scr['sigma1'],
                        scaler_tt, mean_scale_tt, scr['d320_a'])
stat("v2 z_mod (fused norm1+mod)", scr['d320_a'])
compare("norm1_mod output", z_mod_v1, scr['d320_a'])

# ===== BLOCK 0: QKV PROJ =====
print("\n=== BLOCK 0: QKV_PROJ ===")
# V1
qkv_w_v1 = to_tt(state[f"{prefix}.selfattn.QKV.weight"].T.contiguous())
qkv_out_v1 = v1.zeros_tt((SEQ_PADDED, 960), device)
v1.linear_k10(z_mod_v1, qkv_w_v1, qkv_out_v1)
qkv_b_e = v1.expand_bias(state[f"{prefix}.selfattn.QKV.bias"], SEQ_PADDED)
qkv_biased_v1 = v1.zeros_tt((SEQ_PADDED, 960), device)
v1.add_kernel(qkv_out_v1, to_tt(qkv_b_e), qkv_biased_v1)
stat("v1 qkv_biased", qkv_biased_v1)

# V2
v2.linear_bias_k10(scr['d320_a'], dev[f'{prefix}.qkv_w'], dev[f'{prefix}.qkv_bias'], scr['qkv_out'])
stat("v2 qkv_out", scr['qkv_out'])

# Compare Q part (first 320 cols in v1, first 640 cols in v2 due to padding)
qkv_v1_h = tt2t(qkv_biased_v1)
qkv_v2_h = tt2t(scr['qkv_out'])
print("  V1 QKV shape:", qkv_v1_h.shape, "V2 QKV shape:", qkv_v2_h.shape)
# V1 has (96, 960) = 20 heads * 16 per head, V2 has (96, 1920) = 20 heads * 32 per head (padded)
# Compare Q head 0: v1 cols 0:16, v2 cols 0:16 (first 16 of 32)
q_v1_h0 = qkv_v1_h[:, 0:D_HEAD]
q_v2_h0 = qkv_v2_h[:, 0:D_HEAD]
compare("Q head0 [:, 0:16]", q_v1_h0, q_v2_h0)

print("\n=== DONE (block 0 comparison) ===")
ttnn.close_device(device)
