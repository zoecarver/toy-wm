# Pong World Model on Tenstorrent

A diffusion transformer world model trained on 9 hours of pong, running interactively on Tenstorrent hardware. The main implementation is in [`ttlang_sample.py`](ttlang_sample.py).

<video src="https://github.com/zoecarver/toy-wm/raw/main/tt-toy-wm-4x.mov" controls width="480"></video>

## Setup

Requires a Tenstorrent device with `ttnn` and [TT-Lang](https://github.com/tenstorrent/tt-lang/tree/main) installed.

Copy the repo to your target machine and place the weights where the server expects them:

```bash
cp model.pt /tmp/model.pt
```

## Running

`play.py` serves an interactive browser UI. It must be in the same directory as `ttlang_sample.py`.

```bash
python play.py
```

Open http://localhost:8765 in your browser. Use arrow keys to control the paddle.

If the server is running inside Docker or on a remote machine, set up port forwarding first:

```bash
ssh -L 8765:localhost:8765 user@server
```

## Implementation

The model is an mmDiT-style frame-autoregressive diffusion transformer with 8 blocks, sampled via rectified flow matching. Each frame goes through multiple Euler denoising steps, where each step runs the full DiT forward pass (~35ms per step on a single Blackhole card). KV caching across frames means attention sees the full context window without recomputation.

### Forward pass structure

```
Input frame (24x24x3)
    |
    v
[patch_forward]  -- conv2d + SiLU + group_norm + patchify (host/PyTorch)
    |
    v
[conditioning]   -- time embedding + action embedding + SiLU (device)
    |
    v
 x8 DiT blocks:
    |-- [fused_norm_mod]    -- RMSNorm + weight * (1+sigma) + mu  (fused TT-Lang kernel)
    |-- [linear_bias]       -- QKV projection                     (TT-Lang)
    |-- [fused_norm_rope]   -- QK-norm + RoPE in one kernel       (fused TT-Lang kernel)
    |-- [sdpa]              -- scaled dot-product attention        (ttnn)
    |-- [linear_bias]       -- O projection                       (TT-Lang)
    |-- [gated_residual]    -- residual + x * gate                (TT-Lang)
    |-- [fused_norm_mod]    -- RMSNorm + AdaLN for FFN            (fused TT-Lang kernel)
    |-- [linear_bias]       -- GEGLU up+gate projection           (TT-Lang)
    |-- [silu_mul]          -- fused silu(gate) * up              (fused TT-Lang kernel)
    |-- [linear_bias]       -- GEGLU down projection              (TT-Lang)
    |-- [gated_residual]    -- residual + x * gate                (TT-Lang)
    |
    v
[final_mod]      -- final RMSNorm + AdaLN modulation (device)
    |
    v
[unpatch_forward] -- unpatchify + linear (host/PyTorch)
```

### TT-Lang kernels

All kernels use `grid="auto"` for automatic multi-core distribution and stream tiles through dataflow buffers.

| Kernel | What it does | Notes |
|--------|-------------|-------|
| `make_linear_kernel(k)` | Tiled matmul with configurable K-chunk | Parameterized factory; k=10 for d_model, k=40 for d_mid |
| `make_fused_linear_bias_kernel(k, n)` | Matmul + bias add | Fuses two ops into one kernel, batches N output columns to reuse X reads |
| `make_fused_norm_mod_kernel(d)` | RMSNorm + norm weight + AdaLN modulate | Fuses 4 ops (norm, weight mul, scale, shift) into one kernel with two passes over the data |
| `make_fused_norm_rope_kernel(offset)` | QK-norm + RoPE | Fuses RMSNorm + weight multiply + rotation via permutation matrix into one kernel. Three passes: norm stats, normalize + permute, combine with sin/cos |
| `silu_mul_kernel` | Fused SiLU gating for GEGLU | `silu(gate) * up` in one kernel instead of separate silu + mul |
| `adaln_modulate_kernel` | AdaLN: `x * (1 + scale) + shift` | Used for final modulation |
| `gated_residual_kernel` | `residual + x * gate` | Gated residual connection |
| `add_kernel` / `mul_kernel` / `silu_kernel` | Elementwise ops | Building blocks for conditioning path |
| `make_rmsnorm_kernel(d)` | Standalone RMSNorm | Accumulates sum-of-squares tile by tile, broadcasts rsqrt back |
| `mod_broadcast_all` | Broadcast 6 modulation params from row 0 | Single kernel replaces 6 separate broadcast launches |

### Key optimizations

- **Weight preloading**: All model weights are transferred to device DRAM once at startup (~200 tensors). Zero host-device weight transfers during inference.
- **Scratch buffer reuse**: ~30 pre-allocated device tensors are reused across all blocks and timesteps. No per-forward allocation.
- **Fused kernels**: The norm+modulate and norm+RoPE fusions each eliminate multiple DRAM round-trips by keeping intermediate values in L1 between passes.
- **GEGLU fusion**: `silu(gate) * up` runs as a single fused kernel with a wider tile granularity (10 tiles/block) to maximize throughput.
- **Device-side KV cache**: K/V tensors stay on device across frames, extended via `ttnn.concat`. Sliding window trimming via `ttnn.slice`.
- **RoPE tables**: Extended to 100K positions at startup using the model's exact frequency formula (C=5000), so positions never overflow during long generation runs.

### What still runs on host

- **Patch/unpatch**: The patch embedding uses conv2d + SiLU + group_norm, which are nonlinear and not yet ported. This means each Euler step round-trips through the host for patchify/unpatchify (~3ms/step).
- **Modulation broadcast**: TT-Lang kernels exist for this (`mod_broadcast_all`, `final_mod_broadcast`) but they have a reduce/broadcast bug that produces wrong values. Currently bypassed via a host round-trip (~8ms total across 8 blocks). The kernels are defined and ready to swap in once the bug is resolved.

## Files

- `ttlang_sample.py` -- sampling loop with all TT-Lang kernels, weight preloading, and the DiT forward pass
- `play.py` -- HTTP server that generates frames on TT hardware in response to player actions
- `model.pt` -- pretrained weights
- `ttlang/kernels/` -- standalone TT-Lang kernel implementations used during development

## Dependencies

- `torch`
- `ttnn`, `ttl`
- `Pillow` (for PNG encoding in `play.py`)
