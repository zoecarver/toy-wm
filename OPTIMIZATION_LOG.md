# Toy-WM Optimization Log

## Baseline
- **34ms/step**, 0.5s/frame (8 Euler steps, after warmup)
- 144+ kernel calls per step
- Each kernel ~0.05ms launch overhead

## Current Best
- **26ms/step**, 0.2s/frame (~24% improvement from baseline)

## What Worked

### 1. Fix mod_broadcast_all kernel (commit 4efa575)
- **Bug**: reduce_sum(dims=[0]) was summing all 32 tile rows when only row 0 had data
- **Fix**: Removed reduce_sum, just use broadcast(dims=[0]) directly
- **Impact**: Eliminated host round-trip for modulation (~8ms saved across 8 blocks)
- Also fixed final_mod_broadcast with same pattern

### 2. Device-side modulation broadcast (commit 4efa575)
- Replaced ttnn.to_torch / host reshape / ttnn.from_torch with mod_broadcast_all on device
- Pre-baked modulation biases into weight tensors at preload time

### 3. First-frame KV cache on device (commit 4efa575)
- Replaced ttnn.to_torch + ttnn.from_torch with ttnn.clone for initial KV cache
- Avoids host round-trip on first frame

### 4. Fuse final_mod path: 6 kernels to 3 (commit e55689e)
- linear_k10 + add_kernel -> linear_bias_k10 (reuse existing fused kernel)
- rmsnorm + weight_mul + adaln_modulate -> fused_norm_mod (reuse existing fused kernel)
- **Impact**: final_mod 1.1ms -> 0.6ms

### 5. Conditioning fusion (commit e55689e)
- linear_k10 + add_kernel -> linear_bias_k10 for conditioning path

### 6. Fuse linear_bias + gated_residual (pending commit)
- Created `make_fused_linear_bias_gated_res_kernel`: `out = residual + (x @ w + bias) * gate`
- Eliminates 16 kernel launches (2 per block x 8 blocks)
- Eliminates 16 DRAM round-trips (intermediate stays in L1 via DFB)
- **Impact**: 28ms/step -> 26ms/step
- Required 3-step decomposition in compute: matmul -> bias add -> gated residual
  (compiler can't handle `rv + (mmv + bv) * gv` in one expression due to copy_dst limitation)
- Applied to both O proj + gated_res1 (k20) and GEGLU down + gated_res2 (k40)

## What Failed

### 1. TT-Lang V extraction / attn output reshape kernels (reverted in 84a9728)
- Created custom kernels to replace ttnn.slice/reshape/permute for V extraction and attention output
- host_attn went from 5.7ms to 8.1ms (WORSE)
- **Root cause**: TT-Lang kernel launch overhead exceeded the cost of TTNN built-in permute/reshape ops
- **Lesson**: TTNN built-in ops beat TT-Lang for simple data movement (permute/reshape)

### 2. add_silu_kernel (not committed)
- Tried fusing `add_kernel + silu_kernel` into one: `out = silu(a + b)`
- **Compiler error**: `failed to legalize operation 'ttl.copy_dst'`
- **Root cause**: `s = a + b; o.store(s * sigmoid(s))` fails because using intermediate `s`
  twice (once for sigmoid, once for multiply) requires copy_dst which isn't supported
- **Lesson**: Can't reuse intermediate values in expressions involving sigmoid/silu
- The standalone silu_kernel `xv * sigmoid(xv)` works because `xv` is a DFB value, not an intermediate

## Compiler Gotchas

- **copy_dst limitation**: Expressions that reuse an intermediate value in both a math op and
  its argument (e.g., `s * sigmoid(s)` where `s = a + b`) fail. Split into separate DFB steps.
- **4+ DFB inputs in one expression** can trigger copy_dst. Decompose with intermediate DFBs.
- Intermediate DFBs in compute are fine (compute-local, data stays in L1, no DRAM).

## Tips & Tricks

- TTNN built-in ops (reshape, permute, slice) are fast for data movement - don't replace with TT-Lang
- Kernel launch overhead is ~0.05ms each, adds up across 144+ calls
- Profile with `--perf --hw` for DRAM/bandwidth metrics, `--auto-profile --hw` for per-line cycles
- Python timing with time.time() gives wall clock including launch overhead
- Each kernel call includes: compile check + dispatch + execute + sync
- Fusing kernels that share intermediate data is a win even if you need intermediate DFBs
  (L1 round-trip << DRAM round-trip)

## Remaining Bottlenecks (at 26ms/step, per block x8)

| Component | Time | Notes |
|-----------|------|-------|
| host_attn | 5.6ms | ttnn reshape/permute/SDPA - hard to optimize |
| geglu | 5.5ms | linear_bias(320->2560) + silu_mul + fused linear_bias_gated_res(1280->320) |
| modulation | 3.6ms | linear + broadcast |
| o_proj | 2.0ms | fused linear_bias_gated_res (O proj + gated_res1) |
| norm1_mod + norm2_mod | 3.5ms | fused norm+mod kernels |
| qkv_proj | 1.5ms | linear_bias matmul |

## Ideas Not Yet Tried

- Profile individual kernels with --perf to find if any are memory-bound vs compute-bound
- Fuse silu_mul into the preceding linear_bias for GEGLU (hard: need paired up/gate columns)
- Reduce N_STEPS (Euler steps) - trading quality for speed
- Batch multiple kernel calls where possible
- Check if norm_rope kernel could be optimized (it's 3-pass)
- Fuse modulation linear + broadcast into one kernel
- Eliminate unused scratch buffers (d320_b no longer used)
