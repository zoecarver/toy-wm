"""
Inspect model.pt to understand the weight structure and shapes.
"""
import torch

if __name__ == "__main__":
    ckpt = torch.load("/tmp/model.pt", map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    print(f"Type: {type(ckpt)}")
    if isinstance(ckpt, dict):
        print(f"Top-level keys: {list(ckpt.keys())[:20]}")

    if isinstance(state, dict):
        print(f"\nTotal parameters: {len(state)}")
        print(f"\nAll keys and shapes:")
        total_params = 0
        for k, v in sorted(state.items()):
            if hasattr(v, 'shape'):
                print(f"  {k}: {list(v.shape)} ({v.dtype})")
                total_params += v.numel()
            else:
                print(f"  {k}: {type(v)}")
        print(f"\nTotal parameter count: {total_params:,}")
        print(f"Total size (bf16): {total_params * 2 / 1024 / 1024:.1f} MB")
