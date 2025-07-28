import torch
from pathlib import Path
from natten.functional import natten1dav

def debug_1d_av():
    """
    A minimal script to trigger the bug in the na1d_av kernel
    using the captured input tensors from the first layer of the model.
    """
    # --- Configuration ---
    device = 'mps'
    base_input_dir = Path("../all-in-one/tests/test_output")
    
    # --- Load Tensors ---
    attn_file = base_input_dir / f"layer_1_input_attn_{device}.pt"
    value_file = base_input_dir / f"layer_1_input_value_{device}.pt"

    print("--- Loading Tensors for Minimal Test ---")
    if not attn_file.exists() or not value_file.exists():
        print(f"ERROR: Input tensor files not found in {base_input_dir}")
        print("Please run `test_layers.py` in the all-in-one test suite first.")
        return

    attn = torch.load(attn_file).to(device)
    value = torch.load(value_file).to(device)

    print(f"Attention tensor: shape={attn.shape}, device={attn.device}")
    print(f"Value tensor: shape={value.shape}, device={value.device}")

    # These are the hardcoded parameters for the first 1D layer in the model
    kernel_size = 5
    dilation = 1

    # --- Run the Kernel ---
    print("\n--- Calling natten1dav kernel ---")
    try:
        output = natten1dav(attn, value, kernel_size, dilation)
        # Force a read of the output to ensure the kernel executes
        output.cpu()
        print("--- Kernel execution finished ---")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"ERROR: An exception occurred during kernel execution: {e}")

if __name__ == "__main__":
    debug_1d_av()
