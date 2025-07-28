import torch
import pytest
from natten.functional import natten1d
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def execute_natten_forward_pass(query, key, value, rpb, kernel_size, dilation):
    """Helper function to execute a full forward pass and synchronize."""
    try:
        # --- Run Kernel ---
        output = natten1d(query, key, value, rpb, kernel_size, dilation, query.shape[2])
        torch.mps.synchronize() # Force GPU to execute and finish
        logging.info("NATTEN 1D kernel finished successfully.")
        return True
    except Exception as e:
        logging.error(f"NATTEN forward pass failed with exception: {e}", exc_info=True)
        # We return False on exception, but a segfault will crash the process before this.
        return False

class TestNA1DStress:
    """
    Stress tests for 1D Neighborhood Attention designed to trigger
    low-level memory errors and segmentation faults by violating implicit
    assumptions in the C++/Metal backend.
    """

    @pytest.mark.parametrize("kernel_size", [3, 7])
    @pytest.mark.parametrize("dilation", [1, 2])
    @pytest.mark.parametrize("dim", [32, 64])
    def test_non_contiguous_inputs(self, kernel_size, dilation, dim):
        """
        CRITICAL TEST: This test checks for memory errors when input tensors are
        not contiguous in memory. The `permute` operation is a common way this
        happens in real models. A backend that does not correctly handle strides
        will likely crash here.
        """
        logging.info(f"--- Running non-contiguous input test (k={kernel_size}, d={dilation}, dim={dim}) ---")
        
        batch_size, heads, length = 2, 2, 64
        
        # Standard contiguous tensor with a different memory layout
        # (Batch, Length, Heads, Dim) 
        base_tensor = torch.randn(batch_size, length, heads, dim, device='mps')
        
        # Permute to match the expected NATTEN layout (Batch, Heads, Length, Dim)
        # This operation makes the tensor non-contiguous.
        query = base_tensor.permute(0, 2, 1, 3).requires_grad_(False)

        # Second, different base tensor for the key to ensure it's also non-contiguous
        key_base_tensor = torch.randn(batch_size, dim, heads, length, device='mps')
        key = key_base_tensor.permute(0, 2, 3, 1).requires_grad_(False)

        value = torch.randn(batch_size, heads, length, dim, device='mps')
        rpb = torch.randn(heads, 2 * kernel_size - 1, device='mps')

        assert not query.is_contiguous(), "Query tensor should be non-contiguous for this test."
        assert not key.is_contiguous(), "Key tensor should be non-contiguous for this test."

        # --- DEBUG PRINT ---
        print("--------- PYTHON EXPECTED STRIDES ---------")
        print(f"Query Strides (b, h, l, d): {query.stride()}")
        print(f"Key Strides (b, h, l, d):   {key.stride()}")
        print("-----------------------------------------")
        # -------------------

        logging.info("Executing forward pass with non-contiguous query and key...")
        success = execute_natten_forward_pass(query, key, value, rpb, kernel_size, dilation)
        assert success, "Forward pass failed for non-contiguous inputs."

