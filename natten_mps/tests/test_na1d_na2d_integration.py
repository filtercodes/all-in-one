import torch
import logging
from natten.functional import natten1d, natten2d

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_na1d_na2d_integration():
    """
    A targeted integration test to reproduce the command buffer crash.
    This test simulates the core logic of the model's encoder loop by:
    1. Calling the 1D NATTEN kernel.
    2. Permuting and reshaping the output tensor.
    3. Calling the 2D NATTEN kernel with the result.
    This sequence is known to cause an implicit synchronization and crash the MPS backend.
    """
    logging.info("--- Starting 1D -> 2D NATTEN Integration Test ---")

    # --- Configuration ---
    device = 'mps'
    batch_size = 2
    heads = 4
    time_steps = 128
    instruments = 4
    dim = 32

    # --- Tensors for 1D Attention ---
    logging.info("Creating tensors for 1D attention...")
    # Shape for 1D is (Batch * Instruments, Time, Dim)
    q_1d = torch.randn(batch_size * instruments, heads, time_steps, dim, device=device)
    k_1d = torch.randn(batch_size * instruments, heads, time_steps, dim, device=device)
    v_1d = torch.randn(batch_size * instruments, heads, time_steps, dim, device=device)
    rpb_1d = torch.randn(heads, 2 * 7 - 1, device=device)
    
    # --- Tensors for 2D Attention ---
    logging.info("Creating tensors for 2D attention...")
    # Shape for 2D is (Batch, Time, Instruments, Dim)
    q_2d = torch.randn(batch_size, heads, time_steps, instruments, dim, device=device)
    k_2d = torch.randn(batch_size, heads, time_steps, instruments, dim, device=device)
    v_2d = torch.randn(batch_size, heads, time_steps, instruments, dim, device=device)
    rpb_2d = torch.randn(heads, 2 * 5 - 1, 2 * 5 - 1, device=device)

    try:
        # --- Step 1: Execute 1D Attention ---
        logging.info("Executing 1D NATTEN kernel...")
        output_1d = natten1d(q_1d, k_1d, v_1d, rpb_1d, kernel_size=7, dilation=1, original_length=time_steps)
        logging.info(f"1D kernel finished. Output shape: {output_1d.shape}")

        # --- Step 2: Simulate the problematic Python-side transformation ---
        # This is the sequence from the model that causes the implicit sync.
        logging.info("Simulating Python-side permute and reshape...")
        # Reshape from (Batch * Instruments, Heads, Time, Dim) to (Batch, Instruments, Heads, Time, Dim)
        # Then permute and reshape for the 2D kernel
        intermediate = output_1d.reshape(batch_size, instruments, heads, time_steps, dim)
        intermediate = intermediate.permute(0, 2, 3, 1, 4) # (Batch, Heads, Time, Instruments, Dim)
        
        # --- Step 3: Execute 2D Attention ---
        logging.info("Executing 2D NATTEN kernel...")
        output_2d = natten2d(intermediate, k_2d, v_2d, rpb_2d, kernel_size=5, dilation=1, original_height=time_steps, original_width=instruments)
        torch.mps.synchronize() # Wait for the GPU to finish
        logging.info(f"2D kernel finished. Output shape: {output_2d.shape}")

        logging.info("--- INTEGRATION TEST SUCCEEDED ---")

    except Exception as e:
        logging.error("--- INTEGRATION TEST FAILED ---")
        logging.error(f"This test is expected to fail until Plan C is implemented.")
        logging.error(f"An exception occurred: {e}", exc_info=True)
        # We expect this to fail, so we'll re-raise to make the test runner see it.
        raise

if __name__ == "__main__":
    test_na1d_na2d_integration()
