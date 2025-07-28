import torch
from natten.functional import natten1d
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_na1d_unit():
    """
    This test uses small, randomly generated tensors to verify that the
    kernels can execute without crashing and produce outputs of the correct shape.
    """
    logging.info("--- 1D NATTEN Unit Test ---")

    # --- Configuration ---
    device = 'mps'
    batch_size = 1
    heads = 1
    length = 1
    dim = 1
    kernel_size = 1
    dilation = 1

    try:
        # --- Create Tensors ---
        logging.info("Creating random tensors...")
        query = torch.randn(batch_size, heads, length, dim, device=device)
        key = torch.randn(batch_size, heads, length, dim, device=device)
        value = torch.randn(batch_size, heads, length, dim, device=device)
        rpb = torch.randn(heads, 2 * kernel_size - 1, device=device)

        logging.info(f"Query: shape={query.shape}")
        logging.info(f"Key:   shape={key.shape}")
        logging.info(f"Value: shape={value.shape}")
        logging.info(f"RPB:   shape={rpb.shape}")

        # --- Run Kernel ---
        logging.info("Calling natten1d kernel...")
        output = natten1d(query, key, value, rpb, kernel_size, dilation, query.shape[2])
        torch.mps.synchronize() # Wait for the GPU to finish
        logging.info(f"NATTEN 1D kernel finished. Output shape: {output.shape}")

        # Verify output shape
        expected_shape = (value.shape[0], value.shape[2], value.shape[1] * value.shape[3])
        assert output.shape == expected_shape, f"Output shape mismatch! Expected {expected_shape}, got {output.shape}"

        logging.info("--- UNIT TEST SUCCEEDED ---")

    except Exception as e:
        logging.error("--- UNIT TEST FAILED ---")
        logging.error(f"An exception occurred: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    test_na1d_unit()
