import torch
from natten.functional import natten1d
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_na1d_numerical_comparison():
    """
    A focused integration test that uses real tensors captured from a model
    to verify the numerical correctness of the NATTEN 1D kernel against a
    pre-computed CPU ground truth.
    """
    logging.info("--- Starting 1D NATTEN Numerical Comparison Test ---")

    # --- Configuration ---
    device = 'mps'
    base_path = "/Users/stefm1/test/temp/clean_dev/temp_tensors"
    tensor_file = os.path.join(base_path, "debug_tensors.pt")
    cpu_out_file = os.path.join(base_path, "debug_tensors_cpu_out.pt")

    # --- Load Tensors ---
    logging.info(f"Loading input tensors from: {tensor_file}")
    try:
        tensors = torch.load(tensor_file)
        query = tensors['query'].to(device)
        key = tensors['key'].to(device)
        value = tensors['value'].to(device)
        rpb = tensors['rpb'].to(device)
    except FileNotFoundError:
        logging.error(f"FATAL: Debug input tensor file not found at {tensor_file}")
        raise

    logging.info(f"Loading CPU ground truth output from: {cpu_out_file}")
    try:
        # The CPU output should be loaded to the CPU, not the MPS device
        attn_cpu_out = torch.load(cpu_out_file).to('cpu')
    except FileNotFoundError:
        logging.error(f"FATAL: CPU output file not found at {cpu_out_file}")
        raise

    logging.info(f"Query tensor: shape={query.shape}, device={query.device}")
    logging.info(f"Key tensor:   shape={key.shape}, device={key.device}")
    logging.info(f"Value tensor: shape={value.shape}, device={value.device}")
    logging.info(f"RPB tensor:   shape={rpb.shape}, device={rpb.device}")
    logging.info(f"CPU output tensor: shape={attn_cpu_out.shape}, device={attn_cpu_out.device}")

    # Hardcoded parameters from the model's first 1D layer
    kernel_size = 5
    dilation = 1
    
    try:
        # --- Run Kernel ---
        logging.info("Calling natten1d kernel with real data...")
        attn_mps_out = natten1d(
            query, key, value, rpb, 
            kernel_size, dilation, query.shape[2]
        )
        torch.mps.synchronize()
        logging.info(f"NATTEN 1D kernel finished. MPS Output shape: {attn_mps_out.shape}")

        # --- Final Numerical Comparison ---
        logging.info("Comparing MPS kernel output against CPU ground truth...")
        
        # Move MPS output to CPU for comparison
        attn_mps_out_cpu = attn_mps_out.to('cpu')

        # Define tolerance levels. These are chosen based on typical precision
        # differences between CPU (float64) and GPU (float32/16).
        rtol = 1e-1
        atol = 1e-1

        is_close = torch.allclose(attn_mps_out_cpu, attn_cpu_out, rtol=rtol, atol=atol)

        if is_close:
            logging.info(f"✅✅✅ Test Passed: MPS output is numerically close to CPU output.")
            logging.info(f"   (rtol={rtol}, atol={atol})")
        else:
            logging.error(f"❌❌❌ Test Failed: MPS output is NOT numerically close to CPU output.")
            diff = torch.abs(attn_mps_out_cpu - attn_cpu_out)
            max_diff = torch.max(diff)
            mean_diff = torch.mean(diff)
            logging.error(f"   Max absolute difference: {max_diff.item()}")
            logging.error(f"   Mean absolute difference: {mean_diff.item()}")
            logging.error(f"   (rtol={rtol}, atol={atol})")
            assert False, "Numerical comparison failed"

        logging.info("--- NUMERICAL COMPARISON TEST SUCCEEDED ---")

    except Exception as e:
        logging.error(f"--- NUMERICAL COMPARISON TEST FAILED ---")
        logging.error(f"An exception occurred: {e}", exc_info=True)
        raise

if __name__ == "__main__":
     test_na1d_numerical_comparison()
