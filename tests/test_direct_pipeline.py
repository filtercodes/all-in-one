import os
import json
import shutil
from pathlib import Path
import torch
import time

from allin1.demix import demix
from allin1.spectrogram import extract_spectrograms
from allin1.models import load_pretrained_model
from allin1.helpers import run_inference, save_results

def run_allin1_direct(test_name: str, audio_file_str: str, output_dir_str: str, device: str):
    """
    Runs the all-in-one pipeline by directly calling the library functions.
    """
    print(f"--- Running test: {test_name} on {device.upper()} ---")
    start_time = time.time()

    audio_file = Path(audio_file_str)
    output_dir = Path(output_dir_str)
    demix_byproduct_dir = Path('./temp_demix')
    spec_byproduct_dir = Path('./temp_spec')

    print(f"--- Preparing for {test_name} ---")
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Removed existing output directory: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Audio file: {audio_file}")
    if not audio_file.exists():
        print(f"ERROR: Audio file not found at {audio_file}")
        return

    demix_device = 'cpu' if device == 'mps' else device
    inference_device = device

    # --- Step 1: Run Demucs for source separation (if needed) ---
    print(f"--- Running Demucs on {demix_device.upper()} ---")
    if demix_byproduct_dir.exists():
        print(f"Found existing demixed files in {demix_byproduct_dir}, skipping demix.")
        demix_paths = {
            'bass': demix_byproduct_dir / audio_file.name.replace('.wav', '') / 'bass.wav',
            'drums': demix_byproduct_dir / audio_file.name.replace('.wav', '') / 'drums.wav',
            'other': demix_byproduct_dir / audio_file.name.replace('.wav', '') / 'other.wav',
            'vocals': demix_byproduct_dir / audio_file.name.replace('.wav', '') / 'vocals.wav',
        }
    else:
        demix_paths = demix([audio_file], demix_byproduct_dir, demix_device)
    print("Demucs complete.")

    # --- Step 2: Extract spectrograms (CPU operation, if needed) ---
    print("--- Extracting Spectrograms ---")
    if spec_byproduct_dir.exists():
        print(f"Found existing spectrograms in {spec_byproduct_dir}, skipping extraction.")
        spec_paths = [spec_byproduct_dir / audio_file.with_suffix('.npy').name]
    else:
        spec_paths = extract_spectrograms(demix_paths, spec_byproduct_dir, multiprocess=False)
    print("Spectrogram extraction complete.")

    # --- Step 3: Load the model onto the inference device ---
    print(f"--- Loading model on {inference_device.upper()} ---")
    model = load_pretrained_model(
        model_name='harmonix-all',
        device=inference_device,
    )
    print("Model loading complete.")

    # --- Step 4: Run inference ---
    print(f"--- Running Inference on {inference_device.upper()} ---")
    results = []
    with torch.no_grad():
        for path, spec_path in zip([audio_file], spec_paths):
            print(f"Analyzing {path.name}...")
            result = run_inference(
                path=path,
                spec_path=spec_path,
                model=model,
                device=inference_device,
                include_activations=False,
                include_embeddings=False,
            )
            results.append(result)
    print("Inference complete.")

    # --- Step 5: Save the results ---
    print(f"--- Saving results to {output_dir} ---")
    save_results(results, output_dir)
    print("Results saved.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n--- Results for {test_name} on {device.upper()} ---")
    print(f"Execution time: {elapsed_time:.2f} seconds")

    # --- Step 6: Validate the output ---
    print("\n--- Output Validation ---")
    output_file_path = output_dir / audio_file.with_suffix('.json').name
    if not output_file_path.exists():
        print(f"FAILURE: Output file not found at {output_file_path}")
        return

    try:
        with open(output_file_path, 'r') as f:
            data = json.load(f)
        
        beats = data.get('beats', [])
        downbeats = data.get('downbeats', [])

        if beats and downbeats:
            print("SUCCESS: 'beats' and 'downbeats' arrays are present and not empty.")
        else:
            print("FAILURE: 'beats' or 'downbeats' array is missing or empty.")
            if not beats:
                print("- 'beats' array is empty.")
            if not downbeats:
                print("- 'downbeats' array is empty.")

    except json.JSONDecodeError:
        print(f"FAILURE: Could not decode JSON from {output_file_path}")
    except Exception as e:
        print(f"FAILURE: An error occurred during validation: {e}")


def main():
    """
    Runs a series of tests with different backends by calling library functions directly.
    """
    audio_file = "/Users/stefm1/Music/WontDoSteveAustin.wav"
    base_output_dir = "test_output"
    
    # --- USER SETTING ---
    # Set to True to run the MPS test, False to run the CPU test.
    RUN_MPS_TEST = True

    if RUN_MPS_TEST:
        run_allin1_direct(
            test_name="Hybrid MPS Backend Direct Call Test",
            audio_file_str=audio_file,
            output_dir_str=os.path.join(base_output_dir, "mps_direct_run"),
            device="mps"
        )
    else:
        run_allin1_direct(
            test_name="CPU Backend Direct Call Test",
            audio_file_str=audio_file,
            output_dir_str=os.path.join(base_output_dir, "cpu_direct_run"),
            device="cpu"
        )

if __name__ == "__main__":
    main()
