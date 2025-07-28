
import subprocess
import time
import os
import sys
import json

def run_single_test(test_name: str, audio_file: str, output_dir: str, device: str, use_multiprocess: bool):
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, os.path.basename(audio_file).replace('.wav', '.json'))

    print(f"--- Running test: {test_name} on {device.upper()} ---")
    print(f"Audio file: {audio_file}")

    if not os.path.exists(audio_file):
        print(f"ERROR: Audio file not found at {audio_file}")
        return

    command = [
        "allin1",
        audio_file,
        "-o", output_dir,
        "-d", device,
        "--overwrite",
    ]

    if not use_multiprocess:
        command.append("--no-multiprocess")

    print(f"Running command: {' '.join(command)}")

    log_file_path = "mps_test.log"
    with open(log_file_path, 'w') as log_file:
        start_time = time.time()
        env = os.environ.copy()
        env["NATTEN_API"] = "mps"
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=log_file, text=True, check=False, env=env)
        end_time = time.time()

    elapsed_time = end_time - start_time

    print(f"\n--- Results for {test_name} on {device.upper()} ---")
    print(f"Execution time: {elapsed_time:.2f} seconds")

    # Print command output for debugging
    print("\n--- Command Output ---")
    print("Stdout:")
    print(process.stdout)
    print("Stderr:")
    print(process.stderr)

    # Validate the output
    print("\n--- Output Validation ---")
    if not os.path.exists(output_file_path):
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
    Runs a series of tests with different backends.
    """
    audio_file = "/Users/stefm1/Music/WontDoSteveAustin.wav"
    base_output_dir = "test_output"
    
    # --- USER SETTING ---
    # Set to True to run the MPS test, False to run the CPU test.
    RUN_MPS_TEST = True

    if RUN_MPS_TEST:
        # Test: Run on MPS with demucs on CPU
        run_single_test(
            test_name="Hybrid MPS Backend Test",
            audio_file=audio_file,
            output_dir=os.path.join(base_output_dir, "mps_run"),
            device="mps",
            use_multiprocess=False
        )
    else:
        # Test: Run on CPU to confirm baseline
        run_single_test(
            test_name="CPU Backend Test",
            audio_file=audio_file,
            output_dir=os.path.join(base_output_dir, "cpu_run"),
            device="cpu",
            use_multiprocess=False
        )

if __name__ == "__main__":
    main()
