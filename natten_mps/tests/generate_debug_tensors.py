import os
import torch
import numpy as np
from pathlib import Path
from allin1.models import load_pretrained_model

def get_window_start(index, length, kernel_size, neighborhood_size, dilation):
    if dilation <= 1:
        return max(index - neighborhood_size, 0) + (index + neighborhood_size >= length) * (length - index - neighborhood_size - 1)
    ni = index - neighborhood_size * dilation
    if ni < 0:
        return index % dilation
    if index + neighborhood_size * dilation >= length:
        imodd = index % dilation
        a = int(length / dilation) * dilation
        b = length - a
        if imodd < b:
            return length - b + imodd - 2 * neighborhood_size * dilation
        return a + imodd - kernel_size * dilation
    return ni

def get_pb_start(index, length, kernel_size, neighborhood_size, dilation):
    if dilation <= 1:
        return neighborhood_size + (index < neighborhood_size) * (neighborhood_size - index) + (index + neighborhood_size >= length) * (length - index - 1 - neighborhood_size)
    if index - neighborhood_size * dilation < 0:
        return kernel_size - 1 - (index // dilation)
    if index + neighborhood_size * dilation >= length:
        return (length - index - 1) // dilation
    return neighborhood_size

def generate_mps_tensors():
    """
    This test runs the all-in-one model on the MPS device to capture the exact
    tensors that are fed into the second NATTEN call. This generates the
    input tensors required by the `test_na1d_integration.py` script.
    """
    print("--- Generating MPS Input Tensors for Debugging ---")

    # --- Configuration ---
    device = 'mps'
    audio_file_str = "../test_audio.wav"
    output_tensor_file = Path("../debug_tensors.pt")
    spec_byproduct_dir = Path('../temp_spec/')

    # --- Setup Paths ---
    audio_file = Path(audio_file_str)
    output_tensor_file.parent.mkdir(exist_ok=True)
    if output_tensor_file.exists():
        output_tensor_file.unlink()
        print(f"Removed existing tensor file: {output_tensor_file}")

    # --- Step 1: Load Spectrogram ---
    spec_file = spec_byproduct_dir / f"{audio_file.stem}.npy"
    if not spec_file.exists():
        print(f"ERROR: Spectrogram not found at {spec_file}")
        return

    print(f"Loading spectrogram from {spec_file}")
    specs = np.load(spec_file, allow_pickle=True)

    # --- Step 2: Load Model ---
    # Ensure dinat.py uses the MPS API logic
    os.environ['NATTEN_API'] = 'mps'
    print(f"--- Loading model on {device.upper()} ---")
    model = load_pretrained_model(model_name='harmonix-all', device=device)

    # --- Step 3: Manually trace the forward pass ---
    print("--- Manually executing model forward pass on MPS ---")
    with torch.no_grad():
        x = torch.from_numpy(specs).to(device)
        x = x.unsqueeze(0)

        N, K, T, F = x.shape
        x = x.reshape(-1, 1, T, F)
        x = model.models[0].embeddings(x)

        # --- First Attention Block ---
        print("Executing first attention block...")
        attention_block_1 = model.models[0].encoder.layers[0]
        x = attention_block_1(x)[0]
        print("First attention block finished.")

        # --- Second Attention Block ---
        print("Preparing for second attention block...")
        attention_block_2 = model.models[0].encoder.layers[1]

        hidden_states = x
        hidden_states = attention_block_2.timelayer.layernorm_before(hidden_states)
        
        # Manually replicate padding logic from _DinatLayerNd.forward
        if len(hidden_states.shape) > 3:
            _, K_b, T_b, _ = hidden_states.size()
            hidden_states, _ = attention_block_2.timelayer.maybe_pad(hidden_states, K_b, T_b)
        else:
            _, T_b, _ = hidden_states.shape
            hidden_states, _ = attention_block_2.timelayer.maybe_pad(hidden_states, T_b)

        attention_module = attention_block_2.timelayer.attention.self

        query_layer = attention_module.transpose_for_scores(attention_module.query(hidden_states))
        key_layer = attention_module.transpose_for_scores(attention_module.key(hidden_states))
        value_layer = attention_module.transpose_for_scores(attention_module.value(hidden_states))
        rpb = attention_module.rpb

        # --- Step 4: Save the tensors ---
        print(f"!!! SAVING DEBUG TENSORS to {output_tensor_file} !!!")
        torch.save({
            'query': query_layer.cpu(),
            'key': key_layer.cpu(),
            'value': value_layer.cpu(),
            'rpb': rpb.cpu(),
        }, output_tensor_file)
        print("!!! TENSORS SAVED. !!!")

        # --- Step 5: Generate ground-truth indices ---
        print("--- Generating ground-truth indices ---")
        batch_size, heads, length, dim = query_layer.shape
        kernel_size = attention_module.kernel_size
        dilation = attention_module.dilation
        neighborhood_size = kernel_size // 2

        sample_indices = torch.zeros(batch_size, heads, length, kernel_size, dtype=torch.int32)
        pi_indices = torch.zeros(batch_size, heads, length, kernel_size, dtype=torch.int32)

        for b in range(batch_size):
            for h in range(heads):
                for i in range(length):
                    for j in range(kernel_size):
                        sample_i = get_window_start(i, length, kernel_size, neighborhood_size, dilation) + j * dilation
                        pi = get_pb_start(i, length, kernel_size, neighborhood_size, dilation)
                        sample_indices[b, h, i, j] = sample_i
                        pi_indices[b, h, i, j] = pi
        
        ground_truth_indices_file = output_tensor_file.parent / "debug_ground_truth_indices.pt"
        torch.save({
            'sample_indices': sample_indices,
            'pi_indices': pi_indices,
        }, ground_truth_indices_file)
        print(f"--- Ground-truth indices saved to {ground_truth_indices_file} ---")

if __name__ == "__main__":
    generate_mps_tensors()
