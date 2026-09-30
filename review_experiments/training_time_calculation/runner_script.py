import sys
import subprocess
import time
import torch
import pandas as pd

# Import necessary components from your original script to get model info
from training_time_calculation_code import CONFIG, DementiaModel

# ================================================================================
# AUTOMATION CONFIGURATION
# ================================================================================
# This is the name of your existing script
SCRIPT_TO_RUN = "training_time_calculation_code.py"

# Define the experiments you want to run
LOSS_FUNCTIONS_TO_TEST = ["WCE", "FA_FL", "FocalLoss", "LDAM"]
SEEDS_TO_TEST = [42, 256]


# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

def get_gpu_info():
    """Returns the name of the GPU if available, otherwise 'CPU'."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


def get_model_params(model):
    """Calculates and returns the total and trainable parameters of a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def modify_config_in_file(file_path, new_loss, new_seed):
    """
    Reads the script file, replaces the loss and seed values in the CONFIG dict,
    and writes the changes back to the file.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find and replace the relevant lines in the CONFIG dictionary
    for i, line in enumerate(lines):
        if '"TYPE":' in line and "# Options:" in lines[i - 1]:
            lines[i] = f'        "TYPE": "{new_loss}",\n'
        if '"RANDOM_STATE":' in line:
            lines[i] = f'    "RANDOM_STATE": {new_seed},\n'

    with open(file_path, 'w') as f:
        f.writelines(lines)


# ================================================================================
# MAIN EXECUTION SCRIPT
# ================================================================================

if __name__ == '__main__':
    # --- 1. Get Static System and Model Information ---
    gpu_model = get_gpu_info()
    # Instantiate a model once to get parameter counts
    temp_model = DementiaModel(num_classes=len(CONFIG["NEW_CLASSES"]))
    total_params, trainable_params = get_model_params(temp_model)
    del temp_model  # Free up memory

    # --- 2. Run the Experiment Loop ---
    results = []
    total_runs = len(LOSS_FUNCTIONS_TO_TEST) * len(SEEDS_TO_TEST)
    run_count = 0

    print("Starting automated experiment suite...")
    print("-" * 50)

    for loss_type in LOSS_FUNCTIONS_TO_TEST:
        for seed in SEEDS_TO_TEST:
            run_count += 1
            print(f"[{run_count}/{total_runs}] Configuring for: Loss={loss_type}, Seed={seed}...")

            # Dynamically modify the CONFIG in the script file
            modify_config_in_file(SCRIPT_TO_RUN, loss_type, seed)

            print(f"--> Running {SCRIPT_TO_RUN}...")
            start_time = time.time()

            # Execute the script as a separate process and capture output
            # We hide the extensive output by capturing it
            process = subprocess.run(
                [sys.executable, SCRIPT_TO_RUN],
                capture_output=True,
                text=True
            )

            # Optional: Check for errors during the run
            if process.returncode != 0:
                print(f"!! ERROR running experiment: Loss={loss_type}, Seed={seed}")
                print(process.stderr)

            end_time = time.time()
            duration_seconds = end_time - start_time

            # Store results
            results.append({
                "Loss Function": loss_type,
                "Seed": seed,
                "Duration (s)": round(duration_seconds, 2)
            })

    print("-" * 50)
    print("All experiments completed.\n")

    # --- 3. Optional: Restore original config ---
    print("Restoring original configuration in script...")
    modify_config_in_file(SCRIPT_TO_RUN, "FA_FL", 256)  # Change to your preferred defaults

    # --- 4. Print the Final Summary ---
    print("=" * 30)
    print("   EXPERIMENT SUMMARY")
    print("=" * 30)
    print(f"GPU Model: {gpu_model}")
    print(f"Model Parameters (Total):     {total_params:,}")
    print(f"Model Parameters (Trainable): {trainable_params:,}")
    print("-" * 30)

    # Use pandas for a clean, formatted table
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("=" * 30)