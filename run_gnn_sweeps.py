import wandb
import subprocess
import json
import os
import time
import signal

# Define datasets and convolution types to test
#datasets = ["EMLC0", "EMLC1", "AIDS", "PROTEINS"]
datasets = ["EMLC2"]
#conv_types = ["GCN", "GIN", "SAGE", "GAT", "DIR-GCN", "DIR-GIN", "DIR-SAGE", "DIR-GAT"]
conv_types = ["GCN", "DIR-GCN"]
checkpoint_file = "sweep_checkpoint.json"

# Load existing sweeps if checkpoint file exists
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "r") as f:
        sweeps = json.load(f)
else:
    sweeps = {}

# Define common sweep config
sweep_config = {
    "program": "train_gnn.py",
    "method": "bayes",  # Bayesian Optimization
    "metric": {"name": "avg_val_loss", "goal": "minimize"},
    "parameters": {
        "layers": {"values": [4, 6, 8, 10, 12]},
        "dim": {"values": [64, 128, 256, 512]},
        "lr": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3},
    }
}

# Function to save checkpoint
def save_checkpoint():
    with open(checkpoint_file, "w") as f:
        json.dump(sweeps, f, indent=4)

# Function to handle graceful shutdown
def handle_exit(signum, frame):
    print("\n[INFO] Received termination signal. Saving progress and exiting...")
    save_checkpoint()
    exit(0)

# Register signal handlers for safe exit
signal.signal(signal.SIGINT, handle_exit)  # Handle Ctrl+C
signal.signal(signal.SIGTERM, handle_exit) # Handle system termination

# Iterate over datasets and convolution types
for dataset in datasets:
    for conv in conv_types:
        sweep_key = f"{dataset}-{conv}"

        # Skip if sweep already exists
        if sweep_key in sweeps:
            print(f"[INFO] Sweep for {sweep_key} already exists. Resuming if needed...")
            sweep_id = sweeps[sweep_key]
        else:
            print(f"[INFO] Creating new sweep for {sweep_key}")
            sweep_config["parameters"]["dataset"] = {"value": dataset}
            sweep_config["parameters"]["conv"] = {"value": conv}

            # Create W&B sweep
            sweep_id = wandb.sweep(sweep_config, project="gnn_kfold_sweep")
            sweeps[sweep_key] = sweep_id
            save_checkpoint()

        # Start the sweep agent and wait until it finishes
        print(f"[INFO] Running agent for {sweep_key} (Sweep ID: {sweep_id})")

        # Run agent synchronously (blocking call)
        max_trials=20
        subprocess.Popen(["wandb", "agent", sweep_id, "--count", str(max_trials)])
        process = subprocess.Popen(["wandb", "agent", sweep_id, "--count", str(max_trials)])
        process.wait()  # Ensures only one sweep runs at a time

        print(f"[INFO] Completed sweep for {sweep_key}")

print("[INFO] All sweeps finished successfully!")
