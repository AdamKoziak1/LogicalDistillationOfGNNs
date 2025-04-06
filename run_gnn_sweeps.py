import wandb
import subprocess
import json
import os
import signal

# Define datasets and convolution types to test
datasets = [
    "Ego", 
    #"EMLC0", 
    "EMLC1", 
    #"EMLC2", 
    #"EMLC3", 
    #"AIDS", 
    #"PROTEINS"
]
conv_types = [
    "GCN", 
    #"SAGE", 
    #"GAT", 
    "DIR-GCN", 
    #"DIR-SAGE", 
    #"DIR-GAT"
]
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
        "layers": {"values": [4, 6, 8]},
        "dim": {"values": [64, 128, 256, 512]},
        "lr": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3},
        "norm": {"values": [0, 1]},
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

def count_completed_runs(sweep_id):
    sweep_dir = os.path.join("wandb", f"sweep-{sweep_id}")    
    if not os.path.exists(sweep_dir):
        return 0

    return len(os.listdir(os.path.join("wandb", f"sweep-{sweep_id}")))

# Register signal handlers for safe exit
signal.signal(signal.SIGINT, handle_exit)  # Handle Ctrl+C
signal.signal(signal.SIGTERM, handle_exit) # Handle system termination

has_runs_left = True
while has_runs_left:
    # Iterate over datasets and convolution types
    has_runs_left = False
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
                sweep_config["name"] = f"{dataset}-{conv}"
                print(sweep_config)
                # Create W&B sweep
                sweep_id = wandb.sweep(sweep_config, project="gnn_kfold_sweep")
                checkpoint_file = "sweep_checkpoint.json"

                # Load existing sweeps if checkpoint file exists
                if os.path.exists(checkpoint_file):
                    with open(checkpoint_file, "r") as f:
                        sweeps = json.load(f)
                sweeps[sweep_key] = sweep_id
                save_checkpoint()

            completed_runs = count_completed_runs(sweep_id)
            runs_remaining = 25 - completed_runs

            if runs_remaining <= 0:
                print(f"[INFO] Completed sweep for {sweep_key} with {completed_runs} runs.")
                continue

            has_runs_left = True

            #print(f"[INFO] Running agent for {sweep_key} (Sweep ID: {sweep_id}) for {runs_remaining} more runs. (Completed: {completed_runs})")
            runs_remaining = min(runs_remaining, 5)
            # Launch the agent for the remaining runs
            process = subprocess.Popen(["wandb", "agent", f"ramyhn-carleton-university/gnn_kfold_sweep/{sweep_id}", "--count", str(runs_remaining)])
            process.wait()  # Wait for the agent process to finish before checking again
            # Start the sweep agent and wait until it finishes
            print(f"[INFO] Running agent for {sweep_key} (Sweep ID: {sweep_id})")
            #print(f"[INFO] Completed sweep for {sweep_key} with {completed_runs} runs.")


print("[INFO] All sweeps finished successfully!")
