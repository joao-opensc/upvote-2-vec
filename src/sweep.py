"""
This script automates the process of running a Weights & Biases sweep.

It defines the sweep configuration, initializes the sweep, and then automatically
launches multiple agents in parallel to run the experiments.
"""
import argparse
import os
import pprint
import subprocess
import sys
from multiprocessing import Process

import wandb


def run_agent(sweep_id, project_name):
    """Function to be executed by each parallel agent process."""
    print(f"🚀 Starting agent process: {os.getpid()}")
    try:
        wandb.agent(sweep_id, project=project_name, count=10)
    except Exception as e:
        print(f"Agent process {os.getpid()} encountered an error: {e}", file=sys.stderr)


def main():
    """Defines, creates, and runs the sweep."""
    parser = argparse.ArgumentParser(description="Run a W&B sweep with parallel agents.")
    parser.add_argument(
        "--agents", type=int, default=10, help="Number of parallel agents to run."
    )
    parser.add_argument(
        "--project_name", type=str, default="hackernews-score-sweeps-final", help="W&B project name."
    )
    args = parser.parse_args()

    sweep_config = {
        'method': 'bayes',
        'program': 'src/train.py',
        'command': [
            '${env}',
            '${interpreter}',
            '${program}',
            '${args}'
        ],
        'metric': {
            'name': 'val_r2',
            'goal': 'maximize'
        },
        'parameters': {
            # --- Model Architecture ---
            'LEARNING_RATE': {'distribution': 'log_uniform_values', 'min': 1e-3, 'max': 1e-1},
            'WEIGHT_DECAY': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-2},
            'HIDDEN_DIM': {'values': [128, 256, 512]},
            'DROPOUT_RATE': {'distribution': 'uniform', 'min': 0.1, 'max': 0.5},
            'DOMAIN_EMB_DIM': {'values': [32, 64, 96]},
            'USER_EMB_DIM': {'values': [64, 128, 192]},

            # --- Training Constants ---
            'BATCH_SIZE': {'values': [256, 512, 1024]},
            'PATIENCE': {'values': [5, 7, 10]},
            'FACTOR': {'distribution': 'uniform', 'min': 0.1, 'max': 0.5}, # LR scheduler factor

            # --- Data Processing & Augmentation ---
            'NUM_DOMAINS': {'values': [200, 500, 1000]},
            'NUM_USERS': {'values': [1000, 2000, 5000]},
            'MIN_TRESHOLD': {'values': [5000, 10000, 15000]}, # Min samples per bin for augmentation
            'MAX_AUGMENT_PER_BIN': {'values': [10000, 15000, 20000]},
            'TOTAL_BUDGET': {'values': [50000, 100000, 150000]}, # Max total augmented samples

            # --- Constants to keep fixed during sweep for consistency ---
            'NUM_EPOCHS': {'value': 40},
            'VAL_SIZE': {'value': 0.2},
            'TEST_SIZE': {'value': 0.1},
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 5,
            's': 2,
        }
    }

    print("--- Sweep Configuration ---")
    pprint.pprint(sweep_config)
    print("---------------------------")

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project=args.project_name)
    print(f"\n✅ Sweep created with ID: {sweep_id}")

    # --- Launch Agents in Parallel ---
    print(f"\n🔥 Launching {args.agents} parallel agents...")
    processes = []
    
    for _ in range(args.agents):
        # Using multiprocessing.Process to run agents in parallel
        p = Process(target=run_agent, args=(sweep_id, args.project_name))
        p.start()
        processes.append(p)
        print(f"  -> Launched agent with PID: {p.pid}")

    print(f"\n✅ All {args.agents} agents have been launched.")
    print("You can monitor their progress in the W&B dashboard.")
    
    # Wait for all processes to complete
    for p in processes:
        p.join()

    print("\n--- All agent processes have finished. ---")


if __name__ == '__main__':
    # We need to make sure this script can find the `train.py` module.
    # Since this script is in `src`, and train.py is also in `src`,
    # python's import system will handle it correctly when the agent runs.
    main() 