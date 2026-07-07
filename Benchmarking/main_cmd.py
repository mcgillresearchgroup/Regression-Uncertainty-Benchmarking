"""Main file to run the complete workflow: hyperparameter optimization (optional), training, and plotting for all model combinations."""
import subprocess
import sys
import argparse
import torch
import numpy as np
from evaluations.plotters_new import run_plotting

# Model combinations to test: (wrapper, model)
MODEL_COMBINATIONS = [
    ('MVE_Ensemble_Averaged', 'MVE_Default' ),  # MEA-MD
]

DATASETS = ['Combined_Cycle_Power_Plant']  
n = 35
seed_list = [n, n+1, n+2, n+3, n+4]  # Seeds for reproducibility

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train and compare model combinations'
    )

    parser.add_argument(
        '--use-best',
        type=str,
        help='Use best hyperparameters from optimization. Provide path to JSON config file (e.g. best_params_and_all_results/best_parameters.json).'
    )
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Force run hyperparameter optimization instead of using saved best parameters'
    )
    return parser.parse_args()


def run_training(combinations_to_run=None, use_best=None, optimize=False, dataset=None):
    """Run training for specified model combinations."""
    
    for i, (wrapper, model), dataset_i, seed in combinations_to_run:
        print(f"\n[{i}/{len(combinations_to_run)}] Training {wrapper} with {model} on {DATASETS[dataset_i]}")
        print("-" * 80)

        # Build execution command
        cmd = [
            sys.executable, "train.py",
            "-d", DATASETS[dataset_i],
            "-m", model,
            "-w", wrapper,
            "--seed", f"{seed + 102}"
        ]

        if optimize:
            cmd.append("--optimize")
        else:
            # If use_best was explicitly provided as a custom path, use it.
            # Otherwise, default to the standard best parameters JSON file.
            best_path = use_best if use_best is not None else "best_params_and_all_results/best_parameters.json"
            cmd.extend(["--use-best", best_path])

        try:
            result = subprocess.run(cmd, check=True)
            if result.returncode != 0:
                print(f"Warning: Training command returned non-zero exit code: {result.returncode}")
        except subprocess.CalledProcessError as error:
            print(f"Error: Training failed for {wrapper} + {model}")
            print(f"Exit code: {error.returncode}")
            return False
    
    return True


def main():
    """Execute the complete workflow."""
    args = parse_args()
    
    # Determine which combinations to run
    combinations_to_run = [(i, combo, dataset_i, seed) for i, combo in enumerate(MODEL_COMBINATIONS, 1) for dataset_i in range(len(DATASETS)) for seed in seed_list]
    
    print("\nWorkflow Summary:")
    print(f"Dataset: {DATASETS[0]}")
    print(f"Model combinations to train: {len(combinations_to_run)}")
    for idx, (wrapper, model), dataset_i, seed in combinations_to_run:
        print(f"  [{idx}] {wrapper} + {model} on {DATASETS[dataset_i]} with seed {seed}")

    # Run training
    if not run_training(combinations_to_run, use_best=args.use_best, optimize=args.optimize):
        print("\nWorkflow failed during training phase.")
        sys.exit(1)


if __name__ == "__main__":
    main()
    run_plotting(pms=True)