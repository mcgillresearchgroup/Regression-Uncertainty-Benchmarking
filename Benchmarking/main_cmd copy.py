"""Main file to run the complete workflow: hyperparameter optimization (optional), training, and plotting for all model combinations."""
import subprocess
import sys
import argparse
from evaluations.plotters import run_plotting

# Model combinations to test: (wrapper, model)
MODEL_COMBINATIONS = [
    ("MVE_Ensemble_Averaged", "MVE_Default"),  # MEA-MD
]

DATASETS = ["Appliances_Energy_Prediction", "RT-IoT2022"]

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train and compare model combinations'
    )
    parser.add_argument(
        '-c', '--combination',
        type=int,
        choices=range(1, len(MODEL_COMBINATIONS) + 1),
        help=f'Run specific combination (1-{len(MODEL_COMBINATIONS)}). If not specified, runs all combinations.'
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        choices=DATASETS,
        default=DATASETS[0],
        help=f'Dataset to use for training and evaluation (default: first dataset in dataset list: {DATASETS[0]})'
    )
    parser.add_argument(
        '--use-best',
        nargs='?',
        const=None,
        type=str,
        help='Use best hyperparameters from optimization. Provide path to JSON config file (e.g. results/best_parameters.json). If not specified, uses best_parameters.json.'
    )
    return parser.parse_args()


def run_training(combinations_to_run=None, use_best=None, dataset=None):
    """Run training for specified model combinations."""
    
    for i, (wrapper, model), dataset_i in combinations_to_run:
        print(f"\n[{i}/{len(MODEL_COMBINATIONS)}] Training {wrapper} with {model} on {DATASETS[dataset_i]}")
        print("-" * 80)
        
        cmd = [
            sys.executable, "train.py",
            "-d", DATASETS[dataset_i],
            "-m", model,
            "-w", wrapper
        ]

        # Use best argument. If --use-best is used and there is a path argument, it will use that path. 
        # If --use-best is used but there is no path argument, it will use the default best_parameters.json. 
        # If --use-best is not used, it will run optimization.
        if use_best is not None and isinstance(use_best, str):
            cmd.extend(["--use-best", use_best])
        elif use_best is not None:
            cmd.extend(["--use-best"])
        else:
            cmd.append("--optimize")

        # Dataset argument. If --dataset is used and there is a dataset argument, it will run on that dataset. 
        # If --dataset is used but there is no dataset argument, it will run on the default dataset (the first one). 
        # If --dataset is not used, it will run on all the datasets.
        if dataset is not None and isinstance(dataset, str):
            cmd.extend(["--dataset", dataset])
        elif dataset is not None:
            cmd.extend(["--dataset"])

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
    if args.combination:
        combinations_to_run = [(args.combination, MODEL_COMBINATIONS[args.combination - 1], 0)]
    else:
        combinations_to_run = [(i, combo, dataset_i) for i, combo in enumerate(MODEL_COMBINATIONS, 1) for dataset_i in range(len(DATASETS))]
    
    print("\nWorkflow Summary:")
    print(f"Dataset: {DATASETS[0]}")
    print(f"Model combinations to train: {len(combinations_to_run)}")
    for idx, (wrapper, model), dataset_i in combinations_to_run:
        print(f"  [{idx}] {wrapper} + {model} on {DATASETS[dataset_i]}")
    
    # Run training
    if not run_training(combinations_to_run, use_best=args.use_best):
        print("\nWorkflow failed during training phase.")
        sys.exit(1)
    
    # Run plotting
    run_plotting(pre=False)
    


if __name__ == "__main__":
    main()
