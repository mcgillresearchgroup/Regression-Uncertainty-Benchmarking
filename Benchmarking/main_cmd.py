"""Main file to run the complete workflow: hyperparameter optimization (optional), training, and plotting for all model combinations."""
import subprocess
import sys
import argparse

# Model combinations to test: (wrapper, model)
MODEL_COMBINATIONS = [
    ("MVE_Ensemble_Averaged", "MVE_Default"),  # MEA-MD
    ("MVE_Ensemble_Averaged", "MVE_Mean_Head_Extension"),  # MEA-MH
    ("MVE_Ensemble_Multiplicative", "MVE_Default"),  # MEM-MD
    ("MVE_Ensemble_Multiplicative", "MVE_Mean_Head_Extension"),  # MEM-MH
]

DATASET = "Concrete Compressive Strength"

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
        '--use-best',
        action='store_true',
        type=str,
        help='Use best hyperparameters from optimization. Provide path to JSON config file (e.g. results/best_parameters.json). If not specified, runs optimization for each combination.'
    )
    return parser.parse_args()

 
def run_training(combinations_to_run=None, use_best=False):
    """Run training for specified model combinations."""
    if combinations_to_run is None:
        combinations_to_run = [(i, combo) for i, combo in enumerate(MODEL_COMBINATIONS, 1)]
    
    print("=" * 80)
    print("Starting model training workflow (OPTIMIZED)")
    print("=" * 80)
    
    for i, (wrapper, model) in combinations_to_run:
        print(f"\n[{i}/{len(MODEL_COMBINATIONS)}] Training {wrapper} with {model}")
        print("-" * 80)
        
        cmd = [
            sys.executable, "train.py",
            "-d", DATASET,
            "-m", model,
            "-w", wrapper
        ]
        
        if use_best:
            cmd.append("--use-best")
        else:
            cmd.append("--optimize")
        
        try:
            result = subprocess.run(cmd, check=True)
            if result.returncode != 0:
                print(f"Warning: Training command returned non-zero exit code: {result.returncode}")
        except subprocess.CalledProcessError as e:
            print(f"Error: Training failed for {wrapper} + {model}")
            print(f"Exit code: {e.returncode}")
            return False
    
    return True


def run_plotting():
    """Generate comparison plots from trained models."""
    print("\n" + "=" * 80)
    print("Generating comparison plots")
    print("=" * 80)
    
    cmd = [sys.executable, "./evaluations/plot_results.py"]
    
    try:
        result = subprocess.run(cmd, check=True)
        if result.returncode != 0:
            print(f"Warning: Plotting command returned non-zero exit code: {result.returncode}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Error: Plotting failed")
        print(f"Exit code: {e.returncode}")
        return False
    
    return True


def main():
    """Execute the complete workflow."""
    args = parse_args()
    
    # Determine which combinations to run
    if args.combination:
        combinations_to_run = [(args.combination, MODEL_COMBINATIONS[args.combination - 1])]
    else:
        combinations_to_run = [(i, combo) for i, combo in enumerate(MODEL_COMBINATIONS, 1)]
    
    print("\nWorkflow Summary:")
    print(f"Dataset: {DATASET}")
    print(f"Model combinations to train: {len(combinations_to_run)}")
    for idx, (wrapper, model) in combinations_to_run:
        print(f"  [{idx}] {wrapper} + {model}")
    
    # Run training
    if not run_training(combinations_to_run, use_best=args.use_best):
        print("\nWorkflow failed during training phase.")
        sys.exit(1)
    
    # Only run plotting if doing all combinations
    if not args.combination:
        if not run_plotting():
            print("\nWorkflow failed during plotting phase.")
            sys.exit(1)
    
    print("\n" + "=" * 80)
    print("Workflow completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
