"""Main training script with command-line interface for model training and hyperparameter selection."""

import sys
import numpy as np
import json
import torch
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from yaml import warnings
from utils import (
    parse_args, 
    load_dataset, 
    save_results, 
    print_metrics, 
    negative_log_likelihood, 
    get_model_label,
    wrapper_list_dict,
    base_list_dict
)
from hyperparameter_optimization import (
    initialize_model, 
    run_hyperparameter_optimization,
    load_best_parameters,
    save_best_parameters
)


def main():
    """Main execution function."""
    args = parse_args()
    
    # Set seeds robustly inside the child process for both NumPy and PyTorch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
     
    if args.verbose:
        print(f"Configuration (OPTIMIZED):")
        print(f"  Dataset: {args.dataset}")
        print(f"  Base: {args.base}")
        print(f"  Wrapper: {args.wrapper}")
        if args.optimize:
            print(f"  Mode: Hyperparameter Optimization (Optimized: 30 trials with pruning)")
            print(f"  N Trials: {args.n_trials}")
        elif args.use_best:
            print(f"  Mode: Use Best Parameters (if available)")
        else:
            print(f"  Mode: Individual Hyperparameters")
            print(f"  Epochs: {args.epochs}")
            print(f"  N Models: {args.n_models}")
            print(f"  Layers: {args.n_layers}, Size: {args.layer_size}")
            print(f"  Dropout: {args.dropout}, Mean Head Dropout: {args.mean_head_dropout}")
            print(f"  Learning Rate: {args.lr}")
        print()
    
    # Load dataset
    if args.verbose:
        print("Loading dataset...")
    
    X, y, num_features, num_targets = load_dataset(args.dataset)
    if X is None:
        print("Failed to load dataset.")
        sys.exit(1)
        
    print(f"Dataset loaded: {X.shape[0]} samples, {num_features} features, {num_targets} target(s)\n")

    if args.base not in base_list_dict:
        raise ValueError(f"Base '{args.base}' not found in base_list_dict")
    if args.wrapper not in wrapper_list_dict:
        raise ValueError(f"Wrapper '{args.wrapper}' not found in wrapper_list_dict")
    wrapper_class, wrapper_has_variance = wrapper_list_dict[args.wrapper]
    base_class, base_has_variance = base_list_dict[args.base]
    if wrapper_has_variance and not base_has_variance:
        raise ValueError(f"{args.wrapper} requires variance output, but {args.base} does not provide it")
    if not wrapper_has_variance and base_has_variance:
        warnings.warn(f"{args.wrapper} does not use variance, but {args.base} provides it")


    # Determine hyperparameters based on mode
    if args.optimize:
        # Run hyperparameter optimization
        best_params, best_nll, _ = run_hyperparameter_optimization(
            X, y, wrapper_class, base_class,
            num_features, num_targets, seed=args.seed,
            verbose=args.verbose, batch_size=128, n_trials=args.n_trials, use_mc_cv=True, n_replicates=20, train_percent=args.train_percent
        )
        # Save best parameters
        save_best_parameters(best_params, args.dataset, args.wrapper, args.base, args.train_percent, best_nll=best_nll)

        hp = best_params
        
    elif args.use_best is not None:
        # Try to load best parameters
        print(f"Loading best parameters from: {args.use_best}")
        hp = load_best_parameters(args.dataset, args.wrapper, args.base, filepath=args.use_best)
        if hp is None:
            print(f"No saved hyperparameters found for {args.dataset}/{args.wrapper}/{args.base} in {args.use_best}.")
            sys.exit(1)
        print(f"hyperparameters: {hp}")

    # Create model wrapper with selected hyperparameters and run a cross validation training to get final metrics
    # initialize_model drops any hp entries that wrapper_class doesn't accept (e.g. mean_head_*
    # params when base_class isn't MVE_Mean_Head_Extension), so hp can be passed through as-is.
    model_wrapper = initialize_model(
        wrapper_class, base_class, num_features, num_targets,
        **hp, batch_size=128 * 4
    )
    print(f"Batch Size: {128*4}")
    cross_evaluation_folds = 10
    mean_pred, var_pred, y_true = model_wrapper.cross_validate(X, y, n_splits=cross_evaluation_folds)

    # Save results
    results_dir = Path('./best_params_and_all_results')
    filename, metrics, nll = save_results(results_dir, args.dataset, args.wrapper, args.base, mean_pred, var_pred, y_true, hyperparameters=hp)
     
    if args.verbose:
        print("Model Metrics:")
        print_metrics(metrics)
        print(f"\nResults saved to: {filename}\n")
    
    # Print summary with clean labeling
    model_label = get_model_label(args.wrapper, args.base)
    print(f"Training complete for: {model_label}")
    print(f"NLL: {metrics['nll']:.6f}")
    print(f"Saved to: {filename}")


if __name__ == '__main__':
    main()