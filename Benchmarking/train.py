"""Main training script with command-line interface for model training and hyperparameter selection."""

import sys
import numpy as np
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils import (
    parse_args, 
    load_dataset, 
    save_results, 
    print_metrics, 
    negative_log_likelihood, 
    get_model_label,
    wrapper_list_dict,
    base_model_list_dict
)
from hyperparameter_optimization import (
    create_model_wrapper, 
    run_hyperparameter_optimization,
    load_best_parameters,
    save_best_parameters
)


def main():
    """Main execution function."""
    args = parse_args()
    
    # Set seed
    np.random.seed(args.seed)
     
    if args.verbose:
        print(f"Configuration (OPTIMIZED):")
        print(f"  Dataset: {args.dataset}")
        print(f"  Model: {args.model}")
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
    print(f"Dataset loaded: {X.shape[0]} samples, {num_features} features, {num_targets} target(s)\n")
    if X is None:
        print("Failed to load dataset.")
        sys.exit(1)
    
    if args.verbose:
        print(f"Dataset loaded: {X.shape[0]} samples, {num_features} features, {num_targets} target(s)\n")
    
    wrapper_class = wrapper_list_dict[args.wrapper][0]
    
    # Determine hyperparameters based on mode
    if args.optimize:
        
        # Run hyperparameter optimization
        best_params, best_nll, _ = run_hyperparameter_optimization(
            X, y, wrapper_class, args.model, 
            num_features, num_targets,
            n_trials=args.n_trials, verbose=args.verbose, batch_size=128, use_kfold=True, n_splits=10
        )
        
        # Save best parameters
        save_best_parameters(best_params, args.dataset, args.wrapper, args.model, best_nll=best_nll)
        
        hp = {
            'lr': best_params['lr'],
            'epochs': best_params['epochs'],
            'n_layers': best_params['n_layers'],
            'layer_size': best_params['layer_size'],
            'n_models': best_params.get('n_models', 5),
            'mean_head_n_layers': best_params.get('mean_head_n_layers', None),
            'mean_head_layer_size': best_params.get('mean_head_layer_size', None),
            #batch size hp to be added (Currently defaulting to 256)
        }
        
    elif args.use_best is not None:
        # Try to load best parameters
        print(args.use_best)
        best_params = load_best_parameters(args.dataset, args.wrapper, args.model, filepath = args.use_best)
        print(f"best_params: {best_params}")
        params = best_params.get('hyperparameters', best_params)
        
        hp = {
            #'lr': args.lr if args.lr != None else float(params['lr']),
            #'epochs': args.epochs if args.epochs != None else int(params['epochs']),
            #'n_layers': args.n_layers if args.n_layers != None else int(params['n_layers']),
            #'layer_size': args.layer_size if args.layer_size != None else int(params['layer_size']),
            #'n_models': args.n_models if args.n_models != None else int(params.get('n_models', 5)),
            #'mean_head_n_layers': args.mean_head_n_layers if args.mean_head_n_layers != None else params.get('mean_head_n_layers', None),
            #'mean_head_layer_size': args.mean_head_layer_size if args.mean_head_layer_size != None else params.get('mean_head_layer_size', None)
            'lr': float(params['lr']),
            'epochs': int(params['epochs']),
            'n_layers': int(params['n_layers']),
            'layer_size': int(params['layer_size']),
            'n_models': int(params.get('n_models', 5)),
            'mean_head_n_layers': params.get('mean_head_n_layers', None),
            'mean_head_layer_size': params.get('mean_head_layer_size', None)
        }


    # Create model wrapper with selected hyperparameters and run a cross validation training to get final metrics
    model_wrapper = create_model_wrapper(
        wrapper_class, args.model, num_features, num_targets,
        hp['lr'], hp['epochs'], hp['n_layers'], hp['layer_size'], 
        hp['mean_head_n_layers'], hp['mean_head_layer_size'], hp['n_models'], batch_size=128
    )
    print(f"Batch Size: {128}")
    cross_evaluation_folds = 10
    mean_pred, var_pred, y_true = model_wrapper.cross_validate(X, y, n_splits=cross_evaluation_folds)

    # Save results
    results_dir = Path('./best_params_and_all_results')
    filename, metrics, nll = save_results(results_dir, args.dataset, args.model, args.wrapper, mean_pred, var_pred, y_true, hyperparameters=hp)

    # Note: Best parameters are saved during optimization in the optimized version
    # No need to save again after training
     
    if args.verbose:
        print("Model Metrics:")
        print_metrics(metrics)
        print(f"\nResults saved to: {filename}\n")
    
    # Print summary with clean labeling
    model_label = get_model_label(args.wrapper, args.model)
    print(f"Training complete for: {model_label}")
    print(f"NLL: {metrics['nll']:.6f}")
    print(f"Saved to: {filename}")


if __name__ == '__main__':
    main()