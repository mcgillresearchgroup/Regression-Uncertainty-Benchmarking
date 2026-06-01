"""Main training script with command-line interface for model training and hyperparameter selection."""

import argparse
import sys
import numpy as np
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_dictionaries import load_dataset, data_pull_dict
from models_and_wrappers.base_model_list import base_model_list_dict
from models_and_wrappers.model_wrappers_optimized import wrapper_list_dict
from custom_metrics import negative_log_likelihood
from model_naming import get_model_label
from hyperparameter_optimization import (
    create_model_wrapper, 
    run_hyperparameter_optimization,
    load_best_parameters,
    save_best_parameters
)


# Available options
DATASETS = list(data_pull_dict.keys())
MODELS = list(base_model_list_dict.keys())
WRAPPERS = list(wrapper_list_dict.keys())


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train and evaluate regression models with uncertainty quantification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use individual hyperparameters
  python train.py --dataset "Wine_Quality" --model "MVE_Default" --wrapper "MVE_Single" --epochs 50 --n-models 3
  
  # Use best parameters from previous runs
  python train.py --dataset "Concrete Compressive Strength" --model "MLP_Default" --wrapper "MLP_Ensemble" --use-best
  
  # Run hyperparameter optimization with default 30 trials (optimized)
  python train.py -d "Wine_Quality" -m "MVE_Default" -w "MVE_Ensemble_Averaged" --optimize
  
  # Run hyperparameter optimization with custom number of trials
  python train.py -d "Wine_Quality" -m "MVE_Default" -w "MVE_Single" --optimize --n-trials 50
        """)
    
    parser.add_argument(
        '-d', '--dataset',
        type=str,
        required=True,
        choices=DATASETS,
        help=f'Dataset to use. Options: {", ".join(DATASETS)}'
    )
    
    parser.add_argument(
        '-m', '--model',
        type=str,
        required=True,
        choices=MODELS,
        help=f'Base model architecture. Options: {", ".join(MODELS)}'
    )
    
    parser.add_argument(
        '-w', '--wrapper',
        type=str,
        required=True,
        choices=WRAPPERS,
        help=f'Model wrapper for training. Options: {", ".join(WRAPPERS)}'
    )
    
    # Hyperparameter selection mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--optimize',
        action='store_true',
        help='Run hyperparameter optimization with Optuna'
    )
    mode_group.add_argument(
        '--use-best',
        nargs='?',
        const=None,
        type=str,
        help='Use best parameters from previous optimization runs. Optionally specify a custom file path.'
    )
    
    # Hyperparameters (used when not using --optimize or --use-best)
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    
    parser.add_argument(
        '--n-models',
        type=int,
        default=5,
        help='Number of models for ensemble (default: 5, ignored for MVE_Single)'
    )
    
    parser.add_argument(
        '--n-layers',
        type=int,
        default=2,
        help='Number of layers in the neural network (default: 2)'
    )
    
    parser.add_argument(
        '--layer-size',
        type=int,
        default=64,
        help='Size of each hidden layer (default: 64)'
    )
    
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout rate for main body (default: 0.1)'
    )
    
    parser.add_argument(
        '--mean-head-dropout',
        type=float,
        default=0.1,
        help='Dropout rate for mean head (default: 0.1)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    
    # Optimization options
    parser.add_argument(
        '--n-trials',
        type=int,
        default=4,
        help='Number of Optuna trials (default: 48)'
    )
    
    # Other options
    parser.add_argument(
        '--train-size',
        type=float,
        default=0.8,
        help='Fraction of data to use for training (default: 0.8)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def save_results(results_dir, dataset_name, model_name, wrapper_name, mean_pred, var_pred, y_test):
    """Save model predictions and metrics to file."""
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, mean_pred)
    mae = mean_absolute_error(y_test, mean_pred)
    r2 = r2_score(y_test, mean_pred)
    var_pred_clipped = np.clip(var_pred, min=1e-6, max=1e6)
    nll = negative_log_likelihood(y_test, mean_pred, var_pred_clipped)
    
    mean_list = mean_pred.flatten().tolist()
    var_list = var_pred.flatten().tolist()
    y_true_array = np.asarray(y_test).flatten()
    y_true_list = y_true_array.tolist()
    # Create filename
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = results_dir / f"{dataset_name}_{wrapper_name}_{model_name}_{timestamp}.json"
    
    # Prepare results
    results = {
        'dataset': dataset_name,
        'model': model_name,
        'wrapper': wrapper_name,
        'timestamp': timestamp,
        'metrics': {
            'nll': float(nll),
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(np.sqrt(mse)),
            'r2': float(r2),
            'n_samples': int(len(y_test))
        },
        'accuracy': {
            'mean': mean_list,
            'variance': var_list,
            'ground_truth': y_true_list
        }
    }
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    return filename, results['metrics'], nll


def print_metrics(metrics):
    """Print model metrics."""
    print(f"  NLL: {metrics['nll']:.6f}")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  R²: {metrics['r2']:.6f}")
    print(f"  N Samples: {metrics['n_samples']}")


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
    
    if X is None:
        print("Failed to load dataset.")
        sys.exit(1)
    
    if args.verbose:
        print(f"Dataset loaded: {X.shape[0]} samples, {num_features} features, {num_targets} target(s)\n")
    
    wrapper_class = wrapper_list_dict[args.wrapper][0]
    
    # Determine hyperparameters based on mode
    if args.optimize:
        # First, split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=args.train_size, random_state=args.seed
        )
        
        if args.verbose:
            print(f"Train/Test split: {len(X_train)} / {len(X_test)}\n")
        
        # Run hyperparameter optimization with pre-split data
        best_params, best_nll, _ = run_hyperparameter_optimization(
            X_train, y_train, X_test, y_test, wrapper_class, args.model, 
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
            'dropout': best_params['dropout'],
            'mean_head_dropout': best_params['mean_head_dropout'],
            'n_models': best_params.get('n_models', 5)
            #batch size hp to be added (Currently defaulting to 128)
        }
        
        hp_source = "Optuna optimization (optimized)"
    elif args.use_best:
        # Try to load best parameters
        best_params = load_best_parameters(args.dataset, args.wrapper, args.model, file_path = args.use_best)
        params = best_params.get('parameters', best_params)
        
        hp = {
            'lr': args.lr if args.lr != None else float(params['lr']),
            'epochs': args.epochs if args.epochs != None else int(params['epochs']),
            'n_layers': args.n_layers if args.n_layers != None else int(params['n_layers']),
            'layer_size': args.layer_size if args.layer_size != None else int(params['layer_size']),
            'dropout': args.dropout if args.dropout != None else float(params['dropout']),
            'mean_head_dropout': args.mean_head_dropout if args.mean_head_dropout != None else float(params['mean_head_dropout']),
            'n_models': args.n_models if args.n_models != None else int(params.get('n_models', 5))
        }
        hp_source = "best parameters (merged with user-specified values)"

    elif best_params is None:
        if args.verbose:
            print(f'Using default parameters')
        hp = {
            'lr': args.lr,
            'epochs': args.epochs,
            'n_layers': args.n_layers,
            'layer_size': args.layer_size,
            'dropout': args.dropout,
            'mean_head_dropout': args.mean_head_dropout,
            'n_models': args.n_models 
        } 
        hp_source = "default parameters"
    
    # Split data if not already done (in optimize mode it's already split)
    if not args.optimize:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=args.train_size, random_state=args.seed
        )
        if args.verbose and not args.use_best:
            print(f"Train/Test split: {len(X_train)} / {len(X_test)}\n")
    
    # Create model wrapper
    if args.verbose:
        print(f"Creating {args.wrapper} with {args.model} base model...")
        print(f"Using hyperparameters from: {hp_source}\n")
    
    # Create wrapper instance with selected parameters
    try:
        model = create_model_wrapper(
            wrapper_class, args.model, num_features, num_targets,
            hp['lr'], hp['epochs'], hp['n_layers'], hp['layer_size'],
            hp['dropout'], hp['mean_head_dropout'], hp['n_models']
        )
    except TypeError as e:
        print(f"Error creating model: {e}")
        sys.exit(1)
    
    if args.verbose:
        print(f"Model created successfully.\n")
        print("Training model...")
    
    # Train model
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)
    
    if args.verbose:
        print("Training complete.\n")
        print("Generating predictions...")
    
    # Make predictions
    try:
        mean_pred, var_pred = model.predict(X_test)
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)
    
    if args.verbose:
        print(f"Predictions generated: mean shape {mean_pred.shape}, var shape {var_pred.shape}\n")
        print(f"Mean predictions - Min: {mean_pred.min():.4f}, Max: {mean_pred.max():.4f}, Mean: {mean_pred.mean():.4f}")
        print(f"Variance predictions - Min: {var_pred.min():.4f}, Max: {var_pred.max():.4f}, Mean: {var_pred.mean():.4f}\n")
    
    # Save results
    results_dir = Path('./results')
    filename, metrics, nll = save_results(results_dir, args.dataset, args.model, args.wrapper, mean_pred, var_pred, y_test)
    
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
