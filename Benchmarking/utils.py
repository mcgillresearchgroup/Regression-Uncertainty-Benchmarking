import argparse
import json
import numpy as np
import pandas as pd
import pathlib as Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Dictionaries of available datasets, models, and wrappers
from data_dictionaries import load_dataset, data_pull_dict
from models_and_wrappers.base_model_list import base_model_list_dict
from Benchmarking.models_and_wrappers.model_wrappers import wrapper_list_dict

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


def get_model_label(wrapper, model):
    """Convert wrapper and model names to clean abbreviated labels.
    
    Args:
        wrapper: Wrapper name (e.g., 'MVE_Ensemble_Averaged')
        model: Model name (e.g., 'MVE_Default')
    
    Returns:
        Clean abbreviated label (e.g., 'MEA-MD')
    """
    wrapper_map = {
        'MVE_Ensemble': 'MEA',  # Legacy name
        'MVE_Ensemble_Averaged': 'MEA',
        'MVE_Ensemble_Multiplicative': 'MEM',
        'MLP_Ensemble': 'MLP',
    }
    model_map = {
        'MVE_Default': 'MD',
        'MVE_Mean_Head_Extension': 'MMH',
        'MLP_Default': 'MLP',
    }
    
    wrapper_label = wrapper_map.get(wrapper, wrapper)
    model_label = model_map.get(model, model)
    
    return f"{wrapper_label}-{model_label}"


def negative_log_likelihood(y_true, y_pred_mean, y_pred_var, eps=1e-6):
    """Calculate the negative log-likelihood for Gaussian distributed targets.
        Inputs:
            y_true: True target values
            y_pred_mean: Predicted mean values from the model
            y_pred_var: Predicted variance values from the model
            eps: Small constant to prevent division by zero in variance
        Returns:
            The negative log-likelihood.
        """
    y_pred_var = np.maximum(y_pred_var, eps)  # Ensure minimum variance
    nll = 0.5 * np.log(2*np.pi*y_pred_var) + ((y_true - y_pred_mean) ** 2) / (2 * y_pred_var)
    return np.mean(nll)


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