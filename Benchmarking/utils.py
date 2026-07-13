import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch

# Dictionaries of available datasets, models, and wrappers
from data_dictionaries import load_dataset, data_pull_dict
from Benchmarking.models_and_wrappers.base_list import base_list_dict
from models_and_wrappers.model_wrappers import wrapper_list_dict

# Available options
DATASETS = list(data_pull_dict.keys())
BASES = list(base_list_dict.keys())
WRAPPERS = list(wrapper_list_dict.keys())


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train and evaluate regression models with uncertainty quantification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fix later
        """)
    
    parser.add_argument(
        '-d', '--dataset',
        type=str,
        choices=DATASETS,
        default=DATASETS[0],
        help=f'Dataset to use for training and evaluation (default: first dataset in dataset list: {DATASETS[0]}). If not specified, runs all datasets.'
    )
    
    parser.add_argument(
        '-m', '--model',
        type=str,
        required=True,
        choices=BASES,
        help=f'Base architecture. Options: {", ".join(BASES)}'
    )
    
    parser.add_argument(
        '-w', '--wrapper',
        type=str,
        required=True,
        choices=WRAPPERS,
        help=f'Model wrapper for training. Options: {", ".join(WRAPPERS)}'
    )
    
    hpo_mode_group = parser.add_mutually_exclusive_group()
    hpo_mode_group.add_argument(
        '--optimize',
        action='store_true',
        help='Run hyperparameter optimization with Optuna'
    )
    hpo_mode_group.add_argument(
        '--use-best',
        nargs='?',
        const=None,
        type=str,
        help='Use best parameters from previous optimization runs. Optionally specify a custom file path.'
    )
    parser.add_argument(
        '--n_models',
        type=int,
        default=5,
        help='Number of models to train (default: 5)'
    )

    parser.add_argument(
        '--n-trials',
        type=int,
        default=10,
        help='Number of Optuna trials (default: 48)'
    )
    
    parser.add_argument(
        '--train-size',
        type=float,
        default=0.8,
        help='Fraction of data to use for training (default: 0.8)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=420,
        help='Random seed used if hyperparameter optimization is not run (default: 42)'
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
        'MVE_Single': 'MVS',
        'MVE_Ensemble': 'MEA',  # Legacy name
        'MVE_Ensemble_Averaged': 'MEA',
        'MVE_Ensemble_Multiplicative': 'MEM',
        'MLP_Ensemble': 'MLP',
    }
    model_map = {
        'MVE_Default': 'MD',
        'MVE_Mean_Head_Extension': 'MH',
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
            The negative log-likelihood. (as a mean of all samples)
        """
    y_pred_var = np.maximum(y_pred_var, eps)  # Ensure minimum variance
    nll = 0.5 * np.log(2*np.pi*y_pred_var) + ((y_true - y_pred_mean) ** 2) / (2 * y_pred_var)
    return np.mean(nll)

# Add in the saving of the params used in the save results function, so that we can easily track which hyperparameters were used for each result file.
def save_results(results_dir, dataset_name, wrapper_name, base_name, mean_pred, var_pred, y_test, hyperparameters=None):
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
    filename = results_dir / f"{dataset_name}_{wrapper_name}_{base_name}_{timestamp}.json"
    
    # Prepare results
    results = {
        'dataset': dataset_name,
        'base': base_name,
        'wrapper': wrapper_name,
        'timestamp': timestamp,
        'hyperparameters': {hyper: value for hyper, value in (hyperparameters or {}).items()},
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


def save_trained_model(wrapper, save_dir, dataset_name, wrapper_name, base_name, seed):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime('%m%d_%H%M')
    filename = save_dir / f"{dataset_name}_{wrapper_name}_{base_name}_{seed}_{timestamp}.pt"
    
    torch.save(wrapper.get_save_state(), filename)
    return filename


def load_trained_model(filepath, map_location=None):
    state = torch.load(filepath, map_location=map_location, weights_only=False)
    wrapper_class, _ = wrapper_list_dict[state['wrapper_name']]
    return wrapper_class.load_from_state(state)