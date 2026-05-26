"""(Optionally) Runs hyperparameter optimization using optuna."""

import json
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from pathlib import Path
import gc
import sys
from sklearn.model_selection import KFold

from models_and_wrappers.model_wrappers_optimized import MVE_Single
from custom_metrics import negative_log_likelihood


def create_model_wrapper(wrapper_class, base_model, num_features, num_targets, lr, epochs, 
                        n_layers, layer_size, dropout, mean_head_dropout, n_models=5, batch_size=32):
    """Create a model wrapper instance with the specified parameters."""
    return wrapper_class(
        lr=lr,
        epochs=epochs,
        n_models=n_models,
        n_layers=n_layers,
        layer_size=layer_size,
        num_features=num_features,
        num_targets=num_targets,
        dropout=dropout,
        mean_head_dropout=mean_head_dropout,
        base_model=base_model,
        batch_size=batch_size
    )


def create_objective(X_train, y_train, X_test, y_test, wrapper_class, base_model, 
                    num_features, num_targets, verbose=False, batch_size=128, use_kfold=False, n_splits=5):
    """Create an Optuna objective function.
    
    Args:
        use_kfold: If True, use k-fold cross-validation on X_train/y_train instead of X_test/y_test
        n_splits: Number of folds for cross-validation (default: 5)
    """
    def objective(trial):
        # Suggest hyperparameters
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        epochs = trial.suggest_int('epochs', 20, 200, step=20)  # Reduced max from 400
        n_layers = trial.suggest_int('n_layers', 2, 8)  # Reduced max from 10
        layer_size = trial.suggest_int('layer_size', 16, 120, step=8)  # Reduced max from 200
        dropout = trial.suggest_categorical('dropout', [0.0, 0.1])
        mean_head_dropout = trial.suggest_categorical('mean_head_dropout', [0.0, 0.1])
        
        if wrapper_class != MVE_Single:
            n_models = trial.suggest_categorical('n_models', [5])
        else:
            n_models = 1
        
        model = None
        try:
            if use_kfold:
                # K-fold cross-validation
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                scores = []
                
                for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                    # Use .iloc for positional indexing with pandas DataFrames
                    if hasattr(X_train, 'iloc'):
                        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    else:
                        X_tr, X_val = X_train[train_idx], X_train[val_idx]
                        y_tr, y_val = y_train[train_idx], y_train[val_idx]
                    
                    model = create_model_wrapper(
                        wrapper_class, base_model, num_features, num_targets,
                        lr, epochs, n_layers, layer_size, dropout, mean_head_dropout, 
                        n_models, batch_size=batch_size
                    )
                    
                    model.fit(X_tr, y_tr)
                    
                    # Make predictions
                    mean_pred, var_pred = model.predict(X_val)
                    
                    # Calculate NLL
                    nll = negative_log_likelihood(y_val, mean_pred, var_pred)
                    scores.append(nll)
                    
                    # Cleanup after each fold
                    if model is not None:
                        del model
                        model = None
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                
                nll = np.mean(scores)
                if verbose:
                    print(f"  Trial {trial.number} NLL: {nll:.6f} (cv), Layers: {n_layers}, Size: {layer_size}, Epochs: {epochs}")
            else:
                # Single train/test split
                model = create_model_wrapper(
                    wrapper_class, base_model, num_features, num_targets,
                    lr, epochs, n_layers, layer_size, dropout, mean_head_dropout, 
                    n_models, batch_size=batch_size
                )
                
                model.fit(X_train, y_train)
                
                # Make predictions
                mean_pred, var_pred = model.predict(X_test)
                
                # Calculate NLL
                #var_pred = np.clip(var_pred, min=1e-6, max=1e6)
                nll = negative_log_likelihood(y_test, mean_pred, var_pred)
                
                if verbose:
                    print(f"  Trial {trial.number} NLL: {nll:.6f}, Layers: {n_layers}, Size: {layer_size}, Epochs: {epochs}")
            
            return nll
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            if verbose:
                print(f"  Trial {trial.number} failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return float('inf')
        finally:
            # Aggressive cleanup after each trial
            if model is not None:
                del model
            
            # Clear tensors and cache
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    return objective


def run_hyperparameter_optimization(X_train, y_train, X_test, y_test, wrapper_class, 
                                   base_model, num_features, num_targets, 
                                   verbose=False, batch_size=128, n_trials=4, use_kfold=True, n_splits=3):
    """Run optimized hyperparameter optimization.
    
    Args:
        n_trials: Number of trials (default: 48)
        batch_size: Mini-batch size for training (default: 32)
        use_kfold: If True, use k-fold cross-validation instead of single train/test split (default: True)
        n_splits: Number of folds for cross-validation (default: 5)
    """
    
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=np.round(n_trials * 0.125, decimals=0), n_warmup_steps=0)
    
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=pruner
    )
    
    objective = create_objective(
        X_train, y_train, X_test, y_test, wrapper_class, base_model,
        num_features, num_targets, verbose=verbose, batch_size=batch_size,
        use_kfold=use_kfold, n_splits=n_splits
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=not verbose)
    
    return study.best_params, study.best_value, study


def save_best_parameters(best_params, dataset, wrapper_name, model_name, filepath=None, best_nll=None):
    """Save best hyperparameters to JSON file with consistent formatting.
    
    Args:
        best_params: Dictionary of hyperparameters
        dataset: Dataset name
        wrapper_name: Wrapper class name
        model_name: Model name
        filepath: Path to save JSON (default: results/best_parameters.json)
        best_nll: Best NLL value (optional)
    """
    if filepath is None:
        filepath = Path('results/best_parameters.json')
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing parameters or create new dict
    if filepath.exists():
        with open(filepath, 'r') as f:
            all_params = json.load(f)
    else:
        all_params = {}
    
    # Create unique key
    key = f"{dataset}_{wrapper_name}_{model_name}"
    
    # Format with consistent structure
    formatted_entry = {
        "dataset": dataset,
        "wrapper": wrapper_name,
        "model": model_name,
        "parameters": best_params
    }
    
    if best_nll is not None:
        formatted_entry["best_nll"] = best_nll
    
    all_params[key] = formatted_entry
    
    # Save
    with open(filepath, 'w') as f:
        json.dump(all_params, f, indent=4)
    
    print(f"Best parameters saved to {filepath}")
    print(f"Key: {key}")


def load_best_parameters(dataset, wrapper_name, model_name, filepath=None):
    """Load best hyperparameters from JSON file.
    
    Returns the parameters dict, handling both old flat format and new nested format.
    """
    if filepath is None:
        filepath = Path('results/best_parameters.json')
    
    if not filepath.exists():
        return None
    
    with open(filepath, 'r') as f:
        all_params = json.load(f)
    
    key = f"{dataset}_{wrapper_name}_{model_name}"
    entry = all_params.get(key, None)
    
    if entry is None:
        return None
    
    # Handle both old flat format and new nested format
    if isinstance(entry, dict) and "parameters" in entry:
        # New nested format
        return entry["parameters"]
    else:
        # Old flat format
        return entry
