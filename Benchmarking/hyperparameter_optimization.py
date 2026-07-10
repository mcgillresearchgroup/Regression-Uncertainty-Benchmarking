"""(Optionally) Runs hyperparameter optimization using optuna."""

import json
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from pathlib import Path, WindowsPath
import gc
from sklearn.model_selection import KFold, RepeatedKFold, train_test_split
from Benchmarking.models_and_wrappers.base_list import MVE_Mean_Head_Extension
from models_and_wrappers.model_wrappers import MVE_Single
from utils import negative_log_likelihood


def create_model_wrapper(wrapper_class, base_class, num_features, num_targets, lr, epochs, 
                        n_layers, layer_size, mean_head_n_layers, mean_head_layer_size, n_models=5, batch_size=128):
    """Create a model wrapper instance with the specified parameters."""
    return wrapper_class(
        lr=lr,
        epochs=epochs,
        n_models=n_models,
        n_layers=n_layers,
        layer_size=layer_size,
        num_features=num_features,
        num_targets=num_targets,
        base_class=base_class,
        batch_size=batch_size,
        mean_head_n_layers=mean_head_n_layers,
        mean_head_layer_size=mean_head_layer_size
    )


def create_objective(X, y, wrapper_class, base_class, 
                    num_features, num_targets, verbose=False, batch_size=128, use_kfold=True, n_splits=10,
                    n_repeats=1):
    """Create an Optuna objective function.
    
    Args:
        use_kfold: If True, use k-fold cross-validation on X_train/y_train instead of X_test/y_test (default: True)
        n_splits: Number of folds for cross-validation (default: 10)
        n_repeats: Number of times to repeat the k-fold split with a different random seed each time (default: 1, i.e. plain KFold)
    """
    def objective(trial):
  
        params = wrapper_class.suggest_specific_params(trial)
        params.update(base_class.suggest_specific_params(trial))
        model = create_model_wrapper(wrapper_class, base_class, num_features, num_targets, **params, batch_size=batch_size)
        try:
            if use_kfold:
                if n_repeats > 1:
                    kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=44)
                else:
                    kf = KFold(n_splits=n_splits, shuffle=True, random_state=44)
                scores = []
                
                for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                    if hasattr(X, 'iloc'):
                        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    else:
                        X_tr, X_val = X[train_idx], X[val_idx]
                        y_tr, y_val = y[train_idx], y[val_idx]
                    
                    model = create_model_wrapper(
                        wrapper_class, base_class, num_features, num_targets,
                        **params, batch_size=batch_size
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
                    
            else:
                # Single train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, train_size=0.8, random_state=81)
                
                model = create_model_wrapper(
                    wrapper_class, base_class, num_features, num_targets,
                    **params, batch_size=batch_size
                )
                
                model.fit(X_train, y_train)
                
                # Make predictions
                mean_pred, var_pred = model.predict(X_test)
                
                # Calculate NLL
                #var_pred = np.clip(var_pred, min=1e-6, max=1e6)
                nll = negative_log_likelihood(y_test, mean_pred, var_pred)
                
            
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


def run_hyperparameter_optimization(X, y, wrapper_class, 
                                   base_class, num_features, num_targets, 
                                   verbose=False, batch_size=128, n_trials=4, use_kfold=True, n_splits=10,
                                   n_repeats=1):
    """Run optimized hyperparameter optimization.
    
    Args:
        n_trials: Number of trials (default: 48)
        batch_size: Mini-batch size for training (default: 32)
        use_kfold: If True, use k-fold cross-validation instead of single train/test split (default: True)
        n_splits: Number of folds for cross-validation (default: 5)
        n_repeats: Number of repeats for RepeatedKFold; use 1 for plain KFold (default: 1)
    """
    
    sampler = TPESampler(seed=43)
    pruner = MedianPruner(n_startup_trials=np.round(n_trials * 0.125, decimals=0), n_warmup_steps=0)
    
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=pruner
    )
    
    objective = create_objective(
        X, y, wrapper_class, base_class,
        num_features, num_targets, verbose=verbose, batch_size=batch_size,
        use_kfold=use_kfold, n_splits=n_splits, n_repeats=n_repeats
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=not verbose)
    
    return study.best_params, study.best_value, study


def save_best_parameters(best_params, dataset, wrapper_name, model_name, filepath=None, best_nll=None):
    """Save best hyperparameters to JSON file with consistent formatting.
    
    Only overwrites existing parameters if the new NLL is lower than the existing one.
    
    Args:
        best_params: Dictionary of hyperparameters
        dataset: Dataset name
        wrapper_name: Wrapper class name
        model_name: Model name
        filepath: Path to save JSON (default: results/best_parameters.json)
        best_nll: Best NLL value (optional)
    """
    if filepath is None:
        filepath = Path('best_params_and_all_results/best_parameters.json')
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing parameters or create new dict
    if filepath.exists():
        with open(filepath, 'r') as f:
            all_params = json.load(f)
    else:
        all_params = {}
    
    # Create unique key
    key = f"{dataset}_{wrapper_name}_{model_name}"
    
    # Check if key exists and if we should skip based on NLL comparison
    if key in all_params and best_nll is not None:
        existing_entry = all_params[key]
        existing_nll = existing_entry.get("best_nll")
        
        if existing_nll is not None and best_nll >= existing_nll:
            print(f"Skipping save for {key}: new NLL ({best_nll:.6f}) is not lower than existing NLL ({existing_nll:.6f})")
            return
    
    # Format with consistent structure
    formatted_entry = {
        "dataset": dataset,
        "wrapper": wrapper_name,
        "model": model_name,
        "hyperparameters": best_params
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
        filepath = Path('best_params_and_all_results/best_parameters.json')
    else:
        filepath = Path(filepath)
    if not filepath.exists():
        print(f"No best parameters file found at {filepath}")
        return None

    with open(filepath, 'r') as f:
        all_params = json.load(f)
    key = f"{dataset}_{wrapper_name}_{model_name}"
    entry = all_params.get(key, None)

    if entry is None:
        print(f"No best parameters found for {key} in {filepath}")
        return None
    
    # Handle both old flat format and new nested format
    if isinstance(entry, dict) and "hyperparameters" in entry:
        # New nested format
        return entry["hyperparameters"]
    else:
        # Old flat format
        return entry