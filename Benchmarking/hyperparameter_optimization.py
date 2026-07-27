"""(Optionally) Runs hyperparameter optimization using optuna."""

import inspect
import json
import numpy as np
import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from pathlib import Path, WindowsPath
import gc
from sklearn.model_selection import train_test_split
from utils import negative_log_likelihood
from sklearn.model_selection import ShuffleSplit

def initialize_model(wrapper_class, base_class, num_features, num_targets, **hyperparameters):
    """Initialize a model wrapper instance from a wrapper class, a base class, and a bag of
    hyperparameters (e.g. loaded straight from best_parameters.json).

    Args:
        wrapper_class: The wrapper class to instantiate (e.g. MVE_Ensemble_Averaged).
        base_class: The base model class the wrapper trains internally (e.g. MVE_Default).
        num_features: Number of input features.
        num_targets: Number of output targets.
        **hyperparameters: Any wrapper/base hyperparameters (lr, epochs, n_layers, layer_size,
            n_models, batch_size, mean_head_n_layers, mean_head_layer_size, etc). Entries that
            wrapper_class's __init__ doesn't accept are dropped automatically, so this can be
            called with a full hyperparameters dict (e.g. from load_best_parameters) even if it
            contains keys that only apply to a different wrapper/base combination. If
            wrapper_class's __init__ accepts **kwargs (e.g. GP_Wrapper, which passes extras like
            inducing_fraction through), nothing is filtered out -- inspect.signature only exposes
            the catch-all parameter name, not the individual keys it will accept.
    """
    accepted_params = inspect.signature(wrapper_class.__init__).parameters
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in accepted_params.values()
    )
    if accepts_var_kwargs:
        filtered_hyperparameters = dict(hyperparameters)
        extra_params = {}
    else:
        filtered_hyperparameters = {k: v for k, v in hyperparameters.items() if k in accepted_params}
        extra_params = {k: v for k, v in hyperparameters.items() if k not in accepted_params}
    if len(extra_params) > 0:
        raise KeyError(f"Warning: Ignoring extra hyperparameters not accepted by {wrapper_class.__name__}: {list(extra_params.keys())}")
    return wrapper_class(
        base_class=base_class,
        num_features=num_features,
        num_targets=num_targets,
        **filtered_hyperparameters
    )


def create_objective(X, y, wrapper_class, base_class,
                    num_features, num_targets, seed, test_size=0.2, verbose=False, batch_size=128,
                    use_mc_cv=True, n_replicates=20, train_percent=100, n_jobs=1, job_index=0
                    ):
    """Create an Optuna objective function.
    
    Args:
        seed: Master seed used to derive per-job/per-replicate seed streams.
        test_size: Fraction of data held out for validation in each MC-CV replicate (default: 0.2).
        use_mc_cv: If True, use Monte Carlo cross-validation instead of an 80/20 single split (default: True)
        n_replicates: Number of replicates for Monte Carlo cross-validation (default: 20)
        train_percent: Percentage of the training dataset to use for each replicate (default: 100)
        n_jobs: Total number of parallel jobs sharing the seed stream derived from `seed` (default: 1)
        job_index: This job's 0-based index into the `n_jobs` seed stream (default: 0)
    """
    def objective(trial):
        params = wrapper_class.suggest_specific_params(trial)
        params.update(base_class.suggest_specific_params(trial))
        model = None
        job_seeds = np.random.SeedSequence(seed).spawn(n_jobs)
        replicate_seeds = job_seeds[job_index].spawn(n_replicates)

        try:
            if use_mc_cv:
                scores = []
                for rep_idx, seed_seq in enumerate(replicate_seeds):
                    replicate_seed = int(seed_seq.generate_state(1)[0])
                    splitter = ShuffleSplit(n_splits=1, test_size=test_size, random_state=replicate_seed)
                    train_idx, val_idx = next(splitter.split(X))
                    train_size = len(train_idx)
                    train_size = int(train_size * (train_percent / 100))
                    train_idx = train_idx[:train_size]

                    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    model = initialize_model(wrapper_class, base_class, num_features, num_targets, **params)
                    model.fit(X_tr, y_tr)
                    mean_pred, var_pred = model.predict(X_val)
                    
                    nll = negative_log_likelihood(y_val, mean_pred, var_pred)
                    scores.append(nll)
                    
                    # Cleanup after each fold
                    if model is not None:
                        del model
                        model = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                    running_mean = np.mean(scores)
                    trial.report(running_mean, step=rep_idx)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                
                nll = np.mean(scores)
                    
            else:
                # Single train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, train_size=0.8, random_state=81)
                
                model = initialize_model(
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
            print(f"Trial failed due to: {e}")
            return float('inf') 

        finally:
            if 'model' in locals() and model is not None:
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
    return objective


def run_hyperparameter_optimization(X, y, wrapper_class,
                                   base_class, num_features, num_targets,
                                   seed, test_size=0.2, verbose=False, batch_size=128, n_trials=4,
                                   use_mc_cv=True, n_replicates=20, train_percent=100,
                                   n_jobs=1, job_index=0):
    """Run optimized hyperparameter optimization.
    
    Args:
        X: Input features (DataFrame or array-like)
        y: Target values (Series or array-like)
        wrapper_class: The model wrapper class to optimize (e.g., MVE_Ensemble_Averaged)
        base_class: The base model class to optimize (e.g., MVE_Default)
        num_features: Number of input features
        num_targets: Number of output targets
        seed: Master seed used to derive per-job/per-replicate seed streams.
        test_size: Fraction of data held out for validation in each MC-CV replicate (default: 0.2).
        verbose: If True, print detailed logs (default: False)
        batch_size: Batch size for training (default: 128)
        n_trials: Number of Optuna trials to run (default: 4)
        use_mc_cv: If True, use Monte Carlo cross-validation; otherwise, use a single train/test split (default: True)
        n_replicates: Number of replicates for Monte Carlo cross-validation (default: 20)
        train_percent: Percentage of the training dataset to use for each fold (default: 100)
        n_jobs: Total number of parallel jobs sharing the seed stream derived from `seed` (default: 1)
        job_index: This job's 0-based index into the `n_jobs` seed stream (default: 0)
    """
    
    sampler = TPESampler(seed=43)
    pruner = MedianPruner(n_startup_trials=np.round(n_trials * 0.125, decimals=0), n_warmup_steps=5)    
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=pruner
    )
    
    objective = create_objective(
        X, y, wrapper_class, base_class,
        num_features, num_targets, seed=seed, test_size=test_size,
        verbose=verbose, batch_size=batch_size,
        use_mc_cv=use_mc_cv, n_replicates=n_replicates,
        train_percent=train_percent, n_jobs=n_jobs, job_index=job_index
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=not verbose)
    
    return study.best_params, study.best_value, study


def save_best_parameters(best_params, dataset, wrapper_name, model_name, train_percent, filepath=None, best_nll=None):
    """Save best hyperparameters to JSON file with consistent formatting.
    
    Only overwrites existing parameters if the new NLL is lower than the existing one.
    
    Args:
        best_params: Dictionary of hyperparameters
        dataset: Dataset name
        wrapper_name: Wrapper class name
        model_name: Model name
        train_percent: Percentage of the training dataset used for each fold
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
    
    # Create unique key. train_percent is part of the key (not just stored in the entry)
    # because different train_percent runs for the same dataset/wrapper/model are distinct
    # experiments with their own best hyperparameters -- without it, only the single
    # train_percent with the lowest NLL would ever survive in this file.
    key = f"{dataset}_{wrapper_name}_{model_name}_tp{train_percent}"
    
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
        "train_percent": train_percent,
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


def load_best_parameters(dataset, wrapper_name, model_name, train_percent=None, filepath=None):
    """Load best hyperparameters from JSON file.

    Args:
        train_percent: If given, looks up the entry saved for this specific train_percent
            (the current key format). If that's not found, falls back to the old
            pre-train_percent key for backward compatibility with files saved before this
            change, and prints a note that the loaded params weren't specific to this
            train_percent.

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

    old_key = f"{dataset}_{wrapper_name}_{model_name}"
    entry = None
    if train_percent is not None:
        new_key = f"{old_key}_tp{train_percent}"
        entry = all_params.get(new_key, None)
        if entry is None and old_key in all_params:
            print(f"Note: no entry for {new_key}; falling back to legacy key {old_key} "
                  f"(not specific to train_percent={train_percent}).")
            entry = all_params.get(old_key, None)
    else:
        entry = all_params.get(old_key, None)

    if entry is None:
        return None
    
    # Handle both old flat format and new nested format
    if isinstance(entry, dict) and "hyperparameters" in entry:
        # New nested format
        return entry["hyperparameters"]
    else:
        # Old flat format
        return entry