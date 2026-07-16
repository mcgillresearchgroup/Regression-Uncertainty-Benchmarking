"""(Optionally) Runs hyperparameter optimization using optuna."""

import inspect
import json
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from pathlib import Path, WindowsPath
import gc
from sklearn.model_selection import train_test_split
from models_and_wrappers.model_wrappers import MVE_Single
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
            contains keys that only apply to a different wrapper/base combination.
    """
    accepted_params = inspect.signature(wrapper_class.__init__).parameters
    filtered_hyperparameters = {k: v for k, v in hyperparameters.items() if k in accepted_params}
    return wrapper_class(
        base_class=base_class,
        num_features=num_features,
        num_targets=num_targets,
        **filtered_hyperparameters
    )


def create_objective(X, y, wrapper_class, base_class, 
                    num_features, num_targets, args, verbose=False, batch_size=128, use_mc_replicates=True, n_splits=20
                    ):
    """Create an Optuna objective function.
    
    Args:
        use_mc_replicates: If True, use Monte Carlo cross-validation on X_train/y_train instead of X_test/y_test (default: True)
        n_splits: Number of folds for cross-validation (default: 20)
        n_repeats: Number of times to repeat the k-fold split with a different random seed each time (default: 1, i.e. plain KFold)
    """
    def objective(trial):
        params = wrapper_class.suggest_specific_params(trial)
        params.update(base_class.suggest_specific_params(trial))
        model = None
        job_seeds = np.random.SeedSequence(args.seed).spawn(args.n_jobs)
        replicate_seeds = job_seeds[args.job_index].spawn(n_splits)

        try:
            if use_mc_replicates:
                scores = []
                for rep_idx, seed_seq in enumerate(replicate_seeds):
                    seed = int(seed_seq.generate_state(1)[0])
                    splitter = ShuffleSplit(n_splits=1, test_size=args.test_size, random_state=seed)
                    train_idx, val_idx = next(splitter.split(X))

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
                    import torch
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
    pruner = MedianPruner(n_startup_trials=np.round(n_trials * 0.125, decimals=0), n_warmup_steps=5)    
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