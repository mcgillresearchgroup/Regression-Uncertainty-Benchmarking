"""Memory-optimized model wrappers with mini-batch training support."""

import torch
import torch.nn as nn
import numpy as np
import warnings
import gc
import inspect
import gpytorch
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from .base_list import MVE_Mean_Head_Extension, base_list_dict
from abc import ABC, abstractmethod


def data_check(X=None, y=None):
    """Validate that inputs contain no NaNs or infinite values."""
    if X is not None:
        if np.isnan(X).any():
            raise ValueError("NaN detected in input X.")
        if not np.isfinite(X).all():
            raise ValueError("Inf detected in input X.")
    if y is not None:
        if np.isnan(y).any():
            raise ValueError("NaN detected in input y.")
        if not np.isfinite(y).all():
            raise ValueError("Inf detected in input y.")


class Default_Wrapper(ABC):
    default_base = 'MVE_Default'   # name in base_list_dict; used if no base_class is given

    def __init__(self, base_class, num_features, num_targets, lr=0.001, epochs=100, n_models=5,
                 n_layers=2, layer_size=64, batch_size=None,
                 mean_head_n_layers=None, mean_head_layer_size=None):

        # base_class may be passed as an actual class or, for convenience/back-compat, as a
        # name string matching a key in base_list_dict.
        if base_class is None:
            base_class = self.default_base
        if isinstance(base_class, str):
            base_class = base_list_dict[base_class][0]
        self.base_class = base_class

        self.num_features = num_features
        self.num_targets = num_targets
        self.epochs = epochs
        self.lr = lr
        self.n_models = n_models
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.batch_size = batch_size  # None = full batch, int = mini-batch size
        self.mean_head_n_layers = mean_head_n_layers
        self.mean_head_layer_size = mean_head_layer_size
        self.output_variance = base_list_dict.get(self.base_class.__name__, [None, False])[1]
        self.x_scaler = RobustScaler()
        self.y_scaler = RobustScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def fit(self, X, y):
        """Fit the model with optional mini-batch training.
        Inputs:
            X: Input features
            y: Target values
        Returns:
            self: Fitted model instance"""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1) if y.ndim == 1 else np.asarray(y, dtype=np.float32)
        data_check(X=X, y=y)

        #scaling and converting to tensors
        self.num_features = X.shape[1]
        self.num_targets = y.shape[1] if y.ndim > 1 else 1
        X_scaled = self.x_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y)
        X_tensor = torch.from_numpy(X_scaled).float().to(self.device)
        y_tensor = torch.from_numpy(y_scaled).float().to(self.device)
        self.training(X_tensor, y_tensor)
        
        # Clean up GPU memory after training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def create_models(self):
        """Create n_models instances of the specified base model.
        Input:
            self.n_models: Number of models to create
        Returns:
            self.model_set: List of model instances"""
        
        base_kwargs = {k: v for k, v in self.base_class.get_base_specific_params(self).items()}
        self.model_set = []
        for _ in range(self.n_models):
            model = self.base_class(self.num_features, self.num_targets, self.n_layers, self.layer_size, **base_kwargs)
            model = model.to(self.device)
            self.model_set.append(model)

    # Main training section of fit function.
    def training(self, X_tensor, y_tensor):
        """Train models with optional mini-batch support.
        If there is no log(variance), it will use MSE loss. If there is log(variance), it will use GaussianNLLLoss.
        Inputs:
            X_tensor: Scaled input features as a PyTorch tensor
            y_tensor: Scaled target values as a PyTorch tensor
        Returns:
            self.model_set: List of trained model instances"""
        
        self.create_models()
        n_samples = X_tensor.shape[0]
        batch_size = self.batch_size if self.batch_size is not None else n_samples
        
        # Train each model in the ensemble
        for model_idx, model in enumerate(self.model_set):
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            criterion = torch.nn.GaussianNLLLoss()
            rmse_loss_list = []
            training_early_stopped = False

            for epoch in range(self.epochs):
                # Mini-batch training
                rmse_epoch_loss = 0.0
                for batch_start in range(0, n_samples, batch_size):
                    batch_end = min(batch_start + batch_size, n_samples)
                    X_batch = X_tensor[batch_start:batch_end]
                    y_batch = y_tensor[batch_start:batch_end]
                    
                    output = model(X_batch)
                    
                    if self.output_variance:
                        mean_pred, log_var_pred = output
                        var_pred = torch.exp(log_var_pred)
                        loss = criterion(mean_pred.flatten(), y_batch.flatten(), var_pred.flatten())
                    else:
                        mean_pred = output
                        loss = torch.nn.functional.mse_loss(mean_pred.flatten(), y_batch.flatten())
                        
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    rmse_epoch_loss += torch.nn.functional.mse_loss(mean_pred.flatten(), y_batch.flatten()).item() * (batch_end - batch_start)
                print(f"Model {model_idx + 1}/{self.n_models}, Epoch {epoch + 1}/{self.epochs}, RMSE Loss: {rmse_epoch_loss/n_samples:.4f}")
                
                # Early stopping here

            # Move model to CPU after training to free GPU memory
            model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def cross_validate(self, X, y, n_splits=10, train_percent=100, subsample_seed=67):
        """Perform cross-validation and return metrics.
        Inputs:
            X: Input features
            y: Target values
            n_splits: Number of cross-validation folds
            train_percent: Percentage (0-100] of each fold's TRAINING portion to actually
                train on. The validation portion is never subsampled, so folds stay
                comparable across different train_percent values and metrics reflect model
                quality, not a smaller/noisier validation set. Default 100 trains on the
                full training portion of each fold (previous behavior).
            subsample_seed: Seed for the train_percent subsampling (varied per fold so
                different folds don't all keep the exact same relative subset).
        Returns:
            mean_pred: Mean predictions across folds
            var_pred: Variance predictions across folds (if output_variance == True: returns variance, else: returns None)

        A fold that raises during fit/predict (e.g. a rare numerical instability such as
        gpytorch's NotPSDError) is logged and skipped rather than crashing the whole run,
        so one bad fold doesn't cost every other fold's results. Raises if every fold fails.
        """

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=67)
        mean_preds = []
        var_preds = []
        y_true = []
        failed_folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            if train_percent < 100:
                rng = np.random.RandomState(subsample_seed + fold_idx)
                train_size = max(1, int(len(train_idx) * (train_percent / 100)))
                train_idx = rng.choice(train_idx, size=train_size, replace=False)

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            try:
                self.fit(X_train, y_train)
                mean_pred, var_pred = self.predict(X_val)
            except Exception as e:
                print(f"Warning: fold {fold_idx} failed ({e}); skipping this fold.")
                failed_folds.append(fold_idx)
                continue

            mean_preds.append(mean_pred)
            var_preds.append(var_pred)
            y_true.append(y_val)

        if not mean_preds:
            raise RuntimeError(f"All {n_splits} cross-validation folds failed; no results to return.")
        if failed_folds:
            print(f"Warning: {len(failed_folds)}/{n_splits} folds failed and were skipped: {failed_folds}")

        mean_pred = np.concatenate(mean_preds, axis=0)
        var_pred = np.concatenate(var_preds, axis=0) if self.output_variance else None
        y_true = np.concatenate(y_true, axis=0)
        
        return mean_pred, var_pred, y_true
    
    def get_wrapper_specific_params(self):
        return {
            'lr': self.lr,
            'epochs': self.epochs,
            'n_models': self.n_models,
            'n_layers': self.n_layers,
            'layer_size': self.layer_size,
            'batch_size': self.batch_size,
        }

    def get_save_state(self):
        hyperparams = self.get_wrapper_specific_params()
        hyperparams.update(self.model_set[0].get_base_specific_params())
        return {
            'wrapper_name': self.__class__.__name__,
            'base_class': self.base_class.__name__,
            'num_features': self.num_features,
            'num_targets': self.num_targets,
            'model_states': [m.state_dict() for m in self.model_set],
            'x_scaler': self.x_scaler,
            'y_scaler': self.y_scaler,
            'hyperparams': hyperparams,
        }
    
    @classmethod
    def suggest_specific_params(cls, trial):
        return {
            'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
            'epochs': trial.suggest_int('epochs', 20, 200, step=20),
            'n_layers': trial.suggest_int('n_layers', 2, 8),
            'layer_size': trial.suggest_int('layer_size', 16, 120, step=8),
            'batch_size': trial.suggest_int('batch_size', 128, 2048, step=128)
        }

    @classmethod
    def load_from_state(cls, state):
        model = cls(
            base_class=state['base_class'],
            num_features=state['num_features'],
            num_targets=state['num_targets'],
            **state['hyperparams']
        )
        model.create_models()
        for m, sd in zip(model.model_set, state['model_states']):
            m.load_state_dict(sd)
            m.eval()
        model.x_scaler = state['x_scaler']
        model.y_scaler = state['y_scaler']
        return model

    @abstractmethod
    def predict(self, X):
        '''Predict method to be implemented by subclasses.
        Input:
            X: Input features for prediction
        Returns:
            Predictions (mean and log(var) if output_variance == True, else mean only)'''
        pass


class MVE_Ensemble_Averaged(Default_Wrapper):

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        data_check(X=X)
        
        X_scaled = self.x_scaler.transform(X)
        X_tensor = torch.from_numpy(X_scaled).float()
        
        preds_mean_list = []
        preds_var_list = []
        
        for model in self.model_set:
            model.eval()
            model = model.to(self.device)
            X_batch = X_tensor.to(self.device)
            with torch.no_grad():
                mean, log_var = model(X_batch)
                var = torch.exp(log_var)
                preds_mean_list.append(mean.cpu().numpy())
                preds_var_list.append(var.cpu().numpy())
            model.cpu()
        
        # Memory-efficient stacking and processing
        preds_mean = np.stack(preds_mean_list, axis=0)  # (n_models, n_samples, n_targets)
        preds_var = np.stack(preds_var_list, axis=0)
        del preds_mean_list, preds_var_list
        gc.collect()
        
        n_models, n_samples, n_targets = preds_mean.shape
        
        # Reshape for inverse_transform
        preds_mean_reshaped = preds_mean.reshape(-1, n_targets)
        preds_var_reshaped = preds_var.reshape(-1, n_targets)
        
        # Inverse transform
        preds_mean_unscaled = self.y_scaler.inverse_transform(preds_mean_reshaped)
        preds_var_unscaled = preds_var_reshaped * (self.y_scaler.scale_ ** 2)
        
        # Reshape back
        preds_mean_unscaled = preds_mean_unscaled.reshape(n_models, n_samples, n_targets)
        preds_var_unscaled = preds_var_unscaled.reshape(n_models, n_samples, n_targets)
        
        # Ensemble aggregation
        mean_ensemble = preds_mean_unscaled.mean(axis=0)
        aleatoric_var = preds_var_unscaled.mean(axis=0)
        epistemic_var = preds_mean_unscaled.var(axis=0)
        if n_models == 1:
            var_ensemble = aleatoric_var
        else:
            var_ensemble = aleatoric_var + epistemic_var
        return mean_ensemble, var_ensemble


class MLP_Ensemble(Default_Wrapper):
    default_base = 'MLP_Default'

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        data_check(X=X)
        
        X_scaled = self.x_scaler.transform(X)
        X_tensor = torch.from_numpy(X_scaled).float()
        
        preds_mean_list = []
        for model in self.model_set:
            model.eval()
            model = model.to(self.device)
            X_batch = X_tensor.to(self.device)
            with torch.no_grad():
                mean = model(X_batch)
                preds_mean_list.append(mean.cpu().numpy())
            model.cpu()
        
        # Memory-efficient stacking
        preds_mean = np.stack(preds_mean_list, axis=0)  # (n_models, n_samples, n_targets)
        del preds_mean_list
        gc.collect()
        
        n_models, n_samples, n_targets = preds_mean.shape
        preds_mean_reshaped = preds_mean.reshape(-1, n_targets)
        
        # Inverse transform
        preds_mean_unscaled = self.y_scaler.inverse_transform(preds_mean_reshaped)
        preds_mean_unscaled = preds_mean_unscaled.reshape(n_models, n_samples, n_targets)
        
        # Ensemble aggregation (epistemic uncertainty only)
        mean_ensemble = preds_mean_unscaled.mean(axis=0)
        epistemic_var = preds_mean_unscaled.var(axis=0)
        var_ensemble = epistemic_var
        
        return mean_ensemble, var_ensemble


class MVE_Ensemble_Multiplicative(Default_Wrapper):
    """Ensemble using multiplicative aggregation."""

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        data_check(X=X)
        
        X_scaled = self.x_scaler.transform(X)
        X_tensor = torch.from_numpy(X_scaled).float()
        
        preds_mean_list = []
        preds_var_list = []
        
        for model in self.model_set:
            model.eval()
            model = model.to(self.device)
            X_batch = X_tensor.to(self.device)
            with torch.no_grad():
                mean, log_var = model(X_batch)
                var = torch.exp(log_var)
                preds_mean_list.append(mean.cpu().numpy())
                preds_var_list.append(var.cpu().numpy())
            model.cpu()
        
        # Memory-efficient stacking
        preds_mean = np.stack(preds_mean_list, axis=0)
        preds_var = np.stack(preds_var_list, axis=0)
        del preds_mean_list, preds_var_list
        gc.collect()
    
        n_models, n_samples, n_targets = preds_mean.shape
        
        # Reshape for inverse_transform
        preds_mean_reshaped = preds_mean.reshape(-1, n_targets)
        preds_var_reshaped = preds_var.reshape(-1, n_targets)
        
        # Inverse transform
        preds_mean_unscaled = self.y_scaler.inverse_transform(preds_mean_reshaped)
        preds_var_unscaled = preds_var_reshaped * (self.y_scaler.scale_ ** 2)
        
        # Reshape back
        preds_mean_unscaled = preds_mean_unscaled.reshape(n_models, n_samples, n_targets)
        preds_var_unscaled = preds_var_unscaled.reshape(n_models, n_samples, n_targets)
        
        for i in range(n_models - 1):
            if i == 0:
                mean_1 = preds_mean_unscaled[i]
                var_1 = preds_var_unscaled[i]
            mean_2 = preds_mean_unscaled[(i + 1)]
            var_2 = preds_var_unscaled[(i + 1)]
            multiplicative_mean = ((mean_1 * var_2**2) + (mean_2 * var_1**2)) / (var_1**2 + var_2**2)
            multiplicative_var = 1 / ((1 / (var_1 **2)) + (1 / (var_2 **2)))
            mean_1 = multiplicative_mean
            var_1 = multiplicative_var

        return multiplicative_mean, multiplicative_var


from torch.utils.data import TensorDataset, DataLoader
import gpytorch

class GP_Wrapper(Default_Wrapper):
    default_base = 'SVGPModel'

    def __init__(self, base_class=None, num_features=None, num_targets=None, 
                 lr=0.01, epochs=50, num_inducing=500, batch_size=1024, **kwargs):
        # Initialize standard wrapper fields
        super().__init__(
            base_class=base_class, num_features=num_features, num_targets=num_targets,
            lr=lr, epochs=epochs, batch_size=batch_size
        )
        self.num_inducing = num_inducing
        self.kwargs = kwargs
        self.output_variance = True
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)

    def create_models(self):
        """GP models need data tensors to initialize inducing points. 
        Instantiation is deferred to the training method where tensors are ready."""
        self.model_set = []

    def training(self, X_tensor, y_tensor):
        n_samples = X_tensor.shape[0]
        fraction = self.kwargs.get('inducing_fraction', 0.10)
        target_inducing = int(n_samples * fraction)

        num_inducing = max(100, min(target_inducing, 2000))
        if num_inducing > n_samples:
            num_inducing = n_samples
        inducing_idx = np.random.choice(n_samples, num_inducing, replace=False)
        inducing_points = X_tensor[inducing_idx].clone()
        
        model = self.base_class(inducing_points=inducing_points, num_features=self.num_features)
        model = model.to(self.device)
        self.likelihood = self.likelihood.to(self.device)
        self.model_set = [model]
        
        model.train()
        self.likelihood.train()
        
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(self.likelihood.parameters()), lr=self.lr
        )
        
        # Variational ELBO as the loss objective
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, model, num_data=y_tensor.size(0))
        
        batch_size = self.batch_size if self.batch_size is not None else n_samples
        train_dataset = TensorDataset(X_tensor, y_tensor.flatten())
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                # Randomly-sampled inducing points can occasionally end up near-duplicate
                # (especially on datasets with many repeated/discretized feature values),
                # which makes the inducing-point covariance matrix ill-conditioned. The
                # default jitter ceiling (~1e-6) and retry budget are too tight for that;
                # widen both so a marginally-singular matrix doesn't crash the whole job.
                with gpytorch.settings.cholesky_jitter(float_value=1e-3), \
                     gpytorch.settings.cholesky_max_tries(10):
                    output = model(x_batch)
                    loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            print(f"GP Model 1/1, Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss / len(train_loader):.4f}")
            
        # Move back to CPU to save GPU memory
        model.cpu()
        self.likelihood.cpu()
        gc.collect()

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        data_check(X=X)
        
        X_scaled = self.x_scaler.transform(X)
        X_tensor = torch.from_numpy(X_scaled).float().to(self.device)
        
        model = self.model_set[0]
        model.eval()
        self.likelihood.eval()
        model = model.to(self.device)
        self.likelihood = self.likelihood.to(self.device)
        

        with torch.inference_mode(), \
             gpytorch.settings.cholesky_jitter(float_value=1e-3), \
             gpytorch.settings.cholesky_max_tries(10):
            predictions = self.likelihood(model(X_tensor))
            y_pred = predictions.mean.cpu().numpy().reshape(-1, self.num_targets)
            y_var = (predictions.stddev.cpu().numpy() ** 2).reshape(-1, self.num_targets)
            
        model.cpu()
        self.likelihood.cpu()
        del X_tensor, predictions
        
        y_pred_unscaled = self.y_scaler.inverse_transform(y_pred)
        y_var_unscaled = y_var * (self.y_scaler.scale_.reshape(1, -1) ** 2)
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return y_pred_unscaled, y_var_unscaled

    def get_wrapper_specific_params(self):
        return {
            'lr': self.lr,
            'epochs': self.epochs,
            'num_inducing': self.num_inducing,
            'batch_size': self.batch_size,
        }

    @classmethod
    def suggest_specific_params(cls, trial):
        return {
            'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
            'epochs': trial.suggest_int('epochs', 20, 100, step=10),
            'batch_size': trial.suggest_int('batch_size', 128, 2048, step=128),
            'inducing_fraction': trial.suggest_float('inducing_fraction', 0.05, 0.15)
        }


class MVE_Warmup_Phase(Default_Wrapper):
    """Wrapper for MVE with a warmup phase of 10% of epochs for the mean head only, then the full model for the remaining epochs."""

    def training(self, X_tensor, y_tensor):
        """Train models with optional mini-batch support and a warmup phase.
        During warmup (first 10% of epochs), only the mean head is trained using MSE loss,
        regardless of whether variance is output. After warmup, if log(variance) is present,
        training switches to GaussianNLLLoss; otherwise it continues with MSE loss.
        Inputs:
            X_tensor: Scaled input features as a PyTorch tensor
            y_tensor: Scaled target values as a PyTorch tensor
        Returns:
            self.model_set: List of trained model instances"""

        self.create_models()
        n_samples = X_tensor.shape[0]
        batch_size = self.batch_size if self.batch_size is not None else n_samples

        # Number of warmup epochs (mean-only training)
        warmup_epochs = int(0.1 * self.epochs) if self.output_variance else 0

        # Train each model in the ensemble
        for model_idx, model in enumerate(self.model_set):
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            criterion = torch.nn.GaussianNLLLoss()
            rmse_loss_list = []
            training_early_stopped = False

            for epoch in range(self.epochs):
                is_warmup = epoch < warmup_epochs

                # Mini-batch training
                rmse_epoch_loss = 0.0
                for batch_start in range(0, n_samples, batch_size):
                    batch_end = min(batch_start + batch_size, n_samples)
                    X_batch = X_tensor[batch_start:batch_end]
                    y_batch = y_tensor[batch_start:batch_end]

                    output = model(X_batch)

                    if self.output_variance:
                        mean_pred, log_var_pred = output
                        if is_warmup:
                            # Warmup: train only the mean head with MSE loss.
                            # log_var_pred is excluded from the loss so the variance
                            # head receives no gradient signal during this phase.
                            loss = torch.nn.functional.mse_loss(mean_pred.flatten(), y_batch.flatten())
                        else:
                            var_pred = torch.exp(log_var_pred)
                            loss = criterion(mean_pred.flatten(), y_batch.flatten(), var_pred.flatten())
                    else:
                        mean_pred = output
                        loss = torch.nn.functional.mse_loss(mean_pred.flatten(), y_batch.flatten())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    rmse_epoch_loss += torch.nn.functional.mse_loss(mean_pred.flatten(), y_batch.flatten()).item() * (batch_end - batch_start)

                phase_label = "Warmup" if is_warmup else "Full"
                print(f"Model {model_idx + 1}/{self.n_models}, Epoch {epoch + 1}/{self.epochs} [{phase_label}], RMSE Loss: {rmse_epoch_loss/n_samples:.4f}")

                # Early stopping here

            # Move model to CPU after training to free GPU memory
            model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def predict(self, X):
            X = np.asarray(X, dtype=np.float32)
            data_check(X=X)
            
            X_scaled = self.x_scaler.transform(X)
            X_tensor = torch.from_numpy(X_scaled).float()
            
            preds_mean_list = []
            preds_var_list = []
            
            for model in self.model_set:
                model.eval()
                model = model.to(self.device)
                X_batch = X_tensor.to(self.device)
                with torch.no_grad():
                    mean, log_var = model(X_batch)
                    var = torch.exp(log_var)
                    preds_mean_list.append(mean.cpu().numpy())
                    preds_var_list.append(var.cpu().numpy())
                model.cpu()
            
            # Memory-efficient stacking and processing
            preds_mean = np.stack(preds_mean_list, axis=0)  # (n_models, n_samples, n_targets)
            preds_var = np.stack(preds_var_list, axis=0)
            del preds_mean_list, preds_var_list
            gc.collect()
            
            n_models, n_samples, n_targets = preds_mean.shape
            
            # Reshape for inverse_transform
            preds_mean_reshaped = preds_mean.reshape(-1, n_targets)
            preds_var_reshaped = preds_var.reshape(-1, n_targets)
            
            # Inverse transform
            preds_mean_unscaled = self.y_scaler.inverse_transform(preds_mean_reshaped)
            preds_var_unscaled = preds_var_reshaped * (self.y_scaler.scale_ ** 2)
            
            # Reshape back
            preds_mean_unscaled = preds_mean_unscaled.reshape(n_models, n_samples, n_targets)
            preds_var_unscaled = preds_var_unscaled.reshape(n_models, n_samples, n_targets)
            
            # Ensemble aggregation
            mean_ensemble = preds_mean_unscaled.mean(axis=0)
            aleatoric_var = preds_var_unscaled.mean(axis=0)
            epistemic_var = preds_mean_unscaled.var(axis=0)
            if n_models == 1:
                var_ensemble = aleatoric_var
            else:
                var_ensemble = aleatoric_var + epistemic_var
            return mean_ensemble, var_ensemble

    
        
    

wrapper_list_dict = {
    'MVE_Ensemble_Averaged': [MVE_Ensemble_Averaged, True],
    'MLP_Ensemble': [MLP_Ensemble, False],
    'MVE_Ensemble_Multiplicative': [MVE_Ensemble_Multiplicative, True],
    'GP_Wrapper': [GP_Wrapper, True] 
}