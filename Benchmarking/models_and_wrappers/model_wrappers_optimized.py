"""Memory-optimized model wrappers with mini-batch training support."""

import torch
import torch.nn as nn
import numpy as np
import warnings
import gc
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, RegressorMixin
from .base_model_list import base_model_list_dict
from abc import ABC, abstractmethod


class Base_Model_Wrapper(ABC):
    output_variance = None  # To be set by subclasses
    def __init__(self, lr=0.001, epochs=100, n_models=5, n_layers=2, layer_size=64, 
                 num_features=10, num_targets=1, dropout=0.1, mean_head_dropout=0.1, 
                 base_model='MVE_Default', batch_size=None):
        self.epochs = epochs
        self.lr = lr
        self.n_models = n_models
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.num_features = num_features
        self.num_targets = num_targets
        self.dropout = dropout
        self.mean_dropout = mean_head_dropout
        self.base_model = base_model
        self.batch_size = batch_size  # None = full batch, int = mini-batch size
        self.x_scaler = RobustScaler()
        self.y_scaler = RobustScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #NaN and Inf check for input data
    def data_check(self, X=None, y=None):
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


    def fit(self, X, y):
        """Fit the model with optional mini-batch training.
        Inputs:
            X: Input features
            y: Target values
        Returns:
            self: Fitted model instance"""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1) if y.ndim == 1 else np.asarray(y, dtype=np.float32)
        self.data_check(X=X, y=y)

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
        
        # Warnings and Value Errors for incompatible base models and wrappers.
        if self.base_model not in base_model_list_dict:
            raise ValueError(f"Base model '{self.base_model}' not found in base_model_list_dict")
        model_class, model_has_variance = base_model_list_dict[self.base_model]
        if self.output_variance and not model_has_variance:
            raise ValueError(f"{self.__class__.__name__} requires variance output, but {self.base_model} does not provide it")
        if not self.output_variance and model_has_variance:
            warnings.warn(f"{self.__class__.__name__} does not use variance, but {self.base_model} provides it")
        
        # Create model list based on n_models
        self.model_set = []
        for _ in range(self.n_models):
            if 'MLP' in self.base_model:
                model = model_class(self.n_layers, self.layer_size, self.num_features, self.num_targets, self.dropout)
            elif 'Mean_Head' in self.base_model:
                model = model_class(self.n_layers, self.layer_size, self.num_features, self.num_targets, self.dropout, self.mean_dropout)
            else:
                model = model_class(self.n_layers, self.layer_size, self.num_features, self.num_targets, self.dropout)
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

            for epoch in range(self.epochs):
                # Mini-batch training
                epoch_loss = 0.0
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
                    epoch_loss += loss.item() * (batch_end - batch_start)
                
                epoch_loss /= n_samples
            
            # Move model to CPU after training to free GPU memory
            model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    @abstractmethod
    def predict(self, X):
        pass


class MVE_Ensemble_Averaged(Base_Model_Wrapper):
    output_variance = True
    def __init__(self, lr=0.001, epochs=100, n_models=5, n_layers=2, layer_size=64, 
                 num_features=10, num_targets=1, dropout=0.0, mean_head_dropout=0.0, 
                 base_model='MVE_Default', batch_size=None):
        super().__init__(lr, epochs, n_models, n_layers, layer_size, num_features, num_targets, 
                        dropout, mean_head_dropout, base_model, batch_size)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.data_check(X=X)
        
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
        preds_var_unscaled = np.exp(preds_var_reshaped) * (self.y_scaler.scale_ ** 2)
        
        # Reshape back
        preds_mean_unscaled = preds_mean_unscaled.reshape(n_models, n_samples, n_targets)
        preds_var_unscaled = preds_var_unscaled.reshape(n_models, n_samples, n_targets)
        
        # Ensemble aggregation
        mean_ensemble = preds_mean_unscaled.mean(axis=0)
        aleatoric_var = preds_var_unscaled.mean(axis=0)
        epistemic_var = preds_mean_unscaled.var(axis=0)
        var_ensemble = aleatoric_var + epistemic_var

        return mean_ensemble, var_ensemble


class MLP_Ensemble(Base_Model_Wrapper):
    output_variance = False
    def __init__(self, lr=0.001, epochs=100, n_models=5, n_layers=2, layer_size=64, 
                 num_features=10, num_targets=1, dropout=0.0, mean_head_dropout=0.0, 
                 base_model='MLP_Default', batch_size=None):
        super().__init__(lr, epochs, n_models, n_layers, layer_size, num_features, num_targets, 
                        dropout, mean_head_dropout=mean_head_dropout, base_model=base_model, batch_size=batch_size)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.data_check(X=X)
        
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


class MVE_Single(MVE_Ensemble_Averaged):
    """Single MVE model (n_models=1)."""
    def __init__(self, lr=0.001, epochs=100, n_layers=2, layer_size=64, num_features=10, 
                 num_targets=1, dropout=0.0, mean_head_dropout=0.0, base_model='MVE_Default', batch_size=None):
        super().__init__(lr, epochs, n_models=1, n_layers=n_layers, layer_size=layer_size, 
                        num_features=num_features, num_targets=num_targets, 
                        dropout=dropout, mean_head_dropout=mean_head_dropout, 
                        base_model=base_model, batch_size=batch_size)


class MVE_Ensemble_Multiplicative(MVE_Ensemble_Averaged):
    """Ensemble using geometric mean aggregation."""
    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.data_check(X=X)
        
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
        
        # Ensemble aggregation using geometric mean
        mean_ensemble = np.prod(preds_mean_unscaled, axis=0) ** (1 / self.n_models)
        aleatoric_var = preds_var_unscaled.mean(axis=0)
        epistemic_var = np.mean((preds_mean_unscaled - mean_ensemble) ** 2, axis=0)
        var_ensemble = aleatoric_var + epistemic_var

        return mean_ensemble, var_ensemble

# Dictionary of model wrappers. The boolean indicates whether the wrapper requires variance output from the base model.
wrapper_list_dict = {
    'MVE_Ensemble_Averaged': [MVE_Ensemble_Averaged, True],
    'MVE_Single': [MVE_Single, True],
    'MLP_Ensemble': [MLP_Ensemble, False],
    'MVE_Ensemble_Multiplicative': [MVE_Ensemble_Multiplicative, True]
}
