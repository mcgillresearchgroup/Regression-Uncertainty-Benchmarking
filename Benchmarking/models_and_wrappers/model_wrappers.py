"""Memory-optimized model wrappers with mini-batch training support."""

import torch
import torch.nn as nn
import numpy as np
import warnings
import gc
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from .base_model_list import base_model_list_dict
from abc import ABC, abstractmethod


class Base_Model_Wrapper(ABC):
    output_variance = None  # To be set by subclasses
    def __init__(self, lr=0.001, epochs=100, n_models=5, n_layers=2, layer_size=64, 
                 num_features=10, num_targets=1, base_model='MVE_Default', batch_size=None):
        self.epochs = epochs
        self.lr = lr
        self.n_models = n_models
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.num_features = num_features
        self.num_targets = num_targets
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
        
        # Create model list based on n_models. If chosen model_class is MVE_Mean_Head_Extension, it will pass the additional hyperparameters for the mean head extension.
        self.model_set = []
        for _ in range(self.n_models):
            if model_class == 'MVE_Mean_Head_Extension':
                model = model_class(self.n_layers, self.layer_size, self.num_features, self.num_targets, 
                                    mean_head_n_layers=self.mean_head_n_layers, mean_head_layer_size=self.mean_head_layer_size)
            else:
                model = model_class(self.n_layers, self.layer_size, self.num_features, self.num_targets)
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
                    epoch_loss += loss.item() * (batch_end - batch_start)
                    rmse_epoch_loss += torch.nn.functional.mse_loss(mean_pred.flatten(), y_batch.flatten()).item() * (batch_end - batch_start)
                #print(f"Model {model_idx + 1}/{self.n_models}, Epoch {epoch + 1}/{self.epochs}, RMSE Loss: {rmse_epoch_loss}")
                epoch_loss /= n_samples
            
            # Move model to CPU after training to free GPU memory
            model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def cross_validate(self, X, y, n_splits=10):
        """Perform cross-validation and return metrics.
        Inputs:
            X: Input features
            y: Target values
            n_splits: Number of cross-validation folds
        Returns:
            mean_pred: Mean predictions across folds
            var_pred: Variance predictions across folds (if output_variance == True: returns variance, else: returns None)"""
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=67)
        mean_preds = []
        var_preds = []
        y_true = []
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            self.fit(X_train, y_train)
            mean_pred, var_pred = self.predict(X_val)
            mean_preds.append(mean_pred)
            var_preds.append(var_pred)
            y_true.append(y_val)

        mean_pred = np.concatenate(mean_preds, axis=0)
        var_pred = np.concatenate(var_preds, axis=0) if self.output_variance else None
        y_true = np.concatenate(y_true, axis=0)
        
        return mean_pred, var_pred, y_true

    @abstractmethod
    def predict(self, X):
        '''Predict method to be implemented by subclasses.
        Input:
            X: Input features for prediction
        Returns:
            Predictions (mean and log(var) if output_variance == True, else mean only)'''
        pass


class MVE_Ensemble_Averaged(Base_Model_Wrapper):
    output_variance = True
    def __init__(self, lr=0.001, epochs=100, n_models=5, n_layers=2, layer_size=64, 
                 num_features=10, num_targets=1, base_model='MVE_Default', batch_size=None):
        super().__init__(lr, epochs, n_models, n_layers, layer_size, 
                        num_features, num_targets, base_model, batch_size)

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
        preds_var_unscaled = preds_var_reshaped * (self.y_scaler.scale_ ** 2)
        
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
                 num_features=10, num_targets=1, base_model='MLP_Default', batch_size=None):
        super().__init__(lr, epochs, n_models, n_layers, layer_size, 
                        num_features, num_targets, base_model, batch_size)

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
    output_variance = True
    def __init__(self, lr=0.001, epochs=100, n_layers=2, layer_size=64, num_features=10, 
                 num_targets=1, base_model='MVE_Default', batch_size=None):
        super().__init__(lr, epochs, n_layers, layer_size, num_features, 
                        num_targets, base_model, batch_size, n_models=1)
    def predict(self, X):
        mean, var = super().predict(X)
        return mean, var


class MVE_Ensemble_Multiplicative(MVE_Ensemble_Averaged):
    """Ensemble using multiplicative aggregation."""
    output_variance = True
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

# Dictionary of model wrappers. The boolean indicates whether the wrapper requires variance output from the base model.
wrapper_list_dict = {
    'MVE_Ensemble_Averaged': [MVE_Ensemble_Averaged, True],
    'MVE_Single': [MVE_Single, True],
    'MLP_Ensemble': [MLP_Ensemble, False],
    'MVE_Ensemble_Multiplicative': [MVE_Ensemble_Multiplicative, True]
}
