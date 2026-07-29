from abc import abstractmethod
import torch
import torch.nn as nn
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, RegressorMixin
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

torch.manual_seed(82)  # Set seed for reproducibility in training
warnings.filterwarnings('ignore', message='.*X has feature names.*')


# Base model class
class Default_Base(nn.Module):
    def __init__(self, num_features, num_targets, n_layers, layer_size, dropout=0.0):
        super().__init__()
        self.num_features = num_features
        self.num_targets = num_targets
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.size_in = num_features
        self.dropout = dropout
    
    def module_sequence_body(self):
        self.module_sequence_body_list = nn.ModuleList()
        for i in range(self.n_layers - 1):
            self.size_out = self.layer_size
            self.module_sequence_body_list.append(nn.Linear(self.size_in, self.size_out))
            self.module_sequence_body_list.append(nn.ReLU())
            self.size_in = self.layer_size
    
    def module_sequence_head(self):
        pass

    def get_base_specific_params(self):
        return {}
    
    @classmethod
    def suggest_specific_params(cls, trial):
        return {}

    @abstractmethod
    def forward(self, x):
        pass


class MVE_Default(Default_Base):
    def __init__(self, num_features, num_targets, n_layers, layer_size):
        super().__init__(num_features, num_targets, n_layers, layer_size)
        self.module_sequence_body()
        self.module_sequence_head()
    
    def module_sequence_head(self):
        self.mean_head = nn.Linear(self.size_in, self.num_targets)
        self.var_head = nn.Linear(self.size_in, self.num_targets)

    def forward(self, x):
        for layer in self.module_sequence_body_list:
            x = layer(x)
        mean = self.mean_head(x)
        var = self.var_head(x)
        return mean, var
    
    
class MVE_Mean_Head_Extension(Default_Base):
    def __init__(self, num_features, num_targets, n_layers, layer_size, mean_head_n_layers=2, mean_head_layer_size=None):
        super().__init__(num_features, num_targets, n_layers, layer_size)
        self.mean_head_n_layers = mean_head_n_layers
        self.mean_head_layer_size = mean_head_layer_size if mean_head_layer_size is not None else layer_size
        self.module_sequence_body()
        self.module_sequence_head()

    def module_sequence_head(self):
        # Extended mean head with additional layers
        self.module_sequence_head_list = nn.ModuleList()
        for i in range(self.mean_head_n_layers):
            self.size_out = self.mean_head_layer_size
            self.module_sequence_head_list.append(nn.Linear(self.size_in, self.size_out))
            self.module_sequence_head_list.append(nn.ReLU())
            self.size_in = self.mean_head_layer_size
        
        # Final output heads
        self.mean_head = nn.Linear(self.size_in, self.num_targets)
        self.var_head = nn.Linear(self.size_in, self.num_targets)

    def get_base_specific_params(self):
        return {
            'mean_head_n_layers': self.mean_head_n_layers,
            'mean_head_layer_size': self.mean_head_layer_size
        }
    
    @classmethod
    def suggest_specific_params(cls, trial):
        return {
            'mean_head_layer_size': trial.suggest_int('mean_head_layer_size', 16, 120, step=8),
            'mean_head_n_layers': trial.suggest_int('mean_head_n_layers', 1, 4)
        }
    
    def forward(self, x):
        for layer in self.module_sequence_body_list:
            x = layer(x)
        mean = x
        for layer in self.module_sequence_head_list:
            mean = layer(mean)
        mean = self.mean_head(mean)
        var = self.var_head(x)
        return mean, var


class MLP_Default(Default_Base):
    def __init__(self, num_features, num_targets, n_layers, layer_size):
        super().__init__(num_features, num_targets, n_layers, layer_size)
        self.module_sequence_body()
        self.output_head = nn.Linear(self.size_in, num_targets)

    def forward(self, x):
        for layer in self.module_sequence_body_list:
            x = layer(x)
        mean = self.output_head(x)
        return mean


class SVGPModel(ApproximateGP):
    def __init__(self, inducing_points, num_features=None, **kwargs):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        ard_dims = num_features if num_features is not None else inducing_points.size(-1)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ard_dims))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def get_base_specific_params(self):
        return {}
    
    @classmethod
    def suggest_specific_params(cls, trial):
        # Add any base-level hyperparameters for optimization here if needed
        return {}
# Dictionary of base models. The boolean indicates variance output.
base_list_dict = {
    'MVE_Default': [MVE_Default, True],
    'MVE_Mean_Head_Extension': [MVE_Mean_Head_Extension, True],
    'MLP_Default': [MLP_Default, False],
    'SVGPModel': [SVGPModel, True]
}