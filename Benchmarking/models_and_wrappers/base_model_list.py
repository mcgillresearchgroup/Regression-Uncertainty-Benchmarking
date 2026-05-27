from abc import abstractmethod
import torch.nn as nn
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, RegressorMixin

warnings.filterwarnings('ignore', message='.*X has feature names.*')

# Base model class
class BaseModel(nn.Module):
    def __init__(self, n_layers, layer_size, num_features, num_targets = 1, dropout=0.0):
        super().__init__()
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.num_features = num_features
        self.num_targets = num_targets
        self.size_in = num_features
        self.dropout = dropout
    
    # Default architecture
    def module_sequence_body(self):
        self.module_sequence_body_list = nn.ModuleList()
        for i in range(self.n_layers - 1):
            self.size_out = self.layer_size
            self.module_sequence_body_list.append(nn.Linear(self.size_in, self.size_out))
            self.module_sequence_body_list.append(nn.ReLU())
            self.module_sequence_body_list.append(nn.Dropout(self.dropout))  # Add dropout
            self.size_in = self.layer_size
    
    def module_sequence_head(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass


class MVE_Default(BaseModel):
    def __init__(self, n_layers, layer_size, num_features, num_targets, dropout=0.0):
        super().__init__(n_layers, layer_size, num_features, num_targets, dropout)
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
    
    
class MVE_Mean_Head_Extension(BaseModel):
    def __init__(self, n_layers, layer_size, num_features, num_targets, dropout=0.0, mean_head_dropout=0.0, mean_head_n_layers=2, mean_head_layer_size=None):
        super().__init__(n_layers, layer_size, num_features, num_targets, dropout)
        self.mean_head_n_layers = mean_head_n_layers
        self.mean_head_layer_size = mean_head_layer_size if mean_head_layer_size is not None else layer_size
        self.mean_head_dropout = mean_head_dropout
        self.module_sequence_body()
        self.module_sequence_head()

    def module_sequence_head(self):
        # Extended mean head with additional layers
        self.module_sequence_head_list = nn.ModuleList()
        for i in range(self.mean_head_n_layers - 1):
            self.size_out = self.mean_head_layer_size
            self.module_sequence_head_list.append(nn.Linear(self.size_in, self.size_out))
            self.module_sequence_head_list.append(nn.ReLU())
            self.module_sequence_head_list.append(nn.Dropout(self.mean_head_dropout))
            self.size_in = self.mean_head_layer_size
        
    def forward(self, x):
        for layer in self.module_sequence_body_list:
            x = layer(x)
        mean = x
        for layer in self.module_sequence_head_list:
            mean = layer(mean)
        mean = self.mean_head(mean)
        var = self.var_head(x)
        return mean, var
        
class MLP_Default(BaseModel):
    def __init__(self, n_layers, layer_size, num_features, num_targets, dropout=0.0):
        super().__init__(n_layers, layer_size, num_features, num_targets, dropout)
        self.module_sequence_body()
        self.output_head = nn.Linear(self.size_in, num_targets)

    def forward(self, x):
        for layer in self.module_sequence_body_list:
            x = layer(x)
        mean = self.output_head(x)
        return mean
    
# Dictionary of base models. The boolean indicates variance output.
base_model_list_dict = {
    'MVE_Default': [MVE_Default, True],
    'MVE_Mean_Head_Extension': [MVE_Mean_Head_Extension, True],
    'MLP_Default': [MLP_Default, False]
}
