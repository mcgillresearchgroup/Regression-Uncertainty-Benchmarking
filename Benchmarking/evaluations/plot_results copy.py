import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error


def get_model_label(wrapper, model):
    """Convert wrapper and model names to clean abbreviated labels."""
    wrapper_map = {
        'MVE_Ensemble': 'MEA',  # Legacy name
        'MVE_Ensemble_Averaged': 'MEA',
        'MVE_Ensemble_Multiplicative': 'MEM',
        'MLP_Ensemble': 'MLP',
    }
    model_map = {
        'MVE_Default': 'D',
        'MVE_Mean_Head_Extension': 'MH',
        'MLP_Default': 'MLP',
    }
    
    wrapper_label = wrapper_map.get(wrapper, wrapper)
    model_label = model_map.get(model, model)
    
    return f"{wrapper_label}-{model_label}"


# Load results
results_dir = Path('.././best_params_and_all_results')
result_files = sorted(results_dir.glob('*.json'))

