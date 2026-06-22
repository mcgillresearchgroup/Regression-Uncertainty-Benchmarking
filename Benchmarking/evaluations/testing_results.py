import json
from matplotlib.pylab import norm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm
import sys
import seaborn as sns

# Add parent directory to path so we can import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_model_label


results_dir = Path(__file__).parent.parent / 'best_params_and_all_results'
result_files = sorted(results_dir.glob('*.json'))
results_data_ensemble = {}
results_data_old = {}
for file in result_files:
    try:
        with open(file, 'r') as f:
            data = json.load(f)
            key_old = f"{data['wrapper']}-{data['model']}"
            key_ensemble = f"{data['dataset']}-{data['wrapper']}-{data['model']}-{data['hyperparameters']['n_models']}"
            if key_ensemble not in results_data_ensemble:
                results_data_ensemble[key_ensemble] = []
            results_data_ensemble[key_ensemble].append(data)
            if key_old not in results_data_old:
                results_data_old[key_old] = []
            results_data_old[key_old].append(data)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Skipping {file.name}: {e}")
        continue


# Calculate the 25th, 50th, and 75th percentiles of the variance for the concrete dataset for each model and wrapper combination
for key, runs in results_data_ensemble.items():
    variances = [run['accuracy']['variance'] for run in runs]
    flattened_variances = [var for sublist in variances for var in sublist]
    p25 = np.percentile(flattened_variances, 25)
    p50 = np.percentile(flattened_variances, 50)
    p75 = np.percentile(flattened_variances, 75)
    print(f"{key}: 25th percentile: {p25}, 50th percentile: {p50}, 75th percentile: {p75}")