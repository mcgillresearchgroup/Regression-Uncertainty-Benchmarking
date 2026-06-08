import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path so we can import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_model_label



# Load results
results_dir = Path(__file__).parent.parent / 'best_params_and_all_results'
result_files = sorted(results_dir.glob('*.json'))
results_data = {}
for file in result_files:
    try:
        with open(file, 'r') as f:
            data = json.load(f)
            # Skip files without required keys (old results)
            if 'wrapper' not in data or 'model' not in data:
                continue
            key = f"{data['wrapper']}_{data['model']}"
            if key not in results_data:
                results_data[key] = []
            results_data[key].append(data)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Skipping {file.name}: {e}")
        continue
# Do calculations:

# Starting with the histogram

bin_count = 15


# The second plot is the absolute error versus the predictions
plt.figure(figsize=(12, 6))

for key, runs in results_data.items():
    wrapper, model = key.split('_', 1)  # Split into wrapper and model
    wrapper = wrapper  # Get actual wrapper name from results
    model = model     # Get actual model name from results
    
    # Extract wrapper and model from first run's data
    wrapper = runs[0]['wrapper']
    model = runs[0]['model']
    label = get_model_label(wrapper, model)
    
    absolute_errors = []
    for run in runs:
        absolute_error = np.abs(np.array(run['accuracy']['mean']).flatten() - np.array(run['accuracy']['ground_truth']).flatten())
        absolute_errors.extend(absolute_error.tolist())
    
    sorted_absolute_errors = np.sort(absolute_errors)
    prediction_count = list(range(1, len(sorted_absolute_errors) + 1))
    
    plt.plot(sorted_absolute_errors, prediction_count, marker='o', linestyle='-', label=label, ms=.3)

plt.title('Absolute Error vs. Number of Predictions')
plt.ylabel('Number of Predictions')
plt.xlabel('Absolute Error')
plt.legend()
plt.show()


for key, runs in results_data.items():
    wrapper, model = key.split('_', 1)  # Split into wrapper and model
    wrapper = wrapper  # Get actual wrapper name from results
    model = model     # Get actual model name from results
    
    # Extract wrapper and model from first run's data
    wrapper = runs[0]['wrapper']
    model = runs[0]['model']
    label = get_model_label(wrapper, model)
    relative_errors = []
    for run in runs:
        relative_error = (np.abs(np.array(run['accuracy']['mean']).flatten() - np.array(run['accuracy']['ground_truth']).flatten())) / np.array(run['accuracy']['ground_truth']).flatten()
        relative_errors.extend(relative_error.tolist())
    
    sorted_relative_errors = np.sort(relative_errors)
    prediction_count = list(range(1, len(sorted_relative_errors) + 1))
    
    plt.plot(sorted_relative_errors, prediction_count, marker='o', linestyle='-', label=label, ms=.3)

plt.title('Relative Error vs. Number of Predictions')
plt.xlim(0,1)
plt.ylabel('Number of Predictions')
plt.xlabel('Relative Error')
plt.legend()
plt.show()