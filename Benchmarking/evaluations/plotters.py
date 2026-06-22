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

def run_plotting(pae = True, pre = True, hae = True, preesc = True, paeesc = True):
    """Generate comparison plots from trained models.
    Parameters:
        pae: Whether to plot Absolute Error vs. Number of Predictions
        pre: Whether to plot Relative Error vs. Number of Predictions
        hae: Whether to plot Histogram of Absolute Errors
    """
    loaded_data_ensemble, loaded_data_old = load_results()
    if pae:
        plot_absolute_error(loaded_data_old)
    if pre:
        plot_relative_error(loaded_data_old)
    if hae:
        plot_histogram_absolute_error(loaded_data_old)
    if preesc:
        plot_error_ensemble_size_comparisons(loaded_data_ensemble, relative = True)
    if paeesc:
        plot_error_ensemble_size_comparisons(loaded_data_ensemble, absolute = True)


def load_results():
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
    return results_data_ensemble, results_data_old


# Absolute error vs. number of predictions
def plot_absolute_error(results_data_old):
    for key_old, runs in results_data_old.items():
        wrapper, model = key_old.split('-', 1)  # Split into wrapper and model
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
    plt.savefig('absolute_error_vs_predictions.png')
    plt.clf()


# Relative error vs. number of predictions
def plot_relative_error(results_data_old):
    for key_old, runs in results_data_old.items():
        wrapper, model = key_old.split('-', 1)  # Split into wrapper and model
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
    plt.xlim(0, max(sorted_relative_errors))
    plt.ylim(0, max(prediction_count))
    plt.ylabel('Number of Predictions')
    plt.xlabel('Relative Error')
    plt.legend()
    plt.savefig('relative_error_vs_predictions.png')
    plt.clf()


# Histogram of Absolute Errors
def plot_histogram_absolute_error(results_data_old):
    x_max = 2
    y_max = 2
    hist_fig, hist_axs = plt.subplots(y_max, x_max)
    x, y = 0, 0
    bin_count = 15
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    for key_old, runs in results_data_old.items():
        wrapper, model = key_old.split('-', 1)  # Split into wrapper and model
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
        
       
        sns.histplot(absolute_errors, bins=bin_count, kde=True, ax=hist_axs[y, x], color=colors[x+y*x_max % len(colors)], label=label)
        hist_axs[y, x].set_title(f'{label}')
        hist_axs[y, x].set_xlabel('Absolute Error')
        hist_axs[y, x].legend()
        x += 1
        if x >= x_max:
            x = 0
            y += 1
    plt.suptitle('Histogram of Absolute Errors with Fitted Normal Distribution')
    for ax in hist_axs.flat:
        ax.set(ylabel='Frequency')
    plt.tight_layout()
    plt.savefig('histogram_absolute_error.png')
    plt.clf()


def plot_histogram_ensemble_size_comparisons(results_data):  
    pass


def plot_error_ensemble_size_comparisons(results_data_ensemble, relative = False, absolute = False):
    # First does a for loop for each unique dataset/model/wrapper combination, then within that loop it does a for loop for each n_models, and plots the relative error vs. number of predictions for each n_models on the same plot.
    list_of_keys = list(results_data_ensemble.keys())
    unique_combinations = set([f"{key.split('-')[0]}-{key.split('-')[1]}-{key.split('-')[2]}" for key in list_of_keys])  # Get unique dataset x model x wrapper combinations
    for combination in unique_combinations:
        dataset = combination.split('-')[0]
        wrapper = combination.split('-')[1]
        model = combination.split('-')[2]
        label = get_model_label(wrapper, model)
        # Plot absolute error vs. number of predictions for each n_models for this combination
        if absolute:
            for key, runs in results_data_ensemble.items():
                if key.startswith(combination):
                    n_models = key.split('-')[3]
                    absolute_errors = []
                    for run in runs:
                        absolute_error = np.abs(np.array(run['accuracy']['mean']).flatten() - np.array(run['accuracy']['ground_truth']).flatten())
                        absolute_errors.extend(absolute_error.tolist())

                    sorted_absolute_errors = np.sort(absolute_errors)
                    prediction_count = list(range(1, len(sorted_absolute_errors) + 1))

                    plt.plot(sorted_absolute_errors, prediction_count, marker='o', linestyle='-', label=f"{label} (n_models={n_models})", ms=.3)
            plt.title(f'Absolute Error vs. Number of Predictions for {dataset}')
            plt.xlim(0,max(sorted_absolute_errors))
            plt.ylim(0, max(prediction_count))
            plt.ylabel('Number of Predictions')
            plt.xlabel('Absolute Error')
            plt.legend()
            plt.savefig(f'absolute_error_vs_predictions_{dataset}_{wrapper}_{model}.png')
            plt.clf()

        if relative:
            for key, runs in results_data_ensemble.items():
                if key.startswith(combination):
                    n_models = key.split('-')[3]
                    relative_errors = []
                    for run in runs:
                        relative_error = (np.abs(np.array(run['accuracy']['mean']).flatten() - np.array(run['accuracy']['ground_truth']).flatten())) / np.array(run['accuracy']['ground_truth']).flatten()
                        relative_errors.extend(relative_error.tolist())

                    sorted_relative_errors = np.sort(relative_errors)
                    prediction_count = list(range(1, len(sorted_relative_errors) + 1))

                    plt.plot(sorted_relative_errors, prediction_count, marker='o', linestyle='-', label=f"{label} (n_models={n_models})", ms=.3)
            plt.title(f'Relative Error vs. Number of Predictions for {dataset}')
            plt.xlim(0,max(sorted_relative_errors))
            plt.ylim(0, max(prediction_count))
            plt.ylabel('Number of Predictions')
            plt.xlabel('Relative Error')
            plt.legend()
            plt.savefig(f'relative_error_vs_predictions_{dataset}_{wrapper}_{model}.png')
            plt.clf()


run_plotting(pae=False, pre=False, hae=False, preesc=True, paeesc=True)