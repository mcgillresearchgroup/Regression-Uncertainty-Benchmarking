import ijson
import json
from matplotlib.pylab import norm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm
import sys
import seaborn as sns
import uncertainty_toolbox as uct
from collections import defaultdict
sys.path.append('../Benchmarking')
from utils import get_model_label


def run_plotting(pae=False, pre=False, hae=False, preesc=False, paeesc=False, pma=False,
                 psh=False, pres=False, pio=False, pms=False, pve=False):
    """
    pae:    Absolute Error vs. Number of Predictions
    pre:    Relative Error vs. Number of Predictions
    hae:    Histogram of Absolute Errors
    preesc: Relative Error vs. Number of Predictions (ensemble size comparison)
    paeesc: Absolute Error vs. Number of Predictions (ensemble size comparison)
    pma:    Miscalibration Area + Adversarial Group Calibration
    psh:    Sharpness (distribution of predicted std devs)
    pres:   Standardized Residuals vs. Predicted Std (z-score check)
    pio:    Ordered Prediction Intervals
    pms:    Metrics Summary heatmaps (CRPS + variance-error correlation)
    pve:    Bivariate Scatter of Absolute Error vs. Predicted Standard Deviation (Variance-Error Scatter)
    """
    index = build_index()  # cheap, no large arrays loaded

    if pae:
        plot_error(index, error_type='absolute')
    if pre:
        plot_error(index, error_type='relative')
    if hae:
        plot_histogram_absolute_error(index)
    if preesc:
        plot_error_ensemble_size_comparisons(index, error_type='relative')
    if paeesc:
        plot_error_ensemble_size_comparisons(index, error_type='absolute')
    if pma:
        plot_miscalibration_area(index)
    if psh:
        plot_sharpness(index)
    if pio:
        plot_intervals_ordered(index)
    if pms:
        plot_metrics_summary(index)
    if pve:
        plot_variance_error_scatter(index)


def build_index():
    """Scan all result files and record their metadata + path, without loading large arrays."""
    results_dir = Path(__file__).parent.parent / 'best_params_and_all_results'
    result_files = sorted(results_dir.glob('*.json'))
    index = []  # list of dicts: {dataset, wrapper, model, n_models, path}

    for file in result_files:
        try:
            with open(file, 'rb') as f:
                # Stream-parse just the top-level scalar fields we need
                parser = ijson.kvitems(f, '')
                meta = {}
                for key, value in parser:
                    if key in ('dataset', 'wrapper', 'model'):
                        meta[key] = value
                    elif key == 'hyperparameters':
                        meta['n_models'] = value.get('n_models')
                    # once we have everything we need, stop reading further
                    if len(meta) >= 4:
                        break
            if not all(k in meta for k in ('dataset', 'wrapper', 'model', 'n_models')):
                raise KeyError('missing required metadata field')
            index.append({**meta, 'path': file})
        except (ijson.JSONError, KeyError) as e:
            print(f"Skipping {file.name}: {e}")
            continue

    return index


def load_full(entry):
    """Fully load one result file's data (including large arrays)."""
    with open(entry['path'], 'r') as f:
        return json.load(f)
    

def compute_error(run, error_type):
    mean = np.array(run['accuracy']['mean']).flatten()
    ground_truth = np.array(run['accuracy']['ground_truth']).flatten()
    abs_error = np.abs(mean - ground_truth)

    if error_type == 'absolute':
        return abs_error
    elif error_type == 'relative':
        return abs_error / ground_truth
    else:
        raise ValueError(f"Unknown error_type: {error_type}")


def plot_error(index, error_type='absolute'):
    """error_type: 'absolute' or 'relative'"""
    grouped = defaultdict(list)
    for entry in index:
        key = (entry['wrapper'], entry['model'])
        grouped[key].append(entry)

    for (wrapper, model), entries in grouped.items():
        label = get_model_label(wrapper, model)

        errors = []
        for entry in entries:
            run = load_full(entry)
            errors.extend(compute_error(run, error_type).tolist())
            del run

        sorted_errors = np.sort(errors)
        prediction_count = list(range(1, len(sorted_errors) + 1))

        plt.plot(sorted_errors, prediction_count, marker='o', linestyle='-', label=label, ms=.3)

    error_label = error_type.capitalize()
    plt.title(f'{error_label} Error vs. Number of Predictions')
    plt.xlim(0, max(sorted_errors))
    plt.ylim(0, max(prediction_count))
    plt.ylabel('Number of Predictions')
    plt.xlabel(f'{error_label} Error')
    plt.legend()
    plt.savefig(f'{error_type}_error_vs_predictions.png')
    plt.clf()


def plot_histogram_absolute_error(index):
    grouped = defaultdict(list)
    for entry in index:
        key = (entry['wrapper'], entry['model'])
        grouped[key].append(entry)

    x_max, y_max = 2, 2
    hist_fig, hist_axs = plt.subplots(y_max, x_max)
    x, y = 0, 0
    bin_count = 15
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']

    for (wrapper, model), entries in grouped.items():
        label = get_model_label(wrapper, model)

        absolute_errors = []
        for entry in entries:
            run = load_full(entry)
            absolute_error = np.abs(
                np.array(run['accuracy']['mean']).flatten() - np.array(run['accuracy']['ground_truth']).flatten()
            )
            absolute_errors.extend(absolute_error.tolist())
            del run, absolute_error

        sns.histplot(absolute_errors, bins=bin_count, kde=True, ax=hist_axs[y, x],
                     color=colors[(x + y * x_max) % len(colors)], label=label)
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


def plot_error_ensemble_size_comparisons(index, error_type='absolute'):
    grouped = defaultdict(list)
    for entry in index:
        key = (entry['dataset'], entry['wrapper'], entry['model'], entry['n_models'])
        grouped[key].append(entry)

    unique_combinations = sorted({(ds, w, m) for (ds, w, m, _n) in grouped})

    for dataset, wrapper, model in unique_combinations:
        label = get_model_label(wrapper, model)

        for (ds, w, m, n_models), entries in grouped.items():
            if (ds, w, m) != (dataset, wrapper, model):
                continue

            errors = []
            for entry in entries:
                run = load_full(entry)
                errors.extend(compute_error(run, error_type).tolist())
                del run

            sorted_errors = np.sort(errors)
            prediction_count = list(range(1, len(sorted_errors) + 1))

            plt.plot(sorted_errors, prediction_count, marker='o', linestyle='-',
                      label=f"{label} (n_models={n_models})", ms=.3)

        error_label = error_type.capitalize()
        plt.title(f'{error_label} Error vs. Number of Predictions for {dataset}')
        plt.xlim(0, max(sorted_errors))
        plt.ylim(0, max(prediction_count))
        plt.ylabel('Number of Predictions')
        plt.xlabel(f'{error_label} Error')
        plt.legend()
        plt.savefig(f'{error_type}_error_vs_predictions_{dataset}_{wrapper}_{model}.png')
        plt.clf()


def plot_miscalibration_area(index):
    matching_entries = [e for e in index]

    grouped = defaultdict(list)
    for entry in matching_entries:
        key = (entry['dataset'], entry['wrapper'], entry['model'], entry['n_models'])
        grouped[key].append(entry)

    datasets = sorted({k[0] for k in grouped})
    n_models_list = sorted({k[3] for k in grouped})

    for dataset in datasets:
        dataset_groups = {k: v for k, v in grouped.items() if k[0] == dataset}
        max_trials = max(len(v) for v in dataset_groups.values())
        n_rows, n_cols = max_trials, len(n_models_list)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), squeeze=False)

        for col_idx, n_models in enumerate(n_models_list):
            matches = {k: v for k, v in dataset_groups.items() if k[3] == n_models}

            for trial_idx in range(n_rows):
                ax = axes[trial_idx][col_idx]
                plotted_anything = False

                for (ds, wrapper, model, nm), entries in matches.items():
                    if trial_idx >= len(entries):
                        continue
                    entry = entries[trial_idx]
                    run = load_full(entry)
                    label = get_model_label(wrapper, model)

                    pred_mean = np.array(run['accuracy']['mean']).flatten()
                    pred_var = np.array(run['accuracy']['variance']).flatten()
                    te_y = np.array(run['accuracy']['ground_truth']).flatten()
                    pred_std = np.sqrt(pred_var)

                    uct.viz.plot_calibration(pred_mean, pred_std, te_y, ax=ax)
                    ax.get_lines()[-1].set_label(label)
                    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_title('')
                    plotted_anything = True

                    del run, pred_mean, pred_var, te_y, pred_std

                if not plotted_anything:
                    ax.set_visible(False)
                    continue

                ax.set_title(f'trial {trial_idx} | n_models={n_models}', fontsize=9)
                ax.legend(fontsize=6)

        plt.tight_layout()
        fig.suptitle(f'Miscalibration Area for {dataset}')
        fig.savefig(f'miscalibration_area_{dataset}.png')
        plt.close(fig)


def _build_uct_grid(index, suptitle_template, filename_template, uct_plot_fn):
    grouped = defaultdict(list)
    for entry in index:
        key = (entry['dataset'], entry['wrapper'], entry['model'], entry['n_models'])
        grouped[key].append(entry)

    datasets = sorted({k[0] for k in grouped})
    n_models_list = sorted({k[3] for k in grouped})

    for dataset in datasets:
        dataset_groups = {k: v for k, v in grouped.items() if k[0] == dataset}
        max_trials = max(len(v) for v in dataset_groups.values())
        n_rows, n_cols = max_trials, len(n_models_list)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

        for col_idx, n_models in enumerate(n_models_list):
            matches = {k: v for k, v in dataset_groups.items() if k[3] == n_models}

            for trial_idx in range(n_rows):
                ax = axes[trial_idx][col_idx]
                plotted_anything = False

                for (ds, wrapper, model, nm), entries in matches.items():
                    if trial_idx >= len(entries):
                        continue
                    entry = entries[trial_idx]
                    run = load_full(entry)
                    label = get_model_label(wrapper, model)

                    pred_mean = np.array(run['accuracy']['mean']).flatten()
                    pred_var = np.array(run['accuracy']['variance']).flatten()
                    te_y = np.array(run['accuracy']['ground_truth']).flatten()
                    pred_std = np.sqrt(pred_var)

                    uct_plot_fn(pred_mean, pred_std, te_y, ax)
                    ax.get_lines()[-1].set_label(label)
                    ax.set_xlabel('')
                    ax.set_ylabel('')
                    ax.set_title('')
                    plotted_anything = True

                    del run, pred_mean, pred_var, te_y, pred_std

                if not plotted_anything:
                    ax.set_visible(False)
                    continue

                ax.set_title(f'trial {trial_idx} | n_models={n_models}', fontsize=9)
                ax.legend(fontsize=6)

        fig.suptitle(suptitle_template.format(dataset=dataset))
        plt.tight_layout()
        fig.savefig(filename_template.format(dataset=dataset))
        plt.close(fig)


def plot_sharpness(index):
    def _fn(pred_mean, pred_std, te_y, ax):
        uct.viz.plot_sharpness(pred_std, ax=ax)

    _build_uct_grid(
        index,
        suptitle_template='Sharpness for {dataset}',
        filename_template='sharpness_{dataset}.png',
        uct_plot_fn=_fn,
    )


def plot_intervals_ordered(index):
    def _fn(pred_mean, pred_std, te_y, ax):
        uct.viz.plot_intervals_ordered(pred_mean, pred_std, te_y, ax=ax)

    _build_uct_grid(
        index,
        suptitle_template='Ordered Prediction Intervals for {dataset}',
        filename_template='intervals_ordered_{dataset}.png',
        uct_plot_fn=_fn,
    )


def plot_metrics_summary(index):
    from scipy.stats import spearmanr

    grouped = defaultdict(list)
    for entry in index:
        key = (entry['dataset'], entry['wrapper'], entry['model'], entry['n_models'])
        grouped[key].append(entry)

    datasets = sorted({k[0] for k in grouped})
    n_models_list = sorted({k[3] for k in grouped})

    for dataset in datasets:
        dataset_groups = {k: v for k, v in grouped.items() if k[0] == dataset}
        max_trials = max(len(v) for v in dataset_groups.values())

        crps_grid    = defaultdict(list)
        corr_grid    = defaultdict(list)

        for (ds, wrapper, model, n_models), entries in dataset_groups.items():
            for trial_idx, entry in enumerate(entries):
                run = load_full(entry)
                pred_mean = np.array(run['accuracy']['mean']).flatten()
                pred_var  = np.array(run['accuracy']['variance']).flatten()
                te_y      = np.array(run['accuracy']['ground_truth']).flatten()
                pred_std  = np.sqrt(pred_var)

                metrics = uct.metrics.get_all_metrics(pred_mean, pred_std, te_y, verbose=False)
                crps_grid[(n_models, trial_idx)].append(metrics['scoring_rule']['crps'])

                abs_err = np.abs(pred_mean - te_y)
                corr, _ = spearmanr(pred_std, abs_err)
                corr_grid[(n_models, trial_idx)].append(corr)

                del run, pred_mean, pred_var, te_y, pred_std, abs_err

        def _to_matrix(grid):
            mat = np.full((len(n_models_list), max_trials), np.nan)
            for (nm, t), vals in grid.items():
                row = n_models_list.index(nm)
                mat[row, t] = np.mean(vals)
            return mat

        crps_mat = _to_matrix(crps_grid)
        corr_mat = _to_matrix(corr_grid)

        fig, (ax_crps, ax_corr) = plt.subplots(1, 2, figsize=(max_trials * 1.5 + 3, len(n_models_list) * 1.2 + 2))

        sns.heatmap(
            crps_mat,
            ax=ax_crps,
            annot=True, fmt='.3f',
            xticklabels=[f'trial {t}' for t in range(max_trials)],
            yticklabels=[f'n={nm}' for nm in n_models_list],
            cmap='YlOrRd',
            cbar_kws={'label': 'CRPS (lower = better)'},
        )
        ax_crps.set_title('CRPS')
        ax_crps.set_xlabel('Trial')
        ax_crps.set_ylabel('n_models')

        sns.heatmap(
            corr_mat,
            ax=ax_corr,
            annot=True, fmt='.3f',
            xticklabels=[f'trial {t}' for t in range(max_trials)],
            yticklabels=[f'n={nm}' for nm in n_models_list],
            cmap='RdYlGn',
            vmin=-1, vmax=1,
            cbar_kws={'label': 'Spearman corr (std vs |error|)'},
        )
        ax_corr.set_title('Variance–Error Correlation')
        ax_corr.set_xlabel('Trial')
        ax_corr.set_ylabel('n_models')

        fig.suptitle(f'Metrics Summary for {dataset}')
        plt.tight_layout()
        fig.savefig(f'metrics_summary_{dataset}.png')
        plt.close(fig)



run_plotting()