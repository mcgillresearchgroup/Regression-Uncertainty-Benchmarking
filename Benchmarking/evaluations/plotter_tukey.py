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


from pathlib import Path
import json
import re

import re

import re

def build_index(results_dir=None):
    """
    Scan all result files and record their metadata + path.
    Ensures dataset, wrapper, model, trainpercent, timestamp are set when possible.
    """
    if results_dir is None:
        results_dir = Path(__file__).parent.parent / "best_params_and_all_results"
    else:
        results_dir = Path(results_dir)

    result_files = sorted(results_dir.glob("*.json"))
    index = []

    for file_path in result_files:
        try:
            with open(file_path, "r") as f:
                run = json.load(f)

            # Base values from JSON
            dataset = run.get("dataset")
            wrapper = run.get("wrapper")            # 'GP_Wrapper', 'MVE_Ensemble_Averaged', ...
            base = run.get("base") or run.get("model")
            train_percent = run.get("train_percent")  # often missing -> parse from filename
            timestamp = run.get("timestamp")

            fname = file_path.name

            if dataset is None:
                print(f"[index] Skipping {file_path.name}: dataset is None")
                continue

            index.append({
                "path": file_path,
                "filename": fname,
                "dataset": dataset,
                "wrapper": wrapper,
                "base": base,
                "train_percent": train_percent,
                "timestamp": timestamp,
            })

        except Exception as e:
            print(f"[index] Skipping {file_path.name}: {e}")

    print(f"[index] Built index with {len(index)} entries")
    return index

def _safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

def _safe_rmse(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _safe_nll(y_true, y_pred, y_var=None, y_std=None, eps=1e-8):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if y_std is None:
        if y_var is None:
            y_std = np.ones_like(y_pred)
        else:
            y_std = np.sqrt(np.maximum(np.asarray(y_var).reshape(-1), eps))
    else:
        y_std = np.maximum(np.asarray(y_std).reshape(-1), eps)

    y_std = np.maximum(y_std, eps)
    return float(np.mean(0.5 * np.log(2 * np.pi * y_std**2) + 0.5 * ((y_true - y_pred) ** 2) / (y_std**2)))

def _get_run_arrays(run):
    acc = run.get("accuracy", run)

    # prediction mean
    y_pred = acc.get("mean") or acc.get("y_pred")

    # ground-truth: try several likely keys
    y_true = (
        acc.get("ground_truth")
        or acc.get("y_true")
        or run.get("ground_truth")
        or run.get("y_true")
    )

    # predictive variance / std
    y_var = acc.get("variance")
    y_std = acc.get("std")

    fold_id = acc.get("fold_id", run.get("fold_id"))

    # Normalize to None if not usable
    if y_true is None or y_pred is None:
        return None, None, None, None, fold_id

    return y_true, y_pred, y_var, y_std, fold_id

def compute_fold_metrics(run):
    y_true, y_pred, y_var, y_std, fold_id = _get_run_arrays(run)

    # If we don't have both, we cannot compute metrics
    if y_true is None or y_pred is None:
        return []

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    # fold_id handling
    if fold_id is None:
        n = len(y_true)
        fold_id = np.arange(n)
    else:
        fold_id = np.asarray(fold_id).reshape(-1)
        if len(fold_id) != len(y_true):
            fold_id = np.resize(fold_id, len(y_true))

    out = []
    unique_folds = np.unique(fold_id)
    for fid in unique_folds:
        idx = fold_id == fid
        yt = y_true[idx]
        yp = y_pred[idx]

        if yt.size == 0:
            continue

        # variance/std per fold if available
        if y_var is not None:
            yv = np.asarray(y_var).reshape(-1)
            if len(yv) != len(y_true):
                yv = np.resize(yv, len(y_true))
            yv = yv[idx]
        else:
            yv = None

        if y_std is not None:
            ys = np.asarray(y_std).reshape(-1)
            if len(ys) != len(y_true):
                ys = np.resize(ys, len(y_true))
            ys = ys[idx]
        else:
            ys = None

        out.append({
            "fold_id": fid,
            "r2": _safe_r2(yt, yp),
            "rmse": _safe_rmse(yt, yp),
            "nll": _safe_nll(yt, yp, y_var=yv, y_std=ys),
        })
    return out

def load_full(path):
    with open(path, "r") as f:
        return json.load(f)

def _short_label(wrapper, trainpercent):
    """
    Map wrapper + trainpercent to compact label: GP20, GP40, MVE20, MVE40, etc.
    """
    if trainpercent is None:
        tp_str = ""
    else:
        tp_str = str(int(round(float(trainpercent))))

    w = (wrapper or "").lower()
    if "gp" in w and "mve" not in w:
        prefix = "GP"
    elif "mve" in w:
        prefix = "MVE"
    else:
        prefix = wrapper or "UNK"

    return f"{prefix}{tp_str}"


def make_tukey_plots(index, output_dir="tukey_plots"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group entries by (dataset, wrapper, trainpercent)
    grouped = defaultdict(list)
    for entry in index:
        grouped[(entry["dataset"], entry["wrapper"], entry["train_percent"])].append(entry)

    datasets = sorted({e["dataset"] for e in index if e.get("dataset") not in (None, "best")})
    print(f"[tukey] Datasets found: {datasets}")

    metrics = ["r2", "nll", "rmse"]

    for dataset in datasets:
        dataset_entries = {
            k: v for k, v in grouped.items() if k[0] == dataset
        }
        if not dataset_entries:
            print(f"[tukey] No entries for dataset={dataset}, skipping")
            continue

        print(f"[tukey] Dataset={dataset}, combinations={len(dataset_entries)}")

        metric_values = {m: defaultdict(list) for m in metrics}

        # Collect fold-level metrics per (wrapper, trainpercent)
        for (ds, wrapper, tp), entries in dataset_entries.items():
            label = _short_label(wrapper, tp)
            print(f"[tukey]   Collecting for dataset={dataset}, wrapper={wrapper}, tp={tp}, label={label}, files={len(entries)}")

            for entry in entries:
                run = load_full(entry["path"])
                fold_metrics = compute_fold_metrics(run)
                if not fold_metrics:
                    print(f"[tukey]     No fold metrics for {entry['filename']}, skipping")
                    continue
                for fm in fold_metrics:
                    for m in metrics:
                        val = fm[m]
                        if val is None or np.isnan(val):
                            continue
                        metric_values[m][label].append(val)

        # Plot mean ± std as before
        for metric in metrics:
            labels = []
            means = []
            stds = []

            # Prefer GP20, GP40, MVE20, MVE40 order
            for candidate in ["GP20", "GP40", "MVE20", "MVE40"]:
                vals = metric_values[metric].get(candidate, [])
                vals = [v for v in vals if v is not None and not np.isnan(v)]
                if not vals:
                    continue
                labels.append(candidate)
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

            # Add any extra labels, just in case
            extra_labels = sorted(set(metric_values[metric].keys()) - set(labels))
            for label in extra_labels:
                vals = metric_values[metric][label]
                vals = [v for v in vals if v is not None and not np.isnan(v)]
                if not vals:
                    continue
                labels.append(label)
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

            if not labels:
                print(f"[tukey]   No values for dataset={dataset}, metric={metric}, skipping plot")
                continue

            print(f"[tukey]   Plotting dataset={dataset}, metric={metric}, labels={labels}")

            x = np.arange(len(labels))

            plt.figure(figsize=(8, 5))
            plt.errorbar(
                x,
                means,
                yerr=stds,
                fmt="o",
                capsize=5,
                elinewidth=1.5,
                markeredgewidth=1.5,
            )

            plt.xticks(x, labels)
            plt.title(f"{dataset} - {metric.upper()} (mean ± std)")
            plt.ylabel(metric.upper())
            plt.xlabel("Base / trainpercent")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()

            fname = output_dir / f"{dataset}_{metric}_mean_std.png"
            plt.savefig(fname, dpi=200)
            plt.close()
            print(f"[tukey]     Saved {fname}")

def summarize_fold_metrics(index, out_csv="tukey_summary.csv"):
    rows = []
    for entry in index:
        run = load_full(entry["path"])
        fold_metrics = compute_fold_metrics(run)
        for fm in fold_metrics:
            rows.append({
                "dataset": entry["dataset"],
                "wrapper": entry["wrapper"],
                "base": entry["base"],
                "file": Path(entry["path"]).name,
                "fold_id": fm["fold_id"],
                "r2": fm["r2"],
                "nll": fm["nll"],
                "rmse": fm["rmse"],
            })

    import pandas as pd
    df = pd.DataFrame(rows)

    summary = (
        df.groupby(["dataset", "wrapper", "base"], as_index=False)
          .agg(
              r2_mean=("r2", "mean"),
              r2_var=("r2", "var"),
              nll_mean=("nll", "mean"),
              nll_var=("nll", "var"),
              rmse_mean=("rmse", "mean"),
              rmse_var=("rmse", "var"),
          )
    )

    summary.to_csv(out_csv, index=False)
    return df, summary



if __name__ == "__main__":
    print("Building index...")
    index = build_index()
    print(f"Indexed {len(index)} files")

    print("Making Tukey plots...")
    make_tukey_plots(index, output_dir="tukey_plots")

    print("Summarizing fold metrics...")
    df, summary = summarize_fold_metrics(index, out_csv="tukey_summary.csv")

    print("Done.")