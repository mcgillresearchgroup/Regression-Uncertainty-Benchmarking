"""Cache the 5 OpenML-sourced datasets directly from the Hugging Face mirror of the
Grinsztajn et al. tabular-benchmark suite (inria-soda/tabular-benchmark), bypassing
OpenML's API/website entirely. Run this from inside Benchmarking/ (same directory as
data_dictionaries.py) on a node with internet access.

Combined_Cycle_Power_Plant is a UCI dataset, not OpenML, so it isn't affected by an
OpenML outage and isn't included here -- use the normal caching flow for it.
"""
import pandas as pd
import pickle
from pathlib import Path

# name -> (huggingface CSV url, cache filename under dataset_cache/)
# Target column is the LAST column in every reg_num_* file in this benchmark suite
# (confirmed for cpu_act: last column is "usr").
DATASETS_TO_CACHE = {
    "Cpu_Act": (
        "https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/reg_num/cpu_act.csv",
        "openml_44132.pkl",
    ),
    "Ailerons": (
        "https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/reg_num/Ailerons.csv",
        "openml_44135.pkl",
    ),
    "Houses_OpenML": (
        "https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/reg_num/houses.csv",
        "openml_44141.pkl",
    ),
    "Elevators": (
        "https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/reg_num/elevators.csv",
        "openml_44133.pkl",
    ),
    "Pol_OpenML": (
        "https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/reg_num/pol.csv",
        "openml_44134.pkl",
    ),
}

cache_dir = Path("dataset_cache")  # must sit next to data_dictionaries.py
cache_dir.mkdir(parents=True, exist_ok=True)

for name, (url, cache_filename) in DATASETS_TO_CACHE.items():
    cache_path = cache_dir / cache_filename
    if cache_path.exists():
        print(f"{name}: already cached at {cache_path}, skipping.")
        continue

    print(f"{name}: downloading from {url} ...")
    df = pd.read_csv(url)

    data_y = df.iloc[:, [-1]]
    data_x = df.iloc[:, :-1]
    num_features = data_x.shape[1]
    num_targets = data_y.shape[1] if len(data_y.shape) > 1 else 1

    with open(cache_path, "wb") as f:
        pickle.dump((data_x, data_y, num_features, num_targets), f)

    print(f"  cached to {cache_path}: {data_x.shape[0]} samples, "
          f"{num_features} features, target='{data_y.columns[0]}'")

print("\nDone. Re-run your pre-cache verification snippet to confirm everything loads.")