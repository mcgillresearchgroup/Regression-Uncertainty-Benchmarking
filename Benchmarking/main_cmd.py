"""Launcher for HPO training jobs, one per (wrapper, model, dataset, train_percent)
combination. Designed to be invoked once per SLURM array task via --job-index
(see mc_hpo_launch.sh), the same pattern used in mc_replicate_diagnostic.py/.sh.
"""
import subprocess
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

MODEL_COMBINATIONS = [
    ('MVE_Ensemble_Averaged', 'MVE_Default'),
]

DATASETS = ['Combined_Cycle_Power_Plant', 'Cpu_Act', 'Ailerons', 'Houses_OpenML', 'Elevators', 'Pol_OpenML']

# 20% to 100% in increments of 20%. Each train_percent tier gets its own seed (see
# SEED_BASE/SEED_OFFSET below) -- this mirrors the original main_cmd.py pattern of
# n=1, seed_list=[n, n+1, n+2, n+3, n+4], train_percent = 20 * seed, just generalized
# so the mapping is explicit and reusable.
TRAIN_PERCENTS = [20, 40, 60, 80, 100]

# seed passed to train.py = SEED_BASE + (train_percent // 20 - 1) + SEED_OFFSET
#   train_percent=20  -> seed = SEED_BASE + 0 + SEED_OFFSET
#   train_percent=40  -> seed = SEED_BASE + 1 + SEED_OFFSET
#   ...
#   train_percent=100 -> seed = SEED_BASE + 4 + SEED_OFFSET
# SEED_OFFSET=102 matches the original script's `seed + 102` so seeds don't collide with
# small values used elsewhere; SEED_BASE=1 matches the original `n=1`.
SEED_BASE = 1
SEED_OFFSET = 102


def seed_for_train_percent(train_percent):
    """Map a train_percent tier (20, 40, ..., 100) to its unique seed."""
    tier_index = train_percent // 20 - 1  # 0-based: 20->0, 40->1, ..., 100->4
    return SEED_BASE + tier_index + SEED_OFFSET


# Marker directory: one file per successfully-completed job index. This is the source of
# truth for "did this job finish" -- independent of best_parameters.json, whose dedup key
# doesn't include train_percent and so can't reliably tell different train_percent runs
# for the same dataset/wrapper/model apart.
JOB_STATUS_DIR = Path(__file__).resolve().parent / 'job_status'


def build_combinations():
    """Flatten (wrapper, model) x dataset x train_percent into one indexable list, with
    each combination's unique seed attached.

    Index order (slowest -> fastest varying): model combination, dataset, train_percent.
    This is the order SLURM_ARRAY_TASK_ID will walk through.

    NOTE: if you change MODEL_COMBINATIONS/DATASETS/TRAIN_PERCENTS, indices shift and any
    existing job_status/ markers will no longer line up with the same combo. Re-run
    --list (and clear job_status/ if needed) after making such a change.
    """
    combos = []
    for wrapper, model in MODEL_COMBINATIONS:
        for dataset in DATASETS:
            for train_percent in TRAIN_PERCENTS:
                seed = seed_for_train_percent(train_percent)
                combos.append((wrapper, model, dataset, train_percent, seed))
    return combos


def marker_path(job_index):
    return JOB_STATUS_DIR / f"job_{job_index:04d}.done"


def is_done(job_index):
    return marker_path(job_index).exists()


def mark_done(job_index, wrapper, model, dataset, train_percent, seed):
    JOB_STATUS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_index": job_index,
        "wrapper": wrapper,
        "model": model,
        "dataset": dataset,
        "train_percent": train_percent,
        "seed": seed,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(marker_path(job_index), 'w') as f:
        json.dump(payload, f, indent=2)


def pending_job_indices():
    combos = build_combinations()
    return [i for i in range(len(combos)) if not is_done(i)]


def format_array_ranges(indices):
    """Collapse a sorted list of ints into a compact SLURM --array style string,
    e.g. [0,1,2,5,7,8,9] -> '0-2,5,7-9'."""
    if not indices:
        return ""
    indices = sorted(indices)
    ranges = []
    start = prev = indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
            continue
        ranges.append((start, prev))
        start = prev = i
    ranges.append((start, prev))
    return ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in ranges)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a single HPO training job selected by index (for use with a SLURM job array).'
    )
    parser.add_argument(
        '--job-index', type=int,
        help='0-based index into the flattened combination list (pass $SLURM_ARRAY_TASK_ID here).'
    )
    parser.add_argument('--n-trials', type=int, default=100, help='Optuna trials per job (default: 100).')
    parser.add_argument(
        '--list', action='store_true',
        help='Print every (index, wrapper, model, dataset, train_percent, seed) combo and exit '
             '(use this to size --array, e.g. --array=0-N-1).'
    )
    parser.add_argument(
        '--pending', action='store_true',
        help="Print a compact SLURM --array string (e.g. '3,7,12-15') covering only job "
             "indices that don't yet have a job_status/ completion marker, and exit. "
             "Use like: sbatch --array=$(python main_cmd.py --pending) mc_hpo_launch.sh"
    )
    return parser.parse_args()


def run_job(job_index, n_trials):
    combos = build_combinations()
    if not (0 <= job_index < len(combos)):
        print(f"job-index {job_index} out of range (valid: 0-{len(combos) - 1})")
        sys.exit(1)

    wrapper, model, dataset, train_percent, seed = combos[job_index]

    if is_done(job_index):
        print(f"[job {job_index}/{len(combos) - 1}] {wrapper} + {model} on {dataset} "
              f"@ train_percent={train_percent}% (seed={seed}) already marked done -- skipping.")
        return

    print(f"[job {job_index}/{len(combos) - 1}] {wrapper} + {model} on {dataset} "
          f"@ train_percent={train_percent}% (seed={seed})")

    cmd = [
        sys.executable, "train.py",
        "-d", dataset,
        "-m", model,
        "-w", wrapper,
        "--seed", str(seed),
        "--train-percent", str(train_percent),
        "--optimize",
        "--n-trials", str(n_trials),
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error: job {job_index} ({wrapper}+{model} on {dataset} @ {train_percent}%, seed={seed}) "
              f"exited with code {result.returncode}")
        sys.exit(result.returncode)

    mark_done(job_index, wrapper, model, dataset, train_percent, seed)


def main():
    args = parse_args()

    if args.list:
        combos = build_combinations()
        for i, (wrapper, model, dataset, tp, seed) in enumerate(combos):
            status = "DONE" if is_done(i) else "pending"
            print(f"{i}\t{wrapper}\t{model}\t{dataset}\ttrain_percent={tp}\tseed={seed}\t{status}")
        print(f"\nTotal combinations: {len(combos)}  (use --array=0-{len(combos) - 1})")
        return

    if args.pending:
        pending = pending_job_indices()
        total = len(build_combinations())
        if not pending:
            print("", end="")  # empty --array string
            print(f"# All {total} jobs already completed.", file=sys.stderr)
        else:
            print(format_array_ranges(pending))
            print(f"# {len(pending)}/{total} jobs pending.", file=sys.stderr)
        return

    if args.job_index is None:
        print("Error: --job-index is required unless using --list or --pending.")
        sys.exit(1)

    run_job(args.job_index, args.n_trials)


if __name__ == "__main__":
    main()