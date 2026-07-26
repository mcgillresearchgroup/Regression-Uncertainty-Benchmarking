"""Launcher for HPO training jobs, one per (wrapper, model, dataset, train_percent)
combination. Designed to be invoked once per SLURM array task via --job-index
(see mc_hpo_launch.sh), the same pattern used in mc_replicate_diagnostic.py/.sh.
"""
import subprocess
import sys
import argparse


MODEL_COMBINATIONS = [
    ('GP_Wrapper', 'SVGPModel'),
]

DATASETS = ['Combined_Cycle_Power_Plant', 'Cpu_Act', 'Ailerons', 'Houses_OpenML', 'Elevators', 'Pol_OpenML']

# 10% to 100% in increments of 10%
TRAIN_PERCENTS = list(range(10, 101, 10))


DEFAULT_SEED = 1


def build_combinations():
    """Flatten (wrapper, model) x dataset x train_percent into one indexable list.

    Index order (slowest -> fastest varying): model combination, dataset, train_percent.
    This is the order SLURM_ARRAY_TASK_ID will walk through.
    """
    combos = []
    for wrapper, model in MODEL_COMBINATIONS:
        for dataset in DATASETS:
            for train_percent in TRAIN_PERCENTS:
                combos.append((wrapper, model, dataset, train_percent))
    return combos


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a single HPO training job selected by index (for use with a SLURM job array).'
    )
    parser.add_argument(
        '--job-index', type=int, required=True,
        help='0-based index into the flattened combination list (pass $SLURM_ARRAY_TASK_ID here).'
    )
    parser.add_argument('--n-trials', type=int, default=100, help='Optuna trials per job (default: 100).')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Seed passed to train.py.')
    parser.add_argument(
        '--list', action='store_true',
        help='Print every (index, wrapper, model, dataset, train_percent) combo and exit '
             '(use this to size --array, e.g. --array=0-N-1).'
    )
    return parser.parse_args()


def run_job(job_index, n_trials, seed):
    combos = build_combinations()
    if not (0 <= job_index < len(combos)):
        print(f"job-index {job_index} out of range (valid: 0-{len(combos) - 1})")
        sys.exit(1)

    wrapper, model, dataset, train_percent = combos[job_index]
    print(f"[job {job_index}/{len(combos) - 1}] {wrapper} + {model} on {dataset} "
          f"@ train_percent={train_percent}%")

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
        print(f"Error: job {job_index} ({wrapper}+{model} on {dataset} @ {train_percent}%) "
              f"exited with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    args = parse_args()

    if args.list:
        combos = build_combinations()
        for i, (wrapper, model, dataset, tp) in enumerate(combos):
            print(f"{i}\t{wrapper}\t{model}\t{dataset}\ttrain_percent={tp}")
        print(f"\nTotal combinations: {len(combos)}  (use --array=0-{len(combos) - 1})")
        return

    run_job(args.job_index, args.n_trials, args.seed)


if __name__ == "__main__":
    main()