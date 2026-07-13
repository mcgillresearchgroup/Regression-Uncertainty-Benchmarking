import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from utils import load_dataset, negative_log_likelihood, wrapper_list_dict, base_list_dict
from hyperparameter_optimization import create_model_wrapper, load_best_parameters


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dataset', required=True)
    p.add_argument('-w', '--wrapper', required=True)
    p.add_argument('-m', '--base', required=True)
    p.add_argument('--params-file', default=None)
    p.add_argument('--test-size', type=float, default=0.1)
    p.add_argument('--master-seed', type=int, default=1)          # fixed, shared across all jobs
    p.add_argument('--n-jobs', type=int, default=50)               # total array size
    p.add_argument('--replicates-per-job', type=int, default=10)   # 50 * 10 = 500
    p.add_argument('--job-index', type=int, required=True)         # 0-based, from the scheduler
    p.add_argument('--num-threads', type=int, default=None)        # pin BLAS/torch threads per job
    args = p.parse_args()

    if args.num_threads is not None:
        import torch
        torch.set_num_threads(args.num_threads)

    X, y, num_features, num_targets = load_dataset(args.dataset)
    wrapper_class = wrapper_list_dict[args.wrapper][0]
    base_class = base_list_dict[args.base][0]

    best = load_best_parameters(args.dataset, args.wrapper, args.base, filepath=args.params_file)
    fixed_params = best.get('hyperparameters', best)

    # Guaranteed non-overlapping seeds: master -> one child per job -> one grandchild per replicate
    job_seeds = np.random.SeedSequence(args.master_seed).spawn(args.n_jobs)
    replicate_seeds = job_seeds[args.job_index].spawn(args.replicates_per_job)

    records = []
    for rep_idx, seed_seq in enumerate(replicate_seeds):
        seed = int(seed_seq.generate_state(1)[0])
        splitter = ShuffleSplit(n_splits=1, test_size=args.test_size, random_state=seed)
        train_idx, val_idx = next(splitter.split(X))

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = create_model_wrapper(wrapper_class, base_class, num_features, num_targets, **fixed_params)
        model.fit(X_tr, y_tr)
        mean_pred, var_pred = model.predict(X_val)

        records.append({
            'job_index': args.job_index,
            'replicate_index': args.job_index * args.replicates_per_job + rep_idx,
            'seed': seed,
            'nll': negative_log_likelihood(y_val, mean_pred, var_pred),
            'mse': mean_squared_error(y_val, mean_pred),
            'mae': mean_absolute_error(y_val, mean_pred),
            'r2': r2_score(y_val, mean_pred),
        })


    out_name = f'mc_diag_{args.dataset}_{args.wrapper}_{args.base}_job{args.job_index:03d}.csv'
    pd.DataFrame(records).to_csv(out_name, index=False)
    # save the output csv to a folder named "mc_diag_results" in the current working directory
    os.makedirs("mc_diag_results", exist_ok=True)
    os.rename(out_name, os.path.join("mc_diag_results", out_name))


if __name__ == '__main__':
    main()