import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('mc_diag_Combined_Cycle_Power_Plant_MVE_Ensemble_Averaged_MVE_Default_combined.csv')

group_sizes = [1, 2, 5, 10, 20, 25, 50, 75, 100]  # capped well below N=500 to avoid sampling-ceiling artifacts
metrics = ['nll', 'mse', 'mae', 'r2']

rng = np.random.default_rng(0)   # fixed seed -> reproducible figure
n_draws = 10000          # draws per group size; higher = smoother band, more compute


def subsample_means(scores, group_sizes, n_draws, rng):
    return {
        g: np.array([rng.choice(scores, size=g, replace=False).mean() for _ in range(n_draws)])
        for g in group_sizes
    }


fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for ax, metric in zip(axes.flat, metrics):
    scores = df[metric].to_numpy()
    grouped = subsample_means(scores, group_sizes, n_draws, rng)

    gs = sorted(grouped)
    means = np.array([grouped[g].mean() for g in gs])
    stds = np.array([grouped[g].std() for g in gs])
    pop_std = scores.std()
    theoretical = pop_std / np.sqrt(gs)

    for g in gs:
        ax.scatter(np.full(len(grouped[g]), g), grouped[g], alpha=0.05, color='steelblue', s=10)

    ax.plot(gs, means, color='black', lw=2, label='mean of group means')
    ax.fill_between(gs, means - stds, means + stds, color='steelblue', alpha=0.25, label='±1 std (empirical)')
    ax.plot(gs, means + theoretical, '--', color='gray', alpha=0.6, label='±1 std (theoretical, CLT)')
    ax.plot(gs, means - theoretical, '--', color='gray', alpha=0.6)

    ax.set_xscale('log')
    ax.set_xlabel('Replicates averaged (g)')
    ax.set_ylabel(metric.upper())
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('mc_replicate_diagnostic.png', dpi=150)
plt.show()