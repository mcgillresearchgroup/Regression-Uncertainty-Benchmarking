import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error


def get_model_label(wrapper, model):
    """Convert wrapper and model names to clean abbreviated labels."""
    wrapper_map = {
        'MVE_Ensemble': 'MEA',  # Legacy name
        'MVE_Ensemble_Averaged': 'MEA',
        'MVE_Ensemble_Multiplicative': 'MEM',
        'MLP_Ensemble': 'MLP',
    }
    model_map = {
        'MVE_Default': 'D',
        'MVE_Mean_Head_Extension': 'MH',
        'MLP_Default': 'MLP',
    }
    
    wrapper_label = wrapper_map.get(wrapper, wrapper)
    model_label = model_map.get(model, model)
    
    return f"{wrapper_label}-{model_label}"


# Load results
results_dir = Path('.././best_params_and_all_results')
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

# Create comparison plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Concrete Compressive Strength - Model Comparison', fontsize=16, fontweight='bold')

# Prepare data for plotting
models = []
nll_scores = []
mse_scores = []
mae_scores = []
rmse_scores = []
r2_scores = []

for key, runs in results_data.items():
    # Use the last/best run for each wrapper+model combination
    run = runs[-1]
    wrapper = run['wrapper']
    model = run['model']
    
    # Create a clean label
    label = get_model_label(wrapper, model)
    
    models.append(label)
    nll_scores.append(run['metrics']['nll'])
    mse_scores.append(run['metrics']['mse'])
    mae_scores.append(run['metrics']['mae'])
    rmse_scores.append(run['metrics']['rmse'])
    # R² might not be in older result files - use 0 as fallback
    r2_scores.append(run['metrics'].get('r2', 0.0))

# Plot 1: NLL Comparison
ax = axes[0, 0]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = ax.bar(models, nll_scores, color=colors[:len(models)])
ax.set_ylabel('Negative Log Likelihood', fontsize=11, fontweight='bold')
ax.set_title('NLL (Lower is Better)', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, nll_scores)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.4f}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 2: MSE Comparison
ax = axes[0, 1]
bars = ax.bar(models, mse_scores, color=colors[:len(models)])
ax.set_ylabel('Mean Squared Error', fontsize=11, fontweight='bold')
ax.set_title('MSE (Lower is Better)', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, mse_scores)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.2f}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 3: MAE Comparison
ax = axes[1, 0]
bars = ax.bar(models, mae_scores, color=colors[:len(models)])
ax.set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
ax.set_title('MAE (Lower is Better)', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, mae_scores)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.1, f'{val:.4f}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 4: RMSE Comparison
ax = axes[1, 1]
bars = ax.bar(models, rmse_scores, color=colors[:len(models)])
ax.set_ylabel('Root Mean Squared Error', fontsize=11, fontweight='bold')
ax.set_title('RMSE (Lower is Better)', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, rmse_scores)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.1, f'{val:.4f}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 5: R² Comparison
ax = axes[1, 2]
bars = ax.bar(models, r2_scores, color=colors[:len(models)])
ax.set_ylabel('R² Score', fontsize=11, fontweight='bold')
ax.set_title('R² (Higher is Better)', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1])
for i, (bar, val) in enumerate(zip(bars, r2_scores)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.4f}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: model_comparison.png")

# Create prediction vs ground truth plots
n_results = len(results_data)
if n_results == 0:
    print("No results found to plot. Exiting.")
    exit(0)
n_rows = (n_results + n_results - 1) // n_results
fig, axes = plt.subplots(n_rows, n_results, figsize=(5*n_results, 5*n_rows))
axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

fig.suptitle('Actual vs Predicted - Concrete Compressive Strength', fontsize=14, fontweight='bold')

for idx, (key, runs) in enumerate(results_data.items()):
    run = runs[-1]
    ax = axes_flat[idx]
    
    wrapper = run['wrapper']
    model = run['model']
    
    y_true = np.array(run['accuracy']['ground_truth'])
    y_pred = np.array(run['accuracy']['mean'])
    y_var = np.array(run['accuracy']['variance'])
    y_std = np.sqrt(y_var)
    
    # Scatter plot: Actual vs Predicted
    ax.scatter(y_true, y_pred, alpha=0.6, s=50, color='blue', edgecolors='darkblue', linewidth=0.5)
    
    # Add error bars for uncertainty (±2σ)
    ax.errorbar(y_true, y_pred, yerr=2*y_std, fmt='none', ecolor='lightblue', 
                alpha=0.4, linewidth=1, capsize=2, capthick=1)
    
    # Add perfect prediction line (y = x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction', alpha=0.7)
    
    label = get_model_label(wrapper, model)
    title = f'{label}\nNLL: {run["metrics"]["nll"]:.4f}'
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Actual (MPa)', fontsize=10)
    ax.set_ylabel('Predicted (MPa)', fontsize=10)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

# Hide extra subplots if n_results < n_rows*n_cols
for idx in range(n_results, len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.tight_layout()
plt.savefig('predictions_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: predictions_comparison.png")

# Print summary
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
for key, runs in results_data.items():
    run = runs[-1]
    wrapper = run['wrapper']
    model = run['model']
    model_name = get_model_label(wrapper, model)
    print(f"\n{model_name}:")
    print(f"  NLL:  {run['metrics']['nll']:.6f}")
    print(f"  MSE:  {run['metrics']['mse']:.6f}")
    print(f"  MAE:  {run['metrics']['mae']:.6f}")
    print(f"  RMSE: {run['metrics']['rmse']:.6f}")
    r2 = run['metrics'].get('r2', None)
    if r2 is not None:
        print(f"  R²:   {r2:.6f}")
    print(f"  N Samples: {run['metrics']['n_samples']}")

print("\n" + "="*60)
