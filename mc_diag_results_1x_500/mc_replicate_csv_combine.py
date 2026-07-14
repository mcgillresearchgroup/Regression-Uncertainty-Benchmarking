import glob
import pandas as pd

files = glob.glob('mc_diag_Combined_Cycle_Power_Plant_MVE_Ensemble_Averaged_MVE_Default_job*.csv')
combined = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
assert len(combined) == 500, f"expected 500 rows, got {len(combined)} — check for missing/failed jobs"
combined.to_csv('mc_diag_Combined_Cycle_Power_Plant_MVE_Ensemble_Averaged_MVE_Default_combined.csv', index=False)