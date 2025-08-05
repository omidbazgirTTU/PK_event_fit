import torch
import pandas as pd
import numpy as np

# Load the data
pk_data = pd.read_csv('data/pk_simulation_data.csv')
dosing_schedule = pd.read_csv('data/dosing_schedule.csv')

print("Data loaded successfully:")
print(f"PK data shape: {pk_data.shape}")
print(f"Dosing schedule shape: {dosing_schedule.shape}")

days = pk_data['Day'].values
observed_conc = pk_data['Concentration_with_noise (mg/L)'].values
dosing_days = dosing_schedule['Day'].values
dose_amounts = dosing_schedule['Dose_Amount (mg)'].values

print(f"\nData ranges:")
print(f"- Days: {days.min():.3f} to {days.max():.3f}")
print(f"- Concentrations: {observed_conc.min():.3f} to {observed_conc.max():.3f}")
print(f"- Dosing days: {dosing_days}")
print(f"- Dose amounts: {dose_amounts}")
print(f"- Total data points: {len(days)}")

# Check for time issues
time_diffs = np.diff(days)
print(f"\nTime differences:")
print(f"- Min diff: {time_diffs.min():.6f}")
print(f"- Max diff: {time_diffs.max():.6f}")
print(f"- Any negative diffs: {(time_diffs < 0).any()}")

# Check dosing times vs data times
print(f"\nDosing time checks:")
for i, dose_day in enumerate(dosing_days):
    closest_idx = np.argmin(np.abs(days - dose_day))
    closest_time = days[closest_idx]
    print(f"Dose {i+1}: scheduled {dose_day:.3f}, closest data point {closest_time:.3f}, diff {abs(dose_day - closest_time):.6f}")