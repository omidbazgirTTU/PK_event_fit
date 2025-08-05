import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Set random seed for reproducibility
np.random.seed(42)

# Two-compartment PK model parameters
k10 = 0.1  # elimination rate
k12 = 0.05 # central to peripheral
k21 = 0.03 # peripheral to central

def two_compartment_pk(t, y, k10, k12, k21):
    """Two-compartment PK model"""
    C1, C2 = y
    dC1_dt = -k10*C1 - k12*C1 + k21*C2
    dC2_dt = k12*C1 - k21*C2
    return [dC1_dt, dC2_dt]

# Dosing schedule
dosing_times = [2.0, 5.0, 8.0]  # days
dose_amounts = [100.0, 150.0, 120.0]  # mg
dose_effect = 0.2  # mg dose -> concentration units

# Simulation time span
t_start = 0.0
t_end = 12.0
t_eval = np.linspace(t_start, t_end, 200)

# Initial conditions
y0 = [0.0, 0.0]  # no drug initially

# Piecewise integration with dosing events
def simulate_pk_with_dosing():
    all_times = []
    all_concentrations = []
    
    current_state = y0
    current_time = t_start
    
    # Add events at dosing times plus final time
    event_times = sorted(dosing_times + [t_end])
    
    for i, event_time in enumerate(event_times):
        # Integrate from current time to event time
        t_span = [current_time, event_time]
        t_segment = np.linspace(current_time, event_time, 50)
        
        sol = solve_ivp(
            two_compartment_pk,
            t_span,
            current_state,
            args=(k10, k12, k21),
            t_eval=t_segment,
            method='DOP853'
        )
        
        all_times.extend(sol.t)
        all_concentrations.extend(sol.y[0])  # central compartment concentration
        
        # Update state for next segment
        current_state = [sol.y[0, -1], sol.y[1, -1]]
        current_time = event_time
        
        # Apply dose if this is a dosing time
        if event_time in dosing_times:
            dose_idx = dosing_times.index(event_time)
            current_state[0] += dose_amounts[dose_idx] * dose_effect
    
    return np.array(all_times), np.array(all_concentrations)

# Generate the data
times, true_concentrations = simulate_pk_with_dosing()

# Add realistic noise
noise_level = 0.05  # 5% noise
noise = np.random.normal(0, noise_level * np.mean(true_concentrations), len(true_concentrations))
observed_concentrations = true_concentrations + noise

# Ensure no negative concentrations due to noise
observed_concentrations = np.maximum(observed_concentrations, 0.0)

# Create DataFrames
pk_data = pd.DataFrame({
    'Day': times,
    'True_Concentration (mg/L)': true_concentrations,
    'Concentration_with_noise (mg/L)': observed_concentrations
})

dosing_schedule = pd.DataFrame({
    'Day': dosing_times,
    'Dose_Amount (mg)': dose_amounts
})

# Save to CSV files in the data directory
import os
data_dir = '../../data'
os.makedirs(data_dir, exist_ok=True)

pk_data.to_csv(os.path.join(data_dir, 'pk_simulation_data.csv'), index=False)
dosing_schedule.to_csv(os.path.join(data_dir, 'dosing_schedule.csv'), index=False)

print("Generated synthetic PK data:")
print(f"- Time range: {times.min():.1f} to {times.max():.1f} days")
print(f"- Number of observations: {len(times)}")
print(f"- Concentration range: {true_concentrations.min():.3f} to {true_concentrations.max():.3f} mg/L")
print(f"- Dosing times: {dosing_times}")
print(f"- Dose amounts: {dose_amounts}")

# Plot the generated data
plt.figure(figsize=(12, 6))
plt.plot(times, true_concentrations, '-', label='True Concentration', linewidth=2)
plt.plot(times, observed_concentrations, 'o', alpha=0.6, markersize=4, label='Observed (with noise)')

# Add vertical lines for dosing events
for i, (day, dose) in enumerate(zip(dosing_times, dose_amounts)):
    plt.axvline(x=day, color='red', linestyle='--', alpha=0.5)
    plt.annotate(f'{dose} mg', xy=(day, max(true_concentrations)*0.8), 
                xytext=(day+0.1, max(true_concentrations)*0.9),
                arrowprops=dict(arrowstyle='->', color='red'))

plt.xlabel('Time (days)')
plt.ylabel('Concentration (mg/L)')
plt.title('Synthetic PK Data for Neural ODE Training')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot to images directory
images_dir = '../../images'
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'synthetic_pk_data.png'), dpi=300, bbox_inches='tight')
plt.show()

print("\nData saved to:")
print("- data/pk_simulation_data.csv")
print("- data/dosing_schedule.csv")
print("- images/synthetic_pk_data.png")