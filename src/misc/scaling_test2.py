import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the two-compartment model
def two_compartment_euler(y, args):
    k10, k12, k21, k13, k31 = args
    C1, C2 = y
    dC1 = -k10*C1 - k12*C1**2 + k21*C2**2 - k13*C1**3
    dC2 =  k12*C1**2 - k21*C2**2 + k31*C1*C2
    return np.array([dC1, dC2])

# Define the scaled model
def scaled_two_compartment_euler(y, args, t0, t1):
    # T = t0 + t * (t1 - t0)  # Convert scaled time back to real time
    dY_dT = two_compartment_euler(y, args) # Compute derivatives
    return (t1 - t0) * dY_dT  # Apply chain rule scaling

# Initial conditions
y0 = np.array([5.0, 10.0])
args = (0.1, 0.05, 0.05, 0.01, 0.01)
t0, t1 = 0.0, 10.0
num_steps = 20
dt_unscaled = (t1 - t0) / num_steps
dt_scaled = 1.0 / num_steps

# Solve using Euler's method
times_unscaled, ys_unscaled = [t0], [y0]
times_scaled, ys_scaled = [0.0], [y0]

y_unscaled = y0.copy()
y_scaled = y0.copy()

def euler_step(y, dt, args):
    return y + dt * two_compartment_euler(y, args)

def scaled_euler_step(y, dt, args, t0, t1):
    return y + dt * scaled_two_compartment_euler(y, args, t0, t1)

for i in range(num_steps):
    # Unscaled step
    y_unscaled = euler_step(y_unscaled, dt_unscaled, args)
    times_unscaled.append(times_unscaled[-1] + dt_unscaled)
    ys_unscaled.append(y_unscaled.copy())

    # Scaled step
    y_scaled = scaled_euler_step(y_scaled, dt_scaled, args, t0, t1)
    times_scaled.append(times_scaled[-1] + dt_scaled)
    ys_scaled.append(y_scaled.copy())

# Convert to NumPy arrays for easy plotting
ys_unscaled = np.array(ys_unscaled)
ys_scaled = np.array(ys_scaled)
times_rescaled = t0 + np.array(times_scaled) * (t1 - t0)  # Convert scaled time back to real time

# Create a table for side-by-side step-by-step comparison
df = pd.DataFrame({
    "Time (Unscaled)": times_unscaled,
    "C1 (Unscaled)": ys_unscaled[:, 0],
    "C2 (Unscaled)": ys_unscaled[:, 1],
    "Time (Rescaled)": times_rescaled,
    "C1 (Rescaled)": ys_scaled[:, 0],
    "C2 (Rescaled)": ys_scaled[:, 1],
})

print(df.head())

# Plot solutions
plt.figure(figsize=(8, 6))
plt.plot(times_unscaled, ys_unscaled[:, 0], label="C1 (Unscaled)", marker='o')
plt.plot(times_unscaled, ys_unscaled[:, 1], label="C2 (Unscaled)", marker='o')
plt.plot(times_rescaled, ys_scaled[:, 0], '--', label="C1 (Rescaled)", marker='s')
plt.plot(times_rescaled, ys_scaled[:, 1], '--', label="C2 (Rescaled)", marker='s')
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title("Euler Integration: Unscaled vs Rescaled")
plt.legend()
plt.show()
