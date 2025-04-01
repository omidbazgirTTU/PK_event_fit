import jax
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt
import numpy as np

# Define the two-compartment model without dosing
def two_compartment_model(t, y, args):
    k10, k12, k21, _, _ = args
    C1, C2 = y

    # Define the ODEs
    dC1_dt = -k10 * C1 - k12 * C1 + k21 * C2
    dC2_dt = k12 * C1 - k21 * C2

    return jnp.array([dC1_dt, dC2_dt])

# Define the dosing event
def dosing_event(t, y, args):
    _, _, _, dose_times, dose_amounts = args
    dose = jnp.sum(jnp.where(jnp.isclose(t, dose_times), dose_amounts, 0.0))
    C1, C2 = y
    return jnp.array([C1 + dose, C2])

# Parameters
k10, k12, k21 = 0.1, 0.05, 0.03
params = (k10, k12, k21)

# Time points
t0, t1 = 0.0, 10.0
ts = jnp.linspace(t0, t1, 1000)

# Dosing schedule
dose_times = jnp.array([2.0, 5.0, 8.0])
dose_amounts = jnp.array([50.0, 50.0, 50.0])

# Combine time points with dose times
ts = jnp.sort(jnp.concatenate([ts, dose_times]))

# Create a dataset with different initial conditions and dosing
num_samples = 10
initial_conditions = jnp.linspace(0.5, 1.5, num_samples)
dataset = []

for C1_0 in initial_conditions:
    C2_0 = 0.0
    y0 = jnp.array([C1_0, C2_0])
    args = (k10, k12, k21, dose_times, dose_amounts)

    # Solve the ODE with events
    solver = diffrax.Kvaerno4()
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
    saveat = diffrax.SaveAt(ts=ts, dense=True)
    
    # Define the event to apply dosing
    def cond_fn(t, y, args):
        return jnp.isclose(t, dose_times).any()

    def apply_dose(t, y, args):
        return dosing_event(t, y, args)

    event = diffrax.Event(cond_fn=cond_fn, apply_fn=apply_dose)
    
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(two_compartment_model),
        solver,
        t0=t0,
        t1=t1,
        dt0=0.01,  # Start with a small initial time step
        y0=y0,
        args=args,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=10000,  # Allow for more steps to handle stiffness
        event=event
    )

    dataset.append(sol.ys)

# Convert dataset to a numpy array for saving
dataset_np = np.array(dataset)

# Save the dataset
np.savez('pk_dataset.npz', ts=ts, dataset=dataset_np)

# Plot the results
for i, sol in enumerate(dataset):
    plt.plot(ts, sol[:, 0], label=f'Sample {i+1} - Central')
    plt.plot(ts, sol[:, 1], '--', label=f'Sample {i+1} - Peripheral')

plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Two-Compartment PK Model with Dosing')
plt.legend()
plt.show()
