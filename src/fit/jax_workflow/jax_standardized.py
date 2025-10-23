"""
Standardized JAX/Diffrax Implementation for Framework Comparison
Two-compartment PK model with Neural ODE
Network: [2 -> 64 -> 64 -> 2] with ReLU activation
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax
import matplotlib.pyplot as plt
import diffrax
import time
from functools import partial


# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_piecewise_data(filename="data/pk_dataset_2C.npz"):
    """
    Load 2-compartment PK dataset.
    Expected shape: (num_samples, n_segments, steps_per_segment, 2)
    """
    data = np.load(filename)
    all_solutions = data["all_solutions"]  # (100, 4, 200, 2)
    return jnp.array(all_solutions)


# ==============================================================================
# NEURAL NETWORK (MLP)
# ==============================================================================
def init_mlp_params(rng, layer_sizes):
    """
    Initialize MLP parameters.
    layer_sizes: e.g., [2, 64, 64, 2]
    Returns: list of (W, b) tuples
    """
    params = []
    for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
        rng, rng_W, rng_b = jax.random.split(rng, 3)
        W = 0.1 * jax.random.normal(rng_W, (in_dim, out_dim))
        b = jnp.zeros((out_dim,))
        params.append((W, b))
    return params


def mlp_apply(params, x):
    """
    Forward pass through MLP with ReLU activation.
    x: shape (2,) or (batch, 2)
    Returns: shape (2,)
    """
    for (W, b) in params[:-1]:
        x = jnp.dot(x, W) + b
        x = jax.nn.relu(x)
    W_last, b_last = params[-1]
    return jnp.dot(x, W_last) + b_last


# ==============================================================================
# NEURAL ODE DYNAMICS
# ==============================================================================
def neural_ode_rhs(t, y, nn_params):
    """
    Neural ODE right-hand side: dy/dt = f_NN(y)
    y: [A1, A2] - amounts in central and peripheral compartments
    """
    return mlp_apply(nn_params, y)


def scaled_neural_ode(t, y, nn_params, t0, t1):
    """
    Scaled ODE for integration over [0, 1] representing real time [t0, t1].
    dy/dt_scaled = (t1 - t0) * f_NN(y)
    """
    f = neural_ode_rhs(t, y, nn_params)
    return (t1 - t0) * f


# ==============================================================================
# PIECEWISE INTEGRATION WITH DOSING
# ==============================================================================
def apply_dose(y, dose_amount):
    """
    Add dose to central compartment (index 0).
    """
    return y.at[0].add(dose_amount)


def solve_segment_neural_ode(y0, t0, t1, nn_params, steps_per_segment):
    """
    Integrate one segment from t0 to t1 (real time) using scaled time [0, 1].

    Args:
        y0: Initial state [A1, A2]
        t0, t1: Real time bounds
        nn_params: Neural network parameters
        steps_per_segment: Number of time points to save

    Returns:
        ys_segment: Trajectory (steps_per_segment, 2)
        y_final: Final state (2,)
    """
    def ode_scaled(t, y, p):
        return scaled_neural_ode(t, y, p, t0, t1)

    term = diffrax.ODETerm(ode_scaled)
    solver = diffrax.Tsit5()  # 5th-order adaptive Runge-Kutta

    ts_scaled = jnp.linspace(0.0, 1.0, steps_per_segment)
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=1.0,
        dt0=0.01,
        y0=y0,
        args=nn_params,
        saveat=diffrax.SaveAt(ts=ts_scaled),
        max_steps=10_000,
        atol=1e-7,
        rtol=1e-5,
    )
    ys_segment = sol.ys  # (steps_per_segment, 2)
    return ys_segment, ys_segment[-1]


def piecewise_integrate_neural_ode(y0, event_times, event_doses, t_final,
                                   nn_params, steps_per_segment):
    """
    Piecewise integration with dosing events.

    Segments: [0, t1], [t1, t2], [t2, t3], [t3, t_final]
    Doses applied at end of each segment (except final)

    Returns:
        all_segments: shape (n_segments, steps_per_segment, 2)
    """
    segment_times = jnp.concatenate([jnp.array([0.0]), event_times, jnp.array([t_final])])
    segment_doses = jnp.concatenate([event_doses, jnp.array([0.0])])

    t_starts = segment_times[:-1]
    t_ends = segment_times[1:]
    segments = jnp.stack([t_starts, t_ends, segment_doses], axis=-1)
    # segments shape: (n_segments, 3)

    def scan_one_segment(carry, seginfo):
        y_in = carry
        t_start, t_end, dose_amount = seginfo
        ys_seg, y_out = solve_segment_neural_ode(
            y_in, t_start, t_end, nn_params, steps_per_segment
        )
        # Apply dose at end of segment
        y_dosed = apply_dose(y_out, dose_amount)
        return y_dosed, ys_seg

    init_carry = y0
    final_carry, all_segments = lax.scan(scan_one_segment, init_carry, segments)
    return all_segments  # (n_segments, steps_per_segment, 2)


# ==============================================================================
# LOSS FUNCTION
# ==============================================================================
def reconstruction_loss(nn_params, data, event_times, event_doses, t_final, steps_per_segment):
    """
    Mean squared error between predicted and true trajectories.

    Args:
        nn_params: Neural network parameters
        data: True trajectories (num_samples, n_segments, steps_per_segment, 2)
        event_times: Dosing times
        event_doses: Dose amounts
        t_final: Final time
        steps_per_segment: Number of points per segment

    Returns:
        loss: Scalar MSE
    """
    num_samples = data.shape[0]
    y0s = data[:, 0, 0, :]  # Initial conditions (num_samples, 2)

    def single_sample_loss(y0, true_segments):
        pred_segments = piecewise_integrate_neural_ode(
            y0, event_times, event_doses, t_final,
            nn_params, steps_per_segment
        )
        return jnp.mean((pred_segments - true_segments)**2)

    losses = jax.vmap(single_sample_loss)(y0s, data)
    return jnp.mean(losses)


# ==============================================================================
# TRAINING
# ==============================================================================
@jax.jit
def update_step(nn_params, opt_state, data, event_times, event_doses, t_final,
                steps_per_segment, optimizer):
    """
    Single training step with gradient computation and parameter update.
    """
    loss_fn = partial(reconstruction_loss,
                     data=data,
                     event_times=event_times,
                     event_doses=event_doses,
                     t_final=t_final,
                     steps_per_segment=steps_per_segment)

    loss_val, grads = jax.value_and_grad(loss_fn)(nn_params)
    updates, new_opt_state = optimizer.update(grads, opt_state, nn_params)
    new_nn_params = optax.apply_updates(nn_params, updates)
    return new_nn_params, new_opt_state, loss_val


def train_model(data, event_times, event_doses, t_final, n_iters=2000,
                lr=1e-3, seed=42):
    """
    Train the Neural ODE model.

    Returns:
        nn_params: Trained parameters
        losses: Training loss history
        training_time: Total training time (seconds)
    """
    num_samples, n_segments, steps_per_segment, _ = data.shape

    # Initialize network
    rng = jax.random.PRNGKey(seed)
    layer_sizes = [2, 64, 64, 2]  # Standardized architecture
    nn_params = init_mlp_params(rng, layer_sizes)

    # Initialize optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(nn_params)

    # Training loop
    losses = []
    print(f"Training JAX/Diffrax model for {n_iters} iterations...")
    start_time = time.time()

    for i in range(n_iters):
        nn_params, opt_state, loss_val = update_step(
            nn_params, opt_state, data, event_times, event_doses,
            t_final, steps_per_segment, optimizer
        )
        losses.append(float(loss_val))

        if i % 200 == 0:
            print(f"Iteration {i:4d}, Loss: {loss_val:.6f}")

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Final loss: {losses[-1]:.6f}")

    return nn_params, losses, training_time


# ==============================================================================
# EVALUATION AND VISUALIZATION
# ==============================================================================
def evaluate_model(nn_params, data, event_times, event_doses, t_final, sample_idx=0):
    """
    Evaluate model on a specific sample and create visualization.
    """
    num_samples, n_segments, steps_per_segment, _ = data.shape

    # Get predictions for one sample
    y0_sample = data[sample_idx, 0, 0, :]
    pred_segments = piecewise_integrate_neural_ode(
        y0_sample, event_times, event_doses, t_final, nn_params, steps_per_segment
    )

    true_segments = data[sample_idx]

    # Compute metrics
    mse = jnp.mean((pred_segments - true_segments)**2)
    rmse = jnp.sqrt(mse)

    print(f"\nEvaluation Metrics (Sample {sample_idx}):")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")

    # Build time axis for plotting
    segment_boundaries = jnp.array([0.0, 12.0, 24.0, 36.0, 48.0])
    T_plot = []
    A1_true_list, A2_true_list = [], []
    A1_pred_list, A2_pred_list = [], []

    for seg_idx in range(n_segments):
        t_start = segment_boundaries[seg_idx]
        t_end = segment_boundaries[seg_idx + 1]
        t_scaled = jnp.linspace(0, 1, steps_per_segment)
        t_real = t_start + t_scaled * (t_end - t_start)

        T_plot.append(t_real)
        A1_true_list.append(true_segments[seg_idx, :, 0])
        A2_true_list.append(true_segments[seg_idx, :, 1])
        A1_pred_list.append(pred_segments[seg_idx, :, 0])
        A2_pred_list.append(pred_segments[seg_idx, :, 1])

    # Concatenate all segments
    T_plot = np.concatenate([np.array(t) for t in T_plot])
    A1_true = np.concatenate([np.array(a) for a in A1_true_list])
    A2_true = np.concatenate([np.array(a) for a in A2_true_list])
    A1_pred = np.concatenate([np.array(a) for a in A1_pred_list])
    A2_pred = np.concatenate([np.array(a) for a in A2_pred_list])

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(T_plot, A1_true, 'b-', label='True A1 (Central)', linewidth=2)
    plt.plot(T_plot, A1_pred, 'r--', label='Predicted A1', linewidth=2)
    plt.plot(T_plot, A2_true, 'g-', label='True A2 (Peripheral)', linewidth=2)
    plt.plot(T_plot, A2_pred, 'y--', label='Predicted A2', linewidth=2)

    # Mark dosing times
    for dose_time in event_times:
        plt.axvline(x=dose_time, color='gray', linestyle=':', alpha=0.5)

    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Amount (mg)', fontsize=12)
    plt.title(f'JAX/Diffrax Neural ODE - Two-Compartment PK\nRMSE: {rmse:.4f}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return pred_segments, mse, rmse


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    # Load data
    print("=" * 60)
    print("JAX/Diffrax Standardized Implementation")
    print("=" * 60)

    data = load_piecewise_data("data/pk_dataset_2C.npz")
    num_samples, n_segments, steps_per_segment, _ = data.shape
    print(f"Loaded dataset: {data.shape}")
    print(f"  - Samples: {num_samples}")
    print(f"  - Segments: {n_segments}")
    print(f"  - Steps per segment: {steps_per_segment}")
    print(f"  - Compartments: 2")

    # Define dosing schedule (matching 2C_simulator.py)
    event_times = jnp.array([12.0, 24.0, 36.0])  # hours
    event_doses = jnp.array([100.0, 100.0, 100.0])  # mg
    t_final = 48.0
    print(f"\nDosing schedule: {event_doses} mg at t = {event_times} hours")

    # Train model
    nn_params, losses, training_time = train_model(
        data, event_times, event_doses, t_final,
        n_iters=2000, lr=1e-3, seed=42
    )

    # Evaluate model
    pred_segments, mse, rmse = evaluate_model(
        nn_params, data, event_times, event_doses, t_final, sample_idx=0
    )

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.title('JAX/Diffrax Training Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()

    print("\n" + "=" * 60)
    print(f"Training completed!")
    print(f"Total time: {training_time:.2f} seconds")
    print(f"Time per iteration: {training_time/2000*1000:.2f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
