import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax
import matplotlib.pyplot as plt

import diffrax

# ------------------------------------------------------------------------------
# 1) LOAD THE PIECEWISE-DATASET FROM pk_dataset.npz
# ------------------------------------------------------------------------------
def load_piecewise_data(filename="pk_dataset.npz"):
    """
    Expects the file to contain something like:
      all_solutions: shape (num_samples, n_segments, steps_per_segment, 2)
      all_final_states: shape (num_samples, 2)

    n_segments should be 4 if your dosing times are [2., 5., 8.] plus the last segment [8..10].
    steps_per_segment might be 200, etc.
    """
    data = np.load(filename)
    all_solutions = data["all_solutions"]  # shape e.g. (5, 4, 200, 2)
    # all_final_states = data["all_final_states"]  # shape (5, 2) if you need it
    return jnp.array(all_solutions)


# ------------------------------------------------------------------------------
# 2) DEFINE THE NEURAL ODE & PIECEWISE INTEGRATION (with the same dosing logic)
# ------------------------------------------------------------------------------
def apply_dose(y, dose_amount):
    """
    Add 'dose_amount' to the central compartment (index 0).
    """
    return y.at[0].add(dose_amount)

def mlp_apply(params, x):
    """
    A simple MLP: params is a list of (W, b) pairs, 
    x has shape (2,) or (batch, 2). Output shape (2,).
    """
    for (W, b) in params[:-1]:
        x = jnp.dot(x, W) + b
        x = jax.nn.relu(x)
    W_last, b_last = params[-1]
    return jnp.dot(x, W_last) + b_last

def neural_ode_rhs(_t, y, nn_params):
    """
    The learned RHS function f(y). 
    (If you want time dependence, you'd pass t into the MLP as well.)
    """
    return mlp_apply(nn_params, y)

def scaled_neural_ode(t, y, nn_params, t0, t1):
    """
    We integrate scaled time in [0..1], which corresponds to real time T in [t0..t1].
    dY/dt(scaled) = (t1 - t0)*neural_ode_rhs(T, Y), but we'll ignore T in the NN for simplicity.
    """
    # real time (unused in the NN, but you *could* pass it in)
    # T = t0 + t*(t1 - t0)
    f = neural_ode_rhs(t, y, nn_params)
    return (t1 - t0)*f

def solve_segment_neural_ode(y0, t0, t1, nn_params, steps_per_segment):
    """
    Integrate from t0..t1 in real time, by letting the solver run from 0..1 in scaled time.
    Returns (ys_segment, y_final).
    """
    def ode_scaled(t, y, p):
        return scaled_neural_ode(t, y, p, t0, t1)

    term = diffrax.ODETerm(ode_scaled)
    solver = diffrax.Tsit5()

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
    )
    ys_segment = sol.ys  # shape (steps_per_segment, 2)
    return ys_segment, ys_segment[-1]

def piecewise_integrate_neural_ode(y0, event_times, event_doses, t_final,
                                   nn_params, steps_per_segment):
    """
    We have segments:
        [0..2], [2..5], [5..8], [8..10]
    We'll do 'scan' over these segments, each integrated in scaled time [0..1].
    Then apply a dose at the end of each segment (except final).
    Returns shape (n_segments, steps_per_segment, 2).
    """
    segment_times = jnp.concatenate([jnp.array([0.0]), event_times, jnp.array([t_final])])
    # final segment has no subsequent dose
    segment_doses = jnp.concatenate([event_doses, jnp.array([0.0])])

    t_starts = segment_times[:-1]
    t_ends   = segment_times[1:]
    segments = jnp.stack([t_starts, t_ends, segment_doses], axis=-1)
    # shape = (n_segments, 3), n_segments = len(event_times)+1

    def scan_one_segment(carry, seginfo):
        y_in = carry
        t_start, t_end, dose_amount = seginfo
        ys_seg, y_out = solve_segment_neural_ode(
            y_in, t_start, t_end, nn_params, steps_per_segment
        )
        # apply dose
        y_dosed = apply_dose(y_out, dose_amount)
        return y_dosed, ys_seg

    init_carry = y0
    final_carry, all_segments = lax.scan(scan_one_segment, init_carry, segments)
    return all_segments  # shape (n_segments, steps_per_segment, 2)


# ------------------------------------------------------------------------------
# 3) LOSS FUNCTION & OPTIMIZER
# ------------------------------------------------------------------------------
def reconstruction_loss(nn_params, data, event_times, event_doses, t_final):
    """
    data shape: (num_samples, n_segments, steps_per_segment, 2)
       - each sample's piecewise solution.
    We'll replicate the same piecewise integration with the neural net for each sample,
    then compute MSE over all points.
    """
    num_samples = data.shape[0]
    n_segments = data.shape[1]
    steps_per_segment = data.shape[2]

    # For each sample, the initial condition = data[sample, 0, 0, :]
    # i.e. the very first point in that sample.
    y0s = data[:, 0, 0, :]  # shape (num_samples, 2)

    def single_sample_loss(y0, true_segments):
        # Integrate with the neural ODE
        pred_segments = piecewise_integrate_neural_ode(
            y0, event_times, event_doses, t_final,
            nn_params, steps_per_segment
        )  # shape (n_segments, steps_per_segment, 2)
        # Compare with true_segments
        return jnp.mean((pred_segments - true_segments)**2)

    # vectorize over samples
    losses = jax.vmap(single_sample_loss)(y0s, data)
    return jnp.mean(losses)

learning_rate = 1e-3
optimizer = optax.adam(learning_rate)
# @jax.partial(jax.jit, static_argnames=('optimizer',))
@jax.jit
def update_nn_params(nn_params, opt_state, data, event_times, event_doses, t_final):#, optimizer):
    """
    A single optimization step. 
    We'll compute dLoss/dParams, then apply the update via the optimizer.
    """
    loss_val, grads = jax.value_and_grad(reconstruction_loss)(
        nn_params, data, event_times, event_doses, t_final
    )
    updates, new_opt_state = optimizer.update(grads, opt_state, nn_params)
    new_nn_params = optax.apply_updates(nn_params, updates)
    return new_nn_params, new_opt_state, loss_val


# ------------------------------------------------------------------------------
# 4) MLP INIT
# ------------------------------------------------------------------------------
def init_mlp_params(rng, layer_sizes):
    """
    layer_sizes, e.g. [2, 32, 32, 2]
    returns list of (W, b) pairs
    """
    params = []
    for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
        rng, rng_W, rng_b = jax.random.split(rng, 3)
        # small init
        W = 0.1 * jax.random.normal(rng_W, (in_dim, out_dim))
        b = jnp.zeros((out_dim,))
        params.append((W, b))
    return params


# ------------------------------------------------------------------------------
# 5) MAIN: Load data, set event times/doses, train the NN ODE
# ------------------------------------------------------------------------------
def main():
    # 1) Load the piecewise data
    data = load_piecewise_data("pk_dataset_2C.npz")  
    # shape e.g. (num_samples, 4, 200, 2)
    num_samples, n_segments, steps_per_segment, _ = data.shape
    print("Loaded dataset shape:", data.shape)

    # 2) Define the known dosing times from your simulator
    event_times = jnp.array([2.0, 5.0, 8.0])  # shape (3,)
    event_doses = jnp.array([50.0, 50.0, 50.0])
    t_final = 10.0

    # 3) Initialize the NN parameters
    rng = jax.random.PRNGKey(42)
    layer_sizes = [2, 32, 32, 2]  # input=2 compartments, output=2 derivatives
    nn_params = init_mlp_params(rng, layer_sizes)

    # 4) Set up optimizer
    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(nn_params)

    # 5) Training loop
    n_iters = 2000
    for i in range(n_iters):
        nn_params, opt_state, loss_val = update_nn_params(
            nn_params, opt_state, data, event_times, event_doses, t_final#, optimizer
        )
        if i % 200 == 0:
            print(f"Iteration {i}, loss={loss_val:0.6f}")

    print("Training complete!")
    print("Final loss =", loss_val)

    # 6) Quick example: plot predictions vs. true for sample=0
    sample_idx = 0
    # The initial condition is data[sample_idx, 0, 0, :]
    y0_sample = data[sample_idx, 0, 0, :]
    # Re-run the piecewise solver with the NN
    pred_segments = piecewise_integrate_neural_ode(
        y0_sample, event_times, event_doses, t_final, nn_params, steps_per_segment
    )  # shape (4, 200, 2)

    # True data for sample=0
    true_segments = data[sample_idx]  # shape (4, 200, 2)

    # Let's build a time axis for plotting
    segment_boundaries = jnp.array([0.0, 2.0, 5.0, 8.0, 10.0])
    T_plot = []
    C1_true_list = []
    C2_true_list = []
    C1_pred_list = []
    C2_pred_list = []

    for seg_idx in range(n_segments):
        t_start = segment_boundaries[seg_idx]
        t_end   = segment_boundaries[seg_idx+1]
        t_scaled = jnp.linspace(0, 1, steps_per_segment)
        t_real = t_start + t_scaled*(t_end - t_start)

        T_plot.append(t_real)
        C1_true_list.append(true_segments[seg_idx, :, 0])
        C2_true_list.append(true_segments[seg_idx, :, 1])
        C1_pred_list.append(pred_segments[seg_idx, :, 0])
        C2_pred_list.append(pred_segments[seg_idx, :, 1])

    T_plot      = np.concatenate([np.array(tt) for tt in T_plot])
    C1_true_all = np.concatenate([np.array(xx) for xx in C1_true_list])
    C2_true_all = np.concatenate([np.array(xx) for xx in C2_true_list])
    C1_pred_all = np.concatenate([np.array(xx) for xx in C1_pred_list])
    C2_pred_all = np.concatenate([np.array(xx) for xx in C2_pred_list])

    plt.figure(figsize=(8,5))
    plt.plot(T_plot, C1_true_all, 'b', label="True C1")
    plt.plot(T_plot, C1_pred_all, 'r--', label="Predicted C1")
    plt.plot(T_plot, C2_true_all, 'g', label="True C2")
    plt.plot(T_plot, C2_pred_all, 'y--', label="Predicted C2")
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.title("Neural ODE fit to PK piecewise data (Sample 0)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
