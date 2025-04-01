import jax
import jax.numpy as jnp
import optax
from jax import lax
import diffrax
import numpy as np
import matplotlib
matplotlib.use("Agg")  # So we can generate figures without displaying them
import matplotlib.pyplot as plt

import imageio   # for making a GIF
import io
from PIL import Image

# -----------------------------------------------------------------------------
# 1) Extended 2CPT + tumor ODE
# -----------------------------------------------------------------------------
def two_compartment_model_unscaled(t, y, args):
    """
    y: [C1, C2, T]
    args: (k10, k12, k21, r, k_t)
    """
    k10, k12, k21, r, k_t = args
    C1, C2, T = y
    dC1 = -k10*C1 - k12*C1 + k21*C2
    dC2 =  k12*C1 - k21*C2
    dT = r*T - k_t*C2*T
    return jnp.array([dC1, dC2, dT])

def apply_dose(y, dose_amount):
    """Instantaneously add 'dose_amount' to C1. (C2 and T unchanged)."""
    C1, C2, T = y
    return jnp.array([C1 + dose_amount, C2, T])

# -----------------------------------------------------------------------------
# 2) Piecewise integration with scaled time in each segment
# -----------------------------------------------------------------------------
def scaled_two_compartment_model(t, y, args, t0, t1):
    T_real = t0 + t*(t1 - t0)
    dY_dT = two_compartment_model_unscaled(T_real, y, args)
    return (t1 - t0)*dY_dT

def solve_segment(y0, t0, t1, args, steps_per_segment=200):
    def ode_scaled(t, y, p):
        return scaled_two_compartment_model(t, y, p, t0, t1)

    solver = diffrax.Tsit5()
    term = diffrax.ODETerm(ode_scaled)
    ts_scaled = jnp.linspace(0., 1., steps_per_segment)
    sol = diffrax.diffeqsolve(
        term, solver,
        t0=0.0, t1=1.0,
        dt0=0.01,
        y0=y0,
        args=args,
        saveat=diffrax.SaveAt(ts=ts_scaled),
        max_steps=10_000,
    )
    ys_segment = sol.ys  # (steps_per_segment, 3)
    return ys_segment, ys_segment[-1]

def piecewise_integrate_with_events(y0, event_times, event_doses, t_final, args, steps_per_segment=200):
    segment_times = jnp.concatenate([jnp.array([0.0]), event_times, jnp.array([t_final])])
    segment_doses = jnp.concatenate([event_doses, jnp.array([0.0])])  # last => no dose
    t_starts = segment_times[:-1]
    t_ends   = segment_times[1:]
    segments = jnp.stack([t_starts, t_ends, segment_doses], axis=-1)

    def scan_one_segment(carry, seginfo):
        y_in = carry
        t_start, t_end, dose_amnt = seginfo
        ys_seg, y_out = solve_segment(y_in, t_start, t_end, args, steps_per_segment)
        y_dosed = apply_dose(y_out, dose_amnt)
        return y_dosed, ys_seg

    init_carry = y0
    final_carry, all_segments = lax.scan(scan_one_segment, init_carry, segments)
    return all_segments, final_carry

# -----------------------------------------------------------------------------
# 3) Compute AUC of C1 and T
# -----------------------------------------------------------------------------
def compute_AUC(seg_solutions, event_times, t_final):
    n_segments = seg_solutions.shape[0]
    boundary_times = jnp.concatenate([jnp.array([0.0]), event_times, jnp.array([t_final])])
    steps_per_segment = seg_solutions.shape[1]

    total_auc_c1 = 0.0
    total_auc_t  = 0.0

    for seg_idx in range(n_segments):
        t0 = boundary_times[seg_idx]
        t1 = boundary_times[seg_idx+1]
        ts_scaled = jnp.linspace(0., 1., steps_per_segment)
        ts_real = t0 + ts_scaled*(t1 - t0)

        values = seg_solutions[seg_idx]  # (steps_per_segment, 3)
        c1_vals = values[:, 0]
        t_vals  = values[:, 2]

        # jnp.trapezoid integrates y wrt x
        auc_c1_seg = jnp.trapezoid(c1_vals, x=ts_real)
        auc_t_seg  = jnp.trapezoid(t_vals,  x=ts_real)

        total_auc_c1 += auc_c1_seg
        total_auc_t  += auc_t_seg

    return total_auc_c1, total_auc_t

# -----------------------------------------------------------------------------
# 4) The objective function: sum of AUC(C1) and AUC(T)
# -----------------------------------------------------------------------------
def objective_fn(dose_amounts, y0, event_times, t_final, args, steps_per_segment=200):
    seg_solutions, _ = piecewise_integrate_with_events(
        y0, event_times, dose_amounts, t_final, args, steps_per_segment
    )
    auc_c1, auc_t = compute_AUC(seg_solutions, event_times, t_final)
    return auc_c1 + auc_t

# -----------------------------------------------------------------------------
# Re-parameterize "unconstrained" => "dose_amounts = exp(unconstrained)"
# so that the doses stay positive
# -----------------------------------------------------------------------------
k10, k12, k21 = 0.1, 0.05, 0.03
r, k_t = 0.1, 0.02
params = (k10, k12, k21, r, k_t)

dose_times = jnp.array([2.0, 5.0, 8.0])  
y0 = jnp.array([1.0, 0.0, 100.0])
t_final = 10.0
steps_per_segment = 200

@jax.jit
def loss_fn(unconstrained_doses):
    # Map from R^n to positive domain
    dose_amounts = jnp.exp(unconstrained_doses)
    return objective_fn(dose_amounts, y0, dose_times, t_final, params, steps_per_segment)

# -----------------------------------------------------------------------------
# 5) Learning Rate Scheduler & Optimizer
# -----------------------------------------------------------------------------
# We'll define an exponential decay schedule:
schedule = optax.exponential_decay(
    init_value=1e-1,        # initial learning rate
    transition_steps=500,   # how many steps between decays
    decay_rate=0.9,         # multiply LR by 0.9 every 500 steps
    staircase=False         # if True, would do discrete steps, else continuous
)
# Now pass schedule into Adam
optimizer = optax.adam(schedule)

@jax.jit
def update_fn(unconstrained_doses, opt_state):
    grads = jax.grad(loss_fn)(unconstrained_doses)
    updates, new_opt_state = optimizer.update(grads, opt_state, unconstrained_doses)
    new_unconstrained_doses = optax.apply_updates(unconstrained_doses, updates)
    return new_unconstrained_doses, new_opt_state

# -----------------------------------------------------------------------------
# Main loop with positivity constraint + LR schedule + GIF
# -----------------------------------------------------------------------------
def main():
    # Start with some initial guess, e.g. log(50) ~ 3.91
    # so that exp(unconstrained) ~ 50
    init_unconstrained = jnp.log(jnp.array([80.0, 10.0, 10.0]))
    opt_state = optimizer.init(init_unconstrained)

    # Warm-up the JIT
    _ = update_fn(init_unconstrained, opt_state)

    frames = []
    max_iters = 10_000
    tolerance = 1e-6
    prev_obj = None

    unconstrained_doses = init_unconstrained

    for i in range(max_iters):
        unconstrained_doses, opt_state = update_fn(unconstrained_doses, opt_state)
        obj_val = loss_fn(unconstrained_doses)
        
        if prev_obj is not None:
            if jnp.abs(prev_obj - obj_val) < tolerance:
                print(f"Converged at iteration {i} with obj={obj_val:.3f}")
                break
        prev_obj = obj_val
        
        if i % 50 == 0:
            # Convert to actual doses for printing
            dose_amounts = jnp.exp(unconstrained_doses)
            current_lr = schedule(i)
            print(f"Iteration {i}, LR={current_lr:.5f}, objective={obj_val:.3f}, doses={dose_amounts}")

            # Generate a plot for the GIF
            fig, ax1 = plt.subplots(figsize=(6,4))
            seg_solutions, _ = piecewise_integrate_with_events(
                y0, dose_times, dose_amounts, t_final, params, steps_per_segment
            )
            # Flatten
            n_segments = seg_solutions.shape[0]
            boundary_times = jnp.concatenate([jnp.array([0.0]), dose_times, jnp.array([t_final])])
            times_plot = []
            c1_plot = []
            t_plot  = []

            for seg_idx in range(n_segments):
                t0 = boundary_times[seg_idx]
                t1 = boundary_times[seg_idx+1]
                ts_scaled = jnp.linspace(0., 1., steps_per_segment)
                ts_real = t0 + ts_scaled*(t1 - t0)
                vals = seg_solutions[seg_idx]
                times_plot.append(ts_real)
                c1_plot.append(vals[:,0])
                t_plot.append(vals[:,2])

            times_plot = np.concatenate([np.array(tp) for tp in times_plot])
            c1_plot    = np.concatenate([np.array(cp) for cp in c1_plot])
            t_plot     = np.concatenate([np.array(tp) for tp in t_plot])

            # Plot
            ax2 = ax1.twinx()
            ax1.plot(times_plot, c1_plot, label="C1", color="blue")
            ax2.plot(times_plot, t_plot, label="Tumor", color="red")

            ax1.set_xlabel("Time")
            ax1.set_ylabel("C1 (Drug)", color="blue")
            ax2.set_ylabel("Tumor", color="red")
            ax1.set_title(f"Iter={i}, obj={obj_val:.3f}\nDoses={dose_amounts}")

            # Convert plot to an in-memory image
            fig.canvas.draw()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            frames.append(Image.open(buf))
            plt.close(fig)

    # Final
    final_obj = loss_fn(unconstrained_doses)
    final_doses = jnp.exp(unconstrained_doses)
    print(f"Done at iteration {i}, objective={final_obj:.3f}")
    print(f"Final unconstrained params={unconstrained_doses}")
    print(f"Final actual dose amounts={final_doses}")

    # Save the GIF
    if len(frames) > 1:
        frames[0].save("pk_optimization.gif",
                       save_all=True, append_images=frames[1:],
                       loop=0, duration=500)
        print("Saved GIF to pk_optimization.gif")
    else:
        print("Not enough frames to create a GIF.")

if __name__ == "__main__":
    main()
