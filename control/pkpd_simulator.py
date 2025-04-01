import jax
import jax.numpy as jnp
import diffrax
import numpy as np
from jax import lax

# -----------------------------------------------------------------------------
# 1) Define the extended ODE system (two-compartment PK plus tumor dynamics)
# -----------------------------------------------------------------------------
def two_compartment_model_unscaled(t, y, args):
    """
    Extended two-compartment model with a tumor state.
    
    y: array([C1, C2, T])
         C1: drug concentration in the central compartment
         C2: drug concentration in the peripheral compartment
         T : tumor population
    args: tuple(k10, k12, k21, r, k_t)
         k10, k12, k21: rate constants for the PK model
         r            : intrinsic tumor growth rate
         k_t          : kill rate constant by the peripheral drug concentration
    """
    k10, k12, k21, r, k_t = args
    C1, C2, T = y
    dC1 = -k10 * C1 - k12 * C1 + k21 * C2
    dC2 =  k12 * C1 - k21 * C2
    # Tumor dynamics: exponential growth minus drug-induced killing (proportional to C2)
    dT  = r * T - k_t * C2 * T
    return jnp.array([dC1, dC2, dT])


def apply_dose(y, dose_amount):
    """
    Instantaneously add `dose_amount` to the central compartment.
    The peripheral and tumor states remain unchanged.
    
    y: array([C1, C2, T])
    """
    C1, C2, T = y
    return jnp.array([C1 + dose_amount, C2, T])


# -----------------------------------------------------------------------------
# 2) Wrap the ODE to integrate "scaled time" from 0 to 1 for each segment
# -----------------------------------------------------------------------------
def scaled_two_compartment_model(t, y, args, t0, t1):
    """
    Integrate from t0 to t1 in real time, but diffrax sees t in [0,1].
    
    We perform the change of variable:
      T = t0 + t*(t1-t0)
    
    so that:
      dY/dt = (t1 - t0) * dY/dT.
    """
    # Map scaled time t in [0,1] to real time T
    T = t0 + t * (t1 - t0)
    # Evaluate the ODE in real time T
    dY_dT = two_compartment_model_unscaled(T, y, args)
    # Apply the chain rule
    return (t1 - t0) * dY_dT


def solve_segment(y0, t0, t1, args, steps_per_segment=200):
    """
    Integrate from real time t0 to t1, returning a trajectory on a scaled grid.
    
    Returns:
      ys      : array of shape (steps_per_segment, state_dim)
      y_final : final state (same as ys[-1])
    """
    # Wrap the ODE for the scaled integration
    def ode_scaled(t, y, p):
        return scaled_two_compartment_model(t, y, p, t0, t1)

    term = diffrax.ODETerm(ode_scaled)
    solver = diffrax.Tsit5()

    # Request solution at evenly spaced points in the scaled time [0,1]
    ts_scaled = jnp.linspace(0.0, 1.0, steps_per_segment)
    saveat = diffrax.SaveAt(ts=ts_scaled)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=1.0,
        dt0=0.01,
        y0=y0,
        args=args,             # ODE parameters
        saveat=saveat,
        max_steps=10_000,
    )

    ys = sol.ys
    y_final = ys[-1]
    return ys, y_final


# -----------------------------------------------------------------------------
# 3) Use lax.scan to iterate over the event times (dosing events)
# -----------------------------------------------------------------------------
def piecewise_integrate_with_events(y0, event_times, event_doses, t_final,
                                    args, steps_per_segment=200):
    """
    Solve from time 0 to event_times[0], then apply dose, and so on,
    until t_final.
    
    event_times: e.g. [2., 5., 8.]
    event_doses: e.g. [50., 50., 50.]  (must match event_times length)
    t_final: final integration time
    
    Returns:
      all_segments : array of shape (n_segments, steps_per_segment, state_dim)
      final_state  : state after the final (zero) dose.
    """
    # Create segment boundary times including final time
    segment_times = jnp.concatenate([jnp.array([0.0]), event_times, jnp.array([t_final])])
    # Append zero dose for the final segment (no dose)
    segment_doses = jnp.concatenate([event_doses, jnp.array([0.0])])

    # Pair segment start, end, and dose into an array of shape (n_segments, 3)
    t_starts = segment_times[:-1]
    t_ends   = segment_times[1:]
    segments = jnp.stack([t_starts, t_ends, segment_doses], axis=-1)

    def scan_one_segment(carry, seginfo):
        """
        Solve one segment and apply the dose at its end.
        
        carry: the incoming state
        seginfo: [t_start, t_end, dose]
        """
        y_in = carry
        t_start, t_end, dose_amount = seginfo
        ys_segment, y_out = solve_segment(y_in, t_start, t_end, args,
                                          steps_per_segment=steps_per_segment)
        # Apply the dose to the central compartment only.
        y_dosed = apply_dose(y_out, dose_amount)
        return y_dosed, ys_segment

    init_carry = y0
    final_carry, all_segments = lax.scan(scan_one_segment, init_carry, segments)
    return all_segments, final_carry


# -----------------------------------------------------------------------------
# 4) Example usage: loop over multiple initial conditions
# -----------------------------------------------------------------------------
def main():
    # Model parameters for the PK model and tumor dynamics
    k10, k12, k21 = 0.1, 0.05, 0.03
    r, k_t = 0.1, 0.02  # tumor growth rate and drug-induced kill rate
    params = (k10, k12, k21, r, k_t)

    # Simulation settings
    t_final = 10.0
    dose_times = jnp.array([2.0, 5.0, 8.0])
    dose_amounts = jnp.array([50.0, 50.0, 50.0])
    steps_per_segment = 200

    # Create multiple samples with different initial central compartment concentrations.
    # For each sample, we also initialize:
    #   C1: variable between 0.5 and 1.5,
    #   C2: 0.0 (peripheral),
    #   T : tumor population (set to 100.0 for example).
    num_samples = 5
    initial_C1_values = jnp.linspace(0.5, 1.5, num_samples)
    T0 = 100.0  # initial tumor population

    all_solutions = []
    all_final_states = []

    for i in range(num_samples):
        C1_0 = initial_C1_values[i]
        C2_0 = 0.0
        y0 = jnp.array([C1_0, C2_0, T0])

        seg_solutions, final_state = piecewise_integrate_with_events(
            y0=y0,
            event_times=dose_times,
            event_doses=dose_amounts,
            t_final=t_final,
            args=params,
            steps_per_segment=steps_per_segment
        )

        all_solutions.append(seg_solutions)
        all_final_states.append(final_state)

    # Convert the list of solutions to JAX arrays:
    # Shape: (num_samples, n_segments, steps_per_segment, state_dim)
    all_solutions = jnp.stack(all_solutions, axis=0)
    all_final_states = jnp.stack(all_final_states, axis=0)

    # Save the solutions to an .npz file.
    # Note: The state now has 3 components: [Central, Peripheral, Tumor]
    all_solutions_np = np.array(all_solutions)
    all_final_states_np = np.array(all_final_states)
    np.savez("pk_dataset.npz",
             all_solutions=all_solutions_np,
             all_final_states=all_final_states_np)

    print("Shape of all_solutions:", all_solutions_np.shape)
    print("Shape of all_final_states:", all_final_states_np.shape)

    # -------------------------------------------------------------------------
    # Optional: Plotting for one sample (sample index 0)
    # We'll plot the central and peripheral drug concentrations as well as tumor.
    # -------------------------------------------------------------------------
    import matplotlib.pyplot as plt
    
    # Segment boundary times for plotting
    seg_times = jnp.array([0.0, 2.0, 5.0, 8.0, 10.0])
    times_plot = []
    conc_plot_C1 = []
    conc_plot_C2 = []
    tumor_plot = []
    data_sample_0 = all_solutions[0]  # shape (n_segments, steps_per_segment, 3)
    
    for seg_idx in range(len(seg_times) - 1):
        t_start = seg_times[seg_idx]
        t_end   = seg_times[seg_idx + 1]
        seg_t_scaled = jnp.linspace(0., 1., steps_per_segment)
        seg_t_real = t_start + seg_t_scaled * (t_end - t_start)
        seg_y = data_sample_0[seg_idx]  # shape (steps_per_segment, 3)
        
        times_plot.append(seg_t_real)
        conc_plot_C1.append(seg_y[:, 0])
        conc_plot_C2.append(seg_y[:, 1])
        tumor_plot.append(seg_y[:, 2])
    
    times_plot = np.concatenate([np.array(tp) for tp in times_plot])
    conc_plot_C1 = np.concatenate([np.array(cp) for cp in conc_plot_C1])
    conc_plot_C2 = np.concatenate([np.array(cp) for cp in conc_plot_C2])
    tumor_plot = np.concatenate([np.array(tp) for tp in tumor_plot])
    
    # Create subplots: one for PK compartments and one for tumor dynamics.
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot drug concentrations (central and peripheral)
    axs[0].plot(times_plot, conc_plot_C1, label="Central")
    axs[0].plot(times_plot, conc_plot_C2, "--", label="Peripheral")
    axs[0].set_ylabel("Drug Concentration")
    axs[0].legend()
    axs[0].set_title("Two-Compartment PK Model with Dosing Events")
    
    # Plot tumor population
    axs[1].plot(times_plot, tumor_plot, label="Tumor", color="red")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Tumor Population")
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
