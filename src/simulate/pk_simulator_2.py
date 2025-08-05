import jax
import jax.numpy as jnp
import diffrax
import numpy as np
from jax import lax

# -----------------------------------------------------------------------------
# 1) Define the ODE system and the dosing update
# -----------------------------------------------------------------------------
def two_compartment_model(t, y, args):
    """
    The standard two-compartment ODE.
    t: float
    y: array([C1, C2])
    args: tuple(k10, k12, k21)
    """
    k10, k12, k21 = args
    C1, C2 = y
    dC1 = -k10*C1 - k12*C1 + k21*C2
    dC2 =  k12*C1 - k21*C2
    return jnp.array([dC1, dC2])


def apply_dose(y, dose_amount):
    """
    Instantaneously add `dose_amount` to compartment-1 (central).
    """
    C1, C2 = y
    return jnp.array([C1 + dose_amount, C2])


# -----------------------------------------------------------------------------
# 2) Wrap the ODE to integrate "scaled time" from 0 to 1 for each segment
# -----------------------------------------------------------------------------
def solve_segment(y0, args, steps_per_segment=200):
    """
    Integrate from real time t0 to real time t1, storing 'steps_per_segment' states.
    Returns (ys, y_final):
    - ys has shape (steps_per_segment, state_dim)
    - y_final is the last state (same as ys[-1]), but returned for convenience.
    """
    # Wrap the ODE so that diffrax sees t in [0,1]
    def ode_scaled(t, y, p):
        return two_compartment_model(t, y, p)

    term = diffrax.ODETerm(ode_scaled)
    solver = diffrax.Tsit5()

    # We'll request intermediate solutions at 0, 1/(steps-1), 2/(steps-1), ..., 1
    ts_scaled = jnp.linspace(0.0, 1.0, steps_per_segment)
    saveat = diffrax.SaveAt(ts=ts_scaled)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=1.0,
        dt0=0.01,
        y0=y0,
        args=args,             # the ODE parameters
        saveat=saveat,
        max_steps=10_000,
    )

    ys = sol.ys                      # shape (steps_per_segment, state_dim)
    y_final = ys[-1]                # final state at scaled t=1 => real time t1
    return ys, y_final


# -----------------------------------------------------------------------------
# 3) Use lax.scan to iterate over the event times
# -----------------------------------------------------------------------------
def piecewise_integrate_with_events(y0, event_times, event_doses, t_final,
                                    args, steps_per_segment=200):
    """
    Solve from time=0 to time=event_times[0], 
         apply dose,
         then from event_times[0] to event_times[1],
         apply dose,
         ...
         until the last event_time,
         then from there to t_final (with no dose at the end).

    We'll store the solution for each segment (on a scaled [0..1] grid) 
    in an array of shape (n_segments, steps_per_segment, state_dim).

    event_times: e.g. [2., 5., 8.]
    event_doses: e.g. [50., 50., 50.]  (same length as event_times)
    t_final: the final time of integration
    """

    # We'll define a new array of "segment boundary times" that includes the final time
    # e.g. if event_times = [2, 5, 8] and t_final=10.0 => segment_times = [0, 2, 5, 8, 10]
    segment_times = jnp.concatenate([jnp.array([0.0]), event_times, jnp.array([t_final])])
    # We also define dose for each event, but the final segment has no dose afterward,
    # so we append a zero for the final segment:
    segment_doses = jnp.concatenate([event_doses, jnp.array([0.0])])

    # We'll have len(segment_times) - 1 segments to integrate
    # but for convenience we'll pair them up in a scan: 
    # Each iteration sees (t_start, t_end, dose_this_segment)

    # Build an array of shape (n_segments, 3) containing [t_start, t_end, dose].
    # n_segments = len(event_times) + 1
    t_starts = segment_times[:-1]
    t_ends   = segment_times[1:]
    segments = jnp.stack([t_starts, t_ends, segment_doses], axis=-1)  
    # shape = (n_segments, 3)

    def scan_one_segment(carry, seginfo):
        """
        carry = (y_in)
        seginfo = [t_start, t_end, dose]
        We solve from t_start to t_end, store the entire solution, apply dose at the end.
        """
        y_in = carry
        t_start, t_end, dose_amount = seginfo
        ys_segment, y_out = solve_segment(y_in, args,
                                          steps_per_segment=steps_per_segment)
        # Now apply dose
        y_dosed = apply_dose(y_out, dose_amount)
        # We'll return all intermediate solutions for plotting/storing
        return (y_dosed), ys_segment

    # We start with y0
    init_carry = y0
    # Run the scan
    final_carry, all_segments = lax.scan(scan_one_segment, init_carry, segments)
    # all_segments has shape (n_segments, steps_per_segment, 2)

    # final_carry = the state after the very last dose (which is zero dose if final segment)
    return all_segments, final_carry


# -----------------------------------------------------------------------------
# 4) Example usage: loop over multiple initial conditions
# -----------------------------------------------------------------------------
def main():
    # Model parameters
    k10, k12, k21 = 0.1, 0.05, 0.03
    params = (k10, k12, k21)

    # Simulation settings
    t_final = 10.0
    dose_times = jnp.array([2.0, 5.0, 8.0])
    dose_amounts = jnp.array([50.0, 50.0, 50.0])
    steps_per_segment = 200  # how many points per segment

    # We'll produce multiple samples with different initial conditions:
    num_samples = 5
    initial_conditions = jnp.linspace(0.5, 1.5, num_samples)

    # We'll store all solutions from each sample
    # We know we have len(dose_times)+1 = 4 segments; each segment has `steps_per_segment` points
    # So shape = (num_samples, n_segments, steps_per_segment, 2)
    all_solutions = []
    all_final_states = []

    for i in range(num_samples):
        C1_0 = initial_conditions[i]
        C2_0 = 0.0
        y0 = jnp.array([C1_0, C2_0])

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

    # Convert to single JAX arrays (shape = [num_samples, n_segments, steps_per_segment, 2])
    all_solutions = jnp.stack(all_solutions, axis=0)
    all_final_states = jnp.stack(all_final_states, axis=0)

    # Convert to numpy for saving
    all_solutions_np = np.array(all_solutions)
    all_final_states_np = np.array(all_final_states)

    # Save them to an .npz file
    np.savez("pk_dataset.npz",
             all_solutions=all_solutions_np,
             all_final_states=all_final_states_np)

    print("Shape of all_solutions:", all_solutions_np.shape)
    print("Shape of all_final_states:", all_final_states_np.shape)

    # Optional: a quick Matplotlib plot (just for one sample)
    import matplotlib.pyplot as plt
    
    # Flatten the 4 segments for sample=0 into one long time axis
    # We'll do it purely in Python for demonstration:
    seg_times = jnp.array([0.0, 2.0, 5.0, 8.0, 10.0])
    times_plot = []
    conc_plot_C1 = []
    conc_plot_C2 = []
    data_sample_0 = all_solutions[0]  # shape (4 segments, steps_per_segment, 2)
    for seg_idx in range(len(seg_times)-1):
        t_start = seg_times[seg_idx]
        t_end   = seg_times[seg_idx+1]
        # The scaled time in [0..1]
        seg_t_scaled = jnp.linspace(0., 1., steps_per_segment)
        # Map it back to [t_start..t_end]
        seg_t_real = t_start + seg_t_scaled*(t_end - t_start)
        seg_y      = data_sample_0[seg_idx]  # shape (steps_per_segment, 2)

        times_plot.append(seg_t_real)
        conc_plot_C1.append(seg_y[:, 0])
        conc_plot_C2.append(seg_y[:, 1])

    times_plot = np.concatenate([np.array(tp) for tp in times_plot])
    conc_plot_C1 = np.concatenate([np.array(cp) for cp in conc_plot_C1])
    conc_plot_C2 = np.concatenate([np.array(cp) for cp in conc_plot_C2])

    plt.plot(times_plot, conc_plot_C1, label="Central (sample 0)")
    plt.plot(times_plot, conc_plot_C2, "--", label="Peripheral (sample 0)")
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.title("Two-Compartment PK Model with Events (sample=0)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
