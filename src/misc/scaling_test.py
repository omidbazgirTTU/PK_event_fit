import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt


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

# solve the system forward in time 0 to 10
y0 = jnp.array([1.0, 0.0])
args = (0.1, 0.05, 0.05)
t0 = 0.0
t1 = 10.0
steps_per_segment = 200

term = diffrax.ODETerm(two_compartment_model)
solver = diffrax.Tsit5()

# We'll request intermediate solutions at 0, 1/(steps-1), 2/(steps-1), ..., 1
ts_scaled = jnp.linspace(t0, t1, steps_per_segment)
saveat = diffrax.SaveAt(ts=ts_scaled)

sol = diffrax.diffeqsolve(
    term,
    solver,
    t0=t0,
    t1=t1,
    dt0=0.01,
    y0=y0,
    args=args,             # the ODE parameters
    saveat=saveat,
    max_steps=10_000,
)

ys = sol.ys


def scaled_two_compartment_model(t, y, args, t0, t1):
    """
    We want to integrate from t0 to t1 in real time.
    However, diffrax will be told to integrate t from 0..1.
    So we do a change of variable T = t0 + t*(t1 - t0).
    
    dY/dt = dY/dT * dT/dt = dY/dT * (t1 - t0).
    """
    # The "real" time T
    T = t0 + t*(t1 - t0)
    # Derivatives wrt T
    dY_dT = two_compartment_model(T, y, args)
    # Chain rule: dY/dt = (t1 - t0) * dY/dT
    return (t1 - t0) * dY_dT


# Wrap the ODE so that diffrax sees t in [0,1]
def ode_scaled(t, y, p):
    return scaled_two_compartment_model(t, y, p, t0, t1)

term = diffrax.ODETerm(ode_scaled)

# We'll request intermediate solutions at 0, 1/(steps-1), 2/(steps-1), ..., 1
ts_scaled = jnp.linspace(0.0, 1.0, steps_per_segment)
saveat = diffrax.SaveAt(ts=ts_scaled)

sol_scaled = diffrax.diffeqsolve(
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

ys_scaled = sol_scaled.ys     

# plot the solutions to compare
# Print debug info

# Ensure correct time transformation
t_rescaled = t0 + ts_scaled * (t1 - t0)  # Back to original time scale


print(f"Unscaled time range: {sol.ts[0]} to {sol.ts[-1]}")
print(f"Scaled time range: {t_rescaled[0]} to {t_rescaled[-1]}")



plt.plot(sol.ts, ys[:, 0], label='C1 (unscaled)')
plt.plot(sol.ts, ys[:, 1], label='C2 (unscaled)')
plt.plot(t_rescaled, ys_scaled[:, 0], '--', label='C1 (scaled)')
plt.plot(t_rescaled, ys_scaled[:, 1], '--', label='C2 (scaled)')

plt.xlabel("Time")
plt.ylabel("Concentration")
plt.legend()
plt.show()


