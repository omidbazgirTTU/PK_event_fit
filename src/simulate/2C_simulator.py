import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 1. Define mean parameters and covariance
mean_params = np.array([1.0, 10.0, 0.5, 20.0])  # [CL (L/h), V1 (L), Q (L/h), V2 (L)]
# Covariance: variances = (0.1*mean)^2, covariances = 0.02*mean_i*mean_j
var = (0.1 * mean_params) ** 2
cov = np.full((4, 4), 0.02) * np.outer(mean_params, mean_params)
np.fill_diagonal(cov, var)

# 2. Sample parameter sets (log-domain sampling for positivity)
num_samples = 100
samples_log = np.random.multivariate_normal(np.log(mean_params), cov, size=num_samples)
samples = np.exp(samples_log)  # ensure all parameters > 0

# 3. Dose schedule (bolus doses, mg)
dose_times = [12.0, 24.0, 36.0]  # hours
dose_amount = 100.0             # mg

# 4. ODE definition: A[0]=amount central, A[1]=amount peripheral
def two_comp(t, A, CL, V1, Q, V2):
    A1, A2 = A
    dA1 = - (CL / V1) * A1 - (Q / V1) * A1 + (Q / V2) * A2
    dA2 =   (Q / V1) * A1 - (Q / V2) * A2
    return [dA1, dA2]

# 5. Time grid for output
t_eval = np.linspace(0, 48, 1000)

# 6. Simulate for a subset of samples and plot
plt.figure(figsize=(8, 5))
for idx in range(10):  # plot first 10 samples
    CL, V1, Q, V2 = samples[idx]
    A0 = [0.0, 0.0]
    ts = []
    ys = []
    t_start = 0.0

    # Piecewise integration between doses
    for t_dose in dose_times + [48.0]:
        sol = solve_ivp(
            two_comp,
            [t_start, t_dose],
            A0,
            args=(CL, V1, Q, V2),
            t_eval=np.linspace(t_start, t_dose, 200)
        )
        ts.extend(sol.t)
        ys.extend(sol.y[0])  # central compartment
        A0 = [sol.y[0, -1], sol.y[1, -1]]
        if t_dose in dose_times:
            A0[0] += dose_amount  # add bolus to central
        t_start = t_dose

    plt.plot(ts, np.array(ys) / V1, alpha=0.6)

plt.xlabel('Time (h)')
plt.ylabel('Concentration (mg/L)')
plt.title('Two-Compartment PK: 100 mg bolus at 12, 24, 36 h (n=10 samples)')
plt.grid(True)
plt.show()

t_final = 48.0
# 5. Build piecewise dataset
n_segments = len(dose_times) + 1
steps_per_segment = 200
all_solutions = np.zeros((num_samples, n_segments, steps_per_segment, 2))

segment_bounds = [0.0] + dose_times + [t_final]

for i in range(num_samples):
    CL, V1, Q, V2 = samples[i]
    y0 = [0.0, 0.0]
    for seg_idx in range(n_segments):
        t_start = segment_bounds[seg_idx]
        t_end   = segment_bounds[seg_idx+1]
        t_eval  = np.linspace(t_start, t_end, steps_per_segment)
        sol = solve_ivp(
            two_comp,
            [t_start, t_end],
            y0,
            args=(CL, V1, Q, V2),
            t_eval=t_eval
        )
        # store (steps_per_segment, 2)
        all_solutions[i, seg_idx, :, :] = sol.y.T
        # advance initial state and apply next dose if any
        y0 = [sol.y[0, -1], sol.y[1, -1]]
        if seg_idx < len(dose_times):
            y0[0] += dose_amount

# 6. Save to npz for use in full_solution_o1.py
np.savez(
    "pk_dataset_2C.npz",
    all_solutions=all_solutions
)
print(f"Saved pk_dataset_2C.npz with shape {all_solutions.shape}")