"""
Standardized TorchDiffEq Implementation for Framework Comparison
Two-compartment PK model with Neural ODE
Network: [2 -> 64 -> 64 -> 2] with ReLU activation
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import time
from functools import partial

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


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
    return torch.tensor(all_solutions, dtype=torch.float32, device=device)


# ==============================================================================
# NEURAL NETWORK (MLP)
# ==============================================================================
class ODEFunc(nn.Module):
    """
    Neural ODE dynamics function.
    Input: [A1, A2] - amounts in central and peripheral compartments
    Output: [dA1/dt, dA2/dt]
    """
    def __init__(self, hidden_dim=64):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, t, y):
        """
        Args:
            t: Time (scalar or tensor)
            y: State [A1, A2] - shape (..., 2)
        Returns:
            dy/dt: shape (..., 2)
        """
        return self.net(y)


# ==============================================================================
# PIECEWISE INTEGRATION MODEL
# ==============================================================================
class PKNeuralODE(nn.Module):
    """
    Two-compartment PK model with piecewise integration and dosing.
    """
    def __init__(self, event_times, event_doses, hidden_dim=64):
        super(PKNeuralODE, self).__init__()
        self.dynamics = ODEFunc(hidden_dim)
        self.event_times = torch.tensor(event_times, dtype=torch.float32, device=device)
        self.event_doses = torch.tensor(event_doses, dtype=torch.float32, device=device)

    def apply_dose(self, y, dose_amount):
        """
        Add dose to central compartment (index 0).
        Args:
            y: State tensor [..., 2]
            dose_amount: Scalar dose
        Returns:
            y_dosed: State after dosing
        """
        y_dosed = y.clone()
        y_dosed[..., 0] = y_dosed[..., 0] + dose_amount
        return y_dosed

    def solve_segment(self, y0, t_start, t_end, steps_per_segment):
        """
        Integrate one segment from t_start to t_end.

        Args:
            y0: Initial state [..., 2]
            t_start: Start time (scalar)
            t_end: End time (scalar)
            steps_per_segment: Number of time points

        Returns:
            ys_segment: Trajectory (steps_per_segment, ..., 2)
        """
        # Create time points for this segment
        t_eval = torch.linspace(t_start, t_end, steps_per_segment, device=device)

        # Integrate using odeint
        ys_segment = odeint(
            self.dynamics,
            y0,
            t_eval,
            method='dopri5',  # Dormand-Prince 5th order (equivalent to Tsit5)
            atol=1e-7,
            rtol=1e-5,
        )
        return ys_segment

    def piecewise_integrate(self, y0, t_final, steps_per_segment):
        """
        Piecewise integration with dosing events.

        Segments: [0, t1], [t1, t2], [t2, t3], [t3, t_final]
        Doses applied at end of each segment (except final)

        Args:
            y0: Initial state (2,)
            t_final: Final time
            steps_per_segment: Number of points per segment

        Returns:
            all_segments: shape (n_segments, steps_per_segment, 2)
        """
        # Build segment boundaries: [0, 12, 24, 36, 48]
        segment_times = torch.cat([
            torch.tensor([0.0], device=device),
            self.event_times,
            torch.tensor([t_final], device=device)
        ])

        # Doses for each segment (0 for final segment)
        segment_doses = torch.cat([
            self.event_doses,
            torch.tensor([0.0], device=device)
        ])

        n_segments = len(segment_times) - 1
        all_segments = []
        current_state = y0

        for seg_idx in range(n_segments):
            t_start = segment_times[seg_idx]
            t_end = segment_times[seg_idx + 1]
            dose_amount = segment_doses[seg_idx]

            # Integrate this segment
            ys_seg = self.solve_segment(
                current_state, t_start, t_end, steps_per_segment
            )  # (steps_per_segment, 2)

            all_segments.append(ys_seg)

            # Get final state and apply dose
            current_state = ys_seg[-1].clone()
            if dose_amount > 0:
                current_state = self.apply_dose(current_state, dose_amount)

        # Stack all segments: (n_segments, steps_per_segment, 2)
        return torch.stack(all_segments, dim=0)

    def forward(self, y0, t_final, steps_per_segment):
        """
        Forward pass: integrate from initial state.

        Args:
            y0: Initial state (batch_size, 2) or (2,)
            t_final: Final time
            steps_per_segment: Number of points per segment

        Returns:
            Trajectory: (batch_size, n_segments, steps_per_segment, 2)
                        or (n_segments, steps_per_segment, 2) if y0 is (2,)
        """
        if y0.dim() == 1:
            # Single sample
            return self.piecewise_integrate(y0, t_final, steps_per_segment)
        else:
            # Batch of samples
            batch_size = y0.shape[0]
            all_trajs = []
            for i in range(batch_size):
                traj = self.piecewise_integrate(y0[i], t_final, steps_per_segment)
                all_trajs.append(traj)
            return torch.stack(all_trajs, dim=0)


# ==============================================================================
# TRAINING
# ==============================================================================
def train_model(data, event_times, event_doses, t_final, n_iters=2000,
                lr=1e-3, seed=42):
    """
    Train the Neural ODE model.

    Args:
        data: Training data (num_samples, n_segments, steps_per_segment, 2)
        event_times: Dosing times
        event_doses: Dose amounts
        t_final: Final time
        n_iters: Number of training iterations
        lr: Learning rate
        seed: Random seed

    Returns:
        model: Trained model
        losses: Training loss history
        training_time: Total training time (seconds)
    """
    torch.manual_seed(seed)
    num_samples, n_segments, steps_per_segment, _ = data.shape

    # Initialize model
    model = PKNeuralODE(event_times, event_doses, hidden_dim=64).to(device)

    # Initialize optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Extract initial conditions
    y0s = data[:, 0, 0, :]  # (num_samples, 2)

    losses = []
    print(f"Training TorchDiffEq model for {n_iters} iterations...")
    start_time = time.time()

    for i in range(n_iters):
        # Forward pass
        pred = model(y0s, t_final, steps_per_segment)  # (num_samples, n_segments, steps_per_segment, 2)

        # Compute loss
        loss = criterion(pred, data)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        losses.append(loss.item())

        if i % 200 == 0:
            print(f"Iteration {i:4d}, Loss: {loss.item():.6f}")

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Final loss: {losses[-1]:.6f}")

    return model, losses, training_time


# ==============================================================================
# EVALUATION AND VISUALIZATION
# ==============================================================================
def evaluate_model(model, data, event_times, event_doses, t_final, sample_idx=0):
    """
    Evaluate model on a specific sample and create visualization.
    """
    num_samples, n_segments, steps_per_segment, _ = data.shape

    # Get predictions for one sample
    model.eval()
    with torch.no_grad():
        y0_sample = data[sample_idx, 0, 0, :]
        pred_segments = model(y0_sample, t_final, steps_per_segment)

    true_segments = data[sample_idx]

    # Compute metrics
    mse = torch.mean((pred_segments - true_segments)**2).item()
    rmse = np.sqrt(mse)

    print(f"\nEvaluation Metrics (Sample {sample_idx}):")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")

    # Convert to numpy for plotting
    pred_segments = pred_segments.cpu().numpy()
    true_segments = true_segments.cpu().numpy()
    event_times_np = event_times.cpu().numpy() if isinstance(event_times, torch.Tensor) else np.array(event_times)

    # Build time axis for plotting
    segment_boundaries = np.array([0.0, 12.0, 24.0, 36.0, 48.0])
    T_plot = []
    A1_true_list, A2_true_list = [], []
    A1_pred_list, A2_pred_list = [], []

    for seg_idx in range(n_segments):
        t_start = segment_boundaries[seg_idx]
        t_end = segment_boundaries[seg_idx + 1]
        t_real = np.linspace(t_start, t_end, steps_per_segment)

        T_plot.append(t_real)
        A1_true_list.append(true_segments[seg_idx, :, 0])
        A2_true_list.append(true_segments[seg_idx, :, 1])
        A1_pred_list.append(pred_segments[seg_idx, :, 0])
        A2_pred_list.append(pred_segments[seg_idx, :, 1])

    # Concatenate all segments
    T_plot = np.concatenate(T_plot)
    A1_true = np.concatenate(A1_true_list)
    A2_true = np.concatenate(A2_true_list)
    A1_pred = np.concatenate(A1_pred_list)
    A2_pred = np.concatenate(A2_pred_list)

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(T_plot, A1_true, 'b-', label='True A1 (Central)', linewidth=2)
    plt.plot(T_plot, A1_pred, 'r--', label='Predicted A1', linewidth=2)
    plt.plot(T_plot, A2_true, 'g-', label='True A2 (Peripheral)', linewidth=2)
    plt.plot(T_plot, A2_pred, 'y--', label='Predicted A2', linewidth=2)

    # Mark dosing times
    for dose_time in event_times_np:
        plt.axvline(x=dose_time, color='gray', linestyle=':', alpha=0.5)

    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Amount (mg)', fontsize=12)
    plt.title(f'TorchDiffEq Neural ODE - Two-Compartment PK\nRMSE: {rmse:.4f}', fontsize=14)
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
    print("TorchDiffEq Standardized Implementation")
    print("=" * 60)

    data = load_piecewise_data("data/pk_dataset_2C.npz")
    num_samples, n_segments, steps_per_segment, _ = data.shape
    print(f"Loaded dataset: {tuple(data.shape)}")
    print(f"  - Samples: {num_samples}")
    print(f"  - Segments: {n_segments}")
    print(f"  - Steps per segment: {steps_per_segment}")
    print(f"  - Compartments: 2")

    # Define dosing schedule
    event_times = [12.0, 24.0, 36.0]  # hours
    event_doses = [100.0, 100.0, 100.0]  # mg
    t_final = 48.0
    print(f"\nDosing schedule: {event_doses} mg at t = {event_times} hours")

    # Train model
    model, losses, training_time = train_model(
        data, event_times, event_doses, t_final,
        n_iters=2000, lr=1e-3, seed=42
    )

    # Evaluate model
    pred_segments, mse, rmse = evaluate_model(
        model, data, event_times, event_doses, t_final, sample_idx=0
    )

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.title('TorchDiffEq Training Loss', fontsize=14)
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
