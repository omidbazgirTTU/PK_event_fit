import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import time
# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# Neural ODE dynamics network - pure black box
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim=64):
        super(ODEFunc, self).__init__()
        # Neural network with 1 input (concentration) and 1 output (dconc/dt)
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),  # Using SiLU (Swish) activation for smooth dynamics
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, t, y):
        return self.net(y)
# Event function for dosing - factorized form (t-tdose1)(t-tdose2)...
class EventFunc(nn.Module):
    def __init__(self, dosing_times):
        super(EventFunc, self).__init__()
        self.dosing_times = dosing_times
    def forward(self, t, y):
        # Compute the product (t-tdose1)(t-tdose2)... divided by a scaling factor
        # to prevent numerical overflow
        result = torch.ones_like(t)
        for dose_time in self.dosing_times:
            result = result * (t - dose_time)
        # Scale to prevent overflow/underflow
        result = result / (10.0 ** (len(self.dosing_times) - 1))
        return result
# Complete PK model with event handling
class PKNeuralODE(nn.Module):
    def __init__(self, dosing_times, dose_amounts, hidden_dim=64):
        super(PKNeuralODE, self).__init__()
        self.dynamics = ODEFunc(hidden_dim).to(device)
        self.event_func = EventFunc(dosing_times).to(device)
        self.dosing_times = torch.tensor(dosing_times, dtype=torch.float32).to(device)
        self.dose_amounts = torch.tensor(dose_amounts, dtype=torch.float32).to(device)
        # Learnable dose effect parameter - how dose amount translates to concentration
        self.log_dose_effect = nn.Parameter(torch.tensor(np.log(0.2), dtype=torch.float32))
    def get_dose_effect(self):
        return torch.exp(self.log_dose_effect)  # Ensure positive
    def dose_event_handler(self, t, state):
        # Find which dose time this event corresponds to
        dose_idx = None
        for i, dose_time in enumerate(self.dosing_times):
            if torch.abs(t - dose_time) < 1e-6:
                dose_idx = i
                break
        if dose_idx is not None:
            # Apply the dose effect - directly add to concentration
            state = state + self.dose_amounts[dose_idx] * self.get_dose_effect()
        return state
    def forward(self, times):
        # Initial state: no drug in the system (zero concentration)
        y0 = torch.zeros(1, 1, device=device)
        # Create tensor to store all predictions
        predictions = torch.zeros(len(times), device=device)
        current_state = y0
        # Process each time segment between adjacent time points
        for i in range(len(times)):
            t = times[i]
            # Check if current time is a dosing time using event function
            event_val = self.event_func(t.reshape(1), current_state)
            if torch.abs(event_val) < 1e-6:  # Close enough to zero
                # Handle dosing event
                current_state = self.dose_event_handler(t, current_state)
            # Store current concentration
            predictions[i] = current_state.reshape(-1)
            # Integrate to next time point if not last point
            if i < len(times) - 1:
                time_span = torch.tensor([float(t), float(times[i+1])], device=device)
                # Integrate from current time to next time
                next_state = odeint(
                    self.dynamics,
                    current_state,
                    time_span,
                    method='dopri5',
                    atol=1e-7,
                    rtol=1e-5
                )
                current_state = next_state[-1].reshape(1, 1)
        return predictions
def load_data():
    pk_data = pd.read_csv('pk_simulation_data.csv')
    dosing_schedule = pd.read_csv('dosing_schedule.csv')
    return pk_data, dosing_schedule
def train_model(model, times, concentrations, epochs=2000, lr=0.01):
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5, verbose=True)
    # Convert data to PyTorch tensors
    times_tensor = torch.tensor(times, dtype=torch.float32, device=device)
    concentrations_tensor = torch.tensor(concentrations, dtype=torch.float32, device=device)
    # Lists to store losses
    losses = []
    print("Training Neural ODE model...")
    start_time = time.time()
    # Training loop
    for epoch in range(epochs):
        # Forward pass
        predicted_conc = model(times_tensor)
        # Compute loss
        loss = criterion(predicted_conc, concentrations_tensor)
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # Update scheduler
        scheduler.step(loss)
        # Store loss
        losses.append(loss.item())
        # Print progress
        if (epoch+1) % 100 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}, Dose Effect: {model.get_dose_effect().item():.4f}')
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    return losses
def evaluate_model(model, times, observed_conc, true_conc, dosing_days, dose_amounts):
    # Convert data to PyTorch tensors
    times_tensor = torch.tensor(times, dtype=torch.float32, device=device)
    # Get predictions
    model.eval()
    with torch.no_grad():
        predicted_conc = model(times_tensor).cpu().numpy()
    # Calculate metrics
    mse = np.mean((predicted_conc - observed_conc)**2)
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum((observed_conc - predicted_conc)**2) / np.sum((observed_conc - observed_conc.mean())**2)
    print(f"\nModel Evaluation:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²: {r2:.6f}")
    print(f"Learned dose effect parameter: {model.get_dose_effect().item():.4f}")
    # Plot results
    plt.figure(figsize=(14, 10))
    # Plot 1: Model fit vs observed data
    plt.subplot(2, 1, 1)
    plt.plot(times, observed_conc, 'o', alpha=0.7, label='Observed Data')
    plt.plot(times, predicted_conc, '-', linewidth=2, label='Neural ODE Model')
    plt.plot(times, true_conc, '--', color='gray', alpha=0.7, label='True Concentration')
    # Add vertical lines for dosing events
    for day, dose in zip(dosing_days, dose_amounts):
        plt.axvline(x=day, color='r', linestyle='--', alpha=0.3)
        plt.annotate(f'{dose} mg', xy=(day, 0), xytext=(day+0.2, max(observed_conc)/10),
                     arrowprops=dict(arrowstyle='->'))
    plt.xlabel('Time (days)')
    plt.ylabel('Concentration (mg/L)')
    plt.title(f'Neural ODE Model Fit\nRMSE: {rmse:.4f}, R²: {r2:.4f}')
    plt.legend()
    plt.grid(True)
    # Plot 2: Residuals
    plt.subplot(2, 1, 2)
    residuals = observed_conc - predicted_conc
    plt.plot(times, residuals, 'o-')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Time (days)')
    plt.ylabel('Residuals (mg/L)')
    plt.title('Residuals (Observed - Predicted)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('neural_ode_torch_results.png', dpi=300)
    plt.show()
    # Save model predictions to CSV
    results_df = pd.DataFrame({
        'Day': times,
        'Observed': observed_conc,
        'Predicted': predicted_conc,
        'True': true_conc,
        'Residuals': residuals
    })
    results_df.to_csv('neural_ode_torch_results.csv', index=False)
    print("\nResults saved to 'neural_ode_torch_results.csv' and 'neural_ode_torch_results.png'")
    return predicted_conc, mse, rmse, r2
def main():
    # Load data
    pk_data, dosing_schedule = load_data()
    # Extract required data
    days = pk_data['Day'].values
    observed_conc = pk_data['Concentration_with_noise (mg/L)'].values
    true_conc = pk_data['True_Concentration (mg/L)'].values
    dosing_days = dosing_schedule['Day'].values
    dose_amounts = dosing_schedule['Dose_Amount (mg)'].values
    # Create the Neural ODE model
    model = PKNeuralODE(dosing_days, dose_amounts, hidden_dim=64).to(device)
    # Train the model
    losses = train_model(model, days, observed_conc, epochs=2000, lr=0.01)
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig('training_loss.png', dpi=300)
    plt.show()
    # Evaluate the model
    predicted_conc, mse, rmse, r2 = evaluate_model(model, days, observed_conc, true_conc, dosing_days, dose_amounts)
    # Save the trained model
    torch.save(model.state_dict(), 'neural_ode_pk_model.pt')
    print("Model saved to 'neural_ode_pk_model.pt'")
if __name__ == "__main__":
    main()