import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set device
device = torch.device('cpu')  # Force CPU to avoid GPU issues
print(f"Using device: {device}")

# Simple Neural ODE function
class SimpleODEFunc(nn.Module):
    def __init__(self):
        super(SimpleODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
    
    def forward(self, t, y):
        return self.net(y)

# Test simple integration first
def test_simple_ode():
    print("Testing simple ODE integration...")
    
    func = SimpleODEFunc()
    y0 = torch.tensor([[1.0]], dtype=torch.float32)
    t = torch.tensor([0.0, 1.0], dtype=torch.float32)
    
    try:
        sol = odeint(func, y0, t, method='dopri5')
        print(f"Simple ODE test successful: {sol.shape}")
        return True
    except Exception as e:
        print(f"Simple ODE test failed: {e}")
        return False

# Test with actual data (small subset)
def test_with_data():
    print("Testing with actual data (subset)...")
    
    # Load just a small subset of data
    pk_data = pd.read_csv('data/pk_simulation_data.csv')
    
    # Take every 50th point to reduce density
    subset_data = pk_data.iloc[::50].copy().reset_index(drop=True)
    
    times = torch.tensor(subset_data['Day'].values, dtype=torch.float32)
    concentrations = torch.tensor(subset_data['Concentration_with_noise (mg/L)'].values, dtype=torch.float32)
    
    print(f"Subset data: {len(times)} points")
    print(f"Time range: {times.min():.3f} to {times.max():.3f}")
    
    # Simple forward pass without events
    func = SimpleODEFunc()
    y0 = torch.tensor([[0.0]], dtype=torch.float32)
    
    try:
        # Test integration over the time span
        t_span = torch.tensor([float(times[0]), float(times[-1])], dtype=torch.float32)
        sol = odeint(func, y0, t_span, method='dopri5')
        print(f"Data integration test successful: {sol.shape}")
        return True
    except Exception as e:
        print(f"Data integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("PyTorch/torchdiffeq Environment Test")
    print("="*40)
    
    # Test 1: Simple ODE
    test1_passed = test_simple_ode()
    
    # Test 2: With data
    if test1_passed:
        test2_passed = test_with_data()
        
        if test2_passed:
            print("\n✅ Environment setup looks good!")
            print("The issue might be in the specific model implementation.")
        else:
            print("\n❌ Issue with data integration.")
    else:
        print("\n❌ Basic torchdiffeq integration failing.")
    
    print("\nEnvironment check complete.")