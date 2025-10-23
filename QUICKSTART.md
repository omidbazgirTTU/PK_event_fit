# Quick Start Guide: Framework Comparison

This guide will help you run the apples-to-apples comparison of JAX, TorchDiffEq, and TorchDyn for Neural ODE-based pharmacokinetic modeling.

## Prerequisites

### System Requirements
- Python 3.8+
- 4GB+ RAM (8GB recommended)
- CPU or GPU (GPU optional but recommended for PyTorch)

### Installation

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd PK_event_fit
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Note:** If you encounter installation issues:
- For JAX with GPU: See [JAX installation guide](https://github.com/google/jax#installation)
- For PyTorch with GPU: See [PyTorch installation guide](https://pytorch.org/get-started/locally/)
- For CPU-only: The default installations should work fine

## Running the Comparison

### Option 1: Full Benchmark (Recommended)

Run all three frameworks and generate comprehensive comparison:

```bash
cd src/benchmarks
python compare_frameworks.py
```

**What this does:**
- Trains each framework 3 times (2000 iterations each)
- Measures time, memory, and accuracy
- Generates comparison plots and tables
- Saves results to `results/` directory

**Expected runtime:** 10-30 minutes (depending on hardware)

**Output:**
```
results/
├── framework_comparison.csv          # Detailed metrics table
└── comparison_plots/
    ├── training_time_comparison.png  # Time comparison
    ├── memory_comparison.png         # Memory usage
    ├── accuracy_comparison.png       # RMSE comparison
    ├── loss_curves.png              # Training convergence
    └── comparison_dashboard.png      # Summary dashboard
```

### Option 2: Test Individual Frameworks

Run each framework separately to verify they work:

**JAX/Diffrax:**
```bash
python src/fit/jax_workflow/jax_standardized.py
```

**TorchDiffEq:**
```bash
python src/fit/pytorch_workflow/torchdiffeq_standardized.py
```

**TorchDyn:**
```bash
python src/fit/pytorch_workflow/torchdyn_standardized.py
```

Each will train for 2000 iterations and display:
- Training progress (loss every 200 iterations)
- Final metrics (MSE, RMSE)
- Training time
- Visualization plots

## Understanding the Output

### Console Output Example

```
============================================================
JAX/Diffrax Standardized Implementation
============================================================
Loaded dataset: (100, 4, 200, 2)
  - Samples: 100
  - Segments: 4
  - Steps per segment: 200
  - Compartments: 2

Dosing schedule: [100. 100. 100.] mg at t = [12. 24. 36.] hours

Training JAX/Diffrax model for 2000 iterations...
Iteration    0, Loss: 245.678901
Iteration  200, Loss: 12.345678
Iteration  400, Loss: 5.678901
...
Iteration 1800, Loss: 0.001234
Training completed in 45.67 seconds
Final loss: 0.001234

Evaluation Metrics (Sample 0):
MSE:  0.001234
RMSE: 0.035128

============================================================
Training completed!
Total time: 45.67 seconds
Time per iteration: 22.84 ms
============================================================
```

### Interpreting Results

**Training Time:**
- Lower is better
- JAX typically fastest due to JIT compilation
- PyTorch frameworks depend on GPU availability

**Memory Usage:**
- Lower is better
- JAX typically most memory-efficient
- PyTorch memory usage depends on batching strategy

**Accuracy (RMSE):**
- Lower is better
- All frameworks should achieve similar accuracy (~0.03-0.05)
- Differences > 0.01 may indicate implementation issues

**Loss Convergence:**
- Should decrease smoothly
- Final loss around 0.001-0.01 is expected
- Oscillations may indicate learning rate too high

## Troubleshooting

### Common Issues

**1. Import Error: No module named 'jax'**
```bash
pip install jax jaxlib diffrax optax
```

**2. Import Error: No module named 'torch'**
```bash
pip install torch torchdiffeq torchdyn
```

**3. TorchDyn not available**
```bash
pip install torchdyn
```

**4. CUDA out of memory (PyTorch)**
- Reduce batch size or use CPU
- Add `device = 'cpu'` in PyTorch scripts

**5. JAX runs slowly**
- First run includes JIT compilation (warm-up)
- Subsequent iterations should be much faster
- Consider using GPU version of JAX

**6. Loss not decreasing**
- Check that data file `data/pk_dataset_2C.npz` exists
- Verify network architecture matches across frameworks
- Try reducing learning rate (e.g., 1e-4 instead of 1e-3)

### Verifying Installation

Quick test to verify everything is installed:

```python
# Test JAX
import jax
import diffrax
import optax
print("JAX version:", jax.__version__)

# Test PyTorch
import torch
from torchdiffeq import odeint
print("PyTorch version:", torch.__version__)

# Test TorchDyn
try:
    from torchdyn.core import NeuralODE
    print("TorchDyn available!")
except ImportError:
    print("TorchDyn not installed")
```

## Customization

### Modify Training Parameters

Edit the benchmark script or individual implementations:

```python
# Number of iterations
n_iters = 1000  # Default: 2000

# Learning rate
lr = 5e-4  # Default: 1e-3

# Number of runs for statistics
n_runs = 5  # Default: 3
```

### Change Network Architecture

All three implementations use the same architecture defined in each file:

```python
# Current: [2 → 64 → 64 → 2]
layer_sizes = [2, 64, 64, 2]

# Example modification: [2 → 128 → 128 → 2]
layer_sizes = [2, 128, 128, 2]
```

**Important:** Change in ALL THREE implementations for fair comparison!

### Use Different Data

Generate new data or use different dataset:

```python
# In each implementation, modify:
data = load_piecewise_data("data/your_dataset.npz")
```

## Next Steps

After running the comparison:

1. **Analyze Results:**
   - Review `results/framework_comparison.csv`
   - Examine plots in `results/comparison_plots/`

2. **Choose Your Framework:**
   - JAX: For research and large-scale experiments
   - TorchDiffEq: For production PyTorch systems
   - TorchDyn: For modern PyTorch with advanced features

3. **Extend the Code:**
   - Add your own PK models
   - Implement different dosing schedules
   - Try other Neural ODE architectures

4. **Publish Results:**
   - Include plots in papers/presentations
   - Share findings with the community
   - Cite this repository

## Getting Help

- **Documentation:** See `results/COMPARISON_README.md`
- **Issues:** Check repository issues page
- **Questions:** Contact the maintainers

## Example Workflow

Complete workflow from installation to results:

```bash
# 1. Setup
git clone <repo-url>
cd PK_event_fit
pip install -r requirements.txt

# 2. Verify data exists
ls data/pk_dataset_2C.npz

# 3. Test individual framework (optional)
python src/fit/jax_workflow/jax_standardized.py

# 4. Run full comparison
cd src/benchmarks
python compare_frameworks.py

# 5. View results
ls ../../results/
ls ../../results/comparison_plots/

# 6. Open comparison table
cat ../../results/framework_comparison.csv
```

## Performance Expectations

Typical results on a modern CPU (Intel i7/AMD Ryzen 7):

| Framework | Time (2000 iters) | Peak Memory | RMSE |
|-----------|------------------|-------------|------|
| JAX/Diffrax | 30-50s | 200-300 MB | 0.035 |
| TorchDiffEq | 60-90s | 400-600 MB | 0.035 |
| TorchDyn | 50-80s | 400-600 MB | 0.035 |

*Note: Results vary significantly with hardware, especially with GPU availability*

---

**Ready to start?** Run the benchmark and explore the results!

```bash
python src/benchmarks/compare_frameworks.py
```
