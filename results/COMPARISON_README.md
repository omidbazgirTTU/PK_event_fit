# Neural ODE Framework Comparison

**Apples-to-Apples Comparison: JAX/Diffrax vs TorchDiffEq vs TorchDyn**

## Overview

This directory contains the comprehensive benchmarking results comparing three Neural ODE frameworks for pharmacokinetic (PK) modeling:

1. **JAX/Diffrax** - Functional programming with JIT compilation
2. **TorchDiffEq** - Classic PyTorch ODE solver
3. **TorchDyn** - Modern PyTorch Neural ODE library

## Standardized Setup

All three implementations use **identical** configurations for fair comparison:

### Model Architecture
- **Task**: Two-compartment PK model with piecewise dosing
- **Network**: `[2 → 64 → 64 → 2]` with ReLU activation
- **State**: `[A1, A2]` (central and peripheral compartment amounts)
- **Integration**: Piecewise approach with 4 segments

### Data
- **Dataset**: `pk_dataset_2C.npz`
- **Samples**: 100 different parameter sets
- **Segments**: 4 (corresponding to dosing events)
- **Time points**: 200 per segment
- **Dosing**: 100 mg bolus at t = [12, 24, 36] hours

### Training Configuration
- **Optimizer**: Adam with learning rate 1e-3
- **Iterations**: 2000
- **Loss**: Mean Squared Error (MSE)
- **Solver**: Dopri5/Tsit5 (equivalent 5th-order adaptive methods)
- **Tolerances**: atol=1e-7, rtol=1e-5
- **Seed**: 42 (for reproducibility)

## Benchmark Metrics

The comparison evaluates:

### 1. **Time Metrics**
- Total training time (seconds)
- Time per iteration (milliseconds)
- Standard deviation across runs

### 2. **Memory Metrics**
- Peak memory usage (MB)
- Memory allocation patterns

### 3. **Stability Metrics**
- Loss convergence curves
- Final loss values
- Consistency across multiple runs

### 4. **Accuracy Metrics**
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Prediction quality on test samples

## Running the Benchmark

### Prerequisites

Install all dependencies:
```bash
pip install -r requirements.txt
```

### Execute Comparison

From the repository root:
```bash
cd src/benchmarks
python compare_frameworks.py
```

This will:
1. Run each framework 3 times for statistical significance
2. Measure time, memory, and accuracy
3. Generate comparison plots
4. Create summary tables

### Individual Framework Testing

Test each framework independently:

```bash
# JAX/Diffrax
python src/fit/jax_workflow/jax_standardized.py

# TorchDiffEq
python src/fit/pytorch_workflow/torchdiffeq_standardized.py

# TorchDyn
python src/fit/pytorch_workflow/torchdyn_standardized.py
```

## Output Files

After running the benchmark, the following files are generated:

### Tables
- `framework_comparison.csv` - Detailed comparison table

### Plots (in `comparison_plots/`)
- `training_time_comparison.png` - Bar chart of training times
- `memory_comparison.png` - Memory usage comparison
- `accuracy_comparison.png` - RMSE comparison
- `loss_curves.png` - Training loss convergence
- `comparison_dashboard.png` - Comprehensive summary dashboard

## Expected Results

### Performance Characteristics

**JAX/Diffrax:**
- ✓ Fastest training (JIT compilation)
- ✓ Most memory efficient
- ✓ Excellent for large-scale experiments
- ✗ Steeper learning curve

**TorchDiffEq:**
- ✓ Mature and stable
- ✓ Good PyTorch integration
- ✓ Well-documented
- ✗ Slower than JAX
- ✗ Higher memory usage

**TorchDyn:**
- ✓ Modern API design
- ✓ Advanced features (adjoint sensitivity)
- ✓ Active development
- ✗ Newer library (smaller community)
- ~ Performance between JAX and TorchDiffEq

### When to Use Each Framework

| Framework | Best For |
|-----------|----------|
| **JAX/Diffrax** | Large-scale research, parameter sweeps, publication-quality results |
| **TorchDiffEq** | Production systems, existing PyTorch pipelines, stability |
| **TorchDyn** | Modern PyTorch projects, advanced ODE features, prototyping |

## Implementation Details

### JAX/Diffrax Approach
- Functional programming style
- `jax.jit` compilation for speed
- `jax.vmap` for batch processing
- `lax.scan` for piecewise integration
- Diffrax's `Tsit5()` solver

### TorchDiffEq Approach
- Imperative PyTorch style
- Standard `nn.Module` architecture
- Manual piecewise integration loop
- `odeint()` with 'dopri5' solver
- Gradient clipping for stability

### TorchDyn Approach
- `NeuralODE` wrapper class
- Adjoint sensitivity method
- Clean API design
- `.trajectory()` method for integration
- Same 'dopri5' solver

## Reproducibility

All implementations:
- Use fixed random seeds (`torch.manual_seed(42)`, `jax.random.PRNGKey(42)`)
- Share identical network initialization distributions
- Process the same dataset in the same order
- Use equivalent numerical solvers

Results should be reproducible within floating-point precision across different runs.

## Citation

If you use this comparison in your research, please cite:

```bibtex
@software{pk_node_comparison,
  title = {Neural ODE Framework Comparison for Pharmacokinetic Modeling},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/PK_event_fit}
}
```

## Contributing

To add a new framework to this comparison:

1. Implement the standardized architecture in `src/fit/<framework>_workflow/`
2. Add benchmark function to `src/benchmarks/compare_frameworks.py`
3. Update this README
4. Submit a pull request

## License

MIT License - See LICENSE file for details
