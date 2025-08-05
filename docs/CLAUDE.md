# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements Neural Ordinary Differential Equations (Neural ODEs) for pharmacokinetic (PK) modeling with piecewise dosing events. The project uses JAX/Diffrax and PyTorch/torchdiffeq to learn two-compartment PK system dynamics from synthetic data.

## Common Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

### Data Generation
```bash
# Generate synthetic 2-compartment PK data for 100 patients
python src/simulate/2C_simulator.py
```

### Training Neural ODE Models
```bash
# JAX implementation with piecewise integration
python src/fit/jax_workflow/full_solution_o1.py

# PyTorch implementation (alternative)
python src/fit/pytorch_workflow/omid_torchdiffeq.py
```

### PKPD Control Optimization
```bash
# Dose optimization with tumor dynamics
python src/control/pkpd_optim.py
```

## Architecture Overview

### Core Components

1. **Data Simulation** (`src/simulate/`):
   - `2C_simulator.py`: Generates synthetic 2-compartment PK data
   - Parameters sampled from multivariate log-normal distribution
   - Dose schedule: 100mg at t=[12, 24, 36] hours, final time 48 hours
   - Output: `pk_dataset_2C.npz` with shape (100, 4, 200, 2)

2. **Neural ODE Fitting** (`src/fit/`):
   - **JAX workflow**: Primary implementation using Diffrax for ODE solving
   - **PyTorch workflow**: Alternative implementation with torchdiffeq
   - MLP architecture: [2, 32, 32, 2] (compartments → hidden → derivatives)
   - Piecewise integration with time scaling for numerical stability

3. **Control Systems** (`src/control/`):
   - PKPD optimization combining PK dynamics with tumor growth models
   - Dose optimization to maximize drug exposure while considering efficacy

### Key Technical Details

- **Two-compartment ODE system**:
  ```
  dC1/dt = -k10*C1 - k12*C1 + k21*C2
  dC2/dt = k12*C1 - k21*C2
  ```
- **Piecewise dosing**: Discontinuous events handled by state jumps (C1 += dose_amount)
- **Time scaling**: Each segment mapped to [0,1] for integration stability
- **Batched training**: Efficient processing across multiple patient trajectories

### Data Format

Training data (`pk_dataset_2C.npz`):
- `all_solutions`: Shape (num_samples, n_segments, steps_per_segment, 2)
  - num_samples: 100 (different patients)
  - n_segments: 4 (3 dosing intervals + final segment)
  - steps_per_segment: 200 (time points per segment)
  - 2: State dimension [C1, C2] (central, peripheral compartments)

## Key Dependencies

- **JAX ecosystem**: jax, jaxlib, diffrax, optax
- **Scientific computing**: numpy, scipy, matplotlib
- **Neural networks**: equinox (JAX), torch (PyTorch alternative)
- **Visualization**: matplotlib, imageio (for GIF generation)

## Development Notes

- Primary implementation uses JAX for automatic differentiation and JIT compilation
- PyTorch implementation available as alternative/comparison
- All ODE solving uses adaptive timesteppers for accuracy
- Time scaling (segment mapping to [0,1]) crucial for numerical stability
- Loss function: MSE between predicted and true concentration profiles