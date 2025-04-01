# PK Event Fitting with Neural ODEs

This project uses Neural Ordinary Differential Equations (Neural ODEs) to model pharmacokinetic (PK) systems with piecewise dosing events. The approach learns the underlying dynamics of a two-compartment PK system directly from data.

## 📋 Overview

This repository implements:

- **Two-compartment PK model simulation** - Generate synthetic PK data
- **Neural ODE implementation** - Using JAX and Diffrax to model the PK dynamics
- **Piecewise integration** - Handle multiple dosing events
- **Optimization techniques** - Learn ODE parameters from data
- **Visualization tools** - Compare predicted vs. actual concentrations

## 🧠 Model Architecture

The project uses a neural network to approximate the dynamics of a two-compartment PK system:

- **Input**: State vector `[C1, C2]` (central and peripheral compartment concentrations)
- **Hidden layers**: 2 layers with 32 units each using ReLU activation
- **Output**: Derivatives `[dC1/dt, dC2/dt]` for the ODE system

## 💉 Piecewise Dosing

The model handles piecewise dosing with:

- **Default dosing times**: [2.0, 5.0, 8.0] hours
- **Default dose amounts**: 50.0 units per dose
- **Final time**: 10.0 hours
- **Scaled time integration**: Each segment is mapped to [0,1] for stable integration

## 📚 Repository Structure

- `full_solution_o1.py` - Main Neural ODE implementation for PK event fitting
- `pk_simulator.py` - Basic implementation of the two-compartment PK model
- `pk_simulator_2.py` - Enhanced PK simulator generating training datasets
- `pk_simulator_o1.py` - PK simulator with time-scaling for better numerical stability
- `pk_dataset.npz` - Generated dataset with multiple samples
- `control/pkpd_optim.py` - Dose optimization for PKPD control problems
- `control/pkpd_simulator.py` - PKPD simulation with tumor dynamics

## 🔬 Scientific Background

The two-compartment model is a standard approach in pharmacokinetics:

- **Central compartment (C1)**: Represents blood/plasma concentration
- **Peripheral compartment (C2)**: Represents drug distribution in tissues
- **Rate constants**:
  - k10: Elimination rate from central compartment
  - k12: Distribution rate from central to peripheral
  - k21: Distribution rate from peripheral to central

The ODE system is:
```
dC1/dt = -k10*C1 - k12*C1 + k21*C2
dC2/dt = k12*C1 - k21*C2
```

## 📊 Data Format

The training data (`pk_dataset.npz`) contains:
- `all_solutions`: Shape (num_samples, n_segments, steps_per_segment, 2)
- `all_final_states`: Shape (num_samples, 2)

Where:
- num_samples: Number of different initial conditions
- n_segments: 4 segments (dosing at times [2, 5, 8] plus final segment)
- steps_per_segment: Number of time points per segment (typically 200)
- 2: State dimension [C1, C2]

## 🚀 Usage

### Requirements

- Python 3.8+
- JAX ecosystem (jax, jaxlib)
- Diffrax (differential equation solver)
- Optax (optimization)
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install -r requirements.txt
```

### Training the Neural ODE

```bash
python full_solution_o1.py
```

This will:
1. Load the PK dataset
2. Initialize the Neural ODE
3. Train using Adam optimizer
4. Plot predictions vs. true values

### Generating New Data

```bash
python pk_simulator_2.py
```

This generates a new dataset with:
- Multiple initial conditions
- Piecewise dosing events
- Saved as `pk_dataset.npz`

## 🔮 Advanced Usage

### Dose Optimization (PKPD Control)

The control directory contains tools for optimizing dosing regimens:

```bash
python control/pkpd_optim.py
```

This optimizes dose amounts to maximize drug exposure while considering tumor dynamics.

## 📝 License

MIT License 