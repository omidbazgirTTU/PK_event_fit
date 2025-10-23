"""
Comprehensive Framework Comparison: JAX vs TorchDiffEq vs TorchDyn
Benchmarks time, memory, stability, and accuracy across all three implementations
"""
import sys
import os
import time
import tracemalloc
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure matplotlib
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

# ==============================================================================
# MEMORY PROFILING UTILITIES
# ==============================================================================
class MemoryProfiler:
    """Track memory usage during execution"""
    def __init__(self):
        self.process = psutil.Process()
        self.peak_mem = 0
        self.start_mem = 0

    def start(self):
        """Start memory tracking"""
        tracemalloc.start()
        self.start_mem = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_mem = self.start_mem

    def update(self):
        """Update peak memory"""
        current_mem = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_mem = max(self.peak_mem, current_mem)

    def stop(self):
        """Stop tracking and return stats"""
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            'peak_mb': self.peak_mem,
            'allocated_mb': peak / 1024 / 1024,
            'current_mb': self.process.memory_info().rss / 1024 / 1024
        }


# ==============================================================================
# JAX BENCHMARK
# ==============================================================================
def benchmark_jax(n_iters=2000, n_runs=3):
    """
    Benchmark JAX/Diffrax implementation
    """
    print("\n" + "="*70)
    print("BENCHMARKING JAX/DIFFRAX")
    print("="*70)

    try:
        import jax
        import jax.numpy as jnp
        from fit.jax_workflow.jax_standardized import (
            load_piecewise_data, train_model, evaluate_model
        )
    except ImportError as e:
        print(f"ERROR: Failed to import JAX components: {e}")
        return None

    # Load data
    data = load_piecewise_data("data/pk_dataset_2C.npz")
    event_times = jnp.array([12.0, 24.0, 36.0])
    event_doses = jnp.array([100.0, 100.0, 100.0])
    t_final = 48.0

    results = {
        'framework': 'JAX/Diffrax',
        'training_times': [],
        'final_losses': [],
        'eval_mse': [],
        'eval_rmse': [],
        'peak_memory_mb': [],
        'loss_histories': []
    }

    for run in range(n_runs):
        print(f"\n--- Run {run + 1}/{n_runs} ---")

        # Start memory profiling
        mem_profiler = MemoryProfiler()
        mem_profiler.start()

        # Train model
        nn_params, losses, training_time = train_model(
            data, event_times, event_doses, t_final,
            n_iters=n_iters, lr=1e-3, seed=42 + run
        )

        # Update memory
        mem_profiler.update()
        mem_stats = mem_profiler.stop()

        # Evaluate
        pred, mse, rmse = evaluate_model(
            nn_params, data, event_times, event_doses, t_final, sample_idx=0
        )

        # Store results
        results['training_times'].append(training_time)
        results['final_losses'].append(losses[-1])
        results['eval_mse'].append(float(mse))
        results['eval_rmse'].append(float(rmse))
        results['peak_memory_mb'].append(mem_stats['peak_mb'])
        results['loss_histories'].append(losses)

    # Compute statistics
    results['mean_training_time'] = np.mean(results['training_times'])
    results['std_training_time'] = np.std(results['training_times'])
    results['mean_final_loss'] = np.mean(results['final_losses'])
    results['mean_eval_mse'] = np.mean(results['eval_mse'])
    results['mean_eval_rmse'] = np.mean(results['eval_rmse'])
    results['mean_peak_memory'] = np.mean(results['peak_memory_mb'])

    print(f"\n{'='*70}")
    print(f"JAX Summary (n={n_runs} runs):")
    print(f"  Training time: {results['mean_training_time']:.2f} ± {results['std_training_time']:.2f} s")
    print(f"  Final loss: {results['mean_final_loss']:.6f}")
    print(f"  Eval RMSE: {results['mean_eval_rmse']:.6f}")
    print(f"  Peak memory: {results['mean_peak_memory']:.1f} MB")
    print(f"{'='*70}")

    return results


# ==============================================================================
# TORCHDIFFEQ BENCHMARK
# ==============================================================================
def benchmark_torchdiffeq(n_iters=2000, n_runs=3):
    """
    Benchmark TorchDiffEq implementation
    """
    print("\n" + "="*70)
    print("BENCHMARKING TORCHDIFFEQ")
    print("="*70)

    try:
        from fit.pytorch_workflow.torchdiffeq_standardized import (
            load_piecewise_data, train_model, evaluate_model
        )
    except ImportError as e:
        print(f"ERROR: Failed to import TorchDiffEq components: {e}")
        return None

    # Load data
    data = load_piecewise_data("data/pk_dataset_2C.npz")
    event_times = [12.0, 24.0, 36.0]
    event_doses = [100.0, 100.0, 100.0]
    t_final = 48.0

    results = {
        'framework': 'TorchDiffEq',
        'training_times': [],
        'final_losses': [],
        'eval_mse': [],
        'eval_rmse': [],
        'peak_memory_mb': [],
        'loss_histories': []
    }

    for run in range(n_runs):
        print(f"\n--- Run {run + 1}/{n_runs} ---")

        # Start memory profiling
        mem_profiler = MemoryProfiler()
        mem_profiler.start()

        # Train model
        model, losses, training_time = train_model(
            data, event_times, event_doses, t_final,
            n_iters=n_iters, lr=1e-3, seed=42 + run
        )

        # Update memory
        mem_profiler.update()
        mem_stats = mem_profiler.stop()

        # Evaluate
        pred, mse, rmse = evaluate_model(
            model, data, event_times, event_doses, t_final, sample_idx=0
        )

        # Store results
        results['training_times'].append(training_time)
        results['final_losses'].append(losses[-1])
        results['eval_mse'].append(mse)
        results['eval_rmse'].append(rmse)
        results['peak_memory_mb'].append(mem_stats['peak_mb'])
        results['loss_histories'].append(losses)

        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Compute statistics
    results['mean_training_time'] = np.mean(results['training_times'])
    results['std_training_time'] = np.std(results['training_times'])
    results['mean_final_loss'] = np.mean(results['final_losses'])
    results['mean_eval_mse'] = np.mean(results['eval_mse'])
    results['mean_eval_rmse'] = np.mean(results['eval_rmse'])
    results['mean_peak_memory'] = np.mean(results['peak_memory_mb'])

    print(f"\n{'='*70}")
    print(f"TorchDiffEq Summary (n={n_runs} runs):")
    print(f"  Training time: {results['mean_training_time']:.2f} ± {results['std_training_time']:.2f} s")
    print(f"  Final loss: {results['mean_final_loss']:.6f}")
    print(f"  Eval RMSE: {results['mean_eval_rmse']:.6f}")
    print(f"  Peak memory: {results['mean_peak_memory']:.1f} MB")
    print(f"{'='*70}")

    return results


# ==============================================================================
# TORCHDYN BENCHMARK
# ==============================================================================
def benchmark_torchdyn(n_iters=2000, n_runs=3):
    """
    Benchmark TorchDyn implementation
    """
    print("\n" + "="*70)
    print("BENCHMARKING TORCHDYN")
    print("="*70)

    try:
        from fit.pytorch_workflow.torchdyn_standardized import (
            load_piecewise_data, train_model, evaluate_model, TORCHDYN_AVAILABLE
        )

        if not TORCHDYN_AVAILABLE:
            print("ERROR: TorchDyn not installed")
            print("Install with: pip install torchdyn")
            return None

    except ImportError as e:
        print(f"ERROR: Failed to import TorchDyn components: {e}")
        return None

    # Load data
    data = load_piecewise_data("data/pk_dataset_2C.npz")
    event_times = [12.0, 24.0, 36.0]
    event_doses = [100.0, 100.0, 100.0]
    t_final = 48.0

    results = {
        'framework': 'TorchDyn',
        'training_times': [],
        'final_losses': [],
        'eval_mse': [],
        'eval_rmse': [],
        'peak_memory_mb': [],
        'loss_histories': []
    }

    for run in range(n_runs):
        print(f"\n--- Run {run + 1}/{n_runs} ---")

        # Start memory profiling
        mem_profiler = MemoryProfiler()
        mem_profiler.start()

        # Train model
        model, losses, training_time = train_model(
            data, event_times, event_doses, t_final,
            n_iters=n_iters, lr=1e-3, seed=42 + run
        )

        # Update memory
        mem_profiler.update()
        mem_stats = mem_profiler.stop()

        # Evaluate
        pred, mse, rmse = evaluate_model(
            model, data, event_times, event_doses, t_final, sample_idx=0
        )

        # Store results
        results['training_times'].append(training_time)
        results['final_losses'].append(losses[-1])
        results['eval_mse'].append(mse)
        results['eval_rmse'].append(rmse)
        results['peak_memory_mb'].append(mem_stats['peak_mb'])
        results['loss_histories'].append(losses)

        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Compute statistics
    results['mean_training_time'] = np.mean(results['training_times'])
    results['std_training_time'] = np.std(results['training_times'])
    results['mean_final_loss'] = np.mean(results['final_losses'])
    results['mean_eval_mse'] = np.mean(results['eval_mse'])
    results['mean_eval_rmse'] = np.mean(results['eval_rmse'])
    results['mean_peak_memory'] = np.mean(results['peak_memory_mb'])

    print(f"\n{'='*70}")
    print(f"TorchDyn Summary (n={n_runs} runs):")
    print(f"  Training time: {results['mean_training_time']:.2f} ± {results['std_training_time']:.2f} s")
    print(f"  Final loss: {results['mean_final_loss']:.6f}")
    print(f"  Eval RMSE: {results['mean_eval_rmse']:.6f}")
    print(f"  Peak memory: {results['mean_peak_memory']:.1f} MB")
    print(f"{'='*70}")

    return results


# ==============================================================================
# COMPARISON VISUALIZATION
# ==============================================================================
def create_comparison_plots(all_results, output_dir="results/comparison_plots"):
    """
    Create comprehensive comparison plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filter out None results
    all_results = [r for r in all_results if r is not None]

    if len(all_results) == 0:
        print("No results to plot!")
        return

    frameworks = [r['framework'] for r in all_results]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(frameworks)]

    # 1. Training Time Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    training_times = [r['mean_training_time'] for r in all_results]
    training_stds = [r['std_training_time'] for r in all_results]

    bars = ax.bar(frameworks, training_times, yerr=training_stds,
                   capsize=5, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Training Time (seconds)', fontsize=12)
    ax.set_title('Training Time Comparison (2000 iterations)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, time in zip(bars, training_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_time_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/training_time_comparison.png")

    # 2. Memory Usage Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    memory_usage = [r['mean_peak_memory'] for r in all_results]

    bars = ax.bar(frameworks, memory_usage, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Peak Memory Usage (MB)', fontsize=12)
    ax.set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar, mem in zip(bars, memory_usage):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mem:.0f} MB', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/memory_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/memory_comparison.png")

    # 3. Accuracy Comparison (RMSE)
    fig, ax = plt.subplots(figsize=(10, 6))
    rmse_values = [r['mean_eval_rmse'] for r in all_results]

    bars = ax.bar(frameworks, rmse_values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Prediction Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar, rmse in zip(bars, rmse_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rmse:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/accuracy_comparison.png")

    # 4. Training Loss Curves
    fig, ax = plt.subplots(figsize=(12, 6))

    for result, color in zip(all_results, colors):
        # Average loss history across runs
        loss_histories = np.array(result['loss_histories'])
        mean_loss = np.mean(loss_histories, axis=0)
        std_loss = np.std(loss_histories, axis=0)

        iterations = np.arange(len(mean_loss))
        ax.plot(iterations, mean_loss, label=result['framework'],
                color=color, linewidth=2)
        ax.fill_between(iterations, mean_loss - std_loss, mean_loss + std_loss,
                        alpha=0.2, color=color)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Training Loss Convergence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_curves.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/loss_curves.png")

    # 5. Comprehensive Summary Dashboard
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Training time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(frameworks, training_times, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Time (s)', fontsize=10)
    ax1.set_title('Training Time', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Memory
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(frameworks, memory_usage, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Memory (MB)', fontsize=10)
    ax2.set_title('Peak Memory', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # RMSE
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(frameworks, rmse_values, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('RMSE', fontsize=10)
    ax3.set_title('Accuracy (lower is better)', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # Loss curves
    ax4 = fig.add_subplot(gs[1, :])
    for result, color in zip(all_results, colors):
        loss_histories = np.array(result['loss_histories'])
        mean_loss = np.mean(loss_histories, axis=0)
        ax4.plot(mean_loss, label=result['framework'], color=color, linewidth=2)
    ax4.set_xlabel('Iteration', fontsize=11)
    ax4.set_ylabel('Loss', fontsize=11)
    ax4.set_yscale('log')
    ax4.set_title('Training Loss Convergence', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    fig.suptitle('Framework Comparison Dashboard', fontsize=16, fontweight='bold')
    plt.savefig(f"{output_dir}/comparison_dashboard.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/comparison_dashboard.png")


# ==============================================================================
# RESULTS TABLE
# ==============================================================================
def create_results_table(all_results, output_file="results/framework_comparison.csv"):
    """
    Create a detailed comparison table
    """
    # Filter out None results
    all_results = [r for r in all_results if r is not None]

    if len(all_results) == 0:
        print("No results to tabulate!")
        return

    table_data = []
    for result in all_results:
        row = {
            'Framework': result['framework'],
            'Mean Training Time (s)': f"{result['mean_training_time']:.2f}",
            'Std Training Time (s)': f"{result['std_training_time']:.2f}",
            'Time per Iteration (ms)': f"{result['mean_training_time']/2000*1000:.2f}",
            'Final Loss': f"{result['mean_final_loss']:.6f}",
            'Eval MSE': f"{result['mean_eval_mse']:.6f}",
            'Eval RMSE': f"{result['mean_eval_rmse']:.6f}",
            'Peak Memory (MB)': f"{result['mean_peak_memory']:.1f}",
        }
        table_data.append(row)

    df = pd.DataFrame(table_data)

    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nSaved results table to: {output_file}")

    # Print formatted table
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

    return df


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("\n" + "="*80)
    print("NEURAL ODE FRAMEWORK COMPARISON")
    print("JAX/Diffrax vs TorchDiffEq vs TorchDyn")
    print("="*80)
    print("\nConfiguration:")
    print("  - Model: 2-compartment PK with piecewise dosing")
    print("  - Network: [2 → 64 → 64 → 2] with ReLU")
    print("  - Dataset: 100 samples, 4 segments, 200 steps/segment")
    print("  - Training: 2000 iterations, Adam optimizer (lr=1e-3)")
    print("  - Runs per framework: 3 (for statistical significance)")
    print("="*80)

    # Run benchmarks
    n_iters = 2000
    n_runs = 3

    all_results = []

    # JAX
    jax_results = benchmark_jax(n_iters=n_iters, n_runs=n_runs)
    if jax_results:
        all_results.append(jax_results)

    # TorchDiffEq
    torchdiffeq_results = benchmark_torchdiffeq(n_iters=n_iters, n_runs=n_runs)
    if torchdiffeq_results:
        all_results.append(torchdiffeq_results)

    # TorchDyn
    torchdyn_results = benchmark_torchdyn(n_iters=n_iters, n_runs=n_runs)
    if torchdyn_results:
        all_results.append(torchdyn_results)

    # Generate comparison outputs
    if len(all_results) > 0:
        create_results_table(all_results)
        create_comparison_plots(all_results)

        print("\n" + "="*80)
        print("BENCHMARK COMPLETE!")
        print("="*80)
        print("Results saved to:")
        print("  - results/framework_comparison.csv")
        print("  - results/comparison_plots/")
        print("="*80)
    else:
        print("\nERROR: No frameworks successfully benchmarked!")


if __name__ == "__main__":
    main()
