"""
Enhanced Framework Comparison for Professional Blog Post
Generates all figures, tables, and markdown for publication
"""
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))

# Import benchmark tools
from benchmarks.compare_frameworks import (
    benchmark_jax, benchmark_torchdiffeq, benchmark_torchdyn,
    create_comparison_plots, create_results_table
)
from benchmarks.dosing_stability_analysis import (
    DosingStabilityAnalyzer, create_dosing_zoom_plot,
    create_side_by_side_comparison, create_metrics_comparison_table
)
from benchmarks.blog_generator import BlogPostGenerator

# Configure matplotlib for publication quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def extract_trajectory_data(framework_name, results, data, event_times, t_final, steps_per_segment):
    """
    Extract trajectory data from trained model for stability analysis.

    Args:
        framework_name: 'jax', 'torchdiffeq', or 'torchdyn'
        results: Results dict from benchmark
        data: Original dataset
        event_times: Dosing times
        t_final: Final time
        steps_per_segment: Steps per segment

    Returns:
        dict with 'times' and 'trajectory'
    """
    import numpy as np

    # Build time array
    segment_boundaries = [0.0] + list(event_times) + [t_final]
    all_times = []
    all_traj = []

    sample_idx = 0  # Use first sample

    if framework_name == 'jax':
        import jax.numpy as jnp
        from fit.jax_workflow.jax_standardized import piecewise_integrate_neural_ode

        nn_params = results.get('final_params')
        if nn_params is None:
            # Need to get params from last run
            return None

        y0 = data[sample_idx, 0, 0, :]
        pred_segments = piecewise_integrate_neural_ode(
            y0, jnp.array(event_times), jnp.array([100.0]*len(event_times)),
            t_final, nn_params, steps_per_segment
        )

        # Convert to numpy and concatenate
        for seg_idx in range(len(segment_boundaries)-1):
            t_start = segment_boundaries[seg_idx]
            t_end = segment_boundaries[seg_idx+1]
            t_seg = np.linspace(t_start, t_end, steps_per_segment)
            all_times.append(t_seg)
            all_traj.append(np.array(pred_segments[seg_idx]))

    else:  # PyTorch frameworks
        import torch
        model = results.get('final_model')
        if model is None:
            return None

        model.eval()
        with torch.no_grad():
            y0 = torch.tensor(data[sample_idx, 0, 0, :], dtype=torch.float32, device=model.event_times.device)
            pred_segments = model(y0, t_final, steps_per_segment)

        pred_segments = pred_segments.cpu().numpy()

        for seg_idx in range(len(segment_boundaries)-1):
            t_start = segment_boundaries[seg_idx]
            t_end = segment_boundaries[seg_idx+1]
            t_seg = np.linspace(t_start, t_end, steps_per_segment)
            all_times.append(t_seg)
            all_traj.append(pred_segments[seg_idx])

    times = np.concatenate(all_times)
    trajectory = np.concatenate(all_traj)

    return {'times': times, 'trajectory': trajectory}


def run_enhanced_benchmark(n_iters=2000, n_runs=3, output_dir="results/blog_post"):
    """
    Run comprehensive benchmark and generate all blog materials.
    """
    print("\n" + "="*80)
    print("ENHANCED FRAMEWORK COMPARISON FOR BLOG POST")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Iterations: {n_iters}")
    print(f"  - Runs per framework: {n_runs}")
    print(f"  - Output directory: {output_dir}")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)

    # Load data for stability analysis
    import numpy as np
    data = np.load("data/pk_dataset_2C.npz")["all_solutions"]
    event_times = np.array([12.0, 24.0, 36.0])
    event_doses = np.array([100.0, 100.0, 100.0])
    t_final = 48.0
    steps_per_segment = 200

    # ==================================================================
    # 1. Run standard benchmarks
    # ==================================================================
    print("\n" + "="*80)
    print("PHASE 1: Standard Performance Benchmarks")
    print("="*80)

    all_results = []

    jax_results = benchmark_jax(n_iters=n_iters, n_runs=n_runs)
    if jax_results:
        all_results.append(jax_results)

    torchdiffeq_results = benchmark_torchdiffeq(n_iters=n_iters, n_runs=n_runs)
    if torchdiffeq_results:
        all_results.append(torchdiffeq_results)

    torchdyn_results = benchmark_torchdyn(n_iters=n_iters, n_runs=n_runs)
    if torchdyn_results:
        all_results.append(torchdyn_results)

    # Create standard comparison plots
    create_comparison_plots(all_results, output_dir=f"{output_dir}/images")
    perf_table = create_results_table(all_results, output_file=f"{output_dir}/performance_metrics.csv")

    # ==================================================================
    # 2. Dosing stability analysis
    # ==================================================================
    print("\n" + "="*80)
    print("PHASE 2: Dosing Event Stability Analysis")
    print("="*80)

    # Re-run models to extract trajectories for stability analysis
    print("\nExtracting trajectories for stability analysis...")

    # For this, we need to re-run the models and save final parameters
    # Let's do a single run to get final models
    print("\n--- Re-training models for trajectory extraction ---")

    # JAX
    if jax_results:
        print("\nRunning JAX for trajectory extraction...")
        import jax.numpy as jnp
        from fit.jax_workflow.jax_standardized import load_piecewise_data, train_model

        jax_data = load_piecewise_data("data/pk_dataset_2C.npz")
        jax_event_times = jnp.array(event_times)
        jax_event_doses = jnp.array(event_doses)

        jax_params, _, _ = train_model(
            jax_data, jax_event_times, jax_event_doses, t_final,
            n_iters=500, lr=1e-3, seed=42  # Shorter training just for trajectory
        )

        # Extract trajectory
        from fit.jax_workflow.jax_standardized import piecewise_integrate_neural_ode

        y0 = jax_data[0, 0, 0, :]
        jax_pred = piecewise_integrate_neural_ode(
            y0, jax_event_times, jax_event_doses, t_final, jax_params, steps_per_segment
        )

        # Build time array and trajectory
        segment_boundaries = [0.0, 12.0, 24.0, 36.0, 48.0]
        jax_times_list = []
        jax_traj_list = []
        for seg_idx in range(4):
            t_start = segment_boundaries[seg_idx]
            t_end = segment_boundaries[seg_idx+1]
            t_seg = np.linspace(t_start, t_end, steps_per_segment)
            jax_times_list.append(t_seg)
            jax_traj_list.append(np.array(jax_pred[seg_idx]))

        jax_times = np.concatenate(jax_times_list)
        jax_trajectory = np.concatenate(jax_traj_list)

        jax_traj_data = {'times': jax_times, 'trajectory': jax_trajectory}

        # Stability analysis
        jax_analyzer = DosingStabilityAnalyzer(jax_times, jax_trajectory, event_times, compartment_idx=0)
        jax_stability_df = jax_analyzer.analyze_all_doses()
        jax_summary = jax_analyzer.compute_summary_metrics()

        print("\nJAX Stability Metrics:")
        print(jax_stability_df)

    # TorchDiffEq
    if torchdiffeq_results:
        print("\nRunning TorchDiffEq for trajectory extraction...")
        from fit.pytorch_workflow.torchdiffeq_standardized import load_piecewise_data, train_model
        import torch

        torch_data = load_piecewise_data("data/pk_dataset_2C.npz")
        torch_model, _, _ = train_model(
            torch_data, list(event_times), list(event_doses), t_final,
            n_iters=500, lr=1e-3, seed=42
        )

        # Extract trajectory
        torch_model.eval()
        with torch.no_grad():
            y0 = torch_data[0, 0, 0, :]
            torch_pred = torch_model(y0, t_final, steps_per_segment).cpu().numpy()

        # Build time array and trajectory
        torch_times_list = []
        torch_traj_list = []
        for seg_idx in range(4):
            t_start = segment_boundaries[seg_idx]
            t_end = segment_boundaries[seg_idx+1]
            t_seg = np.linspace(t_start, t_end, steps_per_segment)
            torch_times_list.append(t_seg)
            torch_traj_list.append(torch_pred[seg_idx])

        torch_times = np.concatenate(torch_times_list)
        torch_trajectory = np.concatenate(torch_traj_list)

        torch_traj_data = {'times': torch_times, 'trajectory': torch_trajectory}

        # Stability analysis
        torch_analyzer = DosingStabilityAnalyzer(torch_times, torch_trajectory, event_times, compartment_idx=0)
        torch_stability_df = torch_analyzer.analyze_all_doses()
        torch_summary = torch_analyzer.compute_summary_metrics()

        print("\nTorchDiffEq Stability Metrics:")
        print(torch_stability_df)

    # TorchDyn
    if torchdyn_results:
        print("\nRunning TorchDyn for trajectory extraction...")
        from fit.pytorch_workflow.torchdyn_standardized import load_piecewise_data, train_model, TORCHDYN_AVAILABLE
        import torch

        if TORCHDYN_AVAILABLE:
            torchdyn_data = load_piecewise_data("data/pk_dataset_2C.npz")
            torchdyn_model, _, _ = train_model(
                torchdyn_data, list(event_times), list(event_doses), t_final,
                n_iters=500, lr=1e-3, seed=42
            )

            # Extract trajectory
            torchdyn_model.eval()
            with torch.no_grad():
                y0 = torchdyn_data[0, 0, 0, :]
                torchdyn_pred = torchdyn_model(y0, t_final, steps_per_segment).cpu().numpy()

            # Build time array and trajectory
            torchdyn_times_list = []
            torchdyn_traj_list = []
            for seg_idx in range(4):
                t_start = segment_boundaries[seg_idx]
                t_end = segment_boundaries[seg_idx+1]
                t_seg = np.linspace(t_start, t_end, steps_per_segment)
                torchdyn_times_list.append(t_seg)
                torchdyn_traj_list.append(torchdyn_pred[seg_idx])

            torchdyn_times = np.concatenate(torchdyn_times_list)
            torchdyn_trajectory = np.concatenate(torchdyn_traj_list)

            torchdyn_traj_data = {'times': torchdyn_times, 'trajectory': torchdyn_trajectory}

            # Stability analysis
            torchdyn_analyzer = DosingStabilityAnalyzer(torchdyn_times, torchdyn_trajectory, event_times, compartment_idx=0)
            torchdyn_stability_df = torchdyn_analyzer.analyze_all_doses()
            torchdyn_summary = torchdyn_analyzer.compute_summary_metrics()

            print("\nTorchDyn Stability Metrics:")
            print(torchdyn_stability_df)

    # ==================================================================
    # 3. Create dosing event visualizations
    # ==================================================================
    print("\n" + "="*80)
    print("PHASE 3: Generating Dosing Event Visualizations")
    print("="*80)

    # Side-by-side comparison at first dose (t=12h)
    if jax_results and torchdiffeq_results:
        if torchdyn_results and TORCHDYN_AVAILABLE:
            fig, metrics = create_side_by_side_comparison(
                jax_traj_data, torch_traj_data, torchdyn_traj_data,
                dose_time=12.0, window=3.0,
                output_path=f"{output_dir}/images/dosing_event_comparison.png"
            )
            plt.close(fig)
        else:
            # Just JAX and TorchDiffEq
            print("TorchDyn not available, creating comparison with JAX and TorchDiffEq only")

    # Individual zoom plots
    if jax_results:
        fig = create_dosing_zoom_plot(
            jax_times, jax_trajectory, event_times, dose_time=12.0,
            framework_name="JAX/Diffrax", window=3.0, compartment_idx=0
        )
        plt.savefig(f"{output_dir}/images/jax_dosing_zoom.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    if torchdiffeq_results:
        fig = create_dosing_zoom_plot(
            torch_times, torch_trajectory, event_times, dose_time=12.0,
            framework_name="TorchDiffEq", window=3.0, compartment_idx=0
        )
        plt.savefig(f"{output_dir}/images/torchdiffeq_dosing_zoom.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    if torchdyn_results and TORCHDYN_AVAILABLE:
        fig = create_dosing_zoom_plot(
            torchdyn_times, torchdyn_trajectory, event_times, dose_time=12.0,
            framework_name="TorchDyn", window=3.0, compartment_idx=0
        )
        plt.savefig(f"{output_dir}/images/torchdyn_dosing_zoom.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    # ==================================================================
    # 4. Create stability comparison table
    # ==================================================================
    print("\n" + "="*80)
    print("PHASE 4: Creating Stability Comparison Table")
    print("="*80)

    if jax_results and torchdiffeq_results:
        if torchdyn_results and TORCHDYN_AVAILABLE:
            stability_table = create_metrics_comparison_table(
                jax_summary, torch_summary, torchdyn_summary
            )
        else:
            # Create table with just JAX and TorchDiffEq
            stability_table = create_metrics_comparison_table(
                jax_summary, torch_summary, {}
            )

        print("\nStability Metrics Comparison:")
        print(stability_table)

        stability_table.to_csv(f"{output_dir}/stability_metrics.csv", index=False)
        print(f"\nSaved stability metrics to {output_dir}/stability_metrics.csv")

    # ==================================================================
    # 5. Generate blog post
    # ==================================================================
    print("\n" + "="*80)
    print("PHASE 5: Generating Blog Post")
    print("="*80)

    # Prepare results summary
    results_summary = {}
    if len(all_results) > 0:
        # Find fastest
        fastest_idx = np.argmin([r['mean_training_time'] for r in all_results])
        results_summary['fastest'] = all_results[fastest_idx]['framework']
        results_summary['fastest_time'] = f"{all_results[fastest_idx]['mean_training_time']:.1f}"

        # Find most memory efficient
        mem_efficient_idx = np.argmin([r['mean_peak_memory'] for r in all_results])
        results_summary['most_efficient'] = all_results[mem_efficient_idx]['framework']
        results_summary['lowest_memory'] = f"{all_results[mem_efficient_idx]['mean_peak_memory']:.0f}"

        # Find most stable (lowest overshoot)
        if jax_results and torchdiffeq_results:
            overshoots = [jax_summary.get('mean_overshoot_pct', 0), torch_summary.get('mean_overshoot_pct', 0)]
            if torchdyn_results and TORCHDYN_AVAILABLE:
                overshoots.append(torchdyn_summary.get('mean_overshoot_pct', 0))

            most_stable_idx = np.argmin(overshoots)
            framework_names = ['JAX/Diffrax', 'TorchDiffEq']
            if torchdyn_results and TORCHDYN_AVAILABLE:
                framework_names.append('TorchDyn')

            results_summary['most_stable'] = framework_names[most_stable_idx]
            results_summary['lowest_overshoot'] = f"{overshoots[most_stable_idx]:.2f}"

    # Generate blog post
    blog_gen = BlogPostGenerator(output_dir=output_dir)
    blog_file = blog_gen.generate_full_post(
        perf_table=perf_table,
        stability_table=stability_table if 'stability_table' in locals() else None,
        results_summary=results_summary
    )

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}/")
    print(f"\nBlog post: {blog_file}")
    print(f"\nImages directory: {output_dir}/images/")
    print("="*80)

    return blog_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run enhanced framework comparison for blog")
    parser.add_argument('--iters', type=int, default=2000, help='Training iterations')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs per framework')
    parser.add_argument('--output', type=str, default='results/blog_post', help='Output directory')

    args = parser.parse_args()

    run_enhanced_benchmark(n_iters=args.iters, n_runs=args.runs, output_dir=args.output)
