"""
Dosing Event Stability Analysis for Neural ODEs
Analyzes numerical stability, oscillations, and solver behavior at discontinuous dosing events
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from scipy.signal import find_peaks


class DosingStabilityAnalyzer:
    """
    Analyzes stability metrics around dosing events for Neural ODE solvers.

    Key Metrics:
    - Post-dose oscillation amplitude
    - Convergence rate after dosing
    - Overshoot percentage
    - Time to steady state
    - Numerical noise level
    """

    def __init__(self, times, trajectory, dosing_times, compartment_idx=0):
        """
        Args:
            times: Time points (array)
            trajectory: State trajectory (array of shape [time_points, n_states])
            dosing_times: Times when doses are administered
            compartment_idx: Which compartment to analyze (0=central, 1=peripheral)
        """
        self.times = np.array(times)
        self.trajectory = np.array(trajectory)
        self.dosing_times = np.array(dosing_times)
        self.compartment_idx = compartment_idx

        # Extract compartment data
        if len(self.trajectory.shape) > 1:
            self.concentration = self.trajectory[:, compartment_idx]
        else:
            self.concentration = self.trajectory

    def analyze_dose_event(self, dose_time, window_before=2.0, window_after=5.0):
        """
        Analyze stability around a single dosing event.

        Args:
            dose_time: Time of dose administration
            window_before: Time window before dose (hours)
            window_after: Time window after dose (hours)

        Returns:
            dict with stability metrics
        """
        # Find indices for analysis window
        idx_before = np.where((self.times >= dose_time - window_before) &
                             (self.times < dose_time))[0]
        idx_after = np.where((self.times > dose_time) &
                            (self.times <= dose_time + window_after))[0]

        if len(idx_before) == 0 or len(idx_after) == 0:
            return None

        # Pre-dose baseline
        baseline = np.mean(self.concentration[idx_before[-10:]])  # Last 10 points before dose

        # Post-dose trajectory
        post_dose_conc = self.concentration[idx_after]
        post_dose_times = self.times[idx_after]

        # Peak concentration after dose
        peak_idx = np.argmax(post_dose_conc)
        peak_value = post_dose_conc[peak_idx]
        time_to_peak = post_dose_times[peak_idx] - dose_time

        # Immediate post-dose value (for jump size)
        immediate_post_dose = post_dose_conc[0]
        jump_size = immediate_post_dose - baseline

        # Overshoot analysis (oscillations after peak)
        if peak_idx < len(post_dose_conc) - 10:
            # Look for oscillations after peak
            post_peak_conc = post_dose_conc[peak_idx:]

            # Detrend (remove exponential decay)
            # Fit exponential: y = a * exp(-b*t) + c
            from scipy.optimize import curve_fit
            def exp_decay(t, a, b, c):
                return a * np.exp(-b * t) + c

            post_peak_times_rel = post_dose_times[peak_idx:] - post_dose_times[peak_idx]
            try:
                popt, _ = curve_fit(exp_decay, post_peak_times_rel, post_peak_conc,
                                   p0=[peak_value, 0.1, 0], maxfev=5000)
                trend = exp_decay(post_peak_times_rel, *popt)
                residuals = post_peak_conc - trend
                oscillation_amplitude = np.std(residuals)
                max_deviation = np.max(np.abs(residuals))
            except:
                oscillation_amplitude = np.std(post_peak_conc)
                max_deviation = 0
        else:
            oscillation_amplitude = 0
            max_deviation = 0

        # Overshoot percentage
        overshoot_pct = (peak_value - immediate_post_dose) / immediate_post_dose * 100 if immediate_post_dose > 0 else 0

        # Numerical noise (high-frequency fluctuations)
        # Compute second derivative to detect oscillations
        if len(post_dose_conc) > 5:
            diffs = np.diff(post_dose_conc)
            second_diffs = np.diff(diffs)
            noise_level = np.std(second_diffs)
        else:
            noise_level = 0

        # Time to 95% of steady state (approximation)
        if len(post_dose_conc) > 10:
            # Assume last 10% of window is near steady state
            approx_steady_state = np.mean(post_dose_conc[-int(len(post_dose_conc)*0.1):])
            threshold = 0.95 * approx_steady_state

            # Find when concentration drops below threshold
            settling_idx = np.where(post_dose_conc <= threshold)[0]
            if len(settling_idx) > 0:
                time_to_settle = post_dose_times[settling_idx[0]] - dose_time
            else:
                time_to_settle = window_after  # Didn't settle in window
        else:
            time_to_settle = None

        return {
            'dose_time': dose_time,
            'baseline': baseline,
            'jump_size': jump_size,
            'peak_value': peak_value,
            'time_to_peak': time_to_peak,
            'overshoot_pct': overshoot_pct,
            'oscillation_amplitude': oscillation_amplitude,
            'max_deviation': max_deviation,
            'noise_level': noise_level,
            'time_to_settle': time_to_settle,
        }

    def analyze_all_doses(self):
        """
        Analyze stability for all dosing events.

        Returns:
            DataFrame with metrics for each dose
        """
        results = []
        for dose_time in self.dosing_times:
            metrics = self.analyze_dose_event(dose_time)
            if metrics:
                results.append(metrics)

        return pd.DataFrame(results)

    def compute_summary_metrics(self):
        """
        Compute overall stability metrics across all doses.

        Returns:
            dict with summary statistics
        """
        df = self.analyze_all_doses()

        if len(df) == 0:
            return {}

        return {
            'mean_overshoot_pct': df['overshoot_pct'].mean(),
            'max_overshoot_pct': df['overshoot_pct'].max(),
            'mean_oscillation_amplitude': df['oscillation_amplitude'].mean(),
            'max_oscillation_amplitude': df['oscillation_amplitude'].max(),
            'mean_noise_level': df['noise_level'].mean(),
            'mean_time_to_peak': df['time_to_peak'].mean(),
            'consistency_score': 1.0 / (1.0 + df['overshoot_pct'].std()),  # Higher = more consistent
        }


def create_dosing_zoom_plot(times, trajectory, dosing_times, dose_time,
                            framework_name, window=3.0, compartment_idx=0):
    """
    Create a zoomed-in plot around a dosing event.

    Args:
        times: Time array
        trajectory: State trajectory
        dosing_times: All dosing times
        dose_time: Specific dose to zoom on
        framework_name: Name of framework (for title)
        window: Time window around dose (hours)
        compartment_idx: Which compartment to plot

    Returns:
        matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    times = np.array(times)
    trajectory = np.array(trajectory)

    if len(trajectory.shape) > 1:
        conc = trajectory[:, compartment_idx]
    else:
        conc = trajectory

    # Full trajectory
    ax1.plot(times, conc, 'b-', linewidth=2, label='Trajectory')
    for dt in dosing_times:
        ax1.axvline(dt, color='red', linestyle='--', alpha=0.5, linewidth=1.5)

    # Highlight zoom region
    ax1.axvspan(dose_time - window, dose_time + window, alpha=0.2, color='yellow')

    ax1.set_xlabel('Time (hours)', fontsize=11)
    ax1.set_ylabel('Amount (mg)', fontsize=11)
    ax1.set_title(f'{framework_name} - Full Trajectory', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Zoomed view
    idx_zoom = np.where((times >= dose_time - window) & (times <= dose_time + window))[0]

    if len(idx_zoom) > 0:
        zoom_times = times[idx_zoom]
        zoom_conc = conc[idx_zoom]

        ax2.plot(zoom_times, zoom_conc, 'b-', linewidth=2.5, marker='o',
                markersize=3, label='Concentration')
        ax2.axvline(dose_time, color='red', linestyle='--', linewidth=2,
                   label=f'Dose at t={dose_time}h')

        # Analyze stability
        analyzer = DosingStabilityAnalyzer(times, trajectory, dosing_times, compartment_idx)
        metrics = analyzer.analyze_dose_event(dose_time, window_before=window, window_after=window)

        if metrics:
            # Mark peak
            idx_after_dose = np.where(zoom_times > dose_time)[0]
            if len(idx_after_dose) > 0:
                post_dose_conc = zoom_conc[idx_after_dose]
                peak_idx = idx_after_dose[np.argmax(post_dose_conc)]
                ax2.plot(zoom_times[peak_idx], zoom_conc[peak_idx], 'r*',
                        markersize=15, label=f'Peak ({zoom_conc[peak_idx]:.1f} mg)')

            # Add metrics text
            metrics_text = (
                f"Jump: {metrics['jump_size']:.1f} mg\n"
                f"Overshoot: {metrics['overshoot_pct']:.2f}%\n"
                f"Time to peak: {metrics['time_to_peak']:.3f} h\n"
                f"Oscillation: {metrics['oscillation_amplitude']:.4f}"
            )
            ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='wheat', alpha=0.8), fontsize=9, family='monospace')

        ax2.set_xlabel('Time (hours)', fontsize=11)
        ax2.set_ylabel('Amount (mg)', fontsize=11)
        ax2.set_title(f'{framework_name} - Dosing Event Detail (±{window}h window)',
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        ax2.set_xlim(dose_time - window, dose_time + window)

    plt.tight_layout()
    return fig


def create_side_by_side_comparison(jax_data, torch_data, torchdyn_data, dose_time,
                                   window=3.0, output_path='dosing_comparison.png'):
    """
    Create side-by-side comparison of all three frameworks at a dosing event.

    Args:
        jax_data: dict with 'times' and 'trajectory' for JAX
        torch_data: dict with 'times' and 'trajectory' for TorchDiffEq
        torchdyn_data: dict with 'times' and 'trajectory' for TorchDyn
        dose_time: Which dose event to analyze
        window: Time window around dose
        output_path: Where to save figure

    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    frameworks = [
        ('JAX/Diffrax', jax_data, '#1f77b4'),
        ('TorchDiffEq', torch_data, '#ff7f0e'),
        ('TorchDyn', torchdyn_data, '#2ca02c')
    ]

    all_metrics = []

    for ax, (name, data, color) in zip(axes, frameworks):
        times = np.array(data['times'])
        traj = np.array(data['trajectory'])

        if len(traj.shape) > 1:
            conc = traj[:, 0]  # Central compartment
        else:
            conc = traj

        # Zoom to window
        idx_zoom = np.where((times >= dose_time - window) &
                           (times <= dose_time + window))[0]

        if len(idx_zoom) > 0:
            zoom_times = times[idx_zoom]
            zoom_conc = conc[idx_zoom]

            ax.plot(zoom_times, zoom_conc, color=color, linewidth=2.5,
                   marker='o', markersize=4, markevery=5)
            ax.axvline(dose_time, color='red', linestyle='--', linewidth=2, alpha=0.7)

            # Analyze
            analyzer = DosingStabilityAnalyzer(times, traj, [dose_time], 0)
            metrics = analyzer.analyze_dose_event(dose_time, window, window)

            if metrics:
                all_metrics.append((name, metrics))

                # Metrics box
                metrics_text = (
                    f"Jump: {metrics['jump_size']:.1f} mg\n"
                    f"Overshoot: {metrics['overshoot_pct']:.2f}%\n"
                    f"Peak time: {metrics['time_to_peak']:.3f}h\n"
                    f"Oscillation: {metrics['oscillation_amplitude']:.4f}"
                )
                ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round',
                       facecolor='white', alpha=0.9, edgecolor=color, linewidth=2),
                       fontsize=9, family='monospace')

            ax.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Central Compartment (mg)', fontsize=11, fontweight='bold')
            ax.set_title(f'{name}', fontsize=13, fontweight='bold', color=color)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(dose_time - window, dose_time + window)

    fig.suptitle(f'Dosing Event Stability Comparison at t={dose_time}h (±{window}h window)',
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison to {output_path}")

    return fig, all_metrics


def create_metrics_comparison_table(jax_metrics, torch_metrics, torchdyn_metrics):
    """
    Create a comparison table of stability metrics.

    Args:
        jax_metrics: Summary metrics from JAX
        torch_metrics: Summary metrics from TorchDiffEq
        torchdyn_metrics: Summary metrics from TorchDyn

    Returns:
        pandas DataFrame
    """
    data = {
        'Metric': [
            'Mean Overshoot (%)',
            'Max Overshoot (%)',
            'Mean Oscillation Amplitude',
            'Max Oscillation Amplitude',
            'Mean Noise Level',
            'Mean Time to Peak (h)',
            'Consistency Score'
        ],
        'JAX/Diffrax': [
            f"{jax_metrics.get('mean_overshoot_pct', 0):.3f}",
            f"{jax_metrics.get('max_overshoot_pct', 0):.3f}",
            f"{jax_metrics.get('mean_oscillation_amplitude', 0):.4f}",
            f"{jax_metrics.get('max_oscillation_amplitude', 0):.4f}",
            f"{jax_metrics.get('mean_noise_level', 0):.6f}",
            f"{jax_metrics.get('mean_time_to_peak', 0):.3f}",
            f"{jax_metrics.get('consistency_score', 0):.3f}"
        ],
        'TorchDiffEq': [
            f"{torch_metrics.get('mean_overshoot_pct', 0):.3f}",
            f"{torch_metrics.get('max_overshoot_pct', 0):.3f}",
            f"{torch_metrics.get('mean_oscillation_amplitude', 0):.4f}",
            f"{torch_metrics.get('max_oscillation_amplitude', 0):.4f}",
            f"{torch_metrics.get('mean_noise_level', 0):.6f}",
            f"{torch_metrics.get('mean_time_to_peak', 0):.3f}",
            f"{torch_metrics.get('consistency_score', 0):.3f}"
        ],
        'TorchDyn': [
            f"{torchdyn_metrics.get('mean_overshoot_pct', 0):.3f}",
            f"{torchdyn_metrics.get('max_overshoot_pct', 0):.3f}",
            f"{torchdyn_metrics.get('mean_oscillation_amplitude', 0):.4f}",
            f"{torchdyn_metrics.get('max_oscillation_amplitude', 0):.4f}",
            f"{torchdyn_metrics.get('mean_noise_level', 0):.6f}",
            f"{torchdyn_metrics.get('mean_time_to_peak', 0):.3f}",
            f"{torchdyn_metrics.get('consistency_score', 0):.3f}"
        ]
    }

    return pd.DataFrame(data)
