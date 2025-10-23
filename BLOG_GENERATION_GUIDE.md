# Professional Blog Post Generation Guide

This guide explains how to generate publication-quality blog content with comprehensive framework comparisons, including **stability analysis at dosing events**.

## What Gets Generated

Running the enhanced benchmark creates a complete blog post package:

### 📊 Performance Metrics
- Training time comparison (mean ± std)
- Memory usage analysis
- Accuracy metrics (MSE, RMSE)
- Per-iteration timing

### 🎯 Dosing Event Stability Analysis ⭐ NEW
- **Post-dose overshoot percentage**
- **Oscillation amplitude** after dosing
- **Numerical noise levels**
- **Time to peak concentration**
- **Convergence rate** to steady state

### 📈 High-Resolution Figures (300 DPI)
1. `training_time_comparison.png` - Bar chart with error bars
2. `memory_comparison.png` - Memory usage comparison
3. `accuracy_comparison.png` - RMSE comparison
4. `loss_curves.png` - Training convergence with confidence bands
5. `comparison_dashboard.png` - Complete summary
6. **`dosing_event_comparison.png`** ⭐ - Side-by-side dosing stability
7. **`jax_dosing_zoom.png`** ⭐ - JAX zoom at dosing event
8. **`torchdiffeq_dosing_zoom.png`** ⭐ - TorchDiffEq zoom
9. **`torchdyn_dosing_zoom.png`** ⭐ - TorchDyn zoom

### 📝 Markdown Blog Post
Complete, publication-ready blog post with:
- Executive summary
- Methodology section
- Results with embedded figures
- **Detailed dosing stability analysis** ⭐
- Framework recommendations
- Code examples
- Conclusion

### 📋 Data Tables (CSV)
- `performance_metrics.csv` - Timing and memory data
- `stability_metrics.csv` - Dosing event analysis ⭐

## Quick Start

### Option 1: Full Blog Generation (Recommended)

Generate everything needed for a professional blog post:

```bash
cd src/benchmarks
python run_blog_benchmark.py
```

**What this does:**
1. Runs all 3 frameworks (3 times each for statistics)
2. Analyzes dosing event stability
3. Creates all 9 publication-quality figures
4. Generates complete markdown blog post
5. Saves all data tables

**Expected runtime:** 15-40 minutes (depending on hardware)

**Output location:** `results/blog_post/`

### Option 2: Quick Test Run

For faster testing (fewer iterations, fewer runs):

```bash
python run_blog_benchmark.py --iters 500 --runs 1
```

**Runtime:** ~5-10 minutes

### Option 3: Custom Configuration

```bash
python run_blog_benchmark.py --iters 1000 --runs 2 --output my_blog_post
```

**Arguments:**
- `--iters`: Number of training iterations (default: 2000)
- `--runs`: Number of runs per framework (default: 3)
- `--output`: Output directory (default: results/blog_post)

## Understanding the Outputs

### Generated Directory Structure

```
results/blog_post/
├── framework_comparison_blog.md    # Complete blog post (MAIN OUTPUT)
├── performance_metrics.csv         # Performance data table
├── stability_metrics.csv           # Dosing stability data ⭐
└── images/                         # All figures (300 DPI)
    ├── training_time_comparison.png
    ├── memory_comparison.png
    ├── accuracy_comparison.png
    ├── loss_curves.png
    ├── comparison_dashboard.png
    ├── dosing_event_comparison.png  ⭐
    ├── jax_dosing_zoom.png          ⭐
    ├── torchdiffeq_dosing_zoom.png  ⭐
    └── torchdyn_dosing_zoom.png     ⭐
```

### The Blog Post

The generated `framework_comparison_blog.md` includes:

1. **Executive Summary**
   - Key findings
   - Fastest framework
   - Most memory efficient
   - Best stability at dosing events ⭐

2. **Methodology**
   - Problem setup
   - Standardized architecture
   - Evaluation metrics

3. **Results**
   - Performance comparison
   - **Dosing event stability analysis** ⭐
   - Accuracy comparison
   - Training dynamics

4. **Detailed Analysis**
   - Per-framework strengths/weaknesses
   - **Deep dive into dosing event handling** ⭐
   - Recommendations

5. **Code Examples**
   - How to reproduce
   - Quick start commands

6. **Conclusion**
   - Summary and takeaways
   - Citation information

## Dosing Event Stability Metrics

The enhanced benchmark measures critical stability metrics at each dosing event:

### What We Measure

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| **Jump Size** | Magnitude of concentration increase | Verifies dose is applied correctly |
| **Overshoot %** | How much solver overshoots peak | Lower = better numerical stability |
| **Oscillation Amplitude** | Post-dose numerical oscillations | Lower = smoother integration |
| **Time to Peak** | How fast solver finds peak | Solver responsiveness |
| **Noise Level** | High-frequency numerical artifacts | Solver quality indicator |
| **Consistency Score** | Variance across dose events | Solver reliability |

### Side-by-Side Comparison

The `dosing_event_comparison.png` shows all three frameworks handling the same dosing event:

- **Left panel**: JAX/Diffrax behavior
- **Middle panel**: TorchDiffEq behavior
- **Right panel**: TorchDyn behavior

Each panel includes metrics box showing:
- Jump size
- Overshoot percentage
- Time to peak
- Oscillation amplitude

### Individual Zoom Plots

Each framework gets a detailed 2-panel figure:
- **Top**: Full trajectory with highlighted zoom region
- **Bottom**: Zoomed view (±3 hours around dose) with stability metrics

## Using the Blog Content

### For Medium/Dev.to/Substack

1. Copy `framework_comparison_blog.md` content
2. Upload images from `images/` directory
3. Update image links to your hosting URLs
4. Publish!

### For GitHub Pages/Jekyll

1. Copy `framework_comparison_blog.md` to your `_posts/` directory
2. Rename to Jekyll format: `YYYY-MM-DD-framework-comparison.md`
3. Copy `images/` to your assets folder
4. Update image paths in markdown

### For LaTeX/Academic Paper

1. Convert markdown to LaTeX (pandoc recommended)
2. Use images at 300 DPI (already publication quality)
3. Reference CSV tables for precise numbers

## Customization

### Change Figure Style

Edit `run_blog_benchmark.py` to customize matplotlib settings:

```python
# At top of file
plt.rcParams['font.family'] = 'serif'  # Change to serif
plt.rcParams['figure.dpi'] = 600       # Higher resolution
plt.rcParams['font.size'] = 12         # Larger font
```

### Add More Dosing Events to Analysis

Edit dosing window in analysis:

```python
# In run_blog_benchmark.py
window = 5.0  # Increase from 3.0 to see more context
```

### Analyze Different Compartment

To analyze peripheral compartment instead of central:

```python
# In DosingStabilityAnalyzer calls
compartment_idx=1  # Instead of 0
```

### Custom Blog Sections

Edit `blog_generator.py` to add/modify sections:

```python
class BlogPostGenerator:
    def generate_custom_section(self):
        """Add your own section"""
        return """## My Custom Analysis

        Your content here...
        """
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'scipy'"

Scipy is required for stability analysis:
```bash
pip install scipy
```

### Figures look pixelated

Increase DPI in `run_blog_benchmark.py`:
```python
plt.rcParams['savefig.dpi'] = 600  # Higher quality
```

### Stability metrics show NaN

This can happen if trajectory doesn't have enough points. Increase `steps_per_segment`:
```python
steps_per_segment = 500  # Increase from 200
```

### Memory error during benchmark

Reduce number of iterations or runs:
```bash
python run_blog_benchmark.py --iters 1000 --runs 2
```

### TorchDyn not available

Install TorchDyn:
```bash
pip install torchdyn
```

If installation fails, the benchmark will run with JAX and TorchDiffEq only.

## Advanced Usage

### Generate Only Stability Analysis

If you already have trained models and just want stability analysis:

```python
from benchmarks.dosing_stability_analysis import DosingStabilityAnalyzer

# Your trajectory data
analyzer = DosingStabilityAnalyzer(times, trajectory, dosing_times)
metrics_df = analyzer.analyze_all_doses()
summary = analyzer.compute_summary_metrics()

print(metrics_df)
print(summary)
```

### Create Custom Comparison Plots

```python
from benchmarks.dosing_stability_analysis import create_side_by_side_comparison

fig, metrics = create_side_by_side_comparison(
    jax_data={'times': t1, 'trajectory': y1},
    torch_data={'times': t2, 'trajectory': y2},
    torchdyn_data={'times': t3, 'trajectory': y3},
    dose_time=12.0,
    window=5.0,  # Custom window
    output_path='my_comparison.png'
)
```

### Extract Specific Metrics

```python
from benchmarks.dosing_stability_analysis import DosingStabilityAnalyzer

analyzer = DosingStabilityAnalyzer(times, traj, [12.0, 24.0, 36.0])

# Analyze single dose
metrics_12h = analyzer.analyze_dose_event(12.0, window_before=2.0, window_after=5.0)
print(f"Overshoot at 12h: {metrics_12h['overshoot_pct']:.2f}%")

# Get all doses
all_metrics = analyzer.analyze_all_doses()
print(all_metrics[['dose_time', 'overshoot_pct', 'oscillation_amplitude']])
```

## Example Output

After running the full benchmark, you'll see console output like:

```
============================================================
ENHANCED FRAMEWORK COMPARISON FOR BLOG POST
============================================================

PHASE 1: Standard Performance Benchmarks
...

PHASE 2: Dosing Event Stability Analysis

JAX Stability Metrics:
   dose_time  baseline  jump_size  peak_value  overshoot_pct  oscillation_amplitude
0       12.0      0.23     99.85      100.45           0.60              0.0023
1       24.0     45.67     99.92      100.12           0.20              0.0015
2       36.0     23.45     99.78      100.05           0.27              0.0018

PHASE 3: Generating Dosing Event Visualizations
Saved comparison to results/blog_post/images/dosing_event_comparison.png
...

PHASE 5: Generating Blog Post
Blog post generated: results/blog_post/framework_comparison_blog.md

============================================================
BENCHMARK COMPLETE!
============================================================

All outputs saved to: results/blog_post/
Blog post: results/blog_post/framework_comparison_blog.md
Images directory: results/blog_post/images/
```

## Best Practices

1. **Use full runs (n_runs=3) for final blog post** - Gives statistical significance
2. **Test with quick run first** (`--iters 500 --runs 1`) - Verify everything works
3. **Keep output directory organized** - One directory per blog post
4. **Version control your figures** - Git LFS for images
5. **Document any customizations** - Note changes in blog post

## Next Steps

After generating your blog post:

1. ✅ Review `framework_comparison_blog.md`
2. ✅ Check all figures in `images/` directory
3. ✅ Verify metrics in CSV tables
4. ✅ Customize sections as needed
5. ✅ Add your own insights
6. ✅ Publish to your platform!

## Contributing

Have ideas for additional stability metrics or visualizations?

Open an issue or PR at: `github.com/omidbazgirTTU/PK_event_fit`

---

**Pro Tip:** The dosing event stability analysis is what sets this benchmark apart! Emphasize those results in your blog post.

