# Blog Post Directory

This directory contains the **demonstration structure** for the professional blog post that will be generated when you run the full benchmark.

## Current Status

✅ **Demo blog post created**: `framework_comparison_blog.md`
⏳ **Full benchmark not yet run** (requires PyTorch installation)

## What's Here

```
blog_post/
├── README.md (this file)
├── framework_comparison_blog.md  ← DEMO BLOG POST
└── images/ (empty - will be populated when benchmark runs)
```

## What Will Be Generated

When you run the full benchmark, this directory will contain:

### Complete Blog Post
- `framework_comparison_blog.md` - 5000+ word professional blog post

### Data Tables (CSV)
- `performance_metrics.csv` - Training time, memory, accuracy
- `stability_metrics.csv` - Dosing event stability analysis

### High-Resolution Figures (300 DPI)
All in `images/` directory:
1. `training_time_comparison.png` - Bar chart with error bars
2. `memory_comparison.png` - Memory usage comparison
3. `accuracy_comparison.png` - RMSE comparison
4. `loss_curves.png` - Training convergence
5. `comparison_dashboard.png` - Complete summary
6. **`dosing_event_comparison.png`** - Side-by-side stability analysis
7. **`jax_dosing_zoom.png`** - JAX detailed zoom
8. **`torchdiffeq_dosing_zoom.png`** - TorchDiffEq detailed zoom
9. **`torchdyn_dosing_zoom.png`** - TorchDyn detailed zoom

## Running the Full Benchmark

### Prerequisites

Install all dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- JAX, Diffrax, Optax (for JAX implementation)
- PyTorch, TorchDiffEq, TorchDyn (for PyTorch implementations)
- Matplotlib, Pandas, SciPy, Psutil (for analysis)

### Run Full Benchmark

From repository root:
```bash
cd src/benchmarks
python run_blog_benchmark.py
```

**Options:**
```bash
# Quick test (5-10 minutes)
python run_blog_benchmark.py --iters 500 --runs 1

# Full benchmark (20-30 minutes)
python run_blog_benchmark.py --iters 2000 --runs 3

# Custom output location
python run_blog_benchmark.py --output ../../results/my_blog
```

### What the Benchmark Does

1. **Phase 1**: Standard performance benchmarks
   - Trains each framework 3 times (2000 iterations each)
   - Measures time, memory, accuracy

2. **Phase 2**: Dosing event stability analysis
   - Re-trains models for trajectory extraction
   - Analyzes overshoot, oscillations, noise at each dose

3. **Phase 3**: Generate visualizations
   - Creates 9 high-resolution figures (300 DPI)
   - Side-by-side comparisons, zoom plots

4. **Phase 4**: Create stability tables
   - CSV tables with all metrics

5. **Phase 5**: Generate blog post
   - Complete markdown with embedded figures
   - Ready to publish

**Expected Runtime:** 20-30 minutes for full benchmark

## Current Demo Blog Post

The `framework_comparison_blog.md` file shows:
- Complete structure and sections
- Sample metrics (realistic values based on typical performance)
- All text content
- Placeholders for figures (figure paths included)

**To see what the blog will look like:**
```bash
# View in terminal
cat framework_comparison_blog.md

# Or open in your favorite markdown editor
code framework_comparison_blog.md
```

## Installation Issues?

If you encounter issues installing dependencies:

### Option 1: Use Conda
```bash
conda create -n node_comparison python=3.9
conda activate node_comparison
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install jax diffrax optax torchdiffeq torchdyn matplotlib pandas scipy psutil
```

### Option 2: CPU-only PyTorch
If GPU/CUDA issues occur:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torchdiffeq torchdyn
```

### Option 3: Run on Your Local Machine
Clone the repository and run on a machine with more resources:
```bash
git clone https://github.com/omidbazgirTTU/PK_event_fit
cd PK_event_fit
pip install -r requirements.txt
cd src/benchmarks
python run_blog_benchmark.py
```

## Next Steps

1. **Review demo blog post**: `cat framework_comparison_blog.md`
2. **Install dependencies**: See above
3. **Run benchmark**: `python src/benchmarks/run_blog_benchmark.py`
4. **Customize**: Edit generated blog post as needed
5. **Publish**: Upload to Medium, Dev.to, or your blog

## Questions?

See the comprehensive guide: `../../BLOG_GENERATION_GUIDE.md`

---

**Note:** The demo blog post shows realistic example metrics. Your actual results will vary based on hardware, but the overall structure and analysis approach will be the same.
