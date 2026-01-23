# Academic Evaluation Framework

This directory contains the comprehensive evaluation framework for producing publication-ready results from the RL-based cloud resource allocation system.

## Quick Start

### Run All Experiments (Single Command)

```bash
# Full academic evaluation (recommended for paper)
python experiments/run_all_experiments.py --full

# Quick test run (for development/testing)
python experiments/run_all_experiments.py --quick

# Custom configuration
python experiments/run_all_experiments.py --seeds 5 --timesteps 100000 --episodes 30
```

### Run Individual Experiments

```bash
# Multi-seed training (statistical validity)
python experiments/multi_seed_training.py --seeds 10 --timesteps 200000

# Pareto front analysis (energy-acceptance tradeoff)
python experiments/pareto_analysis.py --weights 0.5,0.6,0.7,0.8,0.9,0.95

# Ablation study (component contribution)
python experiments/ablation_study.py

# Generalization test (infrastructure-agnostic validation)
python experiments/generalization_test.py --train-preset medium --test-presets small,large,enterprise

# Generate publication figures
python experiments/generate_plots.py
```

## Experiments Overview

### 1. Multi-Seed Training (`multi_seed_training.py`)

**Purpose**: Establish statistical validity by training with multiple random seeds.

**Output**:
- Mean ± standard deviation for all metrics
- 95% confidence intervals
- Individual seed results for reproducibility

**Key Metrics**:
- Energy per task (kWh)
- Acceptance rate (%)
- SLA compliance rate (%)
- Average episode reward

### 2. Pareto Front Analysis (`pareto_analysis.py`)

**Purpose**: Map the energy-acceptance tradeoff curve by varying reward weights.

**Output**:
- Pareto front visualization
- Optimal operating points
- Tradeoff analysis

**Methodology**:
- Train separate models with energy_weight ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 0.95}
- Identify Pareto-optimal configurations
- Plot the tradeoff frontier

### 3. Ablation Study (`ablation_study.py`)

**Purpose**: Validate that each reward component contributes to performance.

**Configurations**:
| Name | Energy Weight | SLA Weight | Rejection Penalty |
|------|--------------|------------|-------------------|
| full | 0.8 | 0.15 | 0.3 |
| no_energy | 0.0 | 0.5 | 0.3 |
| no_sla | 0.8 | 0.0 | 0.3 |
| no_rejection_penalty | 0.8 | 0.15 | 0.0 |
| energy_only | 1.0 | 0.0 | 0.0 |

**Output**:
- Performance comparison table
- Impact analysis (% change from baseline)

### 4. Generalization Test (`generalization_test.py`)

**Purpose**: Validate the infrastructure-agnostic architecture claim.

**Methodology**:
1. Train on one HW configuration (e.g., `medium` with 3 HW types)
2. Test on different configurations without retraining
3. Measure performance gap

**HW Configurations**:
| Preset | HW Types | Description |
|--------|----------|-------------|
| small | 2 | CPU-Standard, CPU-HighMem |
| medium | 3 | CPU-Standard, GPU-V100, CPU-HighMem |
| large | 4 | CPU-Standard, GPU-V100, GPU-A100, CPU-HighMem |
| enterprise | 6 | All types including DFE, MIC |

### 5. Baseline Comparison

**Purpose**: Compare RL agent against alternative strategies.

**Baselines**:
- **PPO Agent**: Trained RL policy
- **Scoring Allocator**: Multi-objective weighted sum heuristic
- **Random**: Random valid action selection

**Statistical Analysis**:
- Welch's t-test for significance
- Cohen's d for effect size
- 95% confidence intervals

## Output Structure

```
results/academic/
├── data/
│   ├── multi_seed_results.json
│   ├── pareto_results.json
│   ├── pareto_data.csv
│   ├── ablation_results.json
│   ├── ablation_data.csv
│   ├── generalization_results.json
│   └── generalization_data.csv
├── figures/
│   ├── pareto_front.png
│   ├── pareto_front.pdf
│   ├── ablation_study.png
│   ├── ablation_study.pdf
│   ├── generalization.png
│   ├── generalization.pdf
│   ├── multi_seed_distribution.png
│   ├── multi_seed_distribution.pdf
│   ├── baseline_comparison.png
│   └── baseline_comparison.pdf
├── latex/
│   ├── table_multi_seed.tex
│   └── table_baseline.tex
├── models/
│   ├── multi_seed/
│   ├── pareto/
│   ├── ablation/
│   └── generalization/
├── logs/
└── academic_evaluation_report.json
```

## Configuration

Edit `experiments/config.py` to customize:

```python
@dataclass
class ExperimentConfig:
    # Seeds for statistical validity
    seeds: List[int] = [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]
    num_seeds: int = 10

    # Training parameters
    training_timesteps: int = 200000
    evaluation_episodes: int = 50
    episode_length: int = 100

    # Environment settings
    env_preset: str = "medium"
    exec_time_noise: float = 0.15  # ±15% execution time variance
    energy_noise: float = 0.10    # ±10% energy variance

    # PPO hyperparameters
    learning_rate: float = 3e-4
    batch_size: int = 64
    # ... etc
```

## Environment Stochastic Noise

The environment now supports configurable stochastic noise for improved realism:

```python
env = CloudProvisioningEnv(
    preset='medium',
    exec_time_noise=0.15,  # ±15% execution time variance
    energy_noise=0.10      # ±10% energy estimation variance
)
```

This forces the agent to learn robust policies that handle real-world variance.

## Interpreting Results

### Statistical Significance

Results include p-values from Welch's t-test:
- `***` p < 0.001 (highly significant)
- `**` p < 0.01 (very significant)
- `*` p < 0.05 (significant)
- `ns` not significant

### Effect Size (Cohen's d)

| |d| | Interpretation |
|-----|----------------|
| < 0.2 | Negligible |
| 0.2-0.5 | Small |
| 0.5-0.8 | Medium |
| > 0.8 | Large |

### Generalization Gap

| Gap | Interpretation |
|-----|----------------|
| < ±10% | Excellent generalization |
| ±10-25% | Acceptable generalization |
| > ±25% | Poor generalization |

## Paper Integration

### LaTeX Tables

Pre-generated LaTeX tables are saved to `results/academic/latex/`:

```latex
\input{results/academic/latex/table_multi_seed.tex}
\input{results/academic/latex/table_baseline.tex}
```

### Figures

Publication-ready figures (PNG 300dpi + PDF) are in `results/academic/figures/`.

### Reproducibility

The `academic_evaluation_report.json` contains all parameters and results for full reproducibility.

## Recommended Workflow

1. **Development**: Use `--quick` mode for fast iteration
2. **Validation**: Run individual experiments to verify setup
3. **Production**: Use `--full` for final paper results
4. **Figures**: Run `generate_plots.py` after experiments complete

## Troubleshooting

### Out of Memory

Reduce parallel environments or batch size:
```bash
python experiments/run_all_experiments.py --full --batch-size 32
```

### Slow Training

Use GPU and reduce timesteps for testing:
```bash
python experiments/run_all_experiments.py --quick --timesteps 50000
```

### Missing matplotlib

Install visualization dependencies:
```bash
pip install matplotlib seaborn
```
