#!/usr/bin/env python3
"""
Master Orchestration Script for Academic Evaluation.

Runs all experiments and generates a comprehensive results report.
This is the single command entry point for producing publication-ready results.

Usage:
    python run_all_experiments.py [--quick] [--full] [--skip-training]

Examples:
    # Quick run (fewer seeds, shorter training) for testing
    python run_all_experiments.py --quick

    # Full academic run (all experiments, full configuration)
    python run_all_experiments.py --full

    # Skip training, only evaluate existing models
    python run_all_experiments.py --skip-training
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config import ExperimentConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates all experiments and generates reports."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {}
        self.start_time = None
        self.end_time = None

    def run_all(
        self,
        run_multi_seed: bool = True,
        run_pareto: bool = True,
        run_ablation: bool = True,
        run_generalization: bool = True,
        run_baseline_comparison: bool = True
    ) -> Dict[str, Any]:
        """Run all experiments."""
        self.start_time = datetime.now()
        logger.info("=" * 70)
        logger.info("STARTING ACADEMIC EVALUATION SUITE")
        logger.info("=" * 70)
        logger.info(f"Timestamp: {self.start_time.isoformat()}")
        logger.info(f"Results directory: {self.config.results_dir}")
        logger.info("=" * 70)

        experiment_order = [
            ("multi_seed", run_multi_seed, self._run_multi_seed),
            ("pareto", run_pareto, self._run_pareto),
            ("ablation", run_ablation, self._run_ablation),
            ("generalization", run_generalization, self._run_generalization),
            ("baseline_comparison", run_baseline_comparison, self._run_baseline_comparison),
        ]

        total_experiments = sum(1 for _, enabled, _ in experiment_order if enabled)
        completed = 0

        for name, enabled, runner in experiment_order:
            if enabled:
                completed += 1
                logger.info(f"\n[{completed}/{total_experiments}] Running {name} experiment...")
                try:
                    exp_start = time.time()
                    self.results[name] = runner()
                    exp_duration = time.time() - exp_start
                    self.results[name]["duration_sec"] = exp_duration
                    logger.info(f"  Completed in {exp_duration:.1f}s")
                except Exception as e:
                    logger.error(f"  FAILED: {e}")
                    self.results[name] = {"error": str(e)}

        self.end_time = datetime.now()
        total_duration = (self.end_time - self.start_time).total_seconds()

        self._generate_summary_report()

        logger.info("\n" + "=" * 70)
        logger.info("ACADEMIC EVALUATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
        logger.info(f"Results saved to: {self.config.results_dir}")
        logger.info("=" * 70)

        return self.results

    def _run_multi_seed(self) -> Dict[str, Any]:
        """Run multi-seed training experiment."""
        from experiments.multi_seed_training import run_multi_seed_experiment
        return run_multi_seed_experiment(self.config, num_seeds=self.config.num_seeds)

    def _run_pareto(self) -> Dict[str, Any]:
        """Run Pareto front analysis."""
        from experiments.pareto_analysis import run_pareto_analysis
        return run_pareto_analysis(self.config)

    def _run_ablation(self) -> Dict[str, Any]:
        """Run ablation study."""
        from experiments.ablation_study import run_ablation_study
        return run_ablation_study(self.config)

    def _run_generalization(self) -> Dict[str, Any]:
        """Run generalization experiment."""
        from experiments.generalization_test import run_generalization_experiment
        return run_generalization_experiment(self.config)

    def _run_baseline_comparison(self) -> Dict[str, Any]:
        """Run baseline comparison using existing benchmark script."""
        logger.info("  Running baseline comparison (PPO vs Scoring vs Random)...")

        from rl.environment import CloudProvisioningEnv
        from rl.agent import RLAgent
        from entities.allocator.scoring_allocator import ScoringAllocator
        from scripts.benchmark_performance import (
            run_performance_study, run_statistical_analysis,
            adapt_rl_state_to_scoring_request, PerformanceMetrics
        )
        import random

        model_dir = self.config.results_dir / "models" / "multi_seed"
        model_files = list(model_dir.glob("model_seed_*.pth"))
        if model_files:
            model_path = str(model_files[0])
        else:
            model_path = None

        env = CloudProvisioningEnv(
            preset=self.config.env_preset,
            episode_length=self.config.episode_length,
            exec_time_noise=self.config.exec_time_noise,
            energy_noise=self.config.energy_noise
        )

        ppo_agent = RLAgent(model_path=model_path) if model_path else RLAgent()
        scoring_allocator = ScoringAllocator()

        logger.info("    Evaluating PPO Agent...")
        ppo_metrics = run_performance_study(
            env, "ppo", model=ppo_agent, num_episodes=self.config.evaluation_episodes
        )

        logger.info("    Evaluating Scoring Allocator...")
        scoring_metrics = run_performance_study(
            env, "scoring", allocator=scoring_allocator, num_episodes=self.config.evaluation_episodes
        )

        logger.info("    Evaluating Random Strategy...")
        random_metrics = run_performance_study(
            env, "random", num_episodes=self.config.evaluation_episodes
        )

        stats = run_statistical_analysis(ppo_metrics, scoring_metrics)

        return {
            "ppo": {
                "total_energy_kwh": ppo_metrics.total_energy_kwh,
                "acceptance_rate": ppo_metrics.acceptance_rate,
                "sla_compliance_rate": ppo_metrics.sla_compliance_rate,
                "efficiency_index": ppo_metrics.efficiency_index
            },
            "scoring": {
                "total_energy_kwh": scoring_metrics.total_energy_kwh,
                "acceptance_rate": scoring_metrics.acceptance_rate,
                "sla_compliance_rate": scoring_metrics.sla_compliance_rate,
                "efficiency_index": scoring_metrics.efficiency_index
            },
            "random": {
                "total_energy_kwh": random_metrics.total_energy_kwh,
                "acceptance_rate": random_metrics.acceptance_rate,
                "sla_compliance_rate": random_metrics.sla_compliance_rate,
                "efficiency_index": random_metrics.efficiency_index
            },
            "statistical_analysis": stats
        }

    def _generate_summary_report(self):
        """Generate comprehensive summary report."""
        report = {
            "experiment_suite": "RL Cloud Provisioning Academic Evaluation",
            "timestamp_start": self.start_time.isoformat(),
            "timestamp_end": self.end_time.isoformat(),
            "duration_sec": (self.end_time - self.start_time).total_seconds(),
            "configuration": {
                "training_timesteps": self.config.training_timesteps,
                "evaluation_episodes": self.config.evaluation_episodes,
                "num_seeds": self.config.num_seeds,
                "env_preset": self.config.env_preset,
                "exec_time_noise": self.config.exec_time_noise,
                "energy_noise": self.config.energy_noise
            },
            "experiments": self.results,
            "summary": self._compute_summary()
        }

        report_path = self.config.results_dir / "academic_evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Full report saved to: {report_path}")

        self._generate_latex_tables()
        self._print_summary_to_console()

    def _compute_summary(self) -> Dict[str, Any]:
        """Compute high-level summary statistics."""
        summary = {}

        if "multi_seed" in self.results and "statistics" in self.results["multi_seed"]:
            stats = self.results["multi_seed"]["statistics"]
            summary["multi_seed"] = {
                "energy_per_task_mean": stats["energy_per_task"]["mean"],
                "energy_per_task_std": stats["energy_per_task"]["std"],
                "acceptance_rate_mean": stats["acceptance_rate"]["mean"],
                "sla_compliance_mean": stats["sla_compliance_rate"]["mean"]
            }

        if "generalization" in self.results and "average_gaps" in self.results["generalization"]:
            gaps = self.results["generalization"]["average_gaps"]
            summary["generalization"] = {
                "avg_energy_gap_pct": gaps["avg_energy_gap_pct"],
                "avg_acceptance_gap_pct": gaps["avg_acceptance_gap_pct"],
                "generalizes_well": abs(gaps["avg_reward_gap_pct"]) < 15
            }

        if "baseline_comparison" in self.results:
            bc = self.results["baseline_comparison"]
            if "ppo" in bc and "scoring" in bc:
                summary["vs_baseline"] = {
                    "ppo_energy": bc["ppo"]["total_energy_kwh"],
                    "scoring_energy": bc["scoring"]["total_energy_kwh"],
                    "energy_improvement_pct": (
                        (bc["scoring"]["total_energy_kwh"] - bc["ppo"]["total_energy_kwh"])
                        / bc["scoring"]["total_energy_kwh"] * 100
                    ) if bc["scoring"]["total_energy_kwh"] > 0 else 0
                }

        return summary

    def _generate_latex_tables(self):
        """Generate LaTeX tables for the paper."""
        latex_dir = self.config.results_dir / "latex"
        latex_dir.mkdir(exist_ok=True)

        if "multi_seed" in self.results and "statistics" in self.results["multi_seed"]:
            stats = self.results["multi_seed"]["statistics"]
            latex = r"""
\begin{table}[h]
\centering
\caption{Multi-Seed Training Results (n=%d seeds)}
\label{tab:multi_seed}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Mean} & \textbf{Std} \\
\midrule
Energy per Task (kWh) & %.6f & %.6f \\
Acceptance Rate & %.2f\%% & %.2f\%% \\
SLA Compliance & %.2f\%% & %.2f\%% \\
Average Reward & %.2f & %.2f \\
\bottomrule
\end{tabular}
\end{table}
""" % (
                stats["n_seeds"],
                stats["energy_per_task"]["mean"], stats["energy_per_task"]["std"],
                stats["acceptance_rate"]["mean"] * 100, stats["acceptance_rate"]["std"] * 100,
                stats["sla_compliance_rate"]["mean"] * 100, stats["sla_compliance_rate"]["std"] * 100,
                stats["avg_reward"]["mean"], stats["avg_reward"]["std"]
            )
            with open(latex_dir / "table_multi_seed.tex", 'w') as f:
                f.write(latex)

        if "baseline_comparison" in self.results:
            bc = self.results["baseline_comparison"]
            if all(k in bc for k in ["ppo", "scoring", "random"]):
                latex = r"""
\begin{table}[h]
\centering
\caption{Baseline Comparison Results}
\label{tab:baseline}
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{PPO Agent} & \textbf{Scoring} & \textbf{Random} \\
\midrule
Total Energy (kWh) & %.4f & %.4f & %.4f \\
Acceptance Rate & %.2f\%% & %.2f\%% & %.2f\%% \\
SLA Compliance & %.2f\%% & %.2f\%% & %.2f\%% \\
Efficiency (kWh/task) & %.6f & %.6f & %.6f \\
\bottomrule
\end{tabular}
\end{table}
""" % (
                    bc["ppo"]["total_energy_kwh"], bc["scoring"]["total_energy_kwh"], bc["random"]["total_energy_kwh"],
                    bc["ppo"]["acceptance_rate"] * 100, bc["scoring"]["acceptance_rate"] * 100, bc["random"]["acceptance_rate"] * 100,
                    bc["ppo"]["sla_compliance_rate"] * 100, bc["scoring"]["sla_compliance_rate"] * 100, bc["random"]["sla_compliance_rate"] * 100,
                    bc["ppo"]["efficiency_index"], bc["scoring"]["efficiency_index"], bc["random"]["efficiency_index"]
                )
                with open(latex_dir / "table_baseline.tex", 'w') as f:
                    f.write(latex)

        logger.info(f"LaTeX tables saved to: {latex_dir}")

    def _print_summary_to_console(self):
        """Print formatted summary to console."""
        print("\n" + "=" * 80)
        print("ACADEMIC EVALUATION SUMMARY")
        print("=" * 80)

        summary = self._compute_summary()

        if "multi_seed" in summary:
            ms = summary["multi_seed"]
            print(f"\n[Multi-Seed Training]")
            print(f"  Energy/Task: {ms['energy_per_task_mean']:.6f} ± {ms['energy_per_task_std']:.6f} kWh")
            print(f"  Acceptance:  {ms['acceptance_rate_mean']:.2%}")
            print(f"  SLA Compliance: {ms['sla_compliance_mean']:.2%}")

        if "generalization" in summary:
            gen = summary["generalization"]
            status = "GOOD" if gen["generalizes_well"] else "NEEDS IMPROVEMENT"
            print(f"\n[Generalization]")
            print(f"  Avg Energy Gap: {gen['avg_energy_gap_pct']:+.1f}%")
            print(f"  Avg Acceptance Gap: {gen['avg_acceptance_gap_pct']:+.1f}%")
            print(f"  Status: {status}")

        if "vs_baseline" in summary:
            vs = summary["vs_baseline"]
            print(f"\n[vs Scoring Baseline]")
            print(f"  PPO Energy: {vs['ppo_energy']:.4f} kWh")
            print(f"  Scoring Energy: {vs['scoring_energy']:.4f} kWh")
            print(f"  Improvement: {vs['energy_improvement_pct']:+.1f}%")

        print("\n" + "=" * 80)
        print(f"Full results: {self.config.results_dir}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Run all academic evaluation experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Quick test run:
    python run_all_experiments.py --quick

  Full academic evaluation:
    python run_all_experiments.py --full

  Custom configuration:
    python run_all_experiments.py --seeds 5 --timesteps 100000 --episodes 30
        """
    )

    parser.add_argument('--quick', action='store_true',
                        help='Quick run with reduced settings for testing')
    parser.add_argument('--full', action='store_true',
                        help='Full academic run with all experiments')
    parser.add_argument('--seeds', type=int, default=None,
                        help='Number of seeds for multi-seed training')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Training timesteps per experiment')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Evaluation episodes')
    parser.add_argument('--preset', type=str, default=None,
                        help='Environment preset')
    parser.add_argument('--noise', type=float, default=None,
                        help='Execution time noise factor')

    parser.add_argument('--skip-multi-seed', action='store_true',
                        help='Skip multi-seed training')
    parser.add_argument('--skip-pareto', action='store_true',
                        help='Skip Pareto analysis')
    parser.add_argument('--skip-ablation', action='store_true',
                        help='Skip ablation study')
    parser.add_argument('--skip-generalization', action='store_true',
                        help='Skip generalization test')
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Skip baseline comparison')

    args = parser.parse_args()

    config = ExperimentConfig()

    if args.quick:
        config.num_seeds = 3
        config.training_timesteps = 50000
        config.evaluation_episodes = 20
        config.pareto_energy_weights = [0.6, 0.8, 0.95]
        logger.info("Running in QUICK mode (reduced settings)")

    if args.full:
        config.num_seeds = 10
        config.training_timesteps = 200000
        config.evaluation_episodes = 50
        logger.info("Running in FULL mode (complete academic evaluation)")

    if args.seeds:
        config.num_seeds = args.seeds
    if args.timesteps:
        config.training_timesteps = args.timesteps
    if args.episodes:
        config.evaluation_episodes = args.episodes
    if args.preset:
        config.env_preset = args.preset
    if args.noise:
        config.exec_time_noise = args.noise
        config.energy_noise = args.noise * 0.7

    runner = ExperimentRunner(config)
    runner.run_all(
        run_multi_seed=not args.skip_multi_seed,
        run_pareto=not args.skip_pareto,
        run_ablation=not args.skip_ablation,
        run_generalization=not args.skip_generalization,
        run_baseline_comparison=not args.skip_baseline
    )


if __name__ == "__main__":
    main()
