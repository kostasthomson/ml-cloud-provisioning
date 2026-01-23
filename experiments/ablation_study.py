#!/usr/bin/env python3
"""
Ablation Study for Reward Component Analysis.

Systematically removes or modifies reward components to validate
that each design decision contributes to model performance.

Usage:
    python ablation_study.py [--configs full,no_energy,no_sla]
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.environment import CloudProvisioningEnv
from rl.agent import RLAgent
from rl.trainer import PPOTrainer
from rl.reward import RewardCalculator
from rl.schemas import RLTrainingConfig
from experiments.config import ExperimentConfig, setup_experiment_logging

logger = None


@dataclass
class AblationResult:
    """Results from an ablation configuration."""
    config_name: str
    energy_weight: float
    sla_weight: float
    rejection_penalty: float
    avg_energy_per_task: float
    acceptance_rate: float
    sla_compliance_rate: float
    avg_reward: float
    total_energy: float
    total_accepted: int
    total_tasks: int
    training_time_sec: float
    model_path: str


def train_ablation_config(
    config_name: str,
    reward_config: Dict[str, float],
    experiment_config: ExperimentConfig,
    save_dir: Path,
    seed: int = 42
) -> str:
    """Train a model with specific ablation configuration."""
    logger.info(f"Training ablation config '{config_name}'...")
    logger.info(f"  energy_weight={reward_config['energy_weight']}, "
                f"sla_weight={reward_config['sla_weight']}, "
                f"rejection_penalty={reward_config['rejection_penalty']}")

    np.random.seed(seed)

    env = CloudProvisioningEnv(
        preset=experiment_config.env_preset,
        episode_length=experiment_config.episode_length,
        max_steps=experiment_config.training_timesteps,
        seed=seed,
        exec_time_noise=experiment_config.exec_time_noise,
        energy_noise=experiment_config.energy_noise
    )

    env.reward_calculator = RewardCalculator(
        energy_weight=reward_config['energy_weight'],
        sla_weight=reward_config['sla_weight'],
        rejection_penalty=reward_config['rejection_penalty']
    )

    agent = RLAgent(device="auto", embed_dim=64)

    training_config = RLTrainingConfig(
        learning_rate=experiment_config.learning_rate,
        gamma=experiment_config.gamma,
        batch_size=experiment_config.batch_size,
        n_epochs=experiment_config.ppo_epochs,
        clip_range=experiment_config.clip_range,
        total_timesteps=experiment_config.training_timesteps
    )

    trainer = PPOTrainer(agent, training_config)

    start_time = datetime.now()
    trainer.train(env, total_timesteps=experiment_config.training_timesteps)
    training_time = (datetime.now() - start_time).total_seconds()

    model_path = save_dir / f"model_ablation_{config_name}.pth"
    agent.save(str(model_path))

    return str(model_path), training_time


def evaluate_ablation_config(
    model_path: str,
    config_name: str,
    reward_config: Dict[str, float],
    experiment_config: ExperimentConfig,
    training_time: float
) -> AblationResult:
    """Evaluate a trained ablation model."""
    logger.info(f"Evaluating ablation config '{config_name}'...")

    agent = RLAgent(model_path=model_path)

    env = CloudProvisioningEnv(
        preset=experiment_config.env_preset,
        episode_length=experiment_config.episode_length,
        max_steps=experiment_config.episode_length,
        exec_time_noise=experiment_config.exec_time_noise,
        energy_noise=experiment_config.energy_noise
    )

    total_energy = 0.0
    total_accepted = 0
    total_tasks = 0
    total_sla_violations = 0
    total_reward = 0.0

    for episode in range(experiment_config.evaluation_episodes):
        state, _ = env.reset(seed=episode)
        done = False

        while not done:
            action_result, _, _ = agent.predict(state, deterministic=True)
            action = action_result.action

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            total_tasks += 1

            if info.get('accepted', False):
                total_accepted += 1
                total_energy += info.get('energy', 0.0)

                if state.task.deadline is not None:
                    exec_time = info.get('exec_time', 0.0)
                    if exec_time > state.task.deadline:
                        total_sla_violations += 1

            state = next_state

    return AblationResult(
        config_name=config_name,
        energy_weight=reward_config['energy_weight'],
        sla_weight=reward_config['sla_weight'],
        rejection_penalty=reward_config['rejection_penalty'],
        avg_energy_per_task=total_energy / max(total_accepted, 1),
        acceptance_rate=total_accepted / max(total_tasks, 1),
        sla_compliance_rate=1.0 - (total_sla_violations / max(total_accepted, 1)),
        avg_reward=total_reward / experiment_config.evaluation_episodes,
        total_energy=total_energy,
        total_accepted=total_accepted,
        total_tasks=total_tasks,
        training_time_sec=training_time,
        model_path=model_path
    )


def compute_ablation_impact(
    full_result: AblationResult,
    ablation_result: AblationResult
) -> Dict[str, float]:
    """Compute the impact of removing a component."""
    return {
        "energy_change_pct": (
            (ablation_result.avg_energy_per_task - full_result.avg_energy_per_task)
            / max(full_result.avg_energy_per_task, 1e-9) * 100
        ),
        "acceptance_change_pct": (
            (ablation_result.acceptance_rate - full_result.acceptance_rate)
            / max(full_result.acceptance_rate, 1e-9) * 100
        ),
        "sla_change_pct": (
            (ablation_result.sla_compliance_rate - full_result.sla_compliance_rate)
            / max(full_result.sla_compliance_rate, 1e-9) * 100
        ),
        "reward_change_pct": (
            (ablation_result.avg_reward - full_result.avg_reward)
            / max(abs(full_result.avg_reward), 1e-9) * 100
        )
    }


def run_ablation_study(
    config: ExperimentConfig,
    ablation_configs: Optional[Dict[str, Dict]] = None
) -> Dict[str, Any]:
    """Run complete ablation study."""
    global logger
    if logger is None:
        logger = setup_experiment_logging(config, "ablation_study")

    configs = ablation_configs or config.ablation_configs
    logger.info(f"Running ablation study with {len(configs)} configurations")

    save_dir = config.results_dir / "models" / "ablation"
    save_dir.mkdir(parents=True, exist_ok=True)

    ablation_results = []
    for config_name, reward_config in configs.items():
        model_path, training_time = train_ablation_config(
            config_name, reward_config, config, save_dir
        )
        result = evaluate_ablation_config(
            model_path, config_name, reward_config, config, training_time
        )
        ablation_results.append(result)
        logger.info(
            f"  {config_name}: energy={result.avg_energy_per_task:.6f}, "
            f"acceptance={result.acceptance_rate:.2%}, "
            f"SLA={result.sla_compliance_rate:.2%}"
        )

    full_result = next((r for r in ablation_results if r.config_name == "full"), ablation_results[0])

    impact_analysis = {}
    for result in ablation_results:
        if result.config_name != "full":
            impact_analysis[result.config_name] = compute_ablation_impact(full_result, result)

    results = {
        "experiment": "ablation_study",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "ablation_configs": configs,
            "timesteps": config.training_timesteps,
            "evaluation_episodes": config.evaluation_episodes,
            "env_preset": config.env_preset
        },
        "results": [asdict(r) for r in ablation_results],
        "impact_analysis": impact_analysis,
        "baseline_config": "full"
    }

    output_path = config.results_dir / "data" / "ablation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    generate_ablation_csv(ablation_results, config.results_dir / "data" / "ablation_data.csv")

    print_ablation_summary(ablation_results, impact_analysis)

    return results


def generate_ablation_csv(results: List[AblationResult], output_path: Path):
    """Generate CSV file for ablation data."""
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'config_name', 'energy_weight', 'sla_weight', 'rejection_penalty',
            'avg_energy_per_task', 'acceptance_rate', 'sla_compliance_rate',
            'avg_reward', 'total_energy', 'total_accepted', 'total_tasks'
        ])
        for r in results:
            writer.writerow([
                r.config_name, r.energy_weight, r.sla_weight, r.rejection_penalty,
                r.avg_energy_per_task, r.acceptance_rate, r.sla_compliance_rate,
                r.avg_reward, r.total_energy, r.total_accepted, r.total_tasks
            ])
    logger.info(f"Ablation CSV saved to {output_path}")


def print_ablation_summary(
    results: List[AblationResult],
    impact: Dict[str, Dict[str, float]]
):
    """Print formatted summary of ablation study."""
    print("\n" + "=" * 100)
    print("ABLATION STUDY RESULTS")
    print("=" * 100)

    print(f"\n{'Configuration':<25} {'Energy/Task':<15} {'Acceptance':<15} {'SLA Compliance':<15} {'Avg Reward'}")
    print("-" * 100)

    for result in sorted(results, key=lambda r: r.config_name):
        print(
            f"{result.config_name:<25} "
            f"{result.avg_energy_per_task:<15.6f} "
            f"{result.acceptance_rate:<15.2%} "
            f"{result.sla_compliance_rate:<15.2%} "
            f"{result.avg_reward:<15.2f}"
        )

    print("\n" + "=" * 100)
    print("IMPACT ANALYSIS (% change from 'full' configuration)")
    print("=" * 100)
    print(f"\n{'Configuration':<25} {'Energy Δ%':<15} {'Acceptance Δ%':<15} {'SLA Δ%':<15} {'Reward Δ%'}")
    print("-" * 100)

    for config_name, changes in impact.items():
        print(
            f"{config_name:<25} "
            f"{changes['energy_change_pct']:>+14.1f} "
            f"{changes['acceptance_change_pct']:>+14.1f} "
            f"{changes['sla_change_pct']:>+14.1f} "
            f"{changes['reward_change_pct']:>+14.1f}"
        )

    print("=" * 100)
    print("\nInterpretation:")
    print("  - Positive energy change = WORSE (higher energy consumption)")
    print("  - Negative acceptance change = WORSE (fewer tasks accepted)")
    print("  - Negative SLA change = WORSE (more SLA violations)")
    print("  - Negative reward change = WORSE (lower overall performance)")
    print("=" * 100)


def main():
    global logger
    parser = argparse.ArgumentParser(description='Ablation study')
    parser.add_argument('--configs', type=str, default=None,
                        help='Comma-separated config names to test')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Training timesteps per config')
    parser.add_argument('--eval-episodes', type=int, default=None,
                        help='Evaluation episodes')
    args = parser.parse_args()

    config = ExperimentConfig()
    logger = setup_experiment_logging(config, "ablation_study")

    if args.timesteps:
        config.training_timesteps = args.timesteps
    if args.eval_episodes:
        config.evaluation_episodes = args.eval_episodes

    ablation_configs = None
    if args.configs:
        config_names = [c.strip() for c in args.configs.split(',')]
        ablation_configs = {
            name: config.ablation_configs[name]
            for name in config_names
            if name in config.ablation_configs
        }

    run_ablation_study(config, ablation_configs=ablation_configs)


if __name__ == "__main__":
    main()
