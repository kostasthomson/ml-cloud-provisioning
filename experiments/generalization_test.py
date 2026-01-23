#!/usr/bin/env python3
"""
Generalization Experiment for Infrastructure-Agnostic Validation.

Tests whether a model trained on one hardware configuration can generalize
to different configurations without retraining. This validates the key
contribution of the infrastructure-agnostic architecture.

Usage:
    python generalization_test.py [--train-preset medium] [--test-presets small,large,enterprise]
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

from rl.environment import CloudProvisioningEnv, REALISTIC_HW_CONFIGS
from rl.agent import RLAgent
from rl.trainer import PPOTrainer
from rl.schemas import RLTrainingConfig
from experiments.config import ExperimentConfig, setup_experiment_logging

logger = None


@dataclass
class GeneralizationResult:
    """Results from testing on a specific HW configuration."""
    train_preset: str
    test_preset: str
    num_hw_types_train: int
    num_hw_types_test: int
    avg_energy_per_task: float
    acceptance_rate: float
    sla_compliance_rate: float
    avg_reward: float
    total_energy: float
    total_accepted: int
    total_tasks: int
    is_same_config: bool


def train_generalization_model(
    train_preset: str,
    config: ExperimentConfig,
    save_dir: Path,
    seed: int = 42
) -> str:
    """Train a model on the training preset."""
    logger.info(f"Training model on '{train_preset}' preset...")

    hw_configs = REALISTIC_HW_CONFIGS[train_preset]
    logger.info(f"  Training HW types: {len(hw_configs)}")
    for hw in hw_configs:
        logger.info(f"    - {hw.name} (ID: {hw.hw_type_id})")

    np.random.seed(seed)

    env = CloudProvisioningEnv(
        preset=train_preset,
        episode_length=config.episode_length,
        max_steps=config.training_timesteps,
        seed=seed,
        exec_time_noise=config.exec_time_noise,
        energy_noise=config.energy_noise
    )

    agent = RLAgent(device="auto", embed_dim=64)

    training_config = RLTrainingConfig(
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        batch_size=config.batch_size,
        n_epochs=config.ppo_epochs,
        clip_range=config.clip_range,
        total_timesteps=config.training_timesteps
    )

    trainer = PPOTrainer(agent, training_config)
    trainer.train(env, total_timesteps=config.training_timesteps)

    model_path = save_dir / f"model_trained_on_{train_preset}.pth"
    agent.save(str(model_path))

    return str(model_path)


def evaluate_on_preset(
    model_path: str,
    train_preset: str,
    test_preset: str,
    config: ExperimentConfig
) -> GeneralizationResult:
    """Evaluate trained model on a specific test preset."""
    logger.info(f"Evaluating on '{test_preset}' preset...")

    agent = RLAgent(model_path=model_path)

    train_hw_count = len(REALISTIC_HW_CONFIGS[train_preset])
    test_hw_count = len(REALISTIC_HW_CONFIGS[test_preset])

    logger.info(f"  Test HW types: {test_hw_count}")
    for hw in REALISTIC_HW_CONFIGS[test_preset]:
        logger.info(f"    - {hw.name} (ID: {hw.hw_type_id})")

    env = CloudProvisioningEnv(
        preset=test_preset,
        episode_length=config.episode_length,
        max_steps=config.episode_length,
        exec_time_noise=config.exec_time_noise,
        energy_noise=config.energy_noise
    )

    total_energy = 0.0
    total_accepted = 0
    total_tasks = 0
    total_sla_violations = 0
    total_reward = 0.0

    for episode in range(config.evaluation_episodes):
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

    return GeneralizationResult(
        train_preset=train_preset,
        test_preset=test_preset,
        num_hw_types_train=train_hw_count,
        num_hw_types_test=test_hw_count,
        avg_energy_per_task=total_energy / max(total_accepted, 1),
        acceptance_rate=total_accepted / max(total_tasks, 1),
        sla_compliance_rate=1.0 - (total_sla_violations / max(total_accepted, 1)),
        avg_reward=total_reward / config.evaluation_episodes,
        total_energy=total_energy,
        total_accepted=total_accepted,
        total_tasks=total_tasks,
        is_same_config=(train_preset == test_preset)
    )


def compute_generalization_gap(
    same_config_result: GeneralizationResult,
    cross_config_result: GeneralizationResult
) -> Dict[str, float]:
    """Compute the performance gap when generalizing."""
    return {
        "energy_gap_pct": (
            (cross_config_result.avg_energy_per_task - same_config_result.avg_energy_per_task)
            / max(same_config_result.avg_energy_per_task, 1e-9) * 100
        ),
        "acceptance_gap_pct": (
            (cross_config_result.acceptance_rate - same_config_result.acceptance_rate)
            / max(same_config_result.acceptance_rate, 1e-9) * 100
        ),
        "sla_gap_pct": (
            (cross_config_result.sla_compliance_rate - same_config_result.sla_compliance_rate)
            / max(same_config_result.sla_compliance_rate, 1e-9) * 100
        ),
        "reward_gap_pct": (
            (cross_config_result.avg_reward - same_config_result.avg_reward)
            / max(abs(same_config_result.avg_reward), 1e-9) * 100
        )
    }


def run_generalization_experiment(
    config: ExperimentConfig,
    train_preset: str = None,
    test_presets: List[str] = None
) -> Dict[str, Any]:
    """Run complete generalization experiment."""
    global logger
    if logger is None:
        logger = setup_experiment_logging(config, "generalization_test")

    train_preset = train_preset or config.generalization_train_preset
    test_presets = test_presets or config.generalization_test_presets

    logger.info(f"Running generalization experiment")
    logger.info(f"  Training on: {train_preset}")
    logger.info(f"  Testing on: {test_presets}")

    save_dir = config.results_dir / "models" / "generalization"
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = train_generalization_model(train_preset, config, save_dir)

    results = []
    for test_preset in test_presets:
        result = evaluate_on_preset(model_path, train_preset, test_preset, config)
        results.append(result)
        logger.info(
            f"  {test_preset}: energy={result.avg_energy_per_task:.6f}, "
            f"acceptance={result.acceptance_rate:.2%}, "
            f"SLA={result.sla_compliance_rate:.2%}"
        )

    same_config_result = next((r for r in results if r.is_same_config), results[0])
    generalization_gaps = {}
    for result in results:
        if not result.is_same_config:
            generalization_gaps[result.test_preset] = compute_generalization_gap(
                same_config_result, result
            )

    avg_gaps = {
        "avg_energy_gap_pct": np.mean([g["energy_gap_pct"] for g in generalization_gaps.values()]),
        "avg_acceptance_gap_pct": np.mean([g["acceptance_gap_pct"] for g in generalization_gaps.values()]),
        "avg_sla_gap_pct": np.mean([g["sla_gap_pct"] for g in generalization_gaps.values()]),
        "avg_reward_gap_pct": np.mean([g["reward_gap_pct"] for g in generalization_gaps.values()])
    }

    output = {
        "experiment": "generalization_test",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "train_preset": train_preset,
            "test_presets": test_presets,
            "timesteps": config.training_timesteps,
            "evaluation_episodes": config.evaluation_episodes
        },
        "hw_type_counts": {
            preset: len(REALISTIC_HW_CONFIGS[preset])
            for preset in [train_preset] + test_presets
        },
        "results": [asdict(r) for r in results],
        "generalization_gaps": generalization_gaps,
        "average_gaps": avg_gaps,
        "model_path": model_path
    }

    output_path = config.results_dir / "data" / "generalization_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    generate_generalization_csv(results, config.results_dir / "data" / "generalization_data.csv")

    print_generalization_summary(results, generalization_gaps, avg_gaps, train_preset)

    return output


def generate_generalization_csv(results: List[GeneralizationResult], output_path: Path):
    """Generate CSV file for generalization data."""
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'train_preset', 'test_preset', 'num_hw_types_train', 'num_hw_types_test',
            'avg_energy_per_task', 'acceptance_rate', 'sla_compliance_rate',
            'avg_reward', 'is_same_config'
        ])
        for r in results:
            writer.writerow([
                r.train_preset, r.test_preset, r.num_hw_types_train, r.num_hw_types_test,
                r.avg_energy_per_task, r.acceptance_rate, r.sla_compliance_rate,
                r.avg_reward, r.is_same_config
            ])
    logger.info(f"Generalization CSV saved to {output_path}")


def print_generalization_summary(
    results: List[GeneralizationResult],
    gaps: Dict[str, Dict[str, float]],
    avg_gaps: Dict[str, float],
    train_preset: str
):
    """Print formatted summary of generalization experiment."""
    print("\n" + "=" * 100)
    print("GENERALIZATION EXPERIMENT RESULTS")
    print("=" * 100)
    print(f"\nModel trained on: {train_preset}")
    print(f"Training HW types: {results[0].num_hw_types_train}")

    print(f"\n{'Test Preset':<15} {'HW Types':<12} {'Energy/Task':<15} {'Acceptance':<15} {'SLA Compliance':<15} {'Same?'}")
    print("-" * 100)

    for result in results:
        same_str = "Yes" if result.is_same_config else "No"
        print(
            f"{result.test_preset:<15} "
            f"{result.num_hw_types_test:<12} "
            f"{result.avg_energy_per_task:<15.6f} "
            f"{result.acceptance_rate:<15.2%} "
            f"{result.sla_compliance_rate:<15.2%} "
            f"{same_str}"
        )

    print("\n" + "=" * 100)
    print("GENERALIZATION GAP ANALYSIS (% change from same-config baseline)")
    print("=" * 100)
    print(f"\n{'Test Preset':<15} {'Energy Δ%':<15} {'Acceptance Δ%':<15} {'SLA Δ%':<15} {'Reward Δ%'}")
    print("-" * 100)

    for preset, gap in gaps.items():
        print(
            f"{preset:<15} "
            f"{gap['energy_gap_pct']:>+14.1f} "
            f"{gap['acceptance_gap_pct']:>+14.1f} "
            f"{gap['sla_gap_pct']:>+14.1f} "
            f"{gap['reward_gap_pct']:>+14.1f}"
        )

    print("-" * 100)
    print(
        f"{'AVERAGE':<15} "
        f"{avg_gaps['avg_energy_gap_pct']:>+14.1f} "
        f"{avg_gaps['avg_acceptance_gap_pct']:>+14.1f} "
        f"{avg_gaps['avg_sla_gap_pct']:>+14.1f} "
        f"{avg_gaps['avg_reward_gap_pct']:>+14.1f}"
    )

    print("=" * 100)
    print("\nInterpretation:")
    print("  - Small gaps (< ±10%) indicate GOOD generalization")
    print("  - Large gaps (> ±25%) indicate POOR generalization")
    print("  - The model handles different HW configurations:" if abs(avg_gaps['avg_reward_gap_pct']) < 15
          else "  - The model struggles with different HW configurations")
    print("=" * 100)


def main():
    global logger
    parser = argparse.ArgumentParser(description='Generalization experiment')
    parser.add_argument('--train-preset', type=str, default=None,
                        help='Training environment preset')
    parser.add_argument('--test-presets', type=str, default=None,
                        help='Comma-separated test presets')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Training timesteps')
    parser.add_argument('--eval-episodes', type=int, default=None,
                        help='Evaluation episodes')
    args = parser.parse_args()

    config = ExperimentConfig()
    logger = setup_experiment_logging(config, "generalization_test")

    if args.timesteps:
        config.training_timesteps = args.timesteps
    if args.eval_episodes:
        config.evaluation_episodes = args.eval_episodes

    train_preset = args.train_preset
    test_presets = None
    if args.test_presets:
        test_presets = [p.strip() for p in args.test_presets.split(',')]

    run_generalization_experiment(config, train_preset=train_preset, test_presets=test_presets)


if __name__ == "__main__":
    main()
