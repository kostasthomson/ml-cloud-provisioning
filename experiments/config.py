"""
Experiment Configuration for Academic Evaluation.

Centralized configuration for all experiments to ensure reproducibility.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Master configuration for all experiments."""

    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    results_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "results" / "academic")

    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066])
    num_seeds: int = 10

    training_timesteps: int = 200000
    evaluation_episodes: int = 50
    episode_length: int = 100

    env_preset: str = "medium"
    exec_time_noise: float = 0.15
    energy_noise: float = 0.10

    learning_rate: float = 3e-4
    batch_size: int = 64
    ppo_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01

    pareto_energy_weights: List[float] = field(
        default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    )

    ablation_configs: Dict[str, Dict] = field(default_factory=lambda: {
        "full": {"energy_weight": 0.8, "sla_weight": 0.15, "rejection_penalty": 0.3},
        "no_energy": {"energy_weight": 0.0, "sla_weight": 0.5, "rejection_penalty": 0.3},
        "no_sla": {"energy_weight": 0.8, "sla_weight": 0.0, "rejection_penalty": 0.3},
        "no_rejection_penalty": {"energy_weight": 0.8, "sla_weight": 0.15, "rejection_penalty": 0.0},
        "energy_only": {"energy_weight": 1.0, "sla_weight": 0.0, "rejection_penalty": 0.0},
    })

    generalization_train_preset: str = "medium"
    generalization_test_presets: List[str] = field(
        default_factory=lambda: ["small", "medium", "large", "enterprise"]
    )

    def __post_init__(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "models").mkdir(exist_ok=True)
        (self.results_dir / "logs").mkdir(exist_ok=True)
        (self.results_dir / "figures").mkdir(exist_ok=True)
        (self.results_dir / "data").mkdir(exist_ok=True)


DEFAULT_CONFIG = ExperimentConfig()
