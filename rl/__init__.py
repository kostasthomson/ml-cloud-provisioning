"""
Reinforcement Learning module for energy-efficient resource provisioning.

This module provides:
- RLAgent: PPO-based agent for allocation decisions
- RLEnvironment: Gymnasium environment for training
- REST API endpoints for system-agnostic inference and training
"""

from .schemas import (
    RLState,
    RLAction,
    RLExperience,
    RLPredictionRequest,
    RLPredictionResponse,
    RLTrainingConfig,
    RLTrainingStatus,
    RLModelInfo,
    TaskState,
    HWTypeState,
    GlobalState,
    TaskOutcome,
    ExperienceBatch,
)
from .agent import RLAgent
from .state_encoder import StateEncoder
from .reward import RewardCalculator
from .environment import CloudProvisioningEnv
from .trainer import PPOTrainer
from .distributed_trainer import (
    DistributedPPOTrainer,
    VectorizedEnv,
    run_distributed_training,
)

__all__ = [
    "RLState",
    "RLAction",
    "RLExperience",
    "RLPredictionRequest",
    "RLPredictionResponse",
    "RLTrainingConfig",
    "RLTrainingStatus",
    "RLModelInfo",
    "TaskState",
    "HWTypeState",
    "GlobalState",
    "TaskOutcome",
    "ExperienceBatch",
    "RLAgent",
    "StateEncoder",
    "RewardCalculator",
    "CloudProvisioningEnv",
    "PPOTrainer",
    "DistributedPPOTrainer",
    "VectorizedEnv",
    "run_distributed_training",
]
