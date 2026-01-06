"""
Infrastructure-agnostic PPO Agent for energy-efficient resource allocation.

This module provides an RL agent that works with any number of hardware types.
The architecture uses per-HW-type scoring with shared weights, allowing the
same model to work with different infrastructure configurations.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict
from pathlib import Path
import logging
import time

from .schemas import RLState, RLAction, RLTrainingConfig, RLModelInfo
from .state_encoder import StateEncoder
from .reward import RewardCalculator

logger = logging.getLogger(__name__)


class TaskEncoder(nn.Module):
    """Encodes task + global features into an embedding."""

    def __init__(self, input_dim: int, embed_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class HWEncoder(nn.Module):
    """Encodes single HW type features into an embedding."""

    def __init__(self, input_dim: int, embed_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Scorer(nn.Module):
    """Scores a task-HW pair for allocation suitability."""

    def __init__(self, task_dim: int, hw_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(task_dim + hw_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, task_emb: torch.Tensor, hw_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([task_emb, hw_emb], dim=-1)
        return self.network(combined).squeeze(-1)


class PolicyNetwork(nn.Module):
    """
    Infrastructure-agnostic Actor-Critic network.

    Uses per-HW-type scoring with shared weights:
    - TaskEncoder: task features → task embedding
    - HWEncoder: HW features → HW embedding (shared for all types)
    - Scorer: (task_emb, hw_emb) → allocation score (shared for all types)
    - RejectHead: task_emb → reject score
    - ValueHead: task_emb + aggregated HW → state value
    """

    def __init__(
        self,
        task_dim: int = 17,
        hw_dim: int = 16,
        embed_dim: int = 64
    ):
        super().__init__()
        self.task_dim = task_dim
        self.hw_dim = hw_dim
        self.embed_dim = embed_dim

        self.task_encoder = TaskEncoder(task_dim, embed_dim)
        self.hw_encoder = HWEncoder(hw_dim, embed_dim)
        self.scorer = Scorer(embed_dim, embed_dim)

        self.reject_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.value_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        task_vec: torch.Tensor,
        hw_vecs: List[torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for single sample.

        Args:
            task_vec: Task features (task_dim,)
            hw_vecs: List of HW feature tensors, each (hw_dim,)
            valid_mask: Optional mask of valid HW types (num_hw,)

        Returns:
            Tuple of (action_probs, value, scores)
            - action_probs: (num_hw + 1,) - probs for each HW type + reject
            - value: scalar state value
            - scores: (num_hw + 1,) - raw scores before softmax
        """
        task_emb = self.task_encoder(task_vec.unsqueeze(0)).squeeze(0)

        hw_scores = []
        hw_embs = []
        for hw_vec in hw_vecs:
            hw_emb = self.hw_encoder(hw_vec.unsqueeze(0)).squeeze(0)
            hw_embs.append(hw_emb)
            score = self.scorer(task_emb.unsqueeze(0), hw_emb.unsqueeze(0)).squeeze()
            hw_scores.append(score)

        if hw_embs:
            hw_embs_stacked = torch.stack(hw_embs)
            mean_hw_emb = hw_embs_stacked.mean(dim=0)
        else:
            mean_hw_emb = torch.zeros(self.embed_dim, device=task_vec.device)

        reject_score = self.reject_head(task_emb.unsqueeze(0)).squeeze()

        if hw_scores:
            hw_scores_tensor = torch.stack(hw_scores)
        else:
            hw_scores_tensor = torch.tensor([], device=task_vec.device)

        all_scores = torch.cat([hw_scores_tensor, reject_score.unsqueeze(0)])

        if valid_mask is not None:
            full_mask = torch.cat([valid_mask, torch.tensor([True], device=valid_mask.device)])
            all_scores = all_scores.masked_fill(~full_mask, float('-inf'))

        action_probs = torch.softmax(all_scores, dim=0)

        value_input = torch.cat([task_emb, mean_hw_emb])
        value = self.value_head(value_input.unsqueeze(0)).squeeze()

        return action_probs, value, all_scores

    def forward_batch(
        self,
        task_vecs: torch.Tensor,
        hw_vecs_batch: List[List[torch.Tensor]],
        valid_masks: Optional[List[torch.Tensor]] = None
    ) -> Tuple[List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        """
        Batched forward pass.

        Args:
            task_vecs: (batch, task_dim)
            hw_vecs_batch: List of List of hw tensors
            valid_masks: Optional list of masks

        Returns:
            Lists of action_probs and scores, tensor of values
        """
        batch_size = task_vecs.shape[0]
        all_probs = []
        all_values = []
        all_scores = []

        for i in range(batch_size):
            mask = valid_masks[i] if valid_masks else None
            probs, value, scores = self.forward(task_vecs[i], hw_vecs_batch[i], mask)
            all_probs.append(probs)
            all_values.append(value)
            all_scores.append(scores)

        return all_probs, torch.stack(all_values), all_scores


class RLAgent:
    """
    Infrastructure-agnostic PPO-based RL Agent for resource allocation.

    Works with any number of HW types by scoring each type independently.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        embed_dim: int = 64
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.encoder = StateEncoder()
        self.reward_calculator = RewardCalculator()

        self.task_dim = self.encoder.task_dim
        self.hw_dim = self.encoder.hw_dim
        self.embed_dim = embed_dim

        self.policy = PolicyNetwork(
            task_dim=self.task_dim,
            hw_dim=self.hw_dim,
            embed_dim=embed_dim
        ).to(self.device)

        self.is_trained = False
        self.training_timesteps = 0
        self.model_path = model_path
        self.last_training_reward: Optional[float] = None
        self.training_config: Optional[RLTrainingConfig] = None

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def predict(
        self,
        state: RLState,
        deterministic: bool = True
    ) -> Tuple[RLAction, Optional[float], float]:
        """
        Predict action for given state.

        Args:
            state: Current RLState with any number of HW types
            deterministic: If True, select argmax action; if False, sample

        Returns:
            Tuple of (RLAction, state_value, inference_time_ms)
        """
        start_time = time.perf_counter()

        task_vec, hw_list = self.encoder.encode(state)
        valid_hw_types = self.encoder.get_valid_hw_types(state)

        hw_type_ids = [hw_id for hw_id, _ in hw_list]
        hw_vecs = [torch.FloatTensor(hw_vec).to(self.device) for _, hw_vec in hw_list]
        task_tensor = torch.FloatTensor(task_vec).to(self.device)

        valid_mask = torch.tensor(
            [hw_id in valid_hw_types for hw_id in hw_type_ids],
            dtype=torch.bool,
            device=self.device
        )

        with torch.no_grad():
            action_probs, value, _ = self.policy.forward(task_tensor, hw_vecs, valid_mask)

        probs = action_probs.cpu().numpy()
        num_hw = len(hw_type_ids)

        if deterministic:
            action_idx = int(np.argmax(probs))
        else:
            action_idx = int(np.random.choice(len(probs), p=probs))

        if action_idx < num_hw:
            selected_hw_id = hw_type_ids[action_idx]
        else:
            selected_hw_id = None

        inference_time_ms = (time.perf_counter() - start_time) * 1000

        probs_dict = {hw_type_ids[i]: float(probs[i]) for i in range(num_hw)}
        probs_dict[-1] = float(probs[-1])

        action = RLAction.from_hw_type(
            hw_type_id=selected_hw_id,
            confidence=float(probs[action_idx]),
            probs=probs_dict
        )
        state_value = float(value.cpu().numpy())

        return action, state_value, inference_time_ms

    def get_action_probs(self, state: RLState) -> Dict[int, float]:
        """Get action probability distribution for state."""
        task_vec, hw_list = self.encoder.encode(state)
        valid_hw_types = self.encoder.get_valid_hw_types(state)

        hw_type_ids = [hw_id for hw_id, _ in hw_list]
        hw_vecs = [torch.FloatTensor(hw_vec).to(self.device) for _, hw_vec in hw_list]
        task_tensor = torch.FloatTensor(task_vec).to(self.device)

        valid_mask = torch.tensor(
            [hw_id in valid_hw_types for hw_id in hw_type_ids],
            dtype=torch.bool,
            device=self.device
        )

        with torch.no_grad():
            action_probs, _, _ = self.policy.forward(task_tensor, hw_vecs, valid_mask)

        probs = action_probs.cpu().numpy()
        probs_dict = {hw_type_ids[i]: float(probs[i]) for i in range(len(hw_type_ids))}
        probs_dict[-1] = float(probs[-1])

        return probs_dict

    def save(self, path: str):
        """Save model to file."""
        save_dict = {
            'policy_state_dict': self.policy.state_dict(),
            'is_trained': self.is_trained,
            'training_timesteps': self.training_timesteps,
            'task_dim': self.task_dim,
            'hw_dim': self.hw_dim,
            'embed_dim': self.embed_dim,
            'last_training_reward': self.last_training_reward,
            'infrastructure_agnostic': True,
        }
        if self.training_config:
            save_dict['training_config'] = self.training_config.model_dump()

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, path)
        self.model_path = path
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if not checkpoint.get('infrastructure_agnostic', False):
            logger.warning("Loading old non-agnostic model - reinitializing")
            return

        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.is_trained = checkpoint.get('is_trained', True)
        self.training_timesteps = checkpoint.get('training_timesteps', 0)
        self.last_training_reward = checkpoint.get('last_training_reward')

        if 'training_config' in checkpoint:
            self.training_config = RLTrainingConfig(**checkpoint['training_config'])

        self.model_path = path
        logger.info(f"Model loaded from {path}")

    def get_model_info(self) -> RLModelInfo:
        """Get information about current model."""
        return RLModelInfo(
            model_type="PPO-Agnostic",
            is_trained=self.is_trained,
            training_timesteps=self.training_timesteps,
            task_dim=self.task_dim,
            hw_dim=self.hw_dim,
            infrastructure_agnostic=True,
            model_path=self.model_path,
            last_training_reward=self.last_training_reward,
            training_config=self.training_config
        )

    def encode_state(self, state: RLState) -> Tuple[np.ndarray, List[Tuple[int, np.ndarray]]]:
        """Encode state to numpy vectors (for external use)."""
        return self.encoder.encode(state)

    def get_valid_hw_types(self, state: RLState) -> List[int]:
        """Get list of valid HW type IDs for state."""
        return self.encoder.get_valid_hw_types(state)
