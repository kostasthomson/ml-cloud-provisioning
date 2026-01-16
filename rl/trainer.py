"""
Infrastructure-agnostic PPO Trainer for the RL agent.

Implements PPO training that works with variable numbers of hardware types.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path
import logging
from dataclasses import dataclass

from .agent import RLAgent
from .schemas import RLTrainingConfig, RLTrainingStatus, RLExperience, RLState
from .state_encoder import StateEncoder

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """Single experience for training."""
    task_vec: np.ndarray
    hw_vecs: List[np.ndarray]
    hw_type_ids: List[int]
    valid_mask: np.ndarray
    action_idx: int
    reward: float
    value: float
    log_prob: float
    done: bool


class PPOBuffer:
    """Buffer for storing trajectories with variable action spaces."""

    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        self.gamma = gamma
        self.lam = lam
        self.experiences: List[Experience] = []
        self.advantages: List[float] = []
        self.returns: List[float] = []

    def store(self, exp: Experience):
        """Store a single experience."""
        self.experiences.append(exp)

    def finish_path(self, last_value: float = 0.0):
        """Compute GAE-Lambda advantages for the current trajectory."""
        n = len(self.experiences)
        if n == 0:
            return

        rewards = np.array([e.reward for e in self.experiences] + [last_value])
        values = np.array([e.value for e in self.experiences] + [last_value])
        dones = np.array([float(e.done) for e in self.experiences] + [0.0])

        deltas = rewards[:-1] + self.gamma * values[1:] * (1 - dones[:-1]) - values[:-1]

        advantages = np.zeros(n)
        last_adv = 0.0
        for t in reversed(range(n)):
            advantages[t] = deltas[t] + self.gamma * self.lam * (1 - dones[t]) * last_adv
            last_adv = advantages[t]

        returns = advantages + values[:-1]

        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        self.advantages = advantages.tolist()
        self.returns = returns.tolist()

    def get_all(self) -> List[Tuple[Experience, float, float]]:
        """Get all experiences with their advantages and returns."""
        return list(zip(self.experiences, self.advantages, self.returns))

    def clear(self):
        """Clear the buffer."""
        self.experiences = []
        self.advantages = []
        self.returns = []

    def __len__(self):
        return len(self.experiences)


class PPOTrainer:
    """Infrastructure-agnostic PPO training implementation."""

    def __init__(
        self,
        agent: RLAgent,
        config: Optional[RLTrainingConfig] = None
    ):
        self.agent = agent
        self.config = config or RLTrainingConfig()
        self.device = agent.device

        self.optimizer = optim.Adam(
            agent.policy.parameters(),
            lr=self.config.learning_rate
        )

        self.buffer = PPOBuffer(gamma=self.config.gamma)

        self.is_training = False
        self.current_timestep = 0
        self.episodes_completed = 0
        self.episode_rewards: List[float] = []

    def train(
        self,
        env: 'CloudProvisioningEnv',
        total_timesteps: Optional[int] = None,
        callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Train the agent using PPO.

        Args:
            env: Training environment (infrastructure-agnostic)
            total_timesteps: Override config total_timesteps
            callback: Optional callback(timestep, info) called each update

        Returns:
            Training statistics
        """
        total_timesteps = total_timesteps or self.config.total_timesteps
        self.is_training = True
        self.current_timestep = 0
        buffer_size = 2048

        if hasattr(self.config, 'max_steps_per_episode'):
            env.max_steps = self.config.max_steps_per_episode

        state, info = env.reset()
        episode_reward = 0
        episode_length = 0

        training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': []
        }

        logger.info(f"Starting PPO training for {total_timesteps} timesteps")

        while self.current_timestep < total_timesteps:
            self.buffer.clear()

            for _ in range(buffer_size):
                task_vec, hw_list = self.agent.encoder.encode(state)
                valid_hw_types = self.agent.encoder.get_valid_hw_types(state)

                hw_type_ids = [hw_id for hw_id, _ in hw_list]
                hw_vecs = [hw_vec for _, hw_vec in hw_list]

                valid_mask = np.array([hw_id in valid_hw_types for hw_id in hw_type_ids], dtype=bool)

                task_tensor = torch.FloatTensor(task_vec).to(self.device)
                hw_tensors = [torch.FloatTensor(hv).to(self.device) for hv in hw_vecs]
                mask_tensor = torch.BoolTensor(valid_mask).to(self.device)

                with torch.no_grad():
                    action_probs, value, _ = self.agent.policy.forward(
                        task_tensor, hw_tensors, mask_tensor
                    )

                probs = action_probs.cpu().numpy()
                action_idx = np.random.choice(len(probs), p=probs)
                log_prob = np.log(probs[action_idx] + 1e-8)

                if action_idx < len(hw_type_ids):
                    selected_hw_id = hw_type_ids[action_idx]
                else:
                    selected_hw_id = -1

                next_state, reward, done, truncated, info = env.step(selected_hw_id)

                exp = Experience(
                    task_vec=task_vec,
                    hw_vecs=hw_vecs,
                    hw_type_ids=hw_type_ids,
                    valid_mask=valid_mask,
                    action_idx=action_idx,
                    reward=reward,
                    value=value.item(),
                    log_prob=log_prob,
                    done=done or truncated
                )
                self.buffer.store(exp)

                state = next_state
                episode_reward += reward
                episode_length += 1
                self.current_timestep += 1

                if done or truncated:
                    self.buffer.finish_path(0.0)
                    training_stats['episode_rewards'].append(episode_reward)
                    training_stats['episode_lengths'].append(episode_length)
                    self.episodes_completed += 1
                    self.episode_rewards.append(episode_reward)

                    state, info = env.reset()
                    episode_reward = 0
                    episode_length = 0

                if self.current_timestep >= total_timesteps:
                    break

            if len(self.buffer) > 0:
                task_tensor = torch.FloatTensor(task_vec).to(self.device)
                hw_tensors = [torch.FloatTensor(hv).to(self.device) for hv in hw_vecs]
                with torch.no_grad():
                    _, last_value, _ = self.agent.policy.forward(task_tensor, hw_tensors, None)
                self.buffer.finish_path(last_value.item())

                update_stats = self._update()
                training_stats['policy_losses'].append(update_stats['policy_loss'])
                training_stats['value_losses'].append(update_stats['value_loss'])
                training_stats['entropy_losses'].append(update_stats['entropy_loss'])

                if callback:
                    callback(self.current_timestep, update_stats)

        self.is_training = False
        self.agent.is_trained = True
        self.agent.training_timesteps += self.current_timestep
        self.agent.training_config = self.config

        if self.episode_rewards:
            self.agent.last_training_reward = np.mean(self.episode_rewards[-100:])

        avg_reward = self.agent.last_training_reward or 0.0
        logger.info(f"Training completed. Episodes: {self.episodes_completed}, "
                   f"Avg reward: {avg_reward:.3f}")

        return training_stats

    def _update(self) -> Dict[str, float]:
        """Perform PPO update on collected data."""
        data = self.buffer.get_all()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        n_updates = 0

        for _ in range(self.config.n_epochs):
            indices = np.random.permutation(len(data))

            for start in range(0, len(data), self.config.batch_size):
                end = min(start + self.config.batch_size, len(data))
                batch_indices = indices[start:end]

                batch_policy_loss = 0.0
                batch_value_loss = 0.0
                batch_entropy_loss = 0.0

                for idx in batch_indices:
                    exp, advantage, ret = data[idx]

                    task_tensor = torch.FloatTensor(exp.task_vec).to(self.device)
                    hw_tensors = [torch.FloatTensor(hv).to(self.device) for hv in exp.hw_vecs]
                    mask_tensor = torch.BoolTensor(exp.valid_mask).to(self.device)

                    action_probs, value, _ = self.agent.policy.forward(
                        task_tensor, hw_tensors, mask_tensor
                    )

                    new_log_prob = torch.log(action_probs[exp.action_idx] + 1e-8)
                    old_log_prob = exp.log_prob

                    ratio = torch.exp(new_log_prob - old_log_prob)
                    adv_tensor = torch.tensor(advantage, device=self.device)

                    surr1 = ratio * adv_tensor
                    surr2 = torch.clamp(
                        ratio,
                        1 - self.config.clip_range,
                        1 + self.config.clip_range
                    ) * adv_tensor

                    policy_loss = -torch.min(surr1, surr2)

                    ret_tensor = torch.tensor(ret, device=self.device)
                    value_loss = 0.5 * (value - ret_tensor) ** 2

                    entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum()
                    entropy_loss = -0.01 * entropy

                    batch_policy_loss += policy_loss
                    batch_value_loss += value_loss
                    batch_entropy_loss += entropy_loss

                batch_size = len(batch_indices)
                loss = (batch_policy_loss + 0.5 * batch_value_loss + batch_entropy_loss) / batch_size

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.policy.parameters(), 0.5)
                self.optimizer.step()

                total_policy_loss += batch_policy_loss.item() / batch_size
                total_value_loss += batch_value_loss.item() / batch_size
                total_entropy_loss += batch_entropy_loss.item() / batch_size
                n_updates += 1

        return {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy_loss': total_entropy_loss / max(n_updates, 1)
        }

    def get_status(self) -> RLTrainingStatus:
        """Get current training status."""
        avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
        avg_length = len(self.episode_rewards)

        return RLTrainingStatus(
            is_training=self.is_training,
            current_timestep=self.current_timestep,
            total_timesteps=self.config.total_timesteps,
            episodes_completed=self.episodes_completed,
            avg_reward=float(avg_reward),
            avg_episode_length=float(avg_length)
        )

    def train_from_experiences(
        self,
        experiences: List[RLExperience],
        max_steps_per_episode: int = 2048
    ) -> Dict[str, Any]:
        """
        Train from pre-collected experiences (offline RL).

        Args:
            experiences: List of collected experiences
            max_steps_per_episode: Maximum steps before truncating episode (default 2048)
        """
        from .environment import CloudProvisioningEnv
        env = CloudProvisioningEnv(
            experiences=experiences,
            episode_length=max_steps_per_episode,
            max_steps=max_steps_per_episode
        )
        return self.train(env, total_timesteps=len(experiences))
