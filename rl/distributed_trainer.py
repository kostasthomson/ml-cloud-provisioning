"""
Multi-GPU Distributed PPO Trainer using PyTorch DistributedDataParallel.

Best practices implementation:
- Uses torchrun for process launching (recommended by PyTorch)
- DDP for gradient synchronization
- Vectorized environments for parallel data collection
- File-based checkpointing (no inter-process queues)

Usage:
    torchrun --nproc_per_node=4 scripts/train_rl_distributed.py --timesteps 100000
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path
import logging
from dataclasses import dataclass, field
import time
import json

from .agent import PolicyNetwork
from .schemas import RLTrainingConfig, RLState
from .state_encoder import StateEncoder
from .environment import CloudProvisioningEnv

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


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    policy_losses: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)
    entropy_losses: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropy_losses': self.entropy_losses,
        }


class RolloutBuffer:
    """Buffer for storing rollout experiences with GAE computation."""

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.experiences: List[Experience] = []
        self.advantages: np.ndarray = None
        self.returns: np.ndarray = None

    def add(self, exp: Experience):
        self.experiences.append(exp)

    def compute_returns_and_advantages(self, last_value: float = 0.0):
        """Compute GAE-Lambda advantages and returns."""
        n = len(self.experiences)
        if n == 0:
            return

        rewards = np.array([e.reward for e in self.experiences])
        values = np.array([e.value for e in self.experiences])
        dones = np.array([e.done for e in self.experiences], dtype=np.float32)

        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.experiences[t].done)
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

        self.returns = advantages + values
        self.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def get_batches(self, batch_size: int):
        """Yield random mini-batches."""
        n = len(self.experiences)
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]
            yield [(self.experiences[i], self.advantages[i], self.returns[i]) for i in batch_indices]

    def clear(self):
        self.experiences = []
        self.advantages = None
        self.returns = None

    def __len__(self):
        return len(self.experiences)


class VectorizedEnv:
    """Vectorized environment for parallel rollouts."""

    def __init__(self, num_envs: int, **env_kwargs):
        self.num_envs = num_envs
        self.envs = [CloudProvisioningEnv(**env_kwargs) for _ in range(num_envs)]
        self.states: List[RLState] = []

    def reset(self, seed: int = None) -> List[RLState]:
        self.states = []
        for i, env in enumerate(self.envs):
            env_seed = (seed + i) if seed is not None else None
            state, _ = env.reset(seed=env_seed)
            self.states.append(state)
        return self.states

    def step(self, actions: List[int]) -> Tuple[List[RLState], List[float], List[bool], List[Dict]]:
        next_states = []
        rewards = []
        dones = []
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if done:
                reset_state, _ = env.reset()
                next_states.append(reset_state)
                info['terminal_state'] = next_state
            else:
                next_states.append(next_state)

            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        self.states = next_states
        return next_states, rewards, dones, infos


class DistributedPPOTrainer:
    """
    Multi-GPU PPO Trainer using DistributedDataParallel.

    Designed to be launched with torchrun:
        torchrun --nproc_per_node=NUM_GPUS script.py
    """

    def __init__(
        self,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
    ):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0)))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.is_distributed = self.world_size > 1
        self.is_main_process = self.rank == 0

        if self.is_distributed:
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
            self.device = torch.device(f'cuda:{self.local_rank}')
            torch.cuda.set_device(self.local_rank)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = StateEncoder()
        self.policy = PolicyNetwork(
            task_dim=self.encoder.task_dim,
            hw_dim=self.encoder.hw_dim,
            embed_dim=64
        ).to(self.device)

        if self.is_distributed:
            self.policy = DDP(self.policy, device_ids=[self.local_rank])

        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.buffer = RolloutBuffer(gamma=gamma, gae_lambda=gae_lambda)
        self.metrics = TrainingMetrics()

        self.current_timestep = 0
        self.episodes_completed = 0

    def _get_policy_module(self) -> PolicyNetwork:
        """Get underlying policy module (unwrap DDP if needed)."""
        if isinstance(self.policy, DDP):
            return self.policy.module
        return self.policy

    def _select_action(self, state: RLState) -> Tuple[int, int, float, float, List[int], List[np.ndarray], np.ndarray]:
        """Select action for a single state."""
        task_vec, hw_list = self.encoder.encode(state)
        valid_hw_types = self.encoder.get_valid_hw_types(state)

        hw_type_ids = [hw_id for hw_id, _ in hw_list]
        hw_vecs = [hw_vec for _, hw_vec in hw_list]
        valid_mask = np.array([hw_id in valid_hw_types for hw_id in hw_type_ids], dtype=bool)

        task_tensor = torch.FloatTensor(task_vec).to(self.device)
        hw_tensors = [torch.FloatTensor(hv).to(self.device) for hv in hw_vecs]
        mask_tensor = torch.BoolTensor(valid_mask).to(self.device)

        with torch.no_grad():
            policy_module = self._get_policy_module()
            action_probs, value, _ = policy_module.forward(task_tensor, hw_tensors, mask_tensor)

        probs = action_probs.cpu().numpy()
        action_idx = np.random.choice(len(probs), p=probs)
        log_prob = np.log(probs[action_idx] + 1e-8)

        if action_idx < len(hw_type_ids):
            action = hw_type_ids[action_idx]
        else:
            action = -1

        return action, action_idx, log_prob, value.item(), hw_type_ids, hw_vecs, valid_mask, task_vec

    def collect_rollouts(self, vec_env: VectorizedEnv, n_steps: int) -> Tuple[int, List[float]]:
        """Collect rollouts from vectorized environment."""
        states = vec_env.states
        episode_rewards = []
        current_rewards = [0.0] * vec_env.num_envs
        steps = 0

        for _ in range(n_steps):
            actions = []
            for i, state in enumerate(states):
                action, action_idx, log_prob, value, hw_ids, hw_vecs, valid_mask, task_vec = self._select_action(state)
                actions.append(action)

                exp = Experience(
                    task_vec=task_vec,
                    hw_vecs=hw_vecs,
                    hw_type_ids=hw_ids,
                    valid_mask=valid_mask,
                    action_idx=action_idx,
                    reward=0.0,
                    value=value,
                    log_prob=log_prob,
                    done=False
                )
                self.buffer.add(exp)

            next_states, rewards, dones, infos = vec_env.step(actions)

            buffer_idx = len(self.buffer) - vec_env.num_envs
            for i in range(vec_env.num_envs):
                self.buffer.experiences[buffer_idx + i].reward = rewards[i]
                self.buffer.experiences[buffer_idx + i].done = dones[i]
                current_rewards[i] += rewards[i]

                if dones[i]:
                    episode_rewards.append(current_rewards[i])
                    self.metrics.episode_rewards.append(current_rewards[i])
                    self.episodes_completed += 1
                    current_rewards[i] = 0.0

            states = next_states
            steps += vec_env.num_envs

        if states:
            _, _, _, last_value, _, _, _, _ = self._select_action(states[0])
        else:
            last_value = 0.0

        self.buffer.compute_returns_and_advantages(last_value)
        self.current_timestep += steps

        return steps, episode_rewards

    def update_policy(self) -> Dict[str, float]:
        """Update policy using collected rollouts."""
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        n_updates = 0

        policy_module = self._get_policy_module()

        for _ in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                batch_policy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                batch_value_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                batch_entropy = torch.tensor(0.0, device=self.device, requires_grad=True)

                for exp, advantage, ret in batch:
                    task_tensor = torch.FloatTensor(exp.task_vec).to(self.device)
                    hw_tensors = [torch.FloatTensor(hv).to(self.device) for hv in exp.hw_vecs]
                    mask_tensor = torch.BoolTensor(exp.valid_mask).to(self.device)

                    action_probs, value, _ = policy_module.forward(task_tensor, hw_tensors, mask_tensor)

                    new_log_prob = torch.log(action_probs[exp.action_idx] + 1e-8)
                    ratio = torch.exp(new_log_prob - exp.log_prob)

                    adv = torch.tensor(advantage, device=self.device)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * adv
                    policy_loss = -torch.min(surr1, surr2)

                    ret_tensor = torch.tensor(ret, device=self.device)
                    value_loss = (value - ret_tensor) ** 2

                    entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum()

                    batch_policy_loss = batch_policy_loss + policy_loss
                    batch_value_loss = batch_value_loss + value_loss
                    batch_entropy = batch_entropy + entropy

                bs = len(batch)
                loss = (batch_policy_loss / bs +
                       self.value_loss_coef * batch_value_loss / bs -
                       self.entropy_coef * batch_entropy / bs)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += batch_policy_loss.item() / bs
                total_value_loss += batch_value_loss.item() / bs
                total_entropy_loss += batch_entropy.item() / bs
                n_updates += 1

        self.buffer.clear()

        stats = {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy': total_entropy_loss / max(n_updates, 1),
        }

        self.metrics.policy_losses.append(stats['policy_loss'])
        self.metrics.value_losses.append(stats['value_loss'])
        self.metrics.entropy_losses.append(stats['entropy'])

        return stats

    def _gather_scalar(self, value: float) -> float:
        """Gather and sum a scalar value across all processes."""
        if not self.is_distributed:
            return value
        tensor = torch.tensor([value], device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.item()

    def _gather_mean(self, value: float) -> float:
        """Gather and average a scalar value across all processes."""
        total = self._gather_scalar(value)
        return total / self.world_size

    def train(
        self,
        total_timesteps: int,
        num_envs: int = 8,
        rollout_steps: int = 256,
        env_preset: str = 'medium',
        log_interval: int = 5000,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run training loop.

        Args:
            total_timesteps: Total timesteps to train (across all processes)
            num_envs: Number of parallel environments per process
            rollout_steps: Steps to collect before each update
            env_preset: Environment configuration preset
            log_interval: Log progress every N timesteps
            save_path: Path to save model (only main process saves)
        """
        timesteps_per_process = total_timesteps // self.world_size

        vec_env = VectorizedEnv(num_envs, preset=env_preset)
        vec_env.reset(seed=self.rank * 10000)

        if self.is_main_process:
            print("=" * 70)
            print(f"DISTRIBUTED PPO TRAINING")
            print("=" * 70)
            print(f"  GPUs:                 {self.world_size}")
            print(f"  Envs per GPU:         {num_envs}")
            print(f"  Total parallel envs:  {num_envs * self.world_size}")
            print(f"  Timesteps per GPU:    {timesteps_per_process:,}")
            print(f"  Total timesteps:      {total_timesteps:,}")
            print(f"  Rollout steps:        {rollout_steps}")
            print(f"  Batch size:           {self.batch_size}")
            print(f"  PPO epochs:           {self.n_epochs}")
            print(f"  Learning rate:        {self.learning_rate}")
            print("=" * 70)

        print(f"[GPU {self.rank}] Initialized - device: {self.device}, envs: {num_envs}")

        if self.is_distributed:
            dist.barrier()

        start_time = time.time()
        last_log_step = 0
        update_count = 0

        while self.current_timestep < timesteps_per_process:
            steps, ep_rewards = self.collect_rollouts(vec_env, rollout_steps)
            update_stats = self.update_policy()
            update_count += 1

            global_timesteps = self.current_timestep * self.world_size
            progress_pct = (self.current_timestep / timesteps_per_process) * 100

            if (self.current_timestep - last_log_step) >= log_interval:
                elapsed = time.time() - start_time
                local_fps = self.current_timestep / elapsed if elapsed > 0 else 0
                local_avg_reward = np.mean(self.metrics.episode_rewards[-100:]) if self.metrics.episode_rewards else 0

                if self.is_main_process:
                    print(f"[GPU {self.rank}] Step {self.current_timestep:,}/{timesteps_per_process:,} "
                          f"({progress_pct:.1f}%) | Episodes: {self.episodes_completed} | "
                          f"Reward: {local_avg_reward:.2f} | FPS: {local_fps:.0f}")
                    print("-" * 70)
                    print(f"[GLOBAL] Step {global_timesteps:,}/{total_timesteps:,} ({progress_pct:.1f}%)")
                    print(f"         Avg Reward (GPU 0): {local_avg_reward:.2f} | "
                          f"FPS (GPU 0): {local_fps:.0f}")
                    print(f"         Policy Loss: {update_stats['policy_loss']:.4f} | "
                          f"Value Loss: {update_stats['value_loss']:.4f} | "
                          f"Entropy: {update_stats['entropy']:.4f}")
                    if progress_pct > 0:
                        print(f"         Elapsed: {elapsed:.1f}s | "
                              f"ETA: {(elapsed / progress_pct * 100 - elapsed):.1f}s")
                    print("-" * 70)

                last_log_step = self.current_timestep

        print(f"[GPU {self.rank}] Training complete - {self.current_timestep:,} steps, "
              f"{self.episodes_completed} episodes")

        if self.is_distributed:
            dist.barrier()

        elapsed = time.time() - start_time
        total_steps = self.current_timestep * self.world_size

        total_episodes = int(self._gather_scalar(float(self.episodes_completed)))

        if self.is_main_process and save_path:
            self.save(save_path)

        if self.is_main_process:
            print("=" * 70)
            print("TRAINING COMPLETE")
            print("=" * 70)
            print(f"  Total timesteps:  {total_steps:,}")
            print(f"  Total episodes:   {total_episodes}")
            print(f"  Total time:       {elapsed:.1f}s")
            print(f"  Throughput:       {total_steps / elapsed:.0f} steps/sec")
            if save_path:
                print(f"  Model saved to:   {save_path}")
            print("=" * 70)

        return {
            'total_timesteps': total_steps,
            'episodes': self.episodes_completed,
            'avg_reward': np.mean(self.metrics.episode_rewards[-100:]) if self.metrics.episode_rewards else 0,
            'elapsed_time': elapsed,
            'fps': total_steps / elapsed,
            'metrics': self.metrics.to_dict(),
        }

    def save(self, path: str):
        """Save model checkpoint."""
        policy_module = self._get_policy_module()

        save_dict = {
            'policy_state_dict': policy_module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'is_trained': True,
            'training_timesteps': self.current_timestep * self.world_size,
            'task_dim': self.encoder.task_dim,
            'hw_dim': self.encoder.hw_dim,
            'embed_dim': 64,
            'infrastructure_agnostic': True,
            'last_training_reward': np.mean(self.metrics.episode_rewards[-100:]) if self.metrics.episode_rewards else 0,
            'distributed_training': self.is_distributed,
            'world_size': self.world_size,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")

        metrics_path = Path(path).with_suffix('.metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics.to_dict(), f)

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        policy_module = self._get_policy_module()
        policy_module.load_state_dict(checkpoint['policy_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f"Model loaded from {path}")


def cleanup():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()
