"""
Reward calculation for RL agent.

The reward function prioritizes energy efficiency while maintaining SLA compliance.
"""

from typing import Dict, Optional
import math
from .schemas import TaskOutcome, RLState


class RewardCalculator:
    """
    Calculates reward for RL agent based on task outcomes.

    Reward components (aggressive energy focus):
    - Energy efficiency: Strong negative reward proportional to energy consumed
    - SLA compliance: Bonus for meeting deadlines, penalty for violations
    - Rejection penalty: Moderate penalty for rejecting tasks
    - Efficiency bonus: Extra reward for low-energy allocations
    """

    def __init__(
        self,
        energy_weight: float = 0.6,
        sla_weight: float = 0.2,
        rejection_penalty: float = 0.8,
        acceptance_bonus: float = 0.3,
        normalize_energy: bool = True,
        energy_baseline: float = 0.05,
        energy_excellent_threshold: float = 0.03,
        energy_poor_threshold: float = 0.08
    ):
        self.energy_weight = energy_weight
        self.sla_weight = sla_weight
        self.rejection_penalty = rejection_penalty
        self.acceptance_bonus = acceptance_bonus
        self.normalize_energy = normalize_energy
        self.energy_baseline = energy_baseline
        self.energy_excellent_threshold = energy_excellent_threshold
        self.energy_poor_threshold = energy_poor_threshold
        self._running_energy_sum = 0.0
        self._running_energy_count = 0
        self._running_energy_sq_sum = 0.0

    def compute_reward(
        self,
        outcome: TaskOutcome,
        state: Optional[RLState] = None
    ) -> float:
        """
        Compute reward from task outcome.

        Args:
            outcome: TaskOutcome with execution results
            state: Optional RLState for context-aware rewards

        Returns:
            Float reward value (typically in range [-2, 1])
        """
        if not outcome.accepted:
            return -self.rejection_penalty

        reward = self.acceptance_bonus
        energy = outcome.energy_consumed_kwh

        self._update_running_stats(energy)

        energy_reward = self._compute_energy_reward(energy)
        reward += self.energy_weight * energy_reward

        if energy < self.energy_excellent_threshold:
            reward += 0.2
        elif energy > self.energy_poor_threshold:
            reward -= 0.15

        if self._running_energy_count > 10:
            running_mean = self.get_running_mean()
            if energy < running_mean * 0.8:
                reward += 0.1
            elif energy > running_mean * 1.2:
                reward -= 0.1

        if outcome.deadline_met is not None:
            sla_reward = self._compute_sla_reward(outcome)
            reward += self.sla_weight * sla_reward
        else:
            reward += self.sla_weight * 0.2

        return max(-2.0, min(2.0, reward))

    def _update_running_stats(self, energy_kwh: float):
        """Update running statistics for adaptive normalization."""
        self._running_energy_count += 1
        self._running_energy_sum += energy_kwh
        self._running_energy_sq_sum += energy_kwh ** 2

    def get_running_mean(self) -> float:
        """Get running mean of energy consumption."""
        if self._running_energy_count == 0:
            return self.energy_baseline
        return self._running_energy_sum / self._running_energy_count

    def get_running_std(self) -> float:
        """Get running standard deviation of energy consumption."""
        if self._running_energy_count < 2:
            return self.energy_baseline
        mean = self.get_running_mean()
        variance = (self._running_energy_sq_sum / self._running_energy_count) - (mean ** 2)
        return math.sqrt(max(0, variance))

    def _compute_energy_reward(self, energy_kwh: float) -> float:
        """
        Compute energy component of reward with aggressive penalty.
        Lower energy = higher reward, using exponential scaling.
        """
        if self.normalize_energy and self.energy_baseline > 0:
            normalized = energy_kwh / self.energy_baseline
            if normalized <= 0.5:
                return 1.0
            elif normalized <= 1.0:
                return 1.0 - (normalized - 0.5)
            elif normalized <= 2.0:
                return 0.5 - (normalized - 1.0) * 0.75
            else:
                return max(-1.5, -0.25 - (normalized - 2.0) * 0.5)
        else:
            log_energy = math.log10(max(energy_kwh, 1e-6))
            return max(-1.5, 1.0 - (log_energy + 4) / 2.0)

    def _compute_sla_reward(self, outcome: TaskOutcome) -> float:
        """
        Compute SLA component of reward.
        """
        if outcome.deadline_met:
            return 1.0
        elif outcome.sla_violation:
            return -1.0
        else:
            return 0.0

    def compute_reward_from_components(
        self,
        energy_kwh: float,
        accepted: bool,
        deadline_met: Optional[bool] = None,
        sla_violation: bool = False
    ) -> float:
        """
        Convenience method to compute reward from individual components.
        """
        outcome = TaskOutcome(
            task_id="",
            action_taken=0,
            accepted=accepted,
            energy_consumed_kwh=energy_kwh,
            deadline_met=deadline_met,
            sla_violation=sla_violation
        )
        return self.compute_reward(outcome)

    def get_config(self) -> Dict[str, float]:
        """Return current reward configuration."""
        return {
            "energy_weight": self.energy_weight,
            "sla_weight": self.sla_weight,
            "rejection_penalty": self.rejection_penalty,
            "acceptance_bonus": self.acceptance_bonus,
            "energy_baseline": self.energy_baseline,
            "energy_excellent_threshold": self.energy_excellent_threshold,
            "energy_poor_threshold": self.energy_poor_threshold,
            "running_energy_mean": self.get_running_mean(),
            "running_energy_std": self.get_running_std()
        }
