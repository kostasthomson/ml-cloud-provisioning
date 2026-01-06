"""
Reward calculation for RL agent.

The reward function balances energy efficiency, SLA compliance, and resource utilization.
"""

from typing import Dict, Optional
from .schemas import TaskOutcome, RLState


class RewardCalculator:
    """
    Calculates reward for RL agent based on task outcomes.

    Reward components:
    - Energy efficiency: Negative reward proportional to energy consumed
    - SLA compliance: Bonus for meeting deadlines, penalty for violations
    - Rejection penalty: Small penalty for rejecting tasks
    - Acceptance bonus: Small bonus for accepting feasible tasks
    """

    def __init__(
        self,
        energy_weight: float = 0.5,
        sla_weight: float = 0.3,
        rejection_penalty: float = 0.2,
        normalize_energy: bool = True,
        energy_baseline: float = 1.0
    ):
        self.energy_weight = energy_weight
        self.sla_weight = sla_weight
        self.rejection_penalty = rejection_penalty
        self.normalize_energy = normalize_energy
        self.energy_baseline = energy_baseline

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
            Float reward value (typically in range [-1, 1])
        """
        if not outcome.accepted:
            return -self.rejection_penalty

        reward = 0.0

        energy_reward = self._compute_energy_reward(outcome.energy_consumed_kwh)
        reward += self.energy_weight * energy_reward

        if outcome.deadline_met is not None:
            sla_reward = self._compute_sla_reward(outcome)
            reward += self.sla_weight * sla_reward
        else:
            reward += self.sla_weight * 0.5

        reward += 0.1

        return max(-1.0, min(1.0, reward))

    def _compute_energy_reward(self, energy_kwh: float) -> float:
        """
        Compute energy component of reward.
        Lower energy = higher reward.
        """
        if self.normalize_energy and self.energy_baseline > 0:
            normalized = energy_kwh / self.energy_baseline
            return 1.0 - min(normalized, 2.0) / 2.0
        else:
            return max(0.0, 1.0 - energy_kwh / 10.0)

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
            "energy_baseline": self.energy_baseline
        }
