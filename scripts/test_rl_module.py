"""
Test script for verifying RL module components.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl import (
    RLAgent, StateEncoder, RewardCalculator,
    RLState, TaskState, HWTypeState, GlobalState, TaskOutcome
)


def test_state_encoder():
    """Test state encoding."""
    print("Testing StateEncoder...")

    encoder = StateEncoder()
    assert encoder.state_dim == 81, f"Expected 81, got {encoder.state_dim}"

    task = TaskState(
        task_id="test_001",
        num_vms=4,
        vcpus_per_vm=8,
        memory_per_vm=32.0,
        instructions=1e10,
        compatible_hw_types=[1, 2]
    )

    hw_types = [
        HWTypeState(
            hw_type_id=1,
            utilization_cpu=0.3,
            utilization_memory=0.25,
            available_cpus=1400,
            available_memory=9600,
            total_cpus=2000,
            total_memory=12800,
            compute_capability=4400
        ),
        HWTypeState(
            hw_type_id=2,
            utilization_cpu=0.5,
            utilization_memory=0.4,
            available_cpus=1000,
            available_memory=6000,
            total_cpus=2000,
            total_memory=10000,
            compute_capability=8800,
            available_accelerators=8,
            total_accelerators=16,
            accelerator_compute_capability=44000
        )
    ]

    global_state = GlobalState(
        timestamp=100.0,
        total_power_consumption=50000,
        queue_length=3
    )

    state = RLState(task=task, hw_types=hw_types, global_state=global_state)

    encoded = encoder.encode(state)
    assert encoded.shape == (81,), f"Expected (81,), got {encoded.shape}"
    assert encoded.min() >= 0.0, "Values should be >= 0"
    assert encoded.max() <= 1.0, "Values should be <= 1"

    mask = encoder.get_valid_actions_mask(state)
    assert mask.shape == (5,), f"Expected (5,), got {mask.shape}"
    assert mask[4] == True, "Reject should always be valid"

    print(f"  Encoded state shape: {encoded.shape}")
    print(f"  Valid actions mask: {mask}")
    print("  StateEncoder: PASSED")


def test_reward_calculator():
    """Test reward computation."""
    print("\nTesting RewardCalculator...")

    calc = RewardCalculator()

    outcome_accept = TaskOutcome(
        task_id="test",
        action_taken=0,
        accepted=True,
        energy_consumed_kwh=0.5,
        deadline_met=True
    )
    reward_accept = calc.compute_reward(outcome_accept)
    print(f"  Accepted task (deadline met): reward = {reward_accept:.3f}")
    assert reward_accept > 0, "Accepted task with met deadline should have positive reward"

    outcome_reject = TaskOutcome(
        task_id="test",
        action_taken=4,
        accepted=False,
        energy_consumed_kwh=0.0
    )
    reward_reject = calc.compute_reward(outcome_reject)
    print(f"  Rejected task: reward = {reward_reject:.3f}")
    assert reward_reject < 0, "Rejected task should have negative reward"

    outcome_sla_violation = TaskOutcome(
        task_id="test",
        action_taken=0,
        accepted=True,
        energy_consumed_kwh=2.0,
        deadline_met=False,
        sla_violation=True
    )
    reward_sla = calc.compute_reward(outcome_sla_violation)
    print(f"  SLA violation: reward = {reward_sla:.3f}")

    print("  RewardCalculator: PASSED")


def test_agent():
    """Test RL agent."""
    print("\nTesting RLAgent...")

    agent = RLAgent()

    task = TaskState(
        task_id="test_001",
        num_vms=2,
        vcpus_per_vm=4,
        memory_per_vm=16.0,
        instructions=1e9,
        compatible_hw_types=[1, 2, 3]
    )

    hw_types = [
        HWTypeState(
            hw_type_id=i,
            utilization_cpu=0.3,
            utilization_memory=0.25,
            available_cpus=1000,
            available_memory=5000,
            total_cpus=1500,
            total_memory=8000,
            compute_capability=4400 * i,
            available_accelerators=5 if i > 1 else 0,
            total_accelerators=10 if i > 1 else 0
        )
        for i in [1, 2, 3, 4]
    ]

    global_state = GlobalState(timestamp=0.0)
    state = RLState(task=task, hw_types=hw_types, global_state=global_state)

    action, value, time_ms = agent.predict(state, deterministic=True)
    print(f"  Predicted action: {action.action} ({action.action_name})")
    print(f"  Confidence: {action.confidence:.3f}")
    print(f"  State value: {value:.3f}")
    print(f"  Inference time: {time_ms:.2f} ms")

    probs = agent.get_action_probs(state)
    print(f"  Action probabilities: {probs}")

    info = agent.get_model_info()
    print(f"  Model info: trained={info.is_trained}, state_dim={info.state_dim}")

    print("  RLAgent: PASSED")


def main():
    print("=" * 50)
    print("RL Module Tests")
    print("=" * 50)

    test_state_encoder()
    test_reward_calculator()
    test_agent()

    print("\n" + "=" * 50)
    print("All tests PASSED!")
    print("=" * 50)


if __name__ == '__main__':
    main()
