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


def _make_state(num_hw_types=2):
    task = TaskState(
        task_id="test_001",
        num_vms=4,
        vcpus_per_vm=8,
        memory_per_vm=32.0,
        instructions=1e10,
        compatible_hw_types=list(range(1, num_hw_types + 1))
    )

    hw_types = []
    for i in range(1, num_hw_types + 1):
        hw_types.append(HWTypeState(
            hw_type_id=i,
            utilization_cpu=0.3 + 0.1 * i,
            utilization_memory=0.25 + 0.05 * i,
            available_cpus=1400 - 200 * i,
            available_memory=9600 - 1000 * i,
            total_cpus=2000,
            total_memory=12800,
            compute_capability=4400 * i,
            available_accelerators=8 if i > 1 else 0,
            total_accelerators=16 if i > 1 else 0,
            accelerator_compute_capability=44000 if i > 1 else 0,
        ))

    global_state = GlobalState(
        timestamp=100.0,
        total_power_consumption=50000,
        queue_length=3
    )

    return RLState(task=task, hw_types=hw_types, global_state=global_state)


def test_state_encoder():
    print("Testing StateEncoder...")

    encoder = StateEncoder(use_scarcity_features=True, use_capacity_features=True)
    assert encoder.task_dim == 28, f"Expected 28, got {encoder.task_dim}"
    assert encoder.hw_dim == 16, f"Expected 16, got {encoder.hw_dim}"

    state = _make_state(num_hw_types=3)
    task_vec, hw_list = encoder.encode(state)

    assert task_vec.shape == (28,), f"Expected (28,), got {task_vec.shape}"
    assert len(hw_list) == 3, f"Expected 3 HW types, got {len(hw_list)}"
    for hw_id, hw_vec in hw_list:
        assert hw_vec.shape == (16,), f"Expected (16,), got {hw_vec.shape}"

    assert task_vec.min() >= 0.0, "Values should be >= 0"
    assert task_vec.max() <= 1.0, "Values should be <= 1"

    valid = encoder.get_valid_hw_types(state)
    assert isinstance(valid, list), "Should return list"
    print(f"  Task vec shape: {task_vec.shape}")
    print(f"  HW types encoded: {len(hw_list)}")
    print(f"  Valid HW types: {valid}")

    encoder_v2 = StateEncoder(use_scarcity_features=True, use_capacity_features=False)
    assert encoder_v2.task_dim == 22, f"Expected 22, got {encoder_v2.task_dim}"

    encoder_v1 = StateEncoder(use_scarcity_features=False)
    assert encoder_v1.task_dim == 17, f"Expected 17, got {encoder_v1.task_dim}"

    print("  StateEncoder: PASSED")


def test_reward_calculator():
    print("\nTesting RewardCalculator...")

    calc = RewardCalculator()

    assert calc.rejection_penalty == 0.5, f"Expected 0.5, got {calc.rejection_penalty}"
    assert calc.acceptance_bonus == 0.35, f"Expected 0.35, got {calc.acceptance_bonus}"
    assert calc.scarcity_aware == False, f"Expected False, got {calc.scarcity_aware}"

    outcome_accept = TaskOutcome(
        task_id="test",
        action_taken=0,
        accepted=True,
        energy_consumed_kwh=0.02,
        deadline_met=True
    )
    reward_accept = calc.compute_reward(outcome_accept)
    print(f"  Accepted task (deadline met, low energy): reward = {reward_accept:.3f}")
    assert reward_accept > 0, "Accepted task with met deadline and low energy should have positive reward"

    outcome_reject = TaskOutcome(
        task_id="test",
        action_taken=4,
        accepted=False,
        energy_consumed_kwh=0.0
    )
    reward_reject = calc.compute_reward(outcome_reject)
    print(f"  Rejected task: reward = {reward_reject:.3f}")
    assert reward_reject < 0, "Rejected task should have negative reward"
    assert reward_reject == -0.5, f"Expected -0.5, got {reward_reject}"

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

    calc2 = RewardCalculator()
    for i in range(25):
        calc2._update_running_stats(0.04 + i * 0.001)
    assert calc2._running_energy_count == 25
    assert calc2._energy_ema != calc2.energy_baseline, "EMA should have updated"
    print(f"  EMA after 25 samples: {calc2._energy_ema:.6f}")

    print("  RewardCalculator: PASSED")


def test_agent():
    print("\nTesting RLAgent...")

    agent = RLAgent(use_capacity_features=True)
    assert agent.task_dim == 28, f"Expected 28, got {agent.task_dim}"

    state = _make_state(num_hw_types=3)
    action, value, time_ms = agent.predict(state, deterministic=True)
    print(f"  Predicted action: {action.action} ({action.action_name})")
    print(f"  Confidence: {action.confidence:.3f}")
    print(f"  State value: {value:.3f}")
    print(f"  Inference time: {time_ms:.2f} ms")

    probs = agent.get_action_probs(state)
    print(f"  Action probabilities: {probs}")
    assert len(probs) == 4, f"Expected 4 entries (3 HW + reject), got {len(probs)}"
    assert -1 in probs, "Should have reject action (-1)"

    info = agent.get_model_info()
    print(f"  Model info: trained={info.is_trained}, task_dim={info.task_dim}")

    print("  RLAgent: PASSED")


def test_policy_network():
    print("\nTesting PolicyNetwork architecture...")

    from rl.agent import PolicyNetwork
    import torch

    policy = PolicyNetwork(task_dim=28, hw_dim=16, embed_dim=64)

    assert policy.value_head[0].in_features == 192, \
        f"Value head input should be 192 (64*3), got {policy.value_head[0].in_features}"

    assert policy.reject_head[0].in_features == 195, \
        f"Reject head input should be 195 (64*3+3), got {policy.reject_head[0].in_features}"

    has_layernorm = any(isinstance(m, torch.nn.LayerNorm) for m in policy.task_encoder.network)
    assert has_layernorm, "TaskEncoder should have LayerNorm"

    has_layernorm = any(isinstance(m, torch.nn.LayerNorm) for m in policy.hw_encoder.network)
    assert has_layernorm, "HWEncoder should have LayerNorm"

    task_vec = torch.randn(28)
    hw_vecs = [torch.randn(16) for _ in range(3)]
    valid_mask = torch.tensor([True, True, False])

    probs, value, scores = policy.forward(task_vec, hw_vecs, valid_mask)
    assert probs.shape == (4,), f"Expected (4,) probs, got {probs.shape}"
    assert scores.shape == (4,), f"Expected (4,) scores, got {scores.shape}"
    assert probs[2].item() == 0.0, "Masked HW should have 0 probability"

    print(f"  Probs shape: {probs.shape}")
    print(f"  Value: {value.item():.4f}")
    print("  PolicyNetwork: PASSED")


def test_reject_head_capacity_awareness():
    print("\nTesting reject head capacity awareness...")

    from rl.agent import PolicyNetwork
    import torch

    policy = PolicyNetwork(task_dim=28, hw_dim=16, embed_dim=64)

    task_vec = torch.randn(28)
    hw_vecs = [torch.randn(16) for _ in range(3)]
    valid_mask = torch.tensor([True, True, True])
    probs, value, scores = policy.forward(task_vec, hw_vecs, valid_mask)
    assert probs.shape == (4,), f"Expected (4,) probs, got {probs.shape}"
    assert probs[-1].item() >= 0, "Reject prob should be non-negative"
    print(f"  All-valid: reject_prob={probs[-1].item():.4f}")

    valid_mask_none = torch.tensor([False, False, False])
    probs2, _, _ = policy.forward(task_vec, hw_vecs, valid_mask_none)
    assert probs2.shape == (4,), f"Expected (4,), got {probs2.shape}"
    for i in range(3):
        assert probs2[i].item() == 0.0, f"Masked HW {i} should have 0 prob"
    assert probs2[-1].item() > 0, "Reject should be only valid action"
    print(f"  None-valid: reject_prob={probs2[-1].item():.4f}")

    hw_vecs_single = [torch.randn(16)]
    valid_mask_single = torch.tensor([True])
    probs3, _, _ = policy.forward(task_vec, hw_vecs_single, valid_mask_single)
    assert probs3.shape == (2,), f"Expected (2,), got {probs3.shape}"
    print(f"  Single HW: reject_prob={probs3[-1].item():.4f}")

    probs4, _, _ = policy.forward(task_vec, [], None)
    assert probs4.shape == (1,), f"Expected (1,), got {probs4.shape}"
    assert abs(probs4[0].item() - 1.0) < 1e-5, "Empty HW: reject should be 1.0"
    print(f"  Empty HW: reject_prob={probs4[0].item():.4f}")

    print("  Reject head capacity awareness: PASSED")


def test_capacity_scaled_tasks():
    print("\nTesting capacity-scaled task generation...")

    from rl.environment import CloudProvisioningEnv, REALISTIC_HW_CONFIGS

    env_stress = CloudProvisioningEnv(preset='stress_test', seed=42)
    env_medium = CloudProvisioningEnv(preset='medium', seed=42)

    assert env_stress._capacity_scale == 0.25, \
        f"Expected stress_test scale 0.25, got {env_stress._capacity_scale}"
    assert env_medium._capacity_scale == 1.0, \
        f"Expected medium scale 1.0, got {env_medium._capacity_scale}"
    print(f"  stress_test scale: {env_stress._capacity_scale}")
    print(f"  medium scale: {env_medium._capacity_scale}")

    stress_total_cpus = sum(c.total_cpus for c in env_stress.hw_configs)
    stress_total_mem = sum(c.total_memory for c in env_stress.hw_configs)

    stress_vms = []
    medium_vms = []
    n_tasks = 200

    for _ in range(n_tasks):
        task = env_stress._generate_task()
        total_cpu = task.num_vms * task.vcpus_per_vm
        total_mem = task.num_vms * task.memory_per_vm
        assert total_cpu <= stress_total_cpus, \
            f"stress_test task needs {total_cpu} CPUs but only {stress_total_cpus} available"
        stress_vms.append(task.num_vms)

    for _ in range(n_tasks):
        task = env_medium._generate_task()
        medium_vms.append(task.num_vms)

    avg_stress = sum(stress_vms) / len(stress_vms)
    avg_medium = sum(medium_vms) / len(medium_vms)
    assert avg_stress < avg_medium, \
        f"stress_test avg VMs ({avg_stress:.1f}) should be < medium ({avg_medium:.1f})"
    print(f"  stress_test avg num_vms: {avg_stress:.2f}")
    print(f"  medium avg num_vms: {avg_medium:.2f}")

    assert env_stress._max_task_memory == stress_total_mem * 0.5
    import numpy as np
    orig_choice = np.random.choice
    np.random.choice = lambda x, **kw: 'memory_intensive' if x == ['small', 'medium', 'large', 'gpu', 'memory_intensive'] else orig_choice(x, **kw)
    try:
        for _ in range(100):
            task = env_stress._generate_task()
            assert task.memory_per_vm <= env_stress._max_task_memory, \
                f"Memory per VM {task.memory_per_vm} exceeds cap {env_stress._max_task_memory}"
    finally:
        np.random.choice = orig_choice
    print(f"  memory_intensive cap on stress_test: {env_stress._max_task_memory}")

    print("  Capacity-scaled tasks: PASSED")


def main():
    print("=" * 50)
    print("RL Module Tests")
    print("=" * 50)

    test_state_encoder()
    test_reward_calculator()
    test_policy_network()
    test_reject_head_capacity_awareness()
    test_capacity_scaled_tasks()
    test_agent()

    print("\n" + "=" * 50)
    print("All tests PASSED!")
    print("=" * 50)


if __name__ == '__main__':
    main()
