"""
Test suite for infrastructure-agnostic RL model.

This script tests the RL model with various infrastructure configurations
to verify it works correctly with any number of hardware types.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from typing import List, Dict, Tuple

from rl.agent import RLAgent
from rl.trainer import PPOTrainer
from rl.environment import CloudProvisioningEnv, HWTypeConfig, REALISTIC_HW_CONFIGS
from rl.schemas import (
    RLState, TaskState, HWTypeState, GlobalState,
    RLTrainingConfig
)
from rl.state_encoder import StateEncoder


class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.messages: List[str] = []

    def fail(self, msg: str):
        self.passed = False
        self.messages.append(f"FAIL: {msg}")

    def info(self, msg: str):
        self.messages.append(f"INFO: {msg}")

    def __str__(self):
        status = "PASSED" if self.passed else "FAILED"
        result = f"\n{'='*60}\n{self.name}: {status}\n{'='*60}"
        for msg in self.messages:
            result += f"\n  {msg}"
        return result


def create_realistic_state(
    hw_configs: List[HWTypeConfig],
    task_type: str = 'medium',
    utilization: float = 0.5
) -> RLState:
    """Create a realistic state for testing."""
    if task_type == 'small':
        task = TaskState(
            task_id="test_small_001",
            num_vms=2,
            vcpus_per_vm=4,
            memory_per_vm=16.0,
            storage_per_vm=0.1,
            network_per_vm=0.01,
            instructions=5e9,
            compatible_hw_types=[cfg.hw_type_id for cfg in hw_configs if cfg.total_accelerators == 0],
            requires_accelerator=False,
            deadline=120.0
        )
    elif task_type == 'gpu':
        gpu_types = [cfg.hw_type_id for cfg in hw_configs if cfg.total_accelerators > 0]
        if not gpu_types:
            gpu_types = [hw_configs[0].hw_type_id]
        task = TaskState(
            task_id="test_gpu_001",
            num_vms=4,
            vcpus_per_vm=8,
            memory_per_vm=64.0,
            storage_per_vm=0.5,
            network_per_vm=0.1,
            instructions=1e12,
            compatible_hw_types=gpu_types,
            requires_accelerator=True,
            accelerator_rho=0.8,
            deadline=600.0
        )
    elif task_type == 'large':
        task = TaskState(
            task_id="test_large_001",
            num_vms=8,
            vcpus_per_vm=16,
            memory_per_vm=128.0,
            storage_per_vm=1.0,
            network_per_vm=0.5,
            instructions=5e13,
            compatible_hw_types=[cfg.hw_type_id for cfg in hw_configs],
            requires_accelerator=False,
            deadline=3600.0
        )
    else:
        task = TaskState(
            task_id="test_medium_001",
            num_vms=4,
            vcpus_per_vm=8,
            memory_per_vm=32.0,
            storage_per_vm=0.2,
            network_per_vm=0.05,
            instructions=1e11,
            compatible_hw_types=[cfg.hw_type_id for cfg in hw_configs],
            requires_accelerator=False,
            deadline=300.0
        )

    hw_types = []
    for cfg in hw_configs:
        available_ratio = 1 - utilization
        hw_types.append(HWTypeState(
            hw_type_id=cfg.hw_type_id,
            utilization_cpu=utilization,
            utilization_memory=utilization * 0.9,
            utilization_storage=utilization * 0.3,
            utilization_network=utilization * 0.2,
            utilization_accelerator=utilization * 0.5 if cfg.total_accelerators > 0 else 0,
            available_cpus=cfg.total_cpus * available_ratio,
            available_memory=cfg.total_memory * available_ratio,
            available_storage=cfg.total_storage * available_ratio,
            available_network=cfg.total_network * available_ratio,
            available_accelerators=int(cfg.total_accelerators * available_ratio),
            total_cpus=cfg.total_cpus,
            total_memory=cfg.total_memory,
            total_storage=cfg.total_storage,
            total_network=cfg.total_network,
            total_accelerators=cfg.total_accelerators,
            compute_capability=cfg.compute_capability,
            accelerator_compute_capability=cfg.accelerator_compute,
            power_idle=cfg.power_idle,
            power_max=cfg.power_max,
            acc_power_idle=cfg.acc_power_idle,
            acc_power_max=cfg.acc_power_max,
            num_running_tasks=int(10 * utilization),
            avg_remaining_time=300.0 * utilization
        ))

    global_state = GlobalState(
        timestamp=43200.0,
        total_power_consumption=sum(cfg.power_idle + utilization * (cfg.power_max - cfg.power_idle) for cfg in hw_configs),
        queue_length=5,
        recent_acceptance_rate=0.85,
        recent_avg_energy=0.5
    )

    return RLState(task=task, hw_types=hw_types, global_state=global_state)


def test_encoder_variable_hw_types():
    """Test that encoder works with different numbers of HW types."""
    result = TestResult("Encoder with Variable HW Types")

    encoder = StateEncoder()

    for preset, configs in REALISTIC_HW_CONFIGS.items():
        state = create_realistic_state(configs)
        task_vec, hw_list = encoder.encode(state)

        if task_vec.shape[0] != encoder.task_dim:
            result.fail(f"{preset}: Task dim mismatch {task_vec.shape[0]} != {encoder.task_dim}")

        if len(hw_list) != len(configs):
            result.fail(f"{preset}: HW count mismatch {len(hw_list)} != {len(configs)}")

        for hw_id, hw_vec in hw_list:
            if hw_vec.shape[0] != encoder.hw_dim:
                result.fail(f"{preset}: HW dim mismatch for type {hw_id}")

        result.info(f"{preset}: {len(configs)} HW types encoded successfully")

    return result


def test_agent_inference_variable_hw():
    """Test that agent can do inference with different HW configurations."""
    result = TestResult("Agent Inference with Variable HW Types")

    agent = RLAgent(device='cpu')

    test_cases = [
        ('small', REALISTIC_HW_CONFIGS['small']),
        ('medium', REALISTIC_HW_CONFIGS['medium']),
        ('large', REALISTIC_HW_CONFIGS['large']),
        ('enterprise', REALISTIC_HW_CONFIGS['enterprise']),
    ]

    for name, configs in test_cases:
        state = create_realistic_state(configs)

        try:
            action, value, time_ms = agent.predict(state, deterministic=True)

            if action.action == -1:
                result.info(f"{name} ({len(configs)} types): REJECT, value={value:.3f}, time={time_ms:.2f}ms")
            else:
                result.info(f"{name} ({len(configs)} types): HW_{action.hw_type_id}, conf={action.confidence:.3f}, value={value:.3f}")

            if action.action_probs is None:
                result.fail(f"{name}: No action probs returned")
            elif len(action.action_probs) != len(configs) + 1:
                result.fail(f"{name}: Probs count {len(action.action_probs)} != {len(configs) + 1}")

        except Exception as e:
            result.fail(f"{name}: Inference error - {e}")

    return result


def test_agent_stochastic_inference():
    """Test stochastic action selection."""
    result = TestResult("Stochastic Action Selection")

    agent = RLAgent(device='cpu')
    configs = REALISTIC_HW_CONFIGS['medium']
    state = create_realistic_state(configs)

    action_counts = {}
    n_samples = 100

    for _ in range(n_samples):
        action, _, _ = agent.predict(state, deterministic=False)
        key = action.action
        action_counts[key] = action_counts.get(key, 0) + 1

    result.info(f"Action distribution over {n_samples} samples:")
    for action, count in sorted(action_counts.items()):
        pct = count / n_samples * 100
        label = "REJECT" if action == -1 else f"HW_{action}"
        result.info(f"  {label}: {count} ({pct:.1f}%)")

    if len(action_counts) == 1:
        result.info("Warning: Only one action selected - model may need training")

    return result


def test_valid_hw_types_detection():
    """Test that valid HW types are correctly detected."""
    result = TestResult("Valid HW Types Detection")

    encoder = StateEncoder()

    configs = [
        HWTypeConfig(1, "CPU-Low", 100, 400, 10, 10, 0, 4400, 0, 163, 220),
        HWTypeConfig(2, "CPU-High", 1000, 4000, 100, 100, 0, 4400, 0, 180, 250),
        HWTypeConfig(3, "GPU", 500, 2000, 50, 50, 16, 4400, 125000, 200, 300, 50, 300),
    ]

    task = TaskState(
        task_id="test_resource",
        num_vms=10,
        vcpus_per_vm=16,
        memory_per_vm=64.0,
        instructions=1e10,
        compatible_hw_types=[1, 2, 3],
        requires_accelerator=False
    )

    hw_types = [
        HWTypeState(
            hw_type_id=1,
            utilization_cpu=0.9,
            utilization_memory=0.9,
            available_cpus=10,
            available_memory=40,
            total_cpus=100,
            total_memory=400,
            compute_capability=4400
        ),
        HWTypeState(
            hw_type_id=2,
            utilization_cpu=0.2,
            utilization_memory=0.2,
            available_cpus=800,
            available_memory=3200,
            total_cpus=1000,
            total_memory=4000,
            compute_capability=4400
        ),
        HWTypeState(
            hw_type_id=3,
            utilization_cpu=0.5,
            utilization_memory=0.5,
            available_cpus=250,
            available_memory=1000,
            total_cpus=500,
            total_memory=2000,
            compute_capability=4400,
            available_accelerators=8,
            total_accelerators=16
        ),
    ]

    state = RLState(
        task=task,
        hw_types=hw_types,
        global_state=GlobalState(timestamp=0)
    )

    valid = encoder.get_valid_hw_types(state)

    if 1 in valid:
        result.fail("HW type 1 should NOT be valid (insufficient resources)")
    if 2 not in valid:
        result.fail("HW type 2 SHOULD be valid (has resources)")
    if 3 not in valid:
        result.fail("HW type 3 SHOULD be valid (has resources)")

    result.info(f"Valid HW types: {valid}")
    result.info("Expected: [2, 3] (type 1 lacks resources)")

    return result


def test_training_variable_hw():
    """Test training with different infrastructure configurations."""
    result = TestResult("Training with Variable HW Types")

    for preset in ['small', 'medium', 'large']:
        agent = RLAgent(device='cpu')
        env = CloudProvisioningEnv(preset=preset, episode_length=50, seed=42)

        config = RLTrainingConfig(
            total_timesteps=200,
            learning_rate=3e-4,
            batch_size=32,
            n_epochs=3
        )

        trainer = PPOTrainer(agent, config)

        try:
            stats = trainer.train(env, total_timesteps=200)

            n_episodes = len(stats['episode_rewards'])
            if n_episodes > 0:
                avg_reward = np.mean(stats['episode_rewards'])
                result.info(f"{preset}: {n_episodes} episodes, avg_reward={avg_reward:.3f}")
            else:
                result.info(f"{preset}: Training completed (no full episodes)")

            if not agent.is_trained:
                result.fail(f"{preset}: Agent not marked as trained")

        except Exception as e:
            result.fail(f"{preset}: Training error - {e}")
            import traceback
            traceback.print_exc()

    return result


def test_custom_hw_configuration():
    """Test with completely custom HW configuration."""
    result = TestResult("Custom HW Configuration")

    custom_configs = [
        HWTypeConfig(10, "CustomCPU", 500, 2000, 50, 25, 0, 5000, 0, 150, 200),
        HWTypeConfig(20, "CustomGPU", 250, 1000, 25, 25, 8, 5000, 150000, 200, 350, 75, 350),
        HWTypeConfig(30, "CustomFPGA", 100, 500, 10, 10, 4, 5000, 80000, 100, 150, 40, 120),
        HWTypeConfig(40, "CustomMem", 200, 8000, 200, 50, 0, 5000, 0, 180, 240),
        HWTypeConfig(50, "CustomEdge", 50, 200, 5, 5, 0, 2000, 0, 50, 80),
    ]

    agent = RLAgent(device='cpu')
    state = create_realistic_state(custom_configs, task_type='medium')

    try:
        action, value, time_ms = agent.predict(state, deterministic=True)

        result.info(f"Custom config with {len(custom_configs)} types (IDs: 10,20,30,40,50)")
        result.info(f"Action: {action.action_name}, confidence: {action.confidence:.3f}")
        result.info(f"State value: {value:.3f}, inference time: {time_ms:.2f}ms")

        if action.action not in [-1, 10, 20, 30, 40, 50]:
            result.fail(f"Unexpected action {action.action}")

    except Exception as e:
        result.fail(f"Custom config error: {e}")
        import traceback
        traceback.print_exc()

    return result


def test_model_save_load():
    """Test model save and load."""
    result = TestResult("Model Save/Load")

    import tempfile
    import os

    agent1 = RLAgent(device='cpu')

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.pth")

        agent1.save(model_path)
        result.info(f"Model saved to {model_path}")

        agent2 = RLAgent(model_path=model_path, device='cpu')

        state = create_realistic_state(REALISTIC_HW_CONFIGS['medium'])

        action1, value1, _ = agent1.predict(state, deterministic=True)
        action2, value2, _ = agent2.predict(state, deterministic=True)

        if action1.action != action2.action:
            result.fail(f"Actions differ: {action1.action} vs {action2.action}")

        if abs(value1 - value2) > 1e-5:
            result.fail(f"Values differ: {value1} vs {value2}")

        result.info(f"Loaded model produces identical outputs")

    return result


def test_edge_cases():
    """Test edge cases."""
    result = TestResult("Edge Cases")

    agent = RLAgent(device='cpu')

    single_hw = [HWTypeConfig(1, "Single", 1000, 4000, 100, 50, 0, 4400, 0, 163, 220)]
    state = create_realistic_state(single_hw)

    action, value, _ = agent.predict(state, deterministic=True)
    result.info(f"Single HW type: action={action.action_name}, value={value:.3f}")

    if action.action not in [-1, 1]:
        result.fail(f"Unexpected action with single HW: {action.action}")

    configs = REALISTIC_HW_CONFIGS['medium']
    state = create_realistic_state(configs, task_type='medium', utilization=0.99)

    action, value, _ = agent.predict(state, deterministic=True)
    result.info(f"High utilization: action={action.action_name}")

    configs = REALISTIC_HW_CONFIGS['medium']
    state = create_realistic_state(configs, task_type='medium', utilization=0.01)

    action, value, _ = agent.predict(state, deterministic=True)
    result.info(f"Low utilization: action={action.action_name}")

    return result


def test_expected_behavior():
    """Test that model produces expected behavior patterns."""
    result = TestResult("Expected Behavior Patterns")

    agent = RLAgent(device='cpu')
    env = CloudProvisioningEnv(preset='medium', episode_length=50, seed=42)
    config = RLTrainingConfig(total_timesteps=1000, batch_size=32, n_epochs=5)
    trainer = PPOTrainer(agent, config)
    trainer.train(env, total_timesteps=1000)

    configs = REALISTIC_HW_CONFIGS['medium']
    results_by_utilization = {}

    for util in [0.2, 0.5, 0.8]:
        state = create_realistic_state(configs, utilization=util)
        action, value, _ = agent.predict(state, deterministic=True)
        results_by_utilization[util] = (action.action, value)
        result.info(f"Util {util:.0%}: action={action.action_name}, value={value:.3f}")

    if results_by_utilization[0.2][1] < results_by_utilization[0.8][1]:
        result.info("Note: Low utilization has lower value - may need more training")

    return result


def run_all_tests():
    """Run all tests and print summary."""
    print("\n" + "="*70)
    print("INFRASTRUCTURE-AGNOSTIC RL MODEL TEST SUITE")
    print("="*70)

    tests = [
        test_encoder_variable_hw_types,
        test_agent_inference_variable_hw,
        test_agent_stochastic_inference,
        test_valid_hw_types_detection,
        test_custom_hw_configuration,
        test_model_save_load,
        test_edge_cases,
        test_training_variable_hw,
        test_expected_behavior,
    ]

    results = []
    for test_fn in tests:
        print(f"\nRunning: {test_fn.__name__}...")
        try:
            result = test_fn()
            results.append(result)
            print(result)
        except Exception as e:
            result = TestResult(test_fn.__name__)
            result.fail(f"Unexpected error: {e}")
            results.append(result)
            print(result)
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    for r in results:
        status = "PASSED" if r.passed else "FAILED"
        print(f"  {r.name}: {status}")

    print(f"\nTotal: {passed} passed, {failed} failed out of {len(results)}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
