import requests
import json
import sys

BASE_URL = "http://localhost:8000"


def test_get_weights():
    print("Testing GET /weights endpoint...")
    try:
        resp = requests.get(f"{BASE_URL}/weights", timeout=5)
        print(f"  Status: {resp.status_code}")
        result = resp.json()
        print(f"  Weights: {result}")
        expected_keys = ['energy', 'exec_time', 'network', 'ram', 'storage']
        has_all_keys = all(k in result for k in expected_keys)
        return resp.status_code == 200 and has_all_keys
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_set_weights():
    print("\nTesting PUT /weights endpoint...")
    new_weights = {
        "energy": 0.30,
        "exec_time": 0.30,
        "network": 0.15,
        "ram": 0.15,
        "storage": 0.10
    }
    try:
        resp = requests.put(f"{BASE_URL}/weights", json=new_weights, timeout=5)
        print(f"  Status: {resp.status_code}")
        result = resp.json()
        print(f"  Updated weights: {result}")

        weights_match = (
            abs(result['energy'] - 0.30) < 0.01 and
            abs(result['exec_time'] - 0.30) < 0.01 and
            abs(result['network'] - 0.15) < 0.01 and
            abs(result['ram'] - 0.15) < 0.01 and
            abs(result['storage'] - 0.10) < 0.01
        )
        return resp.status_code == 200 and weights_match
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_allocate_scoring_basic():
    print("\nTesting POST /allocate_scoring (basic case)...")

    request_data = {
        "timestamp": 100.0,
        "task_id": "scoring_test_001",
        "num_vms": 4,
        "implementations": [
            {
                "impl_id": 1,
                "instructions": 1e10,
                "vcpus_per_vm": 4,
                "memory_per_vm": 16.0,
                "storage_per_vm": 0.1,
                "network_per_vm": 0.01,
                "requires_accelerator": False,
                "accelerator_rho": 0.0
            }
        ],
        "hw_types": [
            {
                "hw_type_id": 1,
                "hw_type_name": "CPU-only",
                "num_servers": 100,
                "total_cpus": 2000,
                "total_memory": 12800,
                "total_storage": 100,
                "total_network": 40,
                "total_accelerators": 0,
                "available_cpus": 1500,
                "available_memory": 10000,
                "available_storage": 80,
                "available_network": 30,
                "available_accelerators": 0,
                "compute_capability_per_cpu": 4400,
                "accelerator_compute_capability": 0,
                "cpu_idle_power": 163,
                "cpu_max_power": 220,
                "acc_idle_power": 0,
                "acc_max_power": 0,
                "ongoing_tasks": []
            }
        ]
    }

    try:
        resp = requests.post(
            f"{BASE_URL}/allocate_scoring",
            json=request_data,
            timeout=30
        )
        print(f"  Status: {resp.status_code}")
        result = resp.json()
        print(f"  Success: {result.get('success')}")
        print(f"  Selected HW: {result.get('selected_hw_type_id')}")
        print(f"  Selected Impl: {result.get('selected_impl_id')}")
        print(f"  Score: {result.get('total_score')}")
        print(f"  Exec Time: {result.get('estimated_exec_time_sec')} sec")
        print(f"  Energy: {result.get('estimated_energy_kwh')} kWh")

        return resp.status_code == 200 and result.get('success')
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_allocate_scoring_multi_impl():
    print("\nTesting POST /allocate_scoring (multi-implementation)...")

    request_data = {
        "timestamp": 200.0,
        "task_id": "scoring_test_002",
        "num_vms": 4,
        "implementations": [
            {
                "impl_id": 1,
                "instructions": 1e11,
                "vcpus_per_vm": 8,
                "memory_per_vm": 32.0,
                "storage_per_vm": 0.2,
                "network_per_vm": 0.02,
                "requires_accelerator": False,
                "accelerator_rho": 0.0
            },
            {
                "impl_id": 2,
                "instructions": 1e11,
                "vcpus_per_vm": 4,
                "memory_per_vm": 16.0,
                "storage_per_vm": 0.1,
                "network_per_vm": 0.01,
                "requires_accelerator": True,
                "accelerator_rho": 0.85
            }
        ],
        "hw_types": [
            {
                "hw_type_id": 1,
                "hw_type_name": "CPU-only",
                "num_servers": 100,
                "total_cpus": 2000,
                "total_memory": 12800,
                "total_storage": 100,
                "total_network": 40,
                "total_accelerators": 0,
                "available_cpus": 1500,
                "available_memory": 10000,
                "available_storage": 80,
                "available_network": 30,
                "available_accelerators": 0,
                "compute_capability_per_cpu": 4400,
                "accelerator_compute_capability": 0,
                "cpu_idle_power": 163,
                "cpu_max_power": 220,
                "acc_idle_power": 0,
                "acc_max_power": 0,
                "ongoing_tasks": []
            },
            {
                "hw_type_id": 2,
                "hw_type_name": "CPU+GPU",
                "num_servers": 50,
                "total_cpus": 1000,
                "total_memory": 12800,
                "total_storage": 100,
                "total_network": 40,
                "total_accelerators": 200,
                "available_cpus": 800,
                "available_memory": 10000,
                "available_storage": 80,
                "available_network": 30,
                "available_accelerators": 150,
                "compute_capability_per_cpu": 4400,
                "accelerator_compute_capability": 587505,
                "cpu_idle_power": 163,
                "cpu_max_power": 220,
                "acc_idle_power": 32,
                "acc_max_power": 250,
                "ongoing_tasks": []
            }
        ]
    }

    try:
        resp = requests.post(
            f"{BASE_URL}/allocate_scoring",
            json=request_data,
            timeout=30
        )
        print(f"  Status: {resp.status_code}")
        result = resp.json()
        print(f"  Success: {result.get('success')}")
        print(f"  Selected HW: {result.get('selected_hw_type_id')}")
        print(f"  Selected Impl: {result.get('selected_impl_id')}")
        print(f"  Score: {result.get('total_score')}")
        print(f"  Exec Time: {result.get('estimated_exec_time_sec')} sec")
        print(f"  Energy: {result.get('estimated_energy_kwh')} kWh")

        if result.get('all_scores'):
            print(f"\n  All scores ({len(result['all_scores'])}):")
            for score in sorted(result['all_scores'], key=lambda x: x['total_score'] if x['feasible'] else float('inf')):
                status = "FEASIBLE" if score['feasible'] else f"REJECTED: {score.get('rejection_reason')}"
                print(f"    HW{score['hw_type_id']} impl{score['impl_id']}: score={score['total_score']:.4f} [{status}]")
                if score['feasible'] and score.get('metrics'):
                    for metric_name, breakdown in score['metrics'].items():
                        print(f"      {metric_name}: raw={breakdown['raw_value']:.4f}, norm={breakdown['normalized']:.4f}, weighted={breakdown['weighted']:.4f}")

        return resp.status_code == 200 and result.get('success')
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_allocate_scoring_with_ongoing_tasks():
    print("\nTesting POST /allocate_scoring (with ongoing tasks)...")

    request_data = {
        "timestamp": 300.0,
        "task_id": "scoring_test_003",
        "num_vms": 2,
        "implementations": [
            {
                "impl_id": 1,
                "instructions": 5e9,
                "vcpus_per_vm": 4,
                "memory_per_vm": 16.0,
                "storage_per_vm": 0.1,
                "network_per_vm": 0.01,
                "requires_accelerator": False,
                "accelerator_rho": 0.0
            }
        ],
        "hw_types": [
            {
                "hw_type_id": 1,
                "hw_type_name": "CPU-only",
                "num_servers": 100,
                "total_cpus": 2000,
                "total_memory": 12800,
                "total_storage": 100,
                "total_network": 40,
                "total_accelerators": 0,
                "available_cpus": 800,
                "available_memory": 8000,
                "available_storage": 60,
                "available_network": 20,
                "available_accelerators": 0,
                "compute_capability_per_cpu": 4400,
                "accelerator_compute_capability": 0,
                "cpu_idle_power": 163,
                "cpu_max_power": 220,
                "acc_idle_power": 0,
                "acc_max_power": 0,
                "ongoing_tasks": [
                    {
                        "task_id": "ongoing_001",
                        "remaining_instructions": 2e10,
                        "resources_used": {
                            "vcpus": 200,
                            "memory": 1600,
                            "storage": 10,
                            "network": 5,
                            "accelerators": 0
                        },
                        "estimated_remaining_time_sec": 60.0,
                        "accelerator_rho": 0.0
                    },
                    {
                        "task_id": "ongoing_002",
                        "remaining_instructions": 5e10,
                        "resources_used": {
                            "vcpus": 400,
                            "memory": 3200,
                            "storage": 20,
                            "network": 10,
                            "accelerators": 0
                        },
                        "estimated_remaining_time_sec": 120.0,
                        "accelerator_rho": 0.0
                    }
                ]
            }
        ]
    }

    try:
        resp = requests.post(
            f"{BASE_URL}/allocate_scoring",
            json=request_data,
            timeout=30
        )
        print(f"  Status: {resp.status_code}")
        result = resp.json()
        print(f"  Success: {result.get('success')}")
        print(f"  Selected HW: {result.get('selected_hw_type_id')}")
        print(f"  Score: {result.get('total_score')}")
        print(f"  Exec Time: {result.get('estimated_exec_time_sec')} sec (with ongoing tasks)")
        print(f"  Energy: {result.get('estimated_energy_kwh')} kWh")

        return resp.status_code == 200 and result.get('success')
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_allocate_scoring_with_custom_weights():
    print("\nTesting POST /allocate_scoring (with per-request weights)...")

    request_data = {
        "timestamp": 400.0,
        "task_id": "scoring_test_004",
        "num_vms": 4,
        "implementations": [
            {
                "impl_id": 1,
                "instructions": 1e10,
                "vcpus_per_vm": 4,
                "memory_per_vm": 16.0,
                "requires_accelerator": False
            }
        ],
        "hw_types": [
            {
                "hw_type_id": 1,
                "hw_type_name": "CPU-only",
                "num_servers": 100,
                "total_cpus": 2000,
                "total_memory": 12800,
                "total_storage": 100,
                "total_network": 40,
                "available_cpus": 1500,
                "available_memory": 10000,
                "available_storage": 80,
                "available_network": 30,
                "compute_capability_per_cpu": 4400,
                "cpu_idle_power": 163,
                "cpu_max_power": 220,
                "ongoing_tasks": []
            }
        ],
        "weights": {
            "energy": 0.50,
            "exec_time": 0.30,
            "network": 0.10,
            "ram": 0.05,
            "storage": 0.05
        }
    }

    try:
        resp = requests.post(
            f"{BASE_URL}/allocate_scoring",
            json=request_data,
            timeout=30
        )
        print(f"  Status: {resp.status_code}")
        result = resp.json()
        print(f"  Success: {result.get('success')}")

        weights_used = result.get('weights_used', {})
        print(f"  Weights used: energy={weights_used.get('energy')}, exec_time={weights_used.get('exec_time')}")

        weights_correct = (
            abs(weights_used.get('energy', 0) - 0.50) < 0.01 and
            abs(weights_used.get('exec_time', 0) - 0.30) < 0.01
        )

        return resp.status_code == 200 and result.get('success') and weights_correct
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_scoring_statistics():
    print("\nTesting GET /scoring_statistics endpoint...")
    try:
        resp = requests.get(f"{BASE_URL}/scoring_statistics", timeout=5)
        print(f"  Status: {resp.status_code}")
        result = resp.json()
        print(f"  Statistics: {result.get('statistics')}")
        return resp.status_code == 200 and 'statistics' in result
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    print("=" * 70)
    print("SCORING ALLOCATOR INTEGRATION TEST")
    print("=" * 70)

    results = {
        "get_weights": test_get_weights(),
        "set_weights": test_set_weights(),
        "allocate_scoring_basic": test_allocate_scoring_basic(),
        "allocate_scoring_multi_impl": test_allocate_scoring_multi_impl(),
        "allocate_scoring_with_ongoing": test_allocate_scoring_with_ongoing_tasks(),
        "allocate_scoring_custom_weights": test_allocate_scoring_with_custom_weights(),
        "scoring_statistics": test_scoring_statistics()
    }

    print("\n" + "=" * 70)
    print("RESULTS:")
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())
    print("=" * 70)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
