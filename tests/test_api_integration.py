import requests
import json
import sys

BASE_URL = "http://localhost:8000"


def test_health():
    print("Testing health endpoint...")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"  Status: {resp.status_code}")
        print(f"  Response: {resp.json()}")
        return resp.status_code == 200
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_allocate_task():
    print("\nTesting /allocate_task endpoint...")

    request_data = {
        "timestamp": 100.0,
        "cells": [{
            "cell_id": 1,
            "hw_types": [
                {
                    "hw_type_id": 1,
                    "hw_type_name": "CPU-only",
                    "num_servers": 100,
                    "num_cpus_per_server": 20,
                    "memory_per_server": 128.0,
                    "storage_per_server": 1.0,
                    "compute_capability": 4400.0,
                    "accelerators": 0,
                    "num_accelerators_per_server": 0,
                    "accelerator_compute_capability": 0.0,
                    "cpu_power_consumption": [163, 170, 175, 180, 185, 190, 195, 200, 210, 215, 220],
                    "cpu_utilization_bins": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    "cpu_idle_power": 163.0,
                    "accelerator_idle_power": 0.0,
                    "accelerator_max_power": 0.0
                },
                {
                    "hw_type_id": 2,
                    "hw_type_name": "CPU+GPU",
                    "num_servers": 50,
                    "num_cpus_per_server": 20,
                    "memory_per_server": 256.0,
                    "storage_per_server": 2.0,
                    "compute_capability": 4400.0,
                    "accelerators": 1,
                    "num_accelerators_per_server": 4,
                    "accelerator_compute_capability": 587505.34,
                    "cpu_power_consumption": [163, 170, 175, 180, 185, 190, 195, 200, 210, 215, 220],
                    "cpu_utilization_bins": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    "cpu_idle_power": 163.0,
                    "accelerator_idle_power": 32.0,
                    "accelerator_max_power": 250.0
                }
            ],
            "available_resources": {
                1: {"cpu": 1500, "memory": 10000, "storage": 80, "network": 20, "accelerators": 0},
                2: {"cpu": 800, "memory": 10000, "storage": 80, "network": 20, "accelerators": 150}
            },
            "current_utilization": {
                1: {"cpu": 0.25, "memory": 0.22, "network": 0.3},
                2: {"cpu": 0.20, "memory": 0.22, "network": 0.3}
            }
        }],
        "task": {
            "task_id": "test_task_001",
            "application_id": 1,
            "implementation_id": 1,
            "num_vms": 4,
            "vcpus_per_vm": 8,
            "memory_per_vm": 32.0,
            "storage_per_vm": 0.1,
            "network_per_vm": 0.01,
            "requires_accelerator": False,
            "accelerator_utilization": 0.0,
            "estimated_duration": 3600.0
        }
    }

    try:
        resp = requests.post(
            f"{BASE_URL}/allocate_task",
            json=request_data,
            timeout=30
        )
        print(f"  Status: {resp.status_code}")
        result = resp.json()
        print(f"  Success: {result.get('success')}")
        print(f"  Method: {result.get('allocation_method')}")
        print(f"  Energy: {result.get('estimated_energy_cost')} kWh")
        print(f"  VMs allocated: {result.get('num_vms_allocated')}")
        if result.get('vm_allocations'):
            for vm in result['vm_allocations']:
                print(f"    VM {vm['vm_index']}: Cell {vm['cell_id']}, HW {vm['hw_type_id']}, Server {vm['server_index']}")
        return resp.status_code == 200 and result.get('success')
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_allocate_multi_impl():
    print("\nTesting /allocate_multi_impl endpoint...")

    request_data = {
        "timestamp": 200.0,
        "cells": [{
            "cell_id": 1,
            "hw_types": [
                {
                    "hw_type_id": 1,
                    "hw_type_name": "CPU-only",
                    "num_servers": 100,
                    "num_cpus_per_server": 20,
                    "memory_per_server": 128.0,
                    "storage_per_server": 1.0,
                    "compute_capability": 4400.0,
                    "accelerators": 0,
                    "num_accelerators_per_server": 0,
                    "accelerator_compute_capability": 0.0,
                    "cpu_power_consumption": [163, 170, 175, 180, 185, 190, 195, 200, 210, 215, 220],
                    "cpu_utilization_bins": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    "cpu_idle_power": 163.0,
                    "accelerator_idle_power": 0.0,
                    "accelerator_max_power": 0.0
                },
                {
                    "hw_type_id": 2,
                    "hw_type_name": "CPU+GPU",
                    "num_servers": 50,
                    "num_cpus_per_server": 20,
                    "memory_per_server": 256.0,
                    "storage_per_server": 2.0,
                    "compute_capability": 4400.0,
                    "accelerators": 1,
                    "num_accelerators_per_server": 4,
                    "accelerator_compute_capability": 587505.34,
                    "cpu_power_consumption": [163, 170, 175, 180, 185, 190, 195, 200, 210, 215, 220],
                    "cpu_utilization_bins": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    "cpu_idle_power": 163.0,
                    "accelerator_idle_power": 32.0,
                    "accelerator_max_power": 250.0
                }
            ],
            "available_resources": {
                1: {"cpu": 1500, "memory": 10000, "storage": 80, "network": 20, "accelerators": 0},
                2: {"cpu": 800, "memory": 10000, "storage": 80, "network": 20, "accelerators": 150}
            },
            "current_utilization": {
                1: {"cpu": 0.25, "memory": 0.22, "network": 0.3},
                2: {"cpu": 0.20, "memory": 0.22, "network": 0.3}
            }
        }],
        "application_id": 1,
        "task_id": "multi_impl_test_001",
        "implementations": [
            {
                "impl_id": 1,
                "impl_name": "CPU-only",
                "num_vms": 16,
                "vcpus_per_vm": 8,
                "memory_per_vm": 32.0,
                "storage_per_vm": 0.1,
                "network_per_vm": 0.01,
                "requires_accelerator": False,
                "accelerator_utilization": 0.0,
                "estimated_instructions": 1e11
            },
            {
                "impl_id": 2,
                "impl_name": "GPU-accelerated",
                "num_vms": 4,
                "vcpus_per_vm": 4,
                "memory_per_vm": 16.0,
                "storage_per_vm": 0.1,
                "network_per_vm": 0.01,
                "requires_accelerator": True,
                "accelerator_utilization": 0.9,
                "estimated_instructions": 1e11
            }
        ]
    }

    try:
        resp = requests.post(
            f"{BASE_URL}/allocate_multi_impl",
            json=request_data,
            timeout=30
        )
        print(f"  Status: {resp.status_code}")
        result = resp.json()
        print(f"  Success: {result.get('success')}")
        print(f"  Selected impl: {result.get('selected_impl_name')} (ID: {result.get('selected_impl_id')})")
        print(f"  Energy: {result.get('estimated_energy_wh')} Wh")
        print(f"  VMs allocated: {result.get('num_vms_allocated')}")

        if result.get('all_predictions'):
            print(f"\n  All predictions ({len(result['all_predictions'])}):")
            for pred in sorted(result['all_predictions'], key=lambda x: x['predicted_energy_wh']):
                print(f"    {pred['impl_name']} on Cell{pred['cell_id']}_HW{pred['hw_type_id']}: {pred['predicted_energy_wh']:.2f} Wh")

        if result.get('skipped_combinations'):
            print(f"\n  Skipped: {result['skipped_combinations']}")

        return resp.status_code == 200 and result.get('success')
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    print("=" * 60)
    print("SMART ALLOCATOR API INTEGRATION TEST")
    print("=" * 60)

    results = {
        "health": test_health(),
        "allocate_task": test_allocate_task(),
        "allocate_multi_impl": test_allocate_multi_impl()
    }

    print("\n" + "=" * 60)
    print("RESULTS:")
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())
    print("=" * 60)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
