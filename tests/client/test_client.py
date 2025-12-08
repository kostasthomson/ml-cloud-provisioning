"""
Example test client for Smart Task Allocator API.
Demonstrates how to send allocation requests from Python.
"""
import requests
import json
from typing import Dict, Any


class AllocationClient:
    """Client for interacting with Smart Task Allocator API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize client with API base URL."""
        self.base_url = base_url
        self.session = requests.Session()

    def health_check(self) -> Dict[str, Any]:
        """Check if service is healthy."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def allocate_task(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Request task allocation.

        Args:
            request_data: Allocation request dictionary

        Returns:
            Allocation decision dictionary
        """
        response = self.session.post(
            f"{self.base_url}/allocate_task",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()

    def get_statistics(self) -> Dict[str, Any]:
        """Get allocation statistics."""
        response = self.session.get(f"{self.base_url}/statistics")
        response.raise_for_status()
        return response.json()


def create_example_request() -> Dict[str, Any]:
    """Create an example allocation request."""
    return {
        "timestamp": 100.0,
        "cells": [
            {
                "cell_id": 1,
                "hw_types": [
                    {
                        "hw_type_id": 1,
                        "hw_type_name": "CPU",
                        "num_servers": 25000,
                        "num_cpus_per_server": 20,
                        "memory_per_server": 128.0,
                        "storage_per_server": 1.0,
                        "compute_capability": 88000.8,
                        "accelerators": 0,
                        "num_accelerators_per_server": 0,
                        "accelerator_compute_capability": 0.0,
                        "cpu_power_consumption": [
                            163, 170.1, 172.6, 175.4, 179.8, 
                            183.6, 190.0, 196.8, 206.3, 215.9, 220.2
                        ],
                        "cpu_utilization_bins": [
                            0.0, 0.1, 0.2, 0.3, 0.4, 
                            0.5, 0.6, 0.7, 0.8, 0.9, 1.0
                        ],
                        "cpu_idle_power": 163.0,
                        "accelerator_idle_power": 0.0,
                        "accelerator_max_power": 0.0
                    },
                    {
                        "hw_type_id": 2,
                        "hw_type_name": "CPU+GPU",
                        "num_servers": 25000,
                        "num_cpus_per_server": 20,
                        "memory_per_server": 128.0,
                        "storage_per_server": 1.0,
                        "compute_capability": 88000.8,
                        "accelerators": 1,
                        "num_accelerators_per_server": 4,
                        "accelerator_compute_capability": 587505.34,
                        "cpu_power_consumption": [
                            163, 170.1, 172.6, 175.4, 179.8, 
                            183.6, 190.0, 196.8, 206.3, 215.9, 220.2
                        ],
                        "cpu_utilization_bins": [
                            0.0, 0.1, 0.2, 0.3, 0.4, 
                            0.5, 0.6, 0.7, 0.8, 0.9, 1.0
                        ],
                        "cpu_idle_power": 163.0,
                        "accelerator_idle_power": 32.0,
                        "accelerator_max_power": 250.0
                    }
                ],
                "available_resources": {
                    "1": {
                        "cpu": 500000.0,
                        "memory": 3200000.0,
                        "storage": 25000.0,
                        "network": 20.0,
                        "accelerators": 0
                    },
                    "2": {
                        "cpu": 500000.0,
                        "memory": 3200000.0,
                        "storage": 25000.0,
                        "network": 20.0,
                        "accelerators": 100000
                    }
                },
                "current_utilization": {
                    "1": {"cpu": 0.0, "memory": 0.0, "network": 0.0},
                    "2": {"cpu": 0.0, "memory": 0.0, "network": 0.0}
                }
            }
        ],
        "task": {
            "task_id": "task_test_001",
            "application_id": 1,
            "implementation_id": 1,  # CPU implementation
            "num_vms": 2,
            "vcpus_per_vm": 4,
            "memory_per_vm": 8.0,
            "storage_per_vm": 0.02,
            "network_per_vm": 0.0025,
            "requires_accelerator": False,
            "accelerator_utilization": 0.0,
            "estimated_duration": 3600.0
        }
    }


def main():
    """Run example test."""
    print("Smart Task Allocator - Test Client")
    print("=" * 50)

    # Create client
    client = AllocationClient()

    # 1. Health check
    print("\n1. Checking service health...")
    try:
        health = client.health_check()
        print(f"   [OK] Service is healthy")
        print(f"     Version: {health['version']}")
        print(f"     Model: {health['model_type']}")
    except Exception as e:
        print(f"   [ERROR] Health check failed: {e}")
        return

    # 2. Test allocation request
    print("\n2. Sending allocation request...")
    request = create_example_request()

    try:
        decision = client.allocate_task(request)
        print(f"   [OK] Allocation decision received:")
        print(f"     Success: {decision['success']}")
        if decision['success']:
            print(f"     VMs Allocated: {decision['num_vms_allocated']}")
            print(f"     Energy Cost: {decision['estimated_energy_cost']:.4f} kWh")
            print(f"     Reason: {decision['reason']}")
            print(f"     VM Allocations:")
            for vm_alloc in decision['vm_allocations']:
                print(f"       - VM {vm_alloc['vm_index']}: Cell {vm_alloc['cell_id']}, "
                      f"HW Type {vm_alloc['hw_type_id']}, Server {vm_alloc['server_index']}")
        else:
            print(f"     Reason: {decision['reason']}")
    except Exception as e:
        print(f"   [ERROR] Allocation request failed: {e}")
        return

    # 3. Get statistics
    print("\n3. Retrieving statistics...")
    try:
        stats = client.get_statistics()
        print(f"   [OK] Statistics:")
        print(f"     {json.dumps(stats['statistics'], indent=6)}")
    except Exception as e:
        print(f"   [ERROR] Statistics retrieval failed: {e}")

    print("\n" + "=" * 50)
    print("Test completed successfully!")


if __name__ == "__main__":
    main()
