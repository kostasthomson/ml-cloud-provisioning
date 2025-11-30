"""
Allocation logger to track and save all allocation decisions to JSON file.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from config import configuration
from entities import AllocationRequest, AllocationDecision

logger = logging.getLogger(__name__)


class AllocationLogger:
    """Logs and persists allocation decisions to JSON file."""

    def __init__(self, output_file: str = "allocation_decisions.json"):
        self.output_file = Path("output", configuration.model_type, output_file)
        self.decisions: List[Dict[str, Any]] = []
        self.request_count = 0
        self.success_count = 0
        self.rejection_count = 0

        # Ensure directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def log_decision(
            self,
            request: AllocationRequest,
            decision: AllocationDecision
    ) -> None:
        """
        Log an allocation decision with full context.

        Args:
            request: The original allocation request
            decision: The allocation decision made
        """
        self.request_count += 1

        if decision.success:
            self.success_count += 1
        else:
            self.rejection_count += 1

        # Build decision log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_timestamp": request.timestamp,
            "decision_number": self.request_count,
            "task": {
                "task_id": request.task.task_id,
                "application_id": request.task.application_id,
                "implementation_id": request.task.implementation_id,
                "num_vms": request.task.num_vms,
                "vcpus_per_vm": request.task.vcpus_per_vm,
                "memory_per_vm": request.task.memory_per_vm,
                "storage_per_vm": request.task.storage_per_vm,
                "network_per_vm": request.task.network_per_vm,
                "requires_accelerator": request.task.requires_accelerator,
                "accelerator_utilization": request.task.accelerator_utilization,
                "estimated_duration": request.task.estimated_duration
            },
            "system_state": {
                "num_cells": len(request.cells),
                "cells": [
                    {
                        "cell_id": cell.cell_id,
                        "num_hw_types": len(cell.hw_types),
                        "total_servers": sum(hw.num_servers for hw in cell.hw_types),
                    }
                    for cell in request.cells
                ]
            },
            "decision": str(decision)
        }

        self.decisions.append(log_entry)

        # Log to console as well
        logger.info(
            f"Logged decision #{self.request_count}: "
            f"Task {request.task.task_id} -> "
            f"{'SUCCESS' if decision.success else 'REJECTED'}"
        )

    def save_to_file(self) -> bool:
        """Save all logged decisions to JSON file."""
        try:
            output_data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "model": configuration.model_type,
                    "total_requests": self.request_count,
                    "successful_allocations": self.success_count,
                    "rejected_allocations": self.rejection_count,
                    "success_rate": (
                        self.success_count / self.request_count * 100
                        if self.request_count > 0 else 0.0
                    )
                },
                "decisions": self.decisions
            }

            with open(self.output_file, 'w') as f:
                json.dump(output_data, f, indent=2)

            logger.info(
                f"Saved {self.request_count} allocation decisions to {self.output_file}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save allocation log: {e}", exc_info=True)
            return False

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of logged decisions."""
        return {
            "total_requests": self.request_count,
            "successful_allocations": self.success_count,
            "rejected_allocations": self.rejection_count,
            "success_rate": (
                self.success_count / self.request_count * 100
                if self.request_count > 0 else 0.0
            )
        }

    def reset(self) -> None:
        self.decisions: List[Dict[str, Any]] = []
        self.request_count = 0
        self.success_count = 0
        self.rejection_count = 0
