import logging
import pickle

import numpy as np
import torch

from entities import AllocationRequest
from models.nn.neural_network import NeuralNetwork
from . import BaseAllocator

logger = logging.getLogger(__name__)


class NNAllocator(BaseAllocator):
    def __init__(self, path: str):
        super().__init__()
        self.model_path = f"{path}/model.pth"
        self.scaler_path = f"{path}/scaler.pkl"
        self.model = None
        self.scaler = None
        self._load_model()

    def get_method_name(self) -> str:
        return "ml_energy_aware"

    def _load_model(self):
        """Load trained model and scaler from disk."""
        try:
            # Load Scaler
            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

            # Infer input dimensions from scaler
            input_dim = self.scaler.mean_.shape[0]

            # 4 classes for HW Types 1-4 (stored as 0-3 in model)
            num_classes = 4

            # Initialize model architecture (must match training!)
            self.model = NeuralNetwork(input_dim, num_classes)
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            self.model.eval()

            logger.info(f"  ML Model loaded: {input_dim} features → {num_classes} classes")
            logger.info(f"  Model: {self.model_path}")
            logger.info(f"  Scaler: {self.scaler_path}")

        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")
            logger.error("Run train_agent.py first to generate the model!")
            raise
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            raise

    def _extract_features(self, request: AllocationRequest) -> np.ndarray:
        """
        Extract features in EXACT same order as training data.

        Feature vector (75 dimensions):
          - 3 task features: num_vms, cpu_req, mem_req
          - 72 system state features: 3 cells × 4 HW types × 6 metrics

        Returns: Scaled feature array ready for model input
        """
        task = request.task

        # Task features (must match training column order!)
        task_feats = [
            float(task.num_vms),
            float(task.vcpus_per_vm),
            float(task.memory_per_vm)
        ]

        # System state features
        EXPECTED_NUM_CELLS = 3
        EXPECTED_NUM_HW_TYPES = 4

        # Build lookup: (cell_id, hw_type_id) -> (hw_object, cell_object)
        hw_lookup = {}
        for cell in request.cells:
            for hw in cell.hw_types:
                hw_lookup[(cell.cell_id, hw.hw_type_id)] = (hw, cell)

        # Build features with DESCRIPTIVE column names (matches CSV)
        feature_dict = {}

        for cell_id in range(1, EXPECTED_NUM_CELLS + 1):
            for hw_id in range(1, EXPECTED_NUM_HW_TYPES + 1):

                if (cell_id, hw_id) in hw_lookup:
                    hw, cell = hw_lookup[(cell_id, hw_id)]
                    available = cell.available_resources.get(hw_id, {})

                    avail_cpu = available.get('cpu', 0)
                    avail_mem = available.get('memory', 0)
                    avail_stor = available.get('storage', 0)
                    avail_acc = available.get('accelerators', 0)

                    total_cpu = hw.num_servers * hw.num_cpus_per_server
                    total_mem = hw.num_servers * hw.memory_per_server

                    # Utilization ratios (0-1)
                    util_cpu = 1.0 - (avail_cpu / (total_cpu + 1e-9))
                    util_mem = 1.0 - (avail_mem / (total_mem + 1e-9))

                    # Build features (order matters!)
                    feature_dict[f'cell{cell_id}_hw{hw_id}_util_cpu'] = util_cpu
                    feature_dict[f'cell{cell_id}_hw{hw_id}_util_mem'] = util_mem
                    feature_dict[f'cell{cell_id}_hw{hw_id}_avail_cpu'] = avail_cpu
                    feature_dict[f'cell{cell_id}_hw{hw_id}_avail_mem'] = avail_mem
                    feature_dict[f'cell{cell_id}_hw{hw_id}_avail_storage'] = avail_stor
                    feature_dict[f'cell{cell_id}_hw{hw_id}_avail_accelerators'] = avail_acc
                else:
                    # Padding for missing HW types (saturated utilization, zero capacity)
                    feature_dict[f'cell{cell_id}_hw{hw_id}_util_cpu'] = 1.0
                    feature_dict[f'cell{cell_id}_hw{hw_id}_util_mem'] = 1.0
                    feature_dict[f'cell{cell_id}_hw{hw_id}_avail_cpu'] = 0.0
                    feature_dict[f'cell{cell_id}_hw{hw_id}_avail_mem'] = 0.0
                    feature_dict[f'cell{cell_id}_hw{hw_id}_avail_storage'] = 0.0
                    feature_dict[f'cell{cell_id}_hw{hw_id}_avail_accelerators'] = 0.0

        # Combine features in EXACT training order
        full_dict = {}
        full_dict['num_vms'] = task_feats[0]
        full_dict['cpu_req'] = task_feats[1]
        full_dict['mem_req'] = task_feats[2]
        full_dict.update(feature_dict)

        # Convert to numpy array and scale
        feature_array = np.array(list(full_dict.values())).reshape(1, -1)
        scaled_features = self.scaler.transform(feature_array)

        return scaled_features

    def _perform_allocation(self, request: AllocationRequest):
        """
        Perform ML-based allocation.

        Steps:
          1. Extract features from current system state
          2. Run inference to predict optimal HW Type (0-3)
          3. Remap prediction to actual HW Type ID (1-4)
          4. Find cell with sufficient resources of that type
          5. Allocate VMs and return result
        """

        # Step 1: Run inference
        try:
            input_features = self._extract_features(request)
            input_tensor = torch.FloatTensor(input_features)

            with torch.no_grad():
                logits, pred_energy = self.model(input_tensor)
                probs = torch.softmax(logits, dim=1)

                # Model outputs class 0-3
                predicted_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0][predicted_class].item()

        except Exception as e:
            logger.error(f"ML inference failed: {e}", exc_info=True)
            return None

        # Step 2: REMAP model output (0-3) → actual HW Type ID (1-4)
        predicted_hw_type_id = predicted_class + 1

        logger.debug(
            f"ML Prediction: Class={predicted_class} → HW Type {predicted_hw_type_id} "
            f"(confidence={confidence:.2%}, energy={pred_energy.item():.2f})"
        )

        # Step 3: Find cell with this HW type and sufficient resources
        target_hw_type = None
        target_cell = None

        for cell in request.cells:
            for hw in cell.hw_types:
                # Match against actual HW Type ID (1-4)
                if hw.hw_type_id == predicted_hw_type_id:
                    if self._has_sufficient_resources(request.task, hw, cell):
                        target_hw_type = hw
                        target_cell = cell
                        break
            if target_cell:
                break

        # Step 4: Handle allocation failure
        if not target_cell or not target_hw_type:
            logger.warning(
                f"ML predicted HW Type {predicted_hw_type_id} but no resources available. "
                f"Task requires: {request.task.num_vms} VMs × "
                f"{request.task.vcpus_per_vm} vCPUs × {request.task.memory_per_vm} GB"
            )
            return None

        # Step 5: Allocate VMs to servers
        vm_allocations = self._allocate_vms_to_servers(
            request.task, target_cell, target_hw_type
        )

        if not vm_allocations:
            logger.warning("VM allocation failed despite sufficient aggregate resources")
            return None

        # Step 6: Calculate actual energy cost (don't trust NN prediction for logging)
        energy_cost = self._estimate_energy_cost(request.task, target_hw_type, target_cell)

        logger.info(
            f"  ML Allocation: Cell {target_cell.cell_id}, HW Type {predicted_hw_type_id}, "
            f"{len(vm_allocations)} VMs, Energy={energy_cost:.2f} Wh (conf={confidence:.1%})"
        )

        return vm_allocations, energy_cost, f"ML_HW{target_hw_type.hw_type_id}_C{confidence:.2f}"