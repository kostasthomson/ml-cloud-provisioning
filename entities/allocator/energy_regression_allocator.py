import logging
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Optional, Tuple, List, Dict

from entities import AllocationRequest, VMAllocation, CellStatus, HardwareType
from models import EnergyAwareNN
from . import BaseAllocator

logger = logging.getLogger(__name__)


class EnergyRegressionAllocator(BaseAllocator):
    def __init__(self, model_path: str = None, training_data_path: str = None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base_path = Path(__file__).parent.parent.parent
        if model_path is None:
            model_path = str(base_path / "models" / "energy_aware" / "nn" / "model.pth")
        if training_data_path is None:
            training_data_path = str(base_path / "training" / "training_data.csv")

        self.model_path = model_path
        self.training_data_path = training_data_path
        self.feature_names = []
        self.scaler = self._fit_scaler()
        self.model = self._load_model()

        logger.info(f"EnergyRegressionAllocator initialized on {self.device}")
        logger.info(f"Features: {len(self.feature_names)}")

    def get_method_name(self) -> str:
        return "energy_regression"

    def _get_candidate_features(self):
        return {
            'task': ['num_vms', 'cpu_req', 'mem_req'],
            'task_optional': ['storage_req', 'network_req', 'acc_req', 'rho_acc', 'requested_instructions'],
            'hw_state': [
                'util_cpu_before', 'util_mem_before',
                'avail_cpu_before', 'avail_mem_before', 'avail_storage_before', 'avail_accelerators_before',
                'total_cpu', 'total_mem', 'total_storage', 'total_accelerators',
                'avail_network', 'total_network', 'util_network',
                'cpu_idle_power', 'cpu_max_power', 'acc_idle_power', 'acc_max_power',
                'compute_cap_per_cpu', 'compute_cap_acc',
                'running_vms', 'overcommit_cpu', 'overcommit_mem', 'num_tasks'
            ],
            'derived': ['total_cpu_req', 'total_mem_req', 'total_storage_req', 'power_range', 'cpu_util_ratio']
        }

    def _fit_scaler(self):
        logger.info(f"Fitting scaler from {self.training_data_path}...")
        df = pd.read_csv(self.training_data_path)
        df = df[df['accepted'] == 1].copy()
        df = df[df['chosen_hw_type'] > 0].reset_index(drop=True)
        df = df[df['energy_kwh'] > 0].reset_index(drop=True)

        candidates = self._get_candidate_features()
        available_features = []

        for feat in candidates['task']:
            if feat in df.columns:
                available_features.append(feat)

        for feat in candidates['task_optional']:
            if feat in df.columns:
                available_features.append(feat)

        for feat in candidates['hw_state']:
            if feat in df.columns:
                available_features.append(feat)

        df['total_cpu_req'] = df['num_vms'] * df['cpu_req']
        df['total_mem_req'] = df['num_vms'] * df['mem_req']
        available_features.extend(['total_cpu_req', 'total_mem_req'])

        if 'storage_req' in df.columns:
            df['total_storage_req'] = df['num_vms'] * df['storage_req']
            available_features.append('total_storage_req')

        if 'cpu_idle_power' in df.columns and 'cpu_max_power' in df.columns:
            df['power_range'] = df['cpu_max_power'] - df['cpu_idle_power']
            available_features.append('power_range')

        if 'avail_cpu_before' in df.columns and 'total_cpu' in df.columns:
            df['cpu_util_ratio'] = df['avail_cpu_before'] / (df['total_cpu'] + 1e-6)
            available_features.append('cpu_util_ratio')

        self.feature_names = available_features
        logger.info(f"Using {len(self.feature_names)} features: {self.feature_names}")

        X = df[self.feature_names].values
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

        scaler = StandardScaler()
        scaler.fit(X)
        logger.info(f"Scaler fitted on {len(X)} samples")
        return scaler

    def _load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        input_size = checkpoint.get('input_size', len(self.feature_names))
        hidden_size = checkpoint.get('hidden_size', 128)

        model = EnergyAwareNN(input_size, hidden_size, output_size=1)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        logger.info(f"Model loaded: {input_size} inputs, {hidden_size} hidden")
        return model

    def _extract_hw_state(self, cell: CellStatus, hw: HardwareType) -> Dict:
        hw_id = hw.hw_type_id
        available = cell.available_resources.get(hw_id, {})
        utilization = cell.current_utilization.get(hw_id, {})

        total_cpu = hw.num_servers * hw.num_cpus_per_server
        total_mem = hw.num_servers * hw.memory_per_server
        total_storage = hw.num_servers * hw.storage_per_server
        total_accelerators = hw.num_servers * hw.num_accelerators_per_server if hw.accelerators else 0

        avail_cpu = available.get('cpu', total_cpu * 0.7)
        avail_mem = available.get('memory', total_mem * 0.7)
        avail_storage = available.get('storage', total_storage * 0.8)
        avail_accelerators = available.get('accelerators', 0)
        avail_network = available.get('network', 20)

        util_cpu = utilization.get('cpu', 0.3)
        util_mem = utilization.get('memory', 0.3)
        util_network = utilization.get('network', 0.3)

        return {
            'cell_id': cell.cell_id,
            'hw_type_id': hw_id,
            'name': hw.hw_type_name,
            'total_accelerators': total_accelerators,
            'avail_accelerators': avail_accelerators,
            'acc_idle_power': hw.accelerator_idle_power,
            'acc_max_power': hw.accelerator_max_power,
            'compute_cap_acc': hw.accelerator_compute_capability,
            'cpu_idle_power': hw.cpu_idle_power,
            'cpu_max_power': hw.cpu_power_consumption[-1] if hw.cpu_power_consumption else 220.0,
            'compute_cap_per_cpu': hw.compute_capability,
            'util_cpu': util_cpu,
            'util_mem': util_mem,
            'avail_cpu': avail_cpu,
            'avail_mem': avail_mem,
            'avail_storage': avail_storage,
            'total_cpu': total_cpu,
            'total_mem': total_mem,
            'total_storage': total_storage,
            'avail_network': avail_network,
            'total_network': 40,
            'util_network': util_network,
            'running_vms': 0,
            'num_tasks': 0,
            'overcommit_cpu': 1.0,
            'overcommit_mem': 1.0,
        }

    def _build_feature_vector(self, task_impl: Dict, hw_state: Dict) -> np.ndarray:
        features = {
            'num_vms': task_impl['num_vms'],
            'cpu_req': task_impl['cpu_req'],
            'mem_req': task_impl['mem_req'],
            'storage_req': task_impl.get('storage_req', 0.1),
            'network_req': task_impl.get('network_req', 0.01),
            'acc_req': task_impl.get('acc_req', 0),
            'rho_acc': task_impl.get('rho_acc', 0.0),
            'requested_instructions': task_impl.get('requested_instructions', 1e9),
            'util_cpu_before': hw_state.get('util_cpu', 0.3),
            'util_mem_before': hw_state.get('util_mem', 0.3),
            'avail_cpu_before': hw_state.get('avail_cpu', 100000),
            'avail_mem_before': hw_state.get('avail_mem', 500000),
            'avail_storage_before': hw_state.get('avail_storage', 5000),
            'avail_accelerators_before': hw_state.get('avail_accelerators', 0),
            'total_cpu': hw_state.get('total_cpu', 200000),
            'total_mem': hw_state.get('total_mem', 1000000),
            'total_storage': hw_state.get('total_storage', 10000),
            'total_accelerators': hw_state.get('total_accelerators', 0),
            'avail_network': hw_state.get('avail_network', 20),
            'total_network': hw_state.get('total_network', 30),
            'util_network': hw_state.get('util_network', 0.3),
            'cpu_idle_power': hw_state.get('cpu_idle_power', 163.0),
            'cpu_max_power': hw_state.get('cpu_max_power', 220.0),
            'acc_idle_power': hw_state.get('acc_idle_power', 0.0),
            'acc_max_power': hw_state.get('acc_max_power', 0.0),
            'compute_cap_per_cpu': hw_state.get('compute_cap_per_cpu', 4400.0),
            'compute_cap_acc': hw_state.get('compute_cap_acc', 0.0),
            'running_vms': hw_state.get('running_vms', 0),
            'overcommit_cpu': hw_state.get('overcommit_cpu', 1.0),
            'overcommit_mem': hw_state.get('overcommit_mem', 1.0),
            'num_tasks': hw_state.get('num_tasks', 0),
        }

        features['total_cpu_req'] = features['num_vms'] * features['cpu_req']
        features['total_mem_req'] = features['num_vms'] * features['mem_req']
        if 'total_storage_req' in self.feature_names:
            features['total_storage_req'] = features['num_vms'] * features.get('storage_req', 0.1)
        if 'power_range' in self.feature_names:
            features['power_range'] = features['cpu_max_power'] - features['cpu_idle_power']
        if 'cpu_util_ratio' in self.feature_names:
            features['cpu_util_ratio'] = features['avail_cpu_before'] / (features['total_cpu'] + 1e-6)

        vector = np.array([features.get(name, 0.0) for name in self.feature_names], dtype=np.float32)
        return vector

    def _predict_energy(self, feature_vector: np.ndarray) -> float:
        X = self.scaler.transform(feature_vector.reshape(1, -1))
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            energy_wh = self.model(X_tensor).item()
        return max(0, energy_wh)

    def _task_to_impl(self, task) -> Dict:
        return {
            'num_vms': task.num_vms,
            'cpu_req': task.vcpus_per_vm,
            'mem_req': task.memory_per_vm,
            'storage_req': task.storage_per_vm,
            'network_req': task.network_per_vm,
            'acc_req': 1 if task.requires_accelerator else 0,
            'rho_acc': task.accelerator_utilization,
            'requested_instructions': task.estimated_duration * 1e9 if task.estimated_duration else 1e9,
        }

    def _perform_allocation(self, request: AllocationRequest) -> Optional[Tuple[List[VMAllocation], float, str]]:
        task = request.task
        task_impl = self._task_to_impl(task)
        task_needs_accelerator = task.requires_accelerator

        all_candidates = []

        for cell in request.cells:
            for hw in cell.hw_types:
                hw_state = self._extract_hw_state(cell, hw)
                hw_has_accelerator = hw_state.get('total_accelerators', 0) > 0

                if task_needs_accelerator and not hw_has_accelerator:
                    continue

                if not self._has_sufficient_resources(task, hw, cell):
                    continue

                feature_vector = self._build_feature_vector(task_impl, hw_state)
                predicted_energy = self._predict_energy(feature_vector)

                all_candidates.append({
                    'cell': cell,
                    'hw': hw,
                    'hw_state': hw_state,
                    'energy': predicted_energy,
                })

        if not all_candidates:
            logger.warning(f"No valid allocation candidates for task {task.task_id}")
            return None

        best = min(all_candidates, key=lambda x: x['energy'])
        target_cell = best['cell']
        target_hw = best['hw']
        energy_cost = best['energy'] / 1000.0

        vm_allocations = self._allocate_vms_to_servers(task, target_cell, target_hw)

        if not vm_allocations:
            logger.warning("VM allocation failed despite sufficient aggregate resources")
            return None

        reason = f"Energy-optimal: Cell{target_cell.cell_id}_HW{target_hw.hw_type_id}_{best['energy']:.2f}Wh"

        logger.info(
            f"Energy Regression Allocation: Cell {target_cell.cell_id}, "
            f"HW Type {target_hw.hw_type_id} ({target_hw.hw_type_name}), "
            f"{len(vm_allocations)} VMs, Energy={best['energy']:.2f} Wh"
        )

        return vm_allocations, energy_cost, reason

    def allocate_multi_impl(self, cells: List[CellStatus], implementations: List) -> Dict:
        all_predictions = []
        skipped = []

        for impl in implementations:
            impl_dict = {
                'num_vms': impl.num_vms,
                'cpu_req': impl.vcpus_per_vm,
                'mem_req': impl.memory_per_vm,
                'storage_req': impl.storage_per_vm,
                'network_req': impl.network_per_vm,
                'acc_req': 1 if impl.requires_accelerator else 0,
                'rho_acc': impl.accelerator_utilization,
                'requested_instructions': impl.estimated_instructions,
            }
            impl_needs_accelerator = impl.requires_accelerator

            for cell in cells:
                for hw in cell.hw_types:
                    hw_state = self._extract_hw_state(cell, hw)
                    hw_has_accelerator = hw_state.get('total_accelerators', 0) > 0

                    if impl_needs_accelerator and not hw_has_accelerator:
                        skipped.append(f"{impl.impl_name} on Cell{cell.cell_id}_HW{hw.hw_type_id}: no_accelerator")
                        continue

                    feature_vector = self._build_feature_vector(impl_dict, hw_state)
                    predicted_energy = self._predict_energy(feature_vector)

                    all_predictions.append({
                        'impl': impl,
                        'cell': cell,
                        'hw': hw,
                        'hw_state': hw_state,
                        'energy': predicted_energy,
                    })

        if not all_predictions:
            return {
                'success': False,
                'best': None,
                'all_predictions': [],
                'skipped': skipped,
            }

        best = min(all_predictions, key=lambda x: x['energy'])

        return {
            'success': True,
            'best': best,
            'all_predictions': all_predictions,
            'skipped': skipped,
        }
