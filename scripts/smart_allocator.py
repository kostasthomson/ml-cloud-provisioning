import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class EnergyAwareNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(EnergyAwareNN, self).__init__()
        self.input_size = input_size
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size // 4, output_size)
        )

    def forward(self, x):
        return self.network(x)


class SmartProvisioningAgent:
    def __init__(self, model_path, training_data_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_names = self._get_feature_names()
        self.scaler = self._fit_scaler(training_data_path)
        self.model = self._load_model(model_path)
        print(f"Agent initialized with {len(self.feature_names)} features on {self.device}")

    def _get_feature_names(self):
        return [
            'num_vms', 'cpu_req', 'mem_req', 'storage_req', 'network_req',
            'acc_req', 'rho_acc', 'requested_instructions',
            'util_cpu_before', 'util_mem_before',
            'avail_cpu_before', 'avail_mem_before', 'avail_storage_before', 'avail_accelerators_before',
            'total_cpu', 'total_mem', 'total_storage', 'total_accelerators',
            'avail_network', 'total_network', 'util_network',
            'cpu_idle_power', 'cpu_max_power', 'acc_idle_power', 'acc_max_power',
            'compute_cap_per_cpu', 'compute_cap_acc',
            'running_vms', 'overcommit_cpu', 'overcommit_mem', 'num_tasks',
            'total_cpu_req', 'total_mem_req', 'total_storage_req',
            'power_range', 'cpu_util_ratio'
        ]

    def _fit_scaler(self, training_data_path):
        print(f"Fitting scaler from {training_data_path}...")
        df = pd.read_csv(training_data_path)
        df = df[df['accepted'] == 1].copy()
        df = df[df['chosen_hw_type'] > 0].reset_index(drop=True)
        df = df[df['energy_kwh'] > 0].reset_index(drop=True)

        df['total_cpu_req'] = df['num_vms'] * df['cpu_req']
        df['total_mem_req'] = df['num_vms'] * df['mem_req']
        df['total_storage_req'] = df['num_vms'] * df['storage_req']
        df['power_range'] = df['cpu_max_power'] - df['cpu_idle_power']
        df['cpu_util_ratio'] = df['avail_cpu_before'] / (df['total_cpu'] + 1e-6)

        X = df[self.feature_names].values
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

        scaler = StandardScaler()
        scaler.fit(X)
        print(f"Scaler fitted on {len(X)} samples")
        return scaler

    def _load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        input_size = checkpoint.get('input_size', len(self.feature_names))
        hidden_size = checkpoint.get('hidden_size', 128)

        model = EnergyAwareNN(input_size, hidden_size, output_size=1)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        print(f"Model loaded: {input_size} inputs, {hidden_size} hidden")
        return model

    def _build_feature_vector(self, impl, hw_state):
        features = {
            'num_vms': impl['num_vms'],
            'cpu_req': impl['cpu_req'],
            'mem_req': impl['mem_req'],
            'storage_req': impl.get('storage_req', 0.1),
            'network_req': impl.get('network_req', 0.01),
            'acc_req': impl.get('acc_req', 0),
            'rho_acc': impl.get('rho_acc', 0.0),
            'requested_instructions': impl.get('requested_instructions', 1e9),
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
        features['total_storage_req'] = features['num_vms'] * features['storage_req']
        features['power_range'] = features['cpu_max_power'] - features['cpu_idle_power']
        features['cpu_util_ratio'] = features['avail_cpu_before'] / (features['total_cpu'] + 1e-6)

        vector = np.array([features[name] for name in self.feature_names], dtype=np.float32)
        return vector

    def predict_energy(self, feature_vector):
        X = self.scaler.transform(feature_vector.reshape(1, -1))
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            energy_wh = self.model(X_tensor).item()
        return max(0, energy_wh)

    def get_best_allocation(self, candidate_implementations, system_state_metadata):
        all_predictions = []
        skipped = []

        for impl_idx, impl in enumerate(candidate_implementations):
            impl_needs_accelerator = impl.get('acc_req', 0) > 0

            for hw_id, hw_state in system_state_metadata.items():
                hw_has_accelerator = hw_state.get('total_accelerators', 0) > 0

                if impl_needs_accelerator and not hw_has_accelerator:
                    skipped.append((impl_idx, hw_id, 'no_accelerator'))
                    continue

                feature_vector = self._build_feature_vector(impl, hw_state)
                energy = self.predict_energy(feature_vector)

                all_predictions.append({
                    'impl_idx': impl_idx,
                    'impl_name': impl.get('name', f'impl_{impl_idx}'),
                    'hw_id': hw_id,
                    'hw_name': hw_state.get('name', hw_id),
                    'energy': energy,
                })

        if not all_predictions:
            return None, None, None, [], skipped

        best = min(all_predictions, key=lambda x: x['energy'])
        return best['impl_idx'], best['hw_id'], best['energy'], all_predictions, skipped


def main():
    model_path = Path(__file__).parent.parent / "models" / "energy_aware" / "nn" / "model.pth"
    training_data_path = Path(__file__).parent.parent / "training" / "training_data.csv"

    print("=" * 60)
    print("SMART PROVISIONING AGENT - MULTI-IMPLEMENTATION DEMO")
    print("=" * 60)

    agent = SmartProvisioningAgent(str(model_path), str(training_data_path))

    system_state_metadata = {
        'hw_cpu': {
            'name': 'CPU-only',
            'total_accelerators': 0,
            'avail_accelerators': 0,
            'acc_idle_power': 0.0,
            'acc_max_power': 0.0,
            'compute_cap_acc': 0.0,
            'cpu_idle_power': 163.0,
            'cpu_max_power': 220.0,
            'compute_cap_per_cpu': 4400.0,
            'util_cpu': 0.4,
            'util_mem': 0.35,
            'avail_cpu': 150000,
            'avail_mem': 800000,
            'avail_storage': 8000,
            'total_cpu': 240000,
            'total_mem': 1200000,
            'total_storage': 10000,
            'avail_network': 25,
            'total_network': 40,
            'util_network': 0.375,
            'running_vms': 500,
            'num_tasks': 200,
        },
        'hw_gpu': {
            'name': 'CPU+GPU',
            'total_accelerators': 32000,
            'avail_accelerators': 25000,
            'acc_idle_power': 32.0,
            'acc_max_power': 250.0,
            'compute_cap_acc': 587505.34,
            'cpu_idle_power': 163.0,
            'cpu_max_power': 220.0,
            'compute_cap_per_cpu': 4400.0,
            'util_cpu': 0.35,
            'util_mem': 0.30,
            'avail_cpu': 160000,
            'avail_mem': 850000,
            'avail_storage': 8500,
            'total_cpu': 240000,
            'total_mem': 1200000,
            'total_storage': 10000,
            'avail_network': 30,
            'total_network': 40,
            'util_network': 0.25,
            'running_vms': 400,
            'num_tasks': 150,
        },
        'hw_dfe': {
            'name': 'CPU+DFE',
            'total_accelerators': 32000,
            'avail_accelerators': 28000,
            'acc_idle_power': 176.8,
            'acc_max_power': 225.0,
            'compute_cap_acc': 507494.4,
            'cpu_idle_power': 163.0,
            'cpu_max_power': 220.0,
            'compute_cap_per_cpu': 4400.0,
            'util_cpu': 0.25,
            'util_mem': 0.20,
            'avail_cpu': 180000,
            'avail_mem': 900000,
            'avail_storage': 9000,
            'total_cpu': 240000,
            'total_mem': 1200000,
            'total_storage': 10000,
            'avail_network': 35,
            'total_network': 40,
            'util_network': 0.125,
            'running_vms': 200,
            'num_tasks': 80,
        },
        'hw_mic': {
            'name': 'CPU+MIC',
            'total_accelerators': 16000,
            'avail_accelerators': 14000,
            'acc_idle_power': 17.3,
            'acc_max_power': 25.1,
            'compute_cap_acc': 295959.3,
            'cpu_idle_power': 163.0,
            'cpu_max_power': 220.0,
            'compute_cap_per_cpu': 4400.0,
            'util_cpu': 0.30,
            'util_mem': 0.25,
            'avail_cpu': 170000,
            'avail_mem': 870000,
            'avail_storage': 8700,
            'total_cpu': 240000,
            'total_mem': 1200000,
            'total_storage': 10000,
            'avail_network': 32,
            'total_network': 40,
            'util_network': 0.20,
            'running_vms': 300,
            'num_tasks': 120,
        },
    }

    print("\n--- Test: Matrix Multiplication with 3 Implementations ---")
    matrix_mult_implementations = [
        {
            'name': 'CPU-only',
            'num_vms': 16,
            'cpu_req': 32,
            'mem_req': 64,
            'storage_req': 0.1,
            'network_req': 0.01,
            'acc_req': 0,
            'rho_acc': 0.0,
            'requested_instructions': 1e11,
        },
        {
            'name': 'GPU-accelerated',
            'num_vms': 4,
            'cpu_req': 8,
            'mem_req': 32,
            'storage_req': 0.1,
            'network_req': 0.01,
            'acc_req': 1,
            'rho_acc': 0.9,
            'requested_instructions': 1e11,
        },
        {
            'name': 'DFE-optimized',
            'num_vms': 2,
            'cpu_req': 4,
            'mem_req': 16,
            'storage_req': 0.1,
            'network_req': 0.01,
            'acc_req': 1,
            'rho_acc': 0.95,
            'requested_instructions': 1e11,
        },
    ]

    best_impl, best_hw, best_energy, all_preds, skipped = agent.get_best_allocation(
        matrix_mult_implementations, system_state_metadata
    )

    print("Application: Matrix Multiplication")
    print(f"Available implementations: {len(matrix_mult_implementations)}")
    print(f"Available hardware types: {len(system_state_metadata)}")

    if skipped:
        print(f"\nSkipped combinations: {len(skipped)}")
        for impl_idx, hw_id, reason in skipped[:5]:
            impl_name = matrix_mult_implementations[impl_idx].get('name', f'impl_{impl_idx}')
            print(f"  {impl_name} on {hw_id}: {reason}")

    print("\nAll valid predictions (sorted by energy):")
    for pred in sorted(all_preds, key=lambda x: x['energy']):
        marker = " <-- BEST" if pred['impl_idx'] == best_impl and pred['hw_id'] == best_hw else ""
        print(f"  {pred['impl_name']} on {pred['hw_name']}: {pred['energy']:.4f} Wh{marker}")

    print(f"\nOptimal choice:")
    print(f"  Implementation: {matrix_mult_implementations[best_impl]['name']} (index {best_impl})")
    print(f"  Hardware: {system_state_metadata[best_hw]['name']} ({best_hw})")
    print(f"  Predicted energy: {best_energy:.4f} Wh")

    print("\n--- Test: ML Training with 2 Implementations ---")
    ml_training_implementations = [
        {
            'name': 'CPU-distributed',
            'num_vms': 32,
            'cpu_req': 16,
            'mem_req': 128,
            'storage_req': 1.0,
            'network_req': 0.1,
            'acc_req': 0,
            'rho_acc': 0.0,
            'requested_instructions': 5e11,
        },
        {
            'name': 'GPU-accelerated',
            'num_vms': 4,
            'cpu_req': 8,
            'mem_req': 256,
            'storage_req': 2.0,
            'network_req': 0.05,
            'acc_req': 1,
            'rho_acc': 0.95,
            'requested_instructions': 5e11,
        },
    ]

    best_impl, best_hw, best_energy, all_preds, skipped = agent.get_best_allocation(
        ml_training_implementations, system_state_metadata
    )

    print("Application: ML Model Training")
    print("\nAll valid predictions (sorted by energy):")
    for pred in sorted(all_preds, key=lambda x: x['energy']):
        marker = " <-- BEST" if pred['impl_idx'] == best_impl and pred['hw_id'] == best_hw else ""
        print(f"  {pred['impl_name']} on {pred['hw_name']}: {pred['energy']:.4f} Wh{marker}")

    print(f"\nOptimal choice:")
    print(f"  Implementation: {ml_training_implementations[best_impl]['name']} (index {best_impl})")
    print(f"  Hardware: {system_state_metadata[best_hw]['name']} ({best_hw})")
    print(f"  Predicted energy: {best_energy:.4f} Wh")

    print("\n" + "=" * 60)
    print("Demo complete. Agent supports multi-implementation optimization.")
    print("=" * 60)


if __name__ == "__main__":
    main()
