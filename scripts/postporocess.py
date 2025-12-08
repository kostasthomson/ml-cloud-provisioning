import pandas as pd
import numpy as np
from pathlib import Path
from typing import List


class UnifiedEnergyPostProcessor:
    """Processes decisions from multiple brokers into unified dataset"""

    def __init__(self, power_models=None):
        self.power_models = power_models or {
            1: {'P_min': 50, 'P_max': 200},
            2: {'P_min': 75, 'P_max': 300},
            3: {'P_min': 100, 'P_max': 400},
            4: {'P_min': 150, 'P_max': 500}
        }

    def compute_energy(self, row):
        """Compute energy for single task"""
        hw_type = row['chosen_hw_type']
        cpu_util = row['util_cpu_before']
        mem_util = row['util_mem_before']

        if hw_type not in self.power_models:
            return 0.0, 0.0

        u = np.clip(max(cpu_util, mem_util), 0.0, 1.0)
        P_min = self.power_models[hw_type]['P_min']
        P_max = self.power_models[hw_type]['P_max']

        power_W = P_min + (P_max - P_min) * u
        duration_sec = 10.0  # Placeholder
        energy_kWh = (power_W * duration_sec) / (1000.0 * 3600.0)

        return energy_kWh, duration_sec

    def process_broker(self, broker_name: str, base_path: str = '../training/data'):
        """Process decisions from a single broker"""
        input_path = Path(base_path) / broker_name / 'decisions.csv'

        print(f"\\nProcessing {broker_name}...")

        if not input_path.exists():
            print(f"  WARNING: {input_path} not found, skipping")
            return None

        df = pd.read_csv(input_path)
        print(f"  Loaded {len(df)} decisions ({df['accepted'].sum()} accepted)")

        # Filter accepted tasks
        df_accepted = df[df['accepted'] == 1].copy()

        if len(df_accepted) == 0:
            print(f"  WARNING: No accepted tasks, skipping")
            return None

        # Add broker label (for analysis, not training)
        df_accepted['source_broker'] = broker_name

        # Compute energy
        results = df_accepted.apply(self.compute_energy, axis=1)
        df_accepted['energy_kwh'] = results.apply(lambda x: x[0])
        df_accepted['processing_time_sec'] = results.apply(lambda x: x[1])

        print(f"  ✓ Processed {len(df_accepted)} tasks")
        print(f"    Total energy: {df_accepted['energy_kwh'].sum():.4f} kWh")

        return df_accepted

    def process_all_brokers(self,
                            brokers: List[str] = None,
                            base_path: str = '../training/data',
                            output_path: str = '../training/training_data.csv'):
        """
        Combine decisions from all brokers into unified dataset

        Args:
            brokers: List of broker names to process
            base_path: Base directory containing broker outputs
            output_path: Where to save unified dataset

        Returns:
            Combined DataFrame
        """
        if brokers is None:
            brokers = ['traditional', 'sosm', 'improved']

        print("=" * 80)
        print("UNIFIED ENERGY POST-PROCESSING")
        print("=" * 80)

        all_data = []

        for broker in brokers:
            df_broker = self.process_broker(broker, base_path)
            if df_broker is not None:
                all_data.append(df_broker)

        if not all_data:
            print("\\nERROR: No data processed from any broker!")
            return None

        # Combine all broker data
        print(f"\\n{'=' * 80}")
        print("COMBINING DATASETS")
        print("=" * 80)

        df_unified = pd.concat(all_data, ignore_index=True)

        print(f"\\nUnified Dataset Statistics:")
        print(f"  Total tasks: {len(df_unified)}")
        print(f"  Total energy: {df_unified['energy_kwh'].sum():.4f} kWh")
        print(f"  Mean energy per task: {df_unified['energy_kwh'].mean():.6f} kWh")

        print(f"\\nData Distribution by Broker:")
        for broker in brokers:
            count = len(df_unified[df_unified['source_broker'] == broker])
            pct = 100 * count / len(df_unified)
            print(f"  {broker:15s}: {count:6d} ({pct:5.1f}%)")

        print(f"\\nData Distribution by HW Type:")
        for hw_type in sorted(df_unified['chosen_hw_type'].unique()):
            count = len(df_unified[df_unified['chosen_hw_type'] == hw_type])
            pct = 100 * count / len(df_unified)
            energy = df_unified[df_unified['chosen_hw_type'] == hw_type]['energy_kwh'].sum()
            print(f"  HW Type {hw_type}: {count:6d} ({pct:5.1f}%) | Energy: {energy:.4f} kWh")

        # Save unified dataset
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_unified.to_csv(output_path, index=False)

        print(f"\\n✓ Unified dataset saved to: {output_path}")
        print(f"  Ready for ML training!")

        return df_unified


def main():
    """Command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Unified post-processing for all broker decisions'
    )
    parser.add_argument(
        '--base-path',
        type=str,
        default='../training/data',
        help='Base directory containing broker outputs (default: ../output)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../training/training_data.csv',
        help='Output path for unified dataset'
    )
    parser.add_argument(
        '--brokers',
        nargs='+',
        default=['traditional', 'sosm', 'improved'],
        help='List of brokers to process'
    )

    args = parser.parse_args()

    processor = UnifiedEnergyPostProcessor()
    df_unified = processor.process_all_brokers(
        brokers=args.brokers,
        base_path=args.base_path,
        output_path=args.output
    )

    if df_unified is not None:
        print(f"\\n{'=' * 80}")
        print("READY FOR TRAINING")
        print("=" * 80)
        print(f"Use: python train_agent.py --data {args.output}")

    return args.output


if __name__ == "__main__":
    main()
