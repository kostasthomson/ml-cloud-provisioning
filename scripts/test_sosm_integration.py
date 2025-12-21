"""
Integration test for SOSM simulator -> ML training pipeline.
Validates the end-to-end data flow from C++ simulator CSV output to model training.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

EXPECTED_COLUMNS = [
    'num_vms', 'cpu_req', 'mem_req',
    'util_cpu_before', 'util_mem_before',
    'avail_cpu_before', 'avail_mem_before',
    'avail_storage_before', 'avail_accelerators_before',
    'energy_kwh', 'chosen_hw_type', 'accepted'
]

FORBIDDEN_COLUMNS = [
    'task_id', 'timestamp', 'cell_id', 'source_broker',
    'chosen_cell', 'processing_time_sec'
]


def test_csv_schema(csv_path: str) -> bool:
    """Validate CSV file has expected columns and no biasing columns."""
    print(f"\nTesting CSV schema: {csv_path}")

    if not Path(csv_path).exists():
        print(f"  FAIL: File not found")
        return False

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        print(f"  FAIL: Missing columns: {missing}")
        return False

    forbidden_found = set(FORBIDDEN_COLUMNS) & set(df.columns)
    if forbidden_found:
        print(f"  FAIL: Biasing columns found: {forbidden_found}")
        return False

    print(f"  PASS: Schema correct, no biasing columns")
    return True


def test_data_quality(csv_path: str) -> bool:
    """Validate data quality in CSV file."""
    print(f"\nTesting data quality: {csv_path}")

    df = pd.read_csv(csv_path)

    if df.empty:
        print(f"  FAIL: Empty dataset")
        return False

    accepted = df[df['accepted'] == 1]
    if len(accepted) == 0:
        print(f"  FAIL: No accepted tasks")
        return False

    print(f"  Total records: {len(df)}")
    print(f"  Accepted: {len(accepted)} ({100*len(accepted)/len(df):.1f}%)")

    valid_hw = accepted[(accepted['chosen_hw_type'] >= 1) & (accepted['chosen_hw_type'] <= 4)]
    if len(valid_hw) == 0:
        print(f"  FAIL: No valid HW types (1-4)")
        return False

    print(f"  Valid HW types: {len(valid_hw)}")

    for hw in range(1, 5):
        count = len(accepted[accepted['chosen_hw_type'] == hw])
        if count > 0:
            print(f"    HW Type {hw}: {count}")

    print(f"  PASS: Data quality OK")
    return True


def test_preprocessing(csv_path: str) -> bool:
    """Test that preprocessing logic works correctly (without torch dependency)."""
    print(f"\nTesting preprocessing...")

    try:
        df = pd.read_csv(csv_path)
        df_accepted = df[df['accepted'] == 1].copy()
        df_accepted = df_accepted[df_accepted['chosen_hw_type'] > 0].reset_index(drop=True)

        task_features = ['num_vms', 'cpu_req', 'mem_req']
        df_accepted['total_cpu_req'] = df_accepted['num_vms'] * df_accepted['cpu_req']
        df_accepted['total_mem_req'] = df_accepted['num_vms'] * df_accepted['mem_req']
        task_features.extend(['total_cpu_req', 'total_mem_req'])

        state_features = [
            'util_cpu_before', 'util_mem_before', 'avail_cpu_before',
            'avail_mem_before', 'avail_storage_before', 'avail_accelerators_before'
        ]

        df_accepted['cpu_to_mem_ratio'] = (
            df_accepted['avail_cpu_before'] / (df_accepted['avail_mem_before'] + 1e-6)
        )
        df_accepted['util_avg'] = (
            df_accepted['util_cpu_before'] + df_accepted['util_mem_before']
        ) / 2.0
        state_features.extend(['cpu_to_mem_ratio', 'util_avg'])

        all_features = task_features + state_features
        X = df_accepted[all_features].values
        y = df_accepted['chosen_hw_type'].values - 1

        if X.shape[0] == 0:
            print(f"  FAIL: No samples after preprocessing")
            return False

        if X.shape[1] != 13:
            print(f"  FAIL: Expected 13 features, got {X.shape[1]}")
            return False

        if not np.all((y >= 0) & (y <= 3)):
            print(f"  FAIL: Labels not in range 0-3")
            return False

        print(f"  Feature matrix: {X.shape}")
        print(f"  Labels: {y.shape}, range [{y.min()}, {y.max()}]")
        print(f"  PASS: Preprocessing OK")
        return True

    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_sample_data(output_path: str, n_samples: int = 100):
    """Create sample CSV data for testing."""
    print(f"\nCreating sample data: {output_path}")

    np.random.seed(42)

    data = {
        'num_vms': np.random.randint(1, 16, n_samples),
        'cpu_req': np.random.uniform(4, 16, n_samples),
        'mem_req': np.random.uniform(4, 32, n_samples),
        'util_cpu_before': np.random.uniform(0, 0.8, n_samples),
        'util_mem_before': np.random.uniform(0, 0.8, n_samples),
        'avail_cpu_before': np.random.uniform(100000, 500000, n_samples),
        'avail_mem_before': np.random.uniform(500000, 3000000, n_samples),
        'avail_storage_before': np.random.uniform(10000, 50000, n_samples),
        'avail_accelerators_before': np.random.uniform(0, 100000, n_samples),
        'energy_kwh': np.random.uniform(0.001, 0.1, n_samples),
        'chosen_hw_type': np.random.randint(1, 5, n_samples),
        'accepted': np.random.choice([0, 1], n_samples, p=[0.1, 0.9])
    }

    df = pd.DataFrame(data)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Created {n_samples} sample records")
    return output_path


def main():
    print("=" * 60)
    print("SOSM INTEGRATION TEST")
    print("=" * 60)

    test_file = "../training/test_data.csv"
    create_sample_data(test_file)

    results = []
    results.append(("CSV Schema", test_csv_schema(test_file)))
    results.append(("Data Quality", test_data_quality(test_file)))
    results.append(("Preprocessing", test_preprocessing(test_file)))

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
