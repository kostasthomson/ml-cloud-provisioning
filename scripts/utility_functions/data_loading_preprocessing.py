import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(data_path):
    """Load and preprocess decision data from C++ simulator output."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")

    df_accepted = df[df['accepted'] == 1].copy()
    df_accepted = df_accepted.reset_index(drop=True)

    valid_hw = df_accepted['chosen_hw_type'] > 0
    df_accepted = df_accepted[valid_hw].reset_index(drop=True)

    print(f"Accepted tasks: {len(df_accepted)}")

    task_features = ['num_vms', 'cpu_req', 'mem_req']

    df_accepted['total_cpu_req'] = df_accepted['num_vms'] * df_accepted['cpu_req']
    df_accepted['total_mem_req'] = df_accepted['num_vms'] * df_accepted['mem_req']
    task_features.extend(['total_cpu_req', 'total_mem_req'])

    state_features = [
        'util_cpu_before',
        'util_mem_before',
        'avail_cpu_before',
        'avail_mem_before',
        'avail_storage_before',
        'avail_accelerators_before'
    ]

    df_accepted['cpu_to_mem_ratio'] = (
        df_accepted['avail_cpu_before'] / (df_accepted['avail_mem_before'] + 1e-6)
    )
    df_accepted['util_avg'] = (
        df_accepted['util_cpu_before'] + df_accepted['util_mem_before']
    ) / 2.0
    state_features.extend(['cpu_to_mem_ratio', 'util_avg'])

    all_features = task_features + state_features

    print(f"Task features ({len(task_features)}): {task_features}")
    print(f"State features ({len(state_features)}): {state_features}")
    print(f"Total features: {len(all_features)}")

    X = df_accepted[all_features].values
    y = df_accepted['chosen_hw_type'].values - 1

    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution:")
    for hw_type in range(4):
        count = np.sum(y == hw_type)
        if count > 0:
            print(f"  HW Type {hw_type + 1}: {count} ({100 * count / len(y):.1f}%)")

    return X, y, df_accepted


def normalize_features(X_train, X_val, X_test):
    """Normalize features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
