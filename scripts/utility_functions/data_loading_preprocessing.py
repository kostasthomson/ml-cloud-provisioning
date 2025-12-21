import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(data_path, drop_incomplete=True):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records")

    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        print(f"Found {nan_count} NaN values")
        if drop_incomplete:
            before = len(df)
            df = df.dropna()
            print(f"Dropped {before - len(df)} rows, {len(df)} remaining")
        else:
            df = df.fillna(0)

    df_filtered = df[df['accepted'] == 1].copy()
    df_filtered = df_filtered[df_filtered['chosen_hw_type'] > 0].reset_index(drop=True)
    df_filtered = df_filtered[df_filtered['energy_kwh'] > 0].reset_index(drop=True)
    print(f"Valid records: {len(df_filtered)}")

    task_features = ['num_vms', 'cpu_req', 'mem_req']

    optional_task = ['storage_req', 'network_req', 'acc_req', 'rho_acc', 'requested_instructions']
    for feat in optional_task:
        if feat in df_filtered.columns:
            task_features.append(feat)

    hw_state_features = [
        'util_cpu_before', 'util_mem_before',
        'avail_cpu_before', 'avail_mem_before', 'avail_storage_before', 'avail_accelerators_before',
        'total_cpu', 'total_mem', 'total_storage', 'total_accelerators',
        'avail_network', 'total_network', 'util_network',
        'cpu_idle_power', 'cpu_max_power', 'acc_idle_power', 'acc_max_power',
        'compute_cap_per_cpu', 'compute_cap_acc',
        'running_vms', 'overcommit_cpu', 'overcommit_mem', 'num_tasks'
    ]

    available_hw_features = [f for f in hw_state_features if f in df_filtered.columns]

    df_filtered['total_cpu_req'] = df_filtered['num_vms'] * df_filtered['cpu_req']
    df_filtered['total_mem_req'] = df_filtered['num_vms'] * df_filtered['mem_req']

    derived = ['total_cpu_req', 'total_mem_req']

    if 'storage_req' in df_filtered.columns:
        df_filtered['total_storage_req'] = df_filtered['num_vms'] * df_filtered['storage_req']
        derived.append('total_storage_req')

    if 'cpu_idle_power' in df_filtered.columns and 'cpu_max_power' in df_filtered.columns:
        df_filtered['power_range'] = df_filtered['cpu_max_power'] - df_filtered['cpu_idle_power']
        derived.append('power_range')

    if 'avail_cpu_before' in df_filtered.columns and 'total_cpu' in df_filtered.columns:
        df_filtered['cpu_util_ratio'] = df_filtered['avail_cpu_before'] / (df_filtered['total_cpu'] + 1e-6)
        derived.append('cpu_util_ratio')

    all_features = task_features + available_hw_features + derived

    print(f"Task features: {len(task_features)}")
    print(f"HW state features: {len(available_hw_features)}")
    print(f"Derived features: {len(derived)}")
    print(f"Total features: {len(all_features)}")

    X = df_filtered[all_features].values
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

    y = df_filtered['energy_kwh'].values * 1000.0
    y = np.clip(y, 0, np.percentile(y, 99))

    print(f"Feature matrix shape: {X.shape}")
    print(f"Energy target (Wh): min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}")

    hw_type = df_filtered['chosen_hw_type'].values
    print(f"HW type distribution:")
    for t in range(1, 5):
        count = np.sum(hw_type == t)
        if count > 0:
            avg_energy = y[hw_type == t].mean()
            print(f"  Type {t}: {count} samples, avg energy {avg_energy:.6f} kWh")

    return X, y, df_filtered, all_features


def normalize_features(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
