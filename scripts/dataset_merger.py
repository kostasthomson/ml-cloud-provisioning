import pandas as pd
from pathlib import Path
from typing import List, Optional
import glob


class CellDatasetMerger:
    """Merges per-cell decision CSV files from C++ simulator into unified training dataset."""

    def __init__(self, data_dir: str = "../training/simulation_runs"):
        self.data_dir = Path(data_dir)

    def find_all_csv_files(self) -> List[Path]:
        """Find all CSV files across all configuration directories."""
        pattern = self.data_dir / "*" / "cell_*_decisions.csv"
        files = list(glob.glob(str(pattern)))
        return [Path(f) for f in sorted(files)]

    def merge_config_directory(self, config_dir: Path) -> Optional[pd.DataFrame]:
        """Merge all CSV files from a single configuration directory."""
        csv_files = list(config_dir.glob("cell_*_decisions.csv"))

        if not csv_files:
            return None

        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                dfs.append(df)

        if not dfs:
            return None

        return pd.concat(dfs, ignore_index=True)

    def process_all_configs(self, output_path: str = "../training/training_data.csv") -> Optional[pd.DataFrame]:
        """Process all configuration directories and create unified training dataset."""
        print("=" * 60)
        print("MULTI-CONFIG DATASET MERGER (BIAS-FREE)")
        print("=" * 60)

        if not self.data_dir.exists():
            print(f"ERROR: Data directory not found: {self.data_dir}")
            return None

        config_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]

        if not config_dirs:
            print(f"ERROR: No configuration directories found in {self.data_dir}")
            return None

        print(f"Found {len(config_dirs)} configuration directories")

        all_data = []
        for config_dir in sorted(config_dirs):
            config_name = config_dir.name
            df = self.merge_config_directory(config_dir)

            if df is not None and len(df) > 0:
                print(f"  {config_name}: {len(df)} records")
                all_data.append(df)
            else:
                print(f"  {config_name}: No data")

        if not all_data:
            print("ERROR: No data found!")
            return None

        df_unified = pd.concat(all_data, ignore_index=True)
        df_unified = df_unified.sample(frac=1, random_state=42).reset_index(drop=True)

        df_accepted = df_unified[df_unified['accepted'] == 1].copy()

        print(f"\n{'=' * 60}")
        print("DATASET STATISTICS")
        print("=" * 60)
        print(f"Total records: {len(df_unified)}")
        print(f"Accepted: {len(df_accepted)} ({100*len(df_accepted)/len(df_unified):.1f}%)")

        if len(df_accepted) > 0:
            print(f"\nHW Type Distribution (accepted):")
            for hw_type in sorted(df_accepted['chosen_hw_type'].unique()):
                if hw_type > 0:
                    count = len(df_accepted[df_accepted['chosen_hw_type'] == hw_type])
                    print(f"  HW Type {hw_type}: {count} ({100*count/len(df_accepted):.1f}%)")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_unified.to_csv(output_path, index=False)

        print(f"\nSaved to: {output_path}")
        print(f"Columns (bias-free): {list(df_unified.columns)}")

        return df_unified


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Merge CSV files from multiple simulation runs')
    parser.add_argument('--data-dir', type=str,
                        default='../training/simulation_runs',
                        help='Directory containing simulation run subdirectories')
    parser.add_argument('--output', type=str,
                        default='../training/training_data.csv',
                        help='Output path for merged dataset')

    args = parser.parse_args()

    merger = CellDatasetMerger(args.data_dir)
    df = merger.process_all_configs(output_path=args.output)

    if df is not None:
        print(f"\nReady for training: python train_agent.py")

    return args.output


if __name__ == "__main__":
    main()
