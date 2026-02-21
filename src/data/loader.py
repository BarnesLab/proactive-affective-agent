"""Data loader: loads all data types (EMA, sensing, baseline, processed).

Provides unified access to all data sources with consistent formatting.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.mappings import SENSING_COLUMNS, study_id_to_participant_id


class DataLoader:
    """Loads and caches all data types from the data directory."""

    def __init__(
        self,
        data_dir: Path | None = None,
        cancer_survival_dir: Path | None = None,
    ) -> None:
        self.data_dir = data_dir or Path("data")
        self.processed_dir = self.data_dir / "processed"
        self.splits_dir = self.processed_dir / "splits"
        self.memory_dir = self.processed_dir / "memory_documents"

        # Raw data from cancer_survival (large files, not in repo)
        self.cancer_dir = cancer_survival_dir or Path.home() / "Documents" / "cancer_survival"
        self.sensing_dir = self.cancer_dir / "Passive Sensing Data"
        self.ema_dir = self.cancer_dir / "EMA Data"
        self.baseline_dir = self.cancer_dir / "Baseline Data (self report)"

        # Caches
        self._sensing_cache: dict[str, pd.DataFrame] = {}
        self._baseline_cache: pd.DataFrame | None = None

    def load_split(self, group: int, split: str = "train") -> pd.DataFrame:
        """Load a train/test split file.

        Args:
            group: Fold number (1-5).
            split: "train" or "test".

        Returns:
            DataFrame with EMA data + computed state columns.
        """
        path = self.splits_dir / f"group_{group}_{split}.csv"
        df = pd.read_csv(path)
        df["timestamp_local"] = pd.to_datetime(df["timestamp_local"])
        df["date_local"] = pd.to_datetime(df["date_local"]).dt.date
        return df

    def load_all_ema(self) -> pd.DataFrame:
        """Load ALL EMA data by combining all 5 test splits.

        The 5 test splits are non-overlapping and together cover all 399 users.
        Returns a single DataFrame with ~15,984 rows.
        """
        dfs = []
        for group in range(1, 6):
            path = self.splits_dir / f"group_{group}_test.csv"
            if path.exists():
                df = pd.read_csv(path)
                dfs.append(df)
        if not dfs:
            raise FileNotFoundError("No test split files found in " + str(self.splits_dir))
        combined = pd.concat(dfs, ignore_index=True)
        combined["timestamp_local"] = pd.to_datetime(combined["timestamp_local"])
        combined["date_local"] = pd.to_datetime(combined["date_local"]).dt.date
        # Drop duplicates (should be none, but just in case)
        combined = combined.drop_duplicates(subset=["Study_ID", "timestamp_local"])
        return combined

    def load_all_train(self) -> pd.DataFrame:
        """Load ALL training data by combining all 5 train splits, deduplicated.

        Used for the TF-IDF retriever — each user's entries appear in 4 of 5
        training folds, so we deduplicate to get the full unique training set.
        """
        dfs = []
        for group in range(1, 6):
            path = self.splits_dir / f"group_{group}_train.csv"
            if path.exists():
                df = pd.read_csv(path)
                dfs.append(df)
        if not dfs:
            raise FileNotFoundError("No train split files found")
        combined = pd.concat(dfs, ignore_index=True)
        combined["timestamp_local"] = pd.to_datetime(combined["timestamp_local"])
        combined["date_local"] = pd.to_datetime(combined["date_local"]).dt.date
        combined = combined.drop_duplicates(subset=["Study_ID", "timestamp_local"])
        return combined

    def load_daily_ema(self) -> pd.DataFrame:
        """Load daily EMA survey data from cancer_survival directory."""
        path = self.ema_dir / "dailyEMAdata_cleaned_long_10-25-2024.csv"
        df = pd.read_csv(path)
        df["timestamp_local"] = pd.to_datetime(df["timestamp_local"])
        df["date_local"] = pd.to_datetime(df["date_local"]).dt.date
        return df

    def load_weekly_ema(self) -> pd.DataFrame:
        """Load weekly EMA survey data."""
        path = self.ema_dir / "weeklyEMAdata_cleaned_long_10-25-2024.csv"
        return pd.read_csv(path)

    def load_sensing(self, sensor_name: str) -> pd.DataFrame:
        """Load sensing data for a specific sensor.

        Returns DataFrame keyed by (id_participant, dt_feature).
        """
        if sensor_name in self._sensing_cache:
            return self._sensing_cache[sensor_name]

        info = SENSING_COLUMNS[sensor_name]
        path = self.sensing_dir / info["file"]
        df = pd.read_csv(path)
        df[info["date_col"]] = pd.to_datetime(df[info["date_col"]]).dt.date
        # Normalize id_participant to zero-padded 3-char string
        df[info["id_col"]] = df[info["id_col"]].astype(str).str.strip('"').str.zfill(3)

        self._sensing_cache[sensor_name] = df
        return df

    def load_all_sensing(self) -> dict[str, pd.DataFrame]:
        """Load all 8 sensing data files. Returns {sensor_name: DataFrame}."""
        result = {}
        for sensor_name in SENSING_COLUMNS:
            try:
                result[sensor_name] = self.load_sensing(sensor_name)
            except FileNotFoundError:
                pass  # Skip missing sensor files
        return result

    def load_baseline(self) -> pd.DataFrame:
        """Load baseline trait questionnaire data, indexed by Study_ID."""
        if self._baseline_cache is not None:
            return self._baseline_cache

        path = self.baseline_dir / "BUCS_traitdata_cleaned_11-21-2024.csv"
        df = pd.read_csv(path)
        df = df.set_index("Study_ID")
        self._baseline_cache = df
        return df

    def load_memory_for_user(self, study_id: int) -> str | None:
        """Load the pre-generated memory document for a user.

        Tries user_{study_id}_memory.txt first (without zero-padding).
        """
        # Try plain memory first, then with_interventions
        for suffix in ["_memory.txt", "_memory_with_interventions.txt"]:
            path = self.memory_dir / f"user_{study_id}{suffix}"
            if path.exists():
                return path.read_text(encoding="utf-8")

        # Also check cancer_survival memory_documents
        cs_mem_dir = self.cancer_dir / "memory_documents"
        if cs_mem_dir.exists():
            for suffix in ["_memory.txt", "_memory_with_interventions.txt"]:
                path = cs_mem_dir / f"user_{study_id}{suffix}"
                if path.exists():
                    return path.read_text(encoding="utf-8")

        return None

    def load_memory_documents(self) -> dict[str, str]:
        """Load all pre-generated memory documents. Returns {filename: content}."""
        docs = {}
        if self.memory_dir.exists():
            for f in self.memory_dir.glob("*.txt"):
                docs[f.name] = f.read_text(encoding="utf-8")
        return docs

    def get_user_ids(self) -> list[int]:
        """Get list of all Study_IDs from the combined test data."""
        try:
            df = self.load_all_ema()
            return sorted(df["Study_ID"].unique().tolist())
        except FileNotFoundError:
            pass
        # Fallback: eligible IDs
        path = self.cancer_dir / "eligibleIDs.csv"
        if path.exists():
            ids = pd.read_csv(path)
            return sorted(ids["Study_ID"].tolist())
        return []

    def get_sensing_for_user_date(
        self, study_id: int, target_date, sensing_dfs: dict[str, pd.DataFrame] | None = None
    ) -> dict[str, pd.DataFrame]:
        """Get all sensing data for a user on a specific date.

        Args:
            study_id: User's Study_ID (int).
            target_date: The date to query.
            sensing_dfs: Pre-loaded sensing DataFrames (optional, avoids re-loading).

        Returns:
            {sensor_name: DataFrame rows for that user+date}
        """
        pid = study_id_to_participant_id(study_id)
        if sensing_dfs is None:
            sensing_dfs = self.load_all_sensing()

        result = {}
        for sensor_name, df in sensing_dfs.items():
            info = SENSING_COLUMNS[sensor_name]
            mask = (df[info["id_col"]] == pid) & (df[info["date_col"]] == target_date)
            matched = df[mask]
            if not matched.empty:
                result[sensor_name] = matched
        return result
