from __future__ import annotations

import pandas as pd
import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:

    class RegressionDataset(torch.utils.data.Dataset):
        def __init__(self, features, performance, dtype=None):
            if dtype is None:
                dtype = torch.float32
            if hasattr(features, "sort_index"):
                features = features.sort_index()
            if hasattr(performance, "sort_index"):
                performance = performance.sort_index()

            features_np = (
                features.to_numpy()
                if hasattr(features, "to_numpy")
                else np.asarray(features)
            )
            performance_np = (
                performance.to_numpy()
                if hasattr(performance, "to_numpy")
                else np.asarray(performance)
            )

            self.features = torch.from_numpy(features_np).to(dtype)
            self.performance = torch.from_numpy(performance_np).to(dtype)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, index):
            return self.features[index], self.performance[index]

    class RankingDataset(torch.utils.data.Dataset):
        """
        Dataset for ranking-based training.

        Samples at the (dataset, algorithm) level to match ZAP HPO.
        Each sample consists of a triplet: (main, smaller, larger) where
        smaller and larger are algorithms with lower/higher performance
        on the same dataset.
        """

        def __init__(
            self,
            features: pd.DataFrame,
            performance: pd.DataFrame,
            algorithm_features: pd.DataFrame,
            dtype=None,
        ):
            if dtype is None:
                dtype = torch.float32
            performance = performance.melt(
                ignore_index=False, var_name="algo", value_name="performance"
            )
            all_df = features.merge(performance, left_index=True, right_index=True)
            all_df = all_df.merge(algorithm_features, left_on="algo", right_index=True)
            all_df = all_df.sort_index()
            self.all = all_df

            self.features_cols = features.columns.to_list()
            self.algorithm_features_cols = algorithm_features.columns.to_list()
            self._dtype = dtype

            # Build index mapping for efficient access
            # Each entry is (instance_id, row_position_within_instance)
            self._index_map = []
            self._instance_data = {}
            for iid in self.all.index.unique():
                instance_data = self.all.loc[iid]
                if isinstance(instance_data, pd.Series):
                    # Single row case - convert to DataFrame
                    instance_data = instance_data.to_frame().T
                self._instance_data[iid] = instance_data
                for row_idx in range(len(instance_data)):
                    self._index_map.append((iid, row_idx))

        def __len__(self):
            # Return total number of (dataset, algorithm) pairs
            return len(self._index_map)

        def __getitem__(self, index):
            iid, row_idx = self._index_map[index]
            data = self._instance_data[iid]

            # The main point is the specific (dataset, algorithm) pair
            main_point = data.iloc[row_idx]
            main_perf = main_point["performance"]

            # Find algorithms with smaller performance on the same dataset
            smaller_mask = data["performance"] < main_perf
            if smaller_mask.any():
                smaller = data[smaller_mask].sample(1).iloc[0]
            else:
                smaller = main_point

            # Find algorithms with larger performance on the same dataset
            larger_mask = data["performance"] > main_perf
            if larger_mask.any():
                larger = data[larger_mask].sample(1).iloc[0]
            else:
                larger = main_point

            # Extract features
            cols = self.algorithm_features_cols + self.features_cols
            main_feats = torch.tensor(
                main_point[cols].to_numpy().astype(np.float32)
            ).to(self._dtype)
            smaller_feats = torch.tensor(
                smaller[cols].to_numpy().astype(np.float32)
            ).to(self._dtype)
            larger_feats = torch.tensor(larger[cols].to_numpy().astype(np.float32)).to(
                self._dtype
            )

            return (main_feats, smaller_feats, larger_feats), (
                main_point["performance"],
                smaller["performance"],
                larger["performance"],
            )
else:
    # Use Any to silence type checker complaining about union types
    from typing import Any

    class RegressionDataset(Any):  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "PyTorch is not installed. Install it with: pip install torch"
            )

    class RankingDataset(Any):  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "PyTorch is not installed. Install it with: pip install torch"
            )
