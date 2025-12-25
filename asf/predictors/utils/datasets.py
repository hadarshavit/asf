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

        def __len__(self):
            return len(self.all.index.unique())

        def __getitem__(self, index):
            iid = self.all.index.unique()[index]
            data = self.all.loc[iid]

            main = np.random.randint(0, len(data))

            main_point = data.iloc[main]
            smaller = data[data["performance"] < main_point["performance"]]
            if len(smaller) == 0:
                smaller = main_point
            else:
                smaller = smaller.sample(1).iloc[0]
            larger = data[data["performance"] > main_point["performance"]]
            if len(larger) == 0:
                larger = main_point
            else:
                larger = larger.sample(1).iloc[0]

            main_feats = (
                main_point[self.algorithm_features_cols + self.features_cols]
                .to_numpy()
                .astype(np.float32)
            )
            smaller_feats = (
                smaller[self.algorithm_features_cols + self.features_cols]
                .to_numpy()
                .astype(np.float32)
            )
            larger_feats = (
                larger[self.algorithm_features_cols + self.features_cols]
                .to_numpy()
                .astype(np.float32)
            )

            main_feats = torch.tensor(main_feats).to(self._dtype)
            smaller_feats = torch.tensor(smaller_feats).to(self._dtype)
            larger_feats = torch.tensor(larger_feats).to(self._dtype)

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
