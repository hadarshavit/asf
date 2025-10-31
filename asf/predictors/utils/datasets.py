from __future__ import annotations

try:
    import torch

    TORCH_AVAILABLE = True
    DatasetBase = torch.utils.data.Dataset
except ImportError:
    TORCH_AVAILABLE = False

    class DatasetBase:  # minimal stub to allow class definition
        pass


import pandas as pd
import numpy as np


class RegressionDataset(DatasetBase):
    def __init__(self, features, performance, dtype=None):
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed. Install it with: pip install torch"
            )
        if dtype is None:
            dtype = torch.float32
        self.features = torch.from_numpy(features.sort_index().to_numpy()).to(dtype)
        self.performance = torch.from_numpy(performance.sort_index().to_numpy()).to(
            dtype
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.performance[idx]


class RankingDataset(DatasetBase):
    def __init__(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame,
        algorithm_features: pd.DataFrame,
        dtype=None,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed. Install it with: pip install torch"
            )
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

    def __getitem__(self, idx):
        iid = self.all.index.unique()[idx]
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
