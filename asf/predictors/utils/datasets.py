try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import pandas as pd
import numpy as np

if TORCH_AVAILABLE:

    class RegressionDataset(torch.utils.data.Dataset):
        def __init__(self, features, performance, dtype=torch.float32):
            if isinstance(features, pd.DataFrame) and isinstance(
                performance, (pd.DataFrame, pd.Series)
            ):
                self.features = torch.from_numpy(features.sort_index().to_numpy()).to(
                    dtype
                )
                self.performance = torch.from_numpy(
                    performance.sort_index().to_numpy()
                ).to(dtype)
            else:
                self.features = torch.tensor(features, dtype=dtype)
                self.performance = torch.tensor(performance, dtype=dtype)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return self.features[idx], self.performance[idx]

    class RankingDataset(torch.utils.data.Dataset):
        def __init__(
            self,
            features: pd.DataFrame,
            performance: pd.DataFrame,
            algorithm_features: pd.DataFrame,
            dtype=torch.float32,
        ):
            performance = performance.melt(
                ignore_index=False, var_name="algo", value_name="performance"
            )
            all = features.merge(performance, left_index=True, right_index=True)
            all = all.merge(algorithm_features, left_on="algo", right_index=True)
            all = all.sort_index()
            self.all = all

            self.features_cols = features.columns.to_list()
            self.algorithm_features_cols = algorithm_features.columns.to_list()

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

            main_feats = torch.tensor(main_feats).to(torch.float32)
            smaller_feats = torch.tensor(smaller_feats).to(torch.float32)
            larger_feats = torch.tensor(larger_feats).to(torch.float32)

            return (main_feats, smaller_feats, larger_feats), (
                main_point["performance"],
                smaller["performance"],
                larger["performance"],
            )
