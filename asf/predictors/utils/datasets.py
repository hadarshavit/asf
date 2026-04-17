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
            device: str = "cpu",
        ):
            if dtype is None:
                dtype = torch.float32
            self._device = device
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

            # Convert to tensors for fast access, optionally on GPU
            cols = self.algorithm_features_cols + self.features_cols
            self._all_features_tensor = torch.tensor(
                self.all[cols].to_numpy().astype(np.float32)
            ).to(dtype=self._dtype, device=self._device)
            self._all_perf_tensor = torch.tensor(
                self.all["performance"].to_numpy().astype(np.float32)
            ).to(dtype=self._dtype, device=self._device)

            # Pre-calculate indices for smaller/larger per dataset
            self._num_samples = len(self.all)
            self._smaller_indices = [[] for _ in range(self._num_samples)]
            self._larger_indices = [[] for _ in range(self._num_samples)]

            # Group indices by instance_id
            indices_per_iid = {}
            for i, iid in enumerate(self.all.index):
                if iid not in indices_per_iid:
                    indices_per_iid[iid] = []
                indices_per_iid[iid].append(i)

            for iid, i_list in indices_per_iid.items():
                perfs = self._all_perf_tensor[i_list]
                for rel_idx, abs_idx in enumerate(i_list):
                    p = perfs[rel_idx]
                    
                    # smaller
                    s_rel = (perfs < p).nonzero().flatten()
                    if len(s_rel) > 0:
                        self._smaller_indices[abs_idx] = [i_list[j] for j in s_rel.tolist()]
                    
                    # larger
                    l_rel = (perfs > p).nonzero().flatten()
                    if len(l_rel) > 0:
                        self._larger_indices[abs_idx] = [i_list[j] for j in l_rel.tolist()]

        def __len__(self):
            # Return total number of (dataset, algorithm) pairs
            return self._num_samples

        def __getitem__(self, index):
            # The main point is the specific (dataset, algorithm) pair
            main_feats = self._all_features_tensor[index]
            main_perf = self._all_perf_tensor[index]

            # Find algorithms with smaller performance on the same dataset
            if self._smaller_indices[index]:
                s_idx = np.random.choice(self._smaller_indices[index])
                smaller_feats = self._all_features_tensor[s_idx]
                smaller_perf = self._all_perf_tensor[s_idx]
            else:
                smaller_feats = main_feats
                smaller_perf = main_perf

            # Find algorithms with larger performance on the same dataset
            if self._larger_indices[index]:
                l_idx = np.random.choice(self._larger_indices[index])
                larger_feats = self._all_features_tensor[l_idx]
                larger_perf = self._all_perf_tensor[l_idx]
            else:
                larger_feats = main_feats
                larger_perf = main_perf

            return (main_feats, smaller_feats, larger_feats), (
                main_perf,
                smaller_perf,
                larger_perf,
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
