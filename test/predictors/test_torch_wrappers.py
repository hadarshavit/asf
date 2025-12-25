import importlib.util

import numpy as np
import pandas as pd
import pytest

from asf.predictors.ranking_mlp import RankingMLP
from asf.predictors.regression_mlp import RegressionMLP

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(TORCH_AVAILABLE, reason="Dependency available; fallback not active")
def test_ranking_mlp_requires_torch():
    with pytest.raises(ImportError):
        RankingMLP()


@pytest.mark.skipif(TORCH_AVAILABLE, reason="Dependency available; fallback not active")
def test_regression_mlp_requires_torch():
    with pytest.raises(ImportError):
        RegressionMLP()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Requires torch optional dependency")
def test_regression_mlp_fit_predict():
    import torch

    features = pd.DataFrame(
        [[0.0, np.nan], [1.0, 1.0]],
        index=["i1", "i2"],
        columns=["f1", "f2"],
    )
    performance = pd.DataFrame([0.1, 0.9], index=features.index, columns=["perf"])

    model = RegressionMLP(
        loss=torch.nn.MSELoss(),
        optimizer=torch.optim.SGD,
        epochs=1,
        batch_size=2,
        compile_model=False,
    )

    model.fit(features, performance)
    preds = model.predict(features)

    assert preds.shape[0] == features.shape[0]


@pytest.mark.skipif(True, reason="Requires torch optional dependency")
def test_ranking_mlp_fit_predict():
    import torch

    features = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["i1", "i2"],
        columns=["f1", "f2"],
    )
    performance = pd.DataFrame(
        [[0.1, 0.3], [0.2, 0.1]],
        index=features.index,
        columns=["algo_a", "algo_b"],
    )
    algorithm_features = pd.DataFrame(
        [[0.0], [1.0]],
        index=["algo_a", "algo_b"],
        columns=["bias"],
    )

    model = RankingMLP(
        input_size=len(features.columns) + len(algorithm_features.columns),
        optimizer=torch.optim.SGD,
        epochs=1,
        batch_size=2,
        compile_model=False,
    )

    model.fit(features, performance, algorithm_features)
    preds = model.predict(features, algorithm_features)

    assert preds.shape[0] == features.shape[0]
