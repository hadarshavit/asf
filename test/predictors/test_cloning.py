"""Audit all concrete predictor classes, including non-exported implementations."""

import importlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier

from asf.predictors.sklearn_wrapper import SklearnWrapper
from asf.predictors.epm_random_forest import EPMRandomForest
from asf.predictors.epm_extra_trees import EPMExtraTrees


def wrapper_classes():
    classes = []
    for path in sorted(Path("asf/predictors").glob("*.py")):
        module = importlib.import_module(f"asf.predictors.{path.stem}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ == module.__name__ and issubclass(cls, SklearnWrapper):
                if cls is not SklearnWrapper:
                    classes.append(cls)
    return classes


@pytest.mark.parametrize(
    "cls", wrapper_classes(), ids=lambda cls: f"{cls.__module__}.{cls.__name__}"
)
def test_all_sklearn_wrappers_preserve_nondefault_parameters_and_reset_fit(cls):
    if cls.__module__.endswith("xgboost"):
        pytest.importorskip("xgboost")
    model = cls()
    params = model.get_params()
    settings = {
        key: value
        for key, value in {
            "n_estimators": 3,
            "max_depth": 2,
            "random_state": 7,
            "n_jobs": 1,
            "max_iter": 5,
            "C": 0.7,
            "hidden_layer_sizes": (3,),
        }.items()
        if key in params
    }
    model = cls(**settings)
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = (
        np.array([0, 1, 0, 1, 0, 1])
        if is_classifier(model.model_class)
        else np.arange(6, dtype=float)
    )
    fit_kwargs = (
        {"qid": np.array([0, 0, 0, 1, 1, 1])} if "Ranker" in cls.__name__ else {}
    )
    model.fit(X, y, **fit_kwargs)
    copied = clone(model)
    assert type(copied) is cls
    assert copied.get_params() == model.get_params()
    for key, value in settings.items():
        assert copied.get_params()[key] == value
    assert not hasattr(copied.model_class, "n_features_in_")
    copied.fit(X, y, **fit_kwargs)
    np.testing.assert_allclose(copied.predict(X), model.predict(X))


@pytest.mark.parametrize("cls", [EPMRandomForest, EPMExtraTrees])
def test_epm_forests_clone_all_settings_and_discard_trees(cls):
    kwargs = dict(
        n_estimators=3,
        max_depth=2,
        random_state=7,
        log=True,
        bootstrap=True,
        max_samples=0.75,
    )
    if cls is EPMRandomForest:
        kwargs.update(return_var=True, splitter="best")
    model = cls(**kwargs)
    X = pd.DataFrame(np.arange(12, dtype=float).reshape(6, 2))
    y = pd.Series(np.arange(6, dtype=float))
    model.fit(X, y)
    copied = clone(model)
    for key, value in kwargs.items():
        assert copied.get_params()[key] == value
    assert copied.get_params() == model.get_params()
    assert not hasattr(copied, "estimators_")
    assert not hasattr(copied, "trainX")
    copied.fit(X, y)
    np.testing.assert_allclose(copied.predict(X), model.predict(X))


def test_survival_wrapper_clones_configuration_and_resets_forest():
    pytest.importorskip("sksurv")
    from sksurv.util import Surv
    from asf.predictors.survival import RandomSurvivalForestWrapper
    from asf.selectors import SurvivalAnalysis

    estimator = RandomSurvivalForestWrapper(n_estimators=3, max_depth=2, random_state=7)
    X = pd.DataFrame({"f": np.arange(6, dtype=float)})
    estimator.fit(X, Surv.from_arrays([True] * 6, np.arange(1.0, 7.0)))
    copied = clone(estimator)
    assert copied.get_params() == estimator.get_params()
    assert not hasattr(copied.model, "estimators_")
    selector = SurvivalAnalysis(estimator=estimator, budget=5)
    selector.fit(
        X, pd.DataFrame({"a": np.arange(1.0, 7.0), "b": np.arange(6.0, 0.0, -1)})
    )
    assert selector.model.model.n_estimators == 3
    assert len(selector.predict(X)) == len(X)


@pytest.mark.parametrize("ranking", [False, True])
@pytest.mark.parametrize("custom_network", [False, True])
def test_torch_clones_reset_training_state_and_preserve_configuration(
    ranking, custom_network
):
    torch = pytest.importorskip("torch")
    from asf.predictors import RankingMLP, RegressionMLP
    from asf.selectors import JointRanking, PairwiseRegressor

    X = pd.DataFrame({"f": [0.0, 1.0, 2.0]})
    y = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 1.0, 2.0]})
    af = pd.DataFrame({"af": [0.0, 1.0]}, index=y.columns)
    network = torch.nn.Linear(2 if ranking else 1, 1) if custom_network else None
    network_state = (
        None
        if network is None
        else {k: v.clone() for k, v in network.state_dict().items()}
    )
    settings = dict(
        model=network,
        epochs=1,
        seed=7,
        batch_size=2,
        optimizer=torch.optim.SGD,
        learning_rate=0.01,
        weight_decay=0.02,
    )
    estimator = (
        RankingMLP(compile=False, **settings)
        if ranking
        else RegressionMLP(compile_model=False, **settings)
    )
    if ranking:
        estimator.fit(X, y, algorithm_features=af)
    else:
        estimator.fit(X, y["a"])

    copied = clone(estimator)
    for key in [
        "epochs",
        "seed",
        "batch_size",
        "optimizer",
        "learning_rate",
        "weight_decay",
    ]:
        assert copied.get_params()[key] == settings[key]
    assert copied.model is not estimator.model
    if network is not None:
        for key, initial in network_state.items():
            torch.testing.assert_close(network.state_dict()[key], initial)
            torch.testing.assert_close(copied.model.state_dict()[key], initial)
        assert all(
            a.data_ptr() != b.data_ptr()
            for a, b in zip(copied.model.parameters(), estimator.model.parameters())
        )
    else:
        assert copied.model is None

    trained_state = {k: v.clone() for k, v in estimator.model.state_dict().items()}
    selector = (
        JointRanking(estimator=estimator)
        if ranking
        else PairwiseRegressor(estimator=estimator)
    )
    selector.fit(X, y, **({"algorithm_features": af} if ranking else {}))
    assert len(selector.predict(X)) == len(X)
    for key, value in trained_state.items():
        torch.testing.assert_close(estimator.model.state_dict()[key], value)


def test_epm_random_forest_instance_preserves_settings_in_selector():
    from asf.selectors import PairwiseRegressor

    X = pd.DataFrame({"f": [0.0, 1.0, 2.0]})
    y = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 1.0, 2.0]})
    model = EPMRandomForest(n_estimators=3, max_depth=2, random_state=7)
    selector = PairwiseRegressor(estimator=model)
    selector.fit(X, y)
    assert selector.regressors[0].n_estimators == 3
    assert not hasattr(model, "estimators_")
    assert len(selector.predict(X)) == len(X)
