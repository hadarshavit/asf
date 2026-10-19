from typing import Any
import numpy as np
import pytest


@pytest.mark.parametrize("configured, expected", [(None, 7), (3, 3)])
def test_epm_propagates_random_state_without_mutating_estimator(
    configured: Any, expected: Any
) -> None:
    from asf.epm import EPM
    from sklearn.ensemble import RandomForestRegressor

    estimator = RandomForestRegressor(random_state=configured)
    epm = EPM(estimator=estimator, random_state=7)
    assert epm._get_predictor().random_state == expected
    assert estimator.random_state == configured


def test_tune_epm_requires_smac(monkeypatch: Any) -> None:
    # Force SMAC_AVAILABLE to False to test assertion branch
    import asf.epm.epm_tuner as tuner

    monkeypatch.setattr(tuner, "SMAC_AVAILABLE", False, raising=True)

    X = np.random.RandomState(0).randn(10, 3)
    y = np.random.RandomState(1).randn(10)

    class DummyPredictor:
        @staticmethod
        def get_configuration_space() -> None:
            # minimal callable to satisfy access in tune_epm body (not reached when SMAC
            # unavailable)
            return None

    with pytest.raises(RuntimeError):
        tuner.tune_epm(X, y, model_class=DummyPredictor)  # type: ignore[arg-type]


def test_tune_epm_with_mock_smac_returns_configured_epm(monkeypatch: Any) -> None:
    # Arrange: mock SMAC availability and inject fake Scenario/Facade so no external
    # dependency is needed
    import asf.epm.epm_tuner as tuner
    from asf.epm.epm import EPM
    from asf.predictors.abstract_predictor import AbstractPredictor

    rng = np.random.RandomState(42)
    X = rng.rand(12, 3)
    # ensure strictly positive targets for LogNormalization default
    y = np.exp(rng.randn(12))

    class DummyPredictor(AbstractPredictor):
        @staticmethod
        def get_configuration_space(
            cs: Any = None,
            pre_prefix: Any = "",
            parent_param: Any = None,
            parent_value: Any = None,
        ) -> Any:
            return {"any": "space"}

        @staticmethod
        def get_from_configuration(
            configuration: Any, pre_prefix: Any = "", **kwargs: Any
        ) -> Any:
            class Model(AbstractPredictor):
                def fit(self, X: Any, Y: Any, **kwargs: Any) -> None:
                    # simple constant regressor on normalized target
                    self.mean_ = float(np.mean(Y))

                def predict(self, X: Any, **kwargs: Any) -> Any:
                    return np.full(len(X), self.mean_)

                def save(self, file_path: str) -> None:
                    pass

                @classmethod
                def load(cls, file_path: str) -> Any:
                    return cls()

            return Model

        # Implement abstract methods for DummyPredictor itself too, though not used as
        # instance here
        def fit(self, X: Any, Y: Any, **kwargs: Any) -> None:
            pass

        def predict(self, X: Any, **kwargs: Any) -> Any:
            return []

        def save(self, file_path: str) -> None:
            pass

        @classmethod
        def load(cls, file_path: str) -> Any:
            return cls()

    class FakeScenario:
        def __init__(
            self,
            configspace: Any,
            n_trials: Any,
            walltime_limit: Any,
            deterministic: Any,
            output_directory: Any,
            seed: Any,
            **kwargs: Any,
        ) -> None:
            self.seed = seed

    class FakeHPOFacade:
        def __init__(self, scenario: Any, target_function: Any, **kwargs: Any) -> None:
            self.scenario = scenario
            self.target_function = target_function

        def optimize(self) -> Any:
            # Call once to ensure the objective runs without error
            self.target_function(config={"p": 1}, seed=self.scenario.seed)
            return {"p": 2}

    # Patch SMAC classes and availability flag
    monkeypatch.setattr(tuner, "SMAC_AVAILABLE", True, raising=True)
    monkeypatch.setattr(tuner, "Scenario", FakeScenario, raising=True)
    monkeypatch.setattr(
        tuner, "HyperparameterOptimizationFacade", FakeHPOFacade, raising=True
    )

    # Act
    epm = tuner.tune_epm(
        X, y, model_class=DummyPredictor, cv=3, runcount_limit=3, random_state=7
    )

    # Assert
    assert isinstance(epm, EPM)
    assert epm.predictor_class is DummyPredictor
    assert epm.predictor_config == {"p": 2}
    assert epm.random_state == 7

    # And the returned EPM can be fit/predict successfully
    epm.fit(X, y)
    preds = epm.predict(X)
    assert len(preds) == len(y)
    assert np.all(np.isfinite(preds))
