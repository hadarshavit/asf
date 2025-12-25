import numpy as np
import pytest


def test_tune_epm_requires_smac(monkeypatch):
    # Force SMAC_AVAILABLE to False to test assertion branch
    import asf.epm.epm_tuner as tuner

    monkeypatch.setattr(tuner, "SMAC_AVAILABLE", False, raising=True)

    X = np.random.RandomState(0).randn(10, 3)
    y = np.random.RandomState(1).randn(10)

    class DummyPredictor:
        @staticmethod
        def get_configuration_space():
            # minimal callable to satisfy access in tune_epm body (not reached when SMAC unavailable)
            return None

    with pytest.raises(RuntimeError):
        tuner.tune_epm(X, y, model_class=DummyPredictor)  # type: ignore[arg-type]


def test_tune_epm_with_mock_smac_returns_configured_epm(monkeypatch):
    # Arrange: mock SMAC availability and inject fake Scenario/Facade so no external dependency is needed
    import asf.epm.epm_tuner as tuner
    from asf.epm.epm import EPM
    from asf.predictors.abstract_predictor import AbstractPredictor

    rng = np.random.RandomState(42)
    X = rng.rand(12, 3)
    # ensure strictly positive targets for LogNormalization default
    y = np.exp(rng.randn(12))

    class DummyPredictor(AbstractPredictor):
        @staticmethod
        def get_configuration_space(  # type: ignore[override]
            cs=None, pre_prefix="", parent_param=None, parent_value=None
        ):
            return {"any": "space"}

        @staticmethod
        def get_from_configuration(configuration, pre_prefix="", **kwargs):  # type: ignore[override]
            class Model(AbstractPredictor):
                def fit(self, X, Y, **kwargs):
                    # simple constant regressor on normalized target
                    self.mean_ = float(np.mean(Y))

                def predict(self, X, **kwargs):
                    return np.full(len(X), self.mean_)

                def save(self, file_path: str):
                    pass

                @classmethod
                def load(cls, file_path: str):
                    return cls()

            return Model

        # Implement abstract methods for DummyPredictor itself too, though not used as instance here
        def fit(self, X, Y, **kwargs):
            pass

        def predict(self, X, **kwargs):
            return []

        def save(self, file_path: str):
            pass

        @classmethod
        def load(cls, file_path: str):
            return cls()

    class FakeScenario:
        def __init__(
            self,
            configspace,
            n_trials,
            walltime_limit,
            deterministic,
            output_directory,
            seed,
            **kwargs,
        ):
            self.seed = seed

    class FakeHPOFacade:
        def __init__(self, scenario, target_function, **kwargs):
            self.scenario = scenario
            self.target_function = target_function

        def optimize(self):
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
    epm = tuner.tune_epm(X, y, model_class=DummyPredictor, cv=3, runcount_limit=3)

    # Assert
    assert isinstance(epm, EPM)
    assert epm.predictor_class is DummyPredictor
    assert epm.predictor_config == {"p": 2}

    # And the returned EPM can be fit/predict successfully
    epm.fit(X, y)
    preds = epm.predict(X)
    assert len(preds) == len(y)
    assert np.all(np.isfinite(preds))
