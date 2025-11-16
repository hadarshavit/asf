import numpy as np
import pytest


def test_tune_distnet_requires_smac(monkeypatch):
    # Force SMAC_AVAILABLE to False to test assertion branch
    import asf.epm.distnet_tuner as tuner
    from asf.epm.distnet import DistNet

    monkeypatch.setattr(tuner, "SMAC_AVAILABLE", False, raising=True)

    X = np.random.RandomState(0).randn(10, 3)
    # DistNet expects positive runtimes; ensure positivity
    y = np.exp(np.random.RandomState(1).randn(10))

    with pytest.raises(AssertionError):
        tuner.tune_distnet(DistNet, X, y)


def test_tune_distnet_with_mock_smac_returns_configured_distnet(monkeypatch):
    # Arrange: mock SMAC availability and inject fake Scenario/Facade so no external dependency is needed
    import asf.epm.distnet_tuner as tuner
    from asf.epm.distnet import DistNet
    import torch

    rng = np.random.RandomState(42)
    X = rng.rand(12, 4)
    # strictly positive targets (as runtimes)
    y = np.exp(rng.randn(12))

    # Provide a dummy configuration space so Scenario accepts it without importing ConfigSpace
    class DummyCS:
        pass

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
            assert isinstance(configspace, DummyCS)
            self.seed = seed

    class FakeHPOFacade:
        def __init__(self, scenario, target_function, **kwargs):
            self.scenario = scenario
            self.target_function = target_function

        def optimize(self):
            # Call once to ensure the objective runs; force a very small epoch count for speed
            self.target_function(
                config={
                    "epochs": 2,
                    "batch_size": 8,
                    "hidden_layers": 1,
                    "hidden_size": 16,
                    "optimizer": "adam",
                    "lr": 1e-2,
                    "dropout": 0.0,
                    "scheduler": "none",
                    "use_batchnorm": False,
                },
                seed=self.scenario.seed,
            )
            # Return a small best configuration
            return {
                "epochs": 2,
                "batch_size": 8,
                "hidden_layers": 1,
                "hidden_size": 16,
                "optimizer": "adam",
                "lr": 1e-2,
                "dropout": 0.0,
                "scheduler": "none",
                "use_batchnorm": False,
            }

    # Patch SMAC classes and availability flag, and bypass real ConfigSpace by
    # overriding DistNet.get_configuration_space to return a dummy object.
    monkeypatch.setattr(tuner, "SMAC_AVAILABLE", True, raising=True)
    monkeypatch.setattr(tuner, "Scenario", FakeScenario, raising=True)
    monkeypatch.setattr(
        tuner, "HyperparameterOptimizationFacade", FakeHPOFacade, raising=True
    )
    monkeypatch.setattr(
        DistNet,
        "get_configuration_space",
        staticmethod(lambda: DummyCS()),
        raising=True,
    )

    # Act
    dn = tuner.tune_distnet(DistNet, X, y, cv=3, runcount_limit=3, timeout=10)

    # Assert basic type and that it can be fit and predict distribution params
    assert isinstance(dn, DistNet)

    # Fit on the small data quickly
    dn.fit(X, y)

    preds = dn.predict(X)
    assert hasattr(preds, "shape")
    assert preds.shape[0] == len(X)
    assert preds.shape[1] == 2  # lognormal params (s, scale)
    assert torch.isfinite(preds).all()
