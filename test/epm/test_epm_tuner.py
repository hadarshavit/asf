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

    with pytest.raises(AssertionError):
        tuner.tune_epm(X, y, model_class=DummyPredictor)
