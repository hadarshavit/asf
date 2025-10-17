import numpy as np

from asf.preprocessing.schmee_hahn_imputer import schmee_hahn_impute


class SimpleConstantModel:
    def fit(self, X, y):
        # store mean of provided targets
        self.mean_ = float(np.nanmean(y))

    def predict(self, X):
        return np.full(len(X), self.mean_)


def test_schmee_hahn_impute_handles_all_observed():
    # No censored values; imputer should leave values as-is and train model
    y_raw = np.array([10.0, 20.0, 30.0])
    X = np.eye(3)

    y_imp, model = schmee_hahn_impute(lambda: SimpleConstantModel(), y_raw, X)

    assert np.allclose(y_imp, y_raw)
    assert model is not None


def test_schmee_hahn_impute_imputes_censored_values():
    # Some values censored at budget; should be imputed above budget via EM
    rng = np.random.RandomState(0)
    X = rng.randn(6, 2)

    budget = 50.0
    y_raw = np.array([10.0, 25.0, np.nan, 60.0, 15.0, 80.0])  # 60 and 80 censored

    y_imp, model = schmee_hahn_impute(
        lambda: SimpleConstantModel(),
        y_raw,
        X,
        em_max_iter=50,
        em_tol=1e-6,
        budget=budget,
    )

    # NaNs remain NaN
    assert np.isnan(y_imp[2])

    # Observed values unchanged
    assert np.allclose(y_imp[[0, 1, 4]], y_raw[[0, 1, 4]])

    # Censored values should be >= budget (since expectation over tail)
    assert np.all(y_imp[[3, 5]] >= budget)
    assert model is not None


def test_schmee_hahn_impute_all_censored_converges():
    X = np.eye(4)
    budget = 100.0
    y_raw = np.array(
        [100.0, 100.0, 150.0, 200.0]
    )  # all treated as censored (>= budget)

    y_imp, model = schmee_hahn_impute(
        lambda: SimpleConstantModel(),
        y_raw,
        X,
        em_max_iter=20,
        em_tol=1e-8,
        budget=budget,
    )

    # Should produce finite imputations and a fitted model
    assert np.all(np.isfinite(y_imp))
    assert np.all(y_imp >= budget)
    assert model is not None
