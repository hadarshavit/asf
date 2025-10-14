import numpy as np

from sklearn.ensemble import RandomForestClassifier

from asf.predictors.sklearn_wrapper import SklearnWrapper


def test_sklearn_wrapper_fit_predict(classification_data):
    X, y = classification_data
    wrapper = SklearnWrapper(RandomForestClassifier, {"n_estimators": 16})

    wrapper.fit(X, y, sample_weight=np.ones(len(y)))
    predictions = wrapper.predict(X)

    assert predictions.shape == y.shape


def test_sklearn_wrapper_save_load(tmp_path, classification_data):
    X, y = classification_data
    wrapper = SklearnWrapper(RandomForestClassifier, {"n_estimators": 8})
    wrapper.fit(X, y)
    expected = wrapper.predict(X)

    target = tmp_path / "rf_wrapper.joblib"
    wrapper.save(target.as_posix())
    loaded = wrapper.load(target.as_posix())

    assert isinstance(loaded, SklearnWrapper)
    assert np.array_equal(expected, loaded.predict(X))
