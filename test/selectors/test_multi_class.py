import pytest
import numpy as np
import pandas as pd
from asf.selectors import MultiClassClassifier
from asf.predictors import RegressionMLP, XGBoostClassifierWrapper
from asf.predictors.abstract_predictor import AbstractPredictor


class ContiguousClassChecker(AbstractPredictor):
    """Test predictor that fails when labels are not contiguous from zero."""

    def fit(self, X, Y, **kwargs):
        labels = np.asarray(Y)
        unique = np.unique(labels)
        expected = np.arange(unique.size)
        if not np.array_equal(unique, expected):
            raise ValueError("labels must be contiguous")

    def predict(self, X, **kwargs):
        return np.zeros(len(X), dtype=int)

    def save(self, file_path: str) -> None:
        return None

    @classmethod
    def load(cls, file_path: str) -> "AbstractPredictor":
        return cls()


@pytest.mark.parametrize("model_class", [RegressionMLP, XGBoostClassifierWrapper])
def test_multi_class_classifier(
    dummy_performance, dummy_features, model_class, validate_predictions
):
    selector = MultiClassClassifier(model_class=model_class)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_multi_class_classifier_handles_non_contiguous_classes():
    X = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0]})
    # Argmin targets become [0, 2, 0, 2], i.e. classes {0, 2} (non-contiguous).
    Y = pd.DataFrame(
        {
            "algo_a": [0.1, 1.0, 0.2, 1.2],
            "algo_b": [0.9, 0.8, 0.7, 0.6],
            "algo_c": [0.5, 0.1, 0.6, 0.2],
        }
    )

    selector = MultiClassClassifier(model_class=ContiguousClassChecker)
    selector.fit(X, Y)
    preds = selector.predict(X)

    assert set(preds.keys()) == {str(i) for i in X.index}  # type: ignore[attr-defined]
