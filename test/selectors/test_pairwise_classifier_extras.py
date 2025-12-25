import pandas as pd
import pytest

pytest.importorskip("xgboost")

from asf.selectors.pairwise_classifier import PairwiseClassifier
from asf.predictors.xgboost import XGBoostClassifierWrapper


def tiny_data():
    # Use default RangeIndex to be compatible with .loc indexing used in implementation
    X = pd.DataFrame({"f": [0.0, 1.0, 2.0]})
    Y = pd.DataFrame({"a": [1.0, 2.0, 0.5], "b": [0.9, 1.5, 2.0]}, index=X.index)
    return X, Y


def test_pairwise_generate_features_and_predict():
    X, Y = tiny_data()
    sel = PairwiseClassifier(model_class=XGBoostClassifierWrapper, budget=3.0)
    sel.fit(X, Y)
    feats = sel.generate_features(X)
    assert list(feats.columns) == sel.algorithms
    preds = sel.predict(X)
    assert set(preds.keys()) == {str(i) for i in X.index}  # type: ignore[attr-defined]
    for v in preds.values():  # type: ignore[attr-defined]
        assert isinstance(v, list) and len(v) == 1


@pytest.mark.skipif("ConfigSpace" not in globals(), reason="ConfigSpace not imported")
def test_pairwise_config_space_roundtrip():
    # If ConfigSpace is available, exercise building the space and creating a model
    try:
        import ConfigSpace  # noqa: F401
    except Exception:
        pytest.skip("ConfigSpace not installed in environment")

    cs, cs_transform = PairwiseClassifier.get_configuration_space()
    assert cs is not None and isinstance(cs_transform, dict)
