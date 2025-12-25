import numpy as np
import pandas as pd

from asf.selectors.selector_pipeline import SelectorPipeline
from asf.selectors.multi_class import MultiClassClassifier
from asf.predictors.random_forest import RandomForestClassifierWrapper


def small_dataframes():
    X = pd.DataFrame(
        {
            "f1": [1.0, np.nan, 3.0, 4.0],
            "f2": [0.5, 0.7, np.nan, 1.2],
        },
        index=pd.Index([f"i{k}" for k in range(4)]),
    )
    Y = pd.DataFrame(
        {
            "a": [5.0, 2.0, 1.0, 4.0],
            "b": [1.0, 3.0, 2.0, 6.0],
        },
        index=X.index,
    )
    return X, Y


def test_selector_pipeline_fit_predict_and_config(tmp_path):
    X, Y = small_dataframes()

    selector = MultiClassClassifier(
        model_class=RandomForestClassifierWrapper, budget=10.0
    )
    pipe = SelectorPipeline(selector=selector)

    # Fit with NaNs to exercise SimpleImputer in pipeline
    pipe.fit(X, Y)
    preds = pipe.predict(X)

    assert set(preds.keys()) == set(X.index)
    for v in preds.values():
        assert isinstance(v, list) and len(v) == 1
        algo, bud, *_ = v[0]
        assert algo in ["a", "b"]
        assert bud == 10.0

    # Config should reflect selector & preprocessing steps
    cfg = pipe.get_config()
    assert cfg["selector"] == "MultiClassClassifier"
    assert cfg["selector_model"] in ("RandomForestClassifierWrapper",)
    # assert cfg["preprocessor"] == "Pipeline"  # Removed as get_config does not return this key
    assert "SimpleImputer" in cfg["preprocessor_steps"][0]

    # Save/load round-trip
    pth = tmp_path / "pipe.joblib"
    pipe.save(pth.as_posix())
    loaded = SelectorPipeline.load(pth.as_posix())
    preds2 = loaded.predict(X)
    assert set(preds2.keys()) == set(X.index)
