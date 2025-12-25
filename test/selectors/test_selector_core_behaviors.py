import numpy as np
import pandas as pd

from asf.selectors.abstract_selector import AbstractSelector
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector


def tiny_ctor():
    # Simple picklable callable for model_class
    return None


class TinySelector(AbstractSelector):
    def _fit(self, features, performance, **kwargs):
        # Remember shapes
        self._fshape = features.shape
        self._pshape = performance.shape

    def _predict(self, features, performance=None):
        # Return a trivial schedule choosing the first algorithm
        return {idx: [(self.algorithms[0], 1.0)] for idx in features.index}


def test_abstract_selector_numpy_to_dataframe_conversion():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    Y = np.array([[5.0, 1.0], [2.0, 3.0]])
    sel = TinySelector()
    sel.fit(X, Y)

    # ensure conversion happened and columns were set
    assert sel.features == ["f_0", "f_1"]
    assert sel.algorithms == ["algo_0", "algo_1"]

    preds = sel.predict(pd.DataFrame([[0.0, 0.0]], columns=pd.Index(["f_0", "f_1"])))
    assert list(preds.keys())[0] in (0, "i0")  # type: ignore[attr-defined]


def test_abstract_selector_rejects_mixed_types():
    sel = TinySelector()
    # features DataFrame, performance numpy should raise
    X = pd.DataFrame([[1.0, 2.0]], columns=pd.Index(["a", "b"]))
    Y = np.array([[1.0, 2.0]])
    try:
        sel.fit(X, Y)
        assert False, "Expected ValueError for mixed input types"
    except ValueError:
        pass


class TinyModelBased(AbstractModelBasedSelector):
    def __init__(self, **kwargs):
        super().__init__(model_class=tiny_ctor, **kwargs)
        self._saved = False

    def _fit(self, features, performance, **kwargs):
        return None

    def _predict(self, features, performance=None):
        return {idx: [(self.algorithms[0], 2.0)] for idx in features.index}


def test_model_based_selector_save_and_load(tmp_path):
    X = pd.DataFrame({"x": [1, 2]}, index=pd.Index(["i1", "i2"]))
    Y = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 0.1]}, index=pd.Index(X.index))
    sel = TinyModelBased(budget=2.0)
    sel.fit(X, Y)

    p = tmp_path / "sel.joblib"
    sel.save(p.as_posix())
    loaded = TinyModelBased.load(p.as_posix())
    preds = loaded.predict(X)
    assert set(preds.keys()) == set(X.index)  # type: ignore[attr-defined]
