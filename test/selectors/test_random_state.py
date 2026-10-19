"""Random state propagation through component construction and estimator clones."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from asf.epm import EPM
from asf.predictors import SklearnWrapper
from asf.selectors import PerformanceModel, SATzilla, SelectorPipeline
from asf.selectors.selector_tuner import _create_pipeline


def test_configured_pipeline_passes_tuner_state_to_model() -> None:
    cs = SelectorPipeline.get_configuration_space(selector_class=[PerformanceModel])
    pipeline = _create_pipeline(
        cs.get_default_configuration(),
        budget=100,
        maximize=False,
        selector_kwargs={},
        feature_groups=None,
        feature_selector=PCA(n_components=1, svd_solver="randomized"),
        random_state=7,
    )
    assert pipeline.random_state == 7
    assert pipeline.selector.random_state == 7
    model = pipeline.selector._make_model()
    assert model.model_class.random_state == 7
    assert pipeline.feature_selector.random_state == 7


def test_nested_estimator_states_are_seeded_without_mutating_template() -> None:
    template = SklearnWrapper(
        Pipeline,
        steps=[("scale", StandardScaler()), ("forest", RandomForestRegressor())],
    )
    predictor = EPM(estimator=template, random_state=7)._get_predictor()
    assert predictor.model_class["forest"].random_state == 7
    assert template.model_class["forest"].random_state is None


def test_satzilla_propagates_state_to_label_and_runtime_models() -> None:
    X = pd.DataFrame({"f": np.arange(8, dtype=float)})
    y = pd.DataFrame({"a": [1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0]})
    selector = SATzilla(
        epm_estimator=RandomForestRegressor(n_estimators=2), random_state=7
    )
    selector.fit(X, y, labels=["sat", "unsat"] * 4)
    assert selector.label_classifier.model_class.random_state == 7
    for epm in selector.epms["a"].values():
        assert epm.predictor.random_state == 7


def test_epm_predictions_are_reproducible() -> None:
    X = pd.DataFrame({"f": np.arange(12, dtype=float)})
    y = pd.Series([1.0, 5.0, 2.0, 7.0, 3.0, 9.0, 4.0, 10.0, 6.0, 11.0, 8.0, 12.0])
    template = RandomForestRegressor(n_estimators=3, max_depth=2)
    first = EPM(estimator=template, random_state=7).fit(X, y)
    second = EPM(estimator=template, random_state=7).fit(X, y)
    np.testing.assert_array_equal(first.predict(X), second.predict(X))
    assert not hasattr(template, "estimators_")


def test_custom_clone_protocol_does_not_require_sklearn_parameter_api() -> None:
    class CustomPredictor:
        def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
            pass

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.ones(len(X))

        def __sklearn_clone__(self) -> "CustomPredictor":
            return type(self)()

    template = CustomPredictor()
    predictor = EPM(estimator=template, random_state=7)._get_predictor()
    assert isinstance(predictor, CustomPredictor)
    assert predictor is not template
