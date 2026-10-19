import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from asf.selectors.performance_model import PerformanceModel
from asf.selectors.selector_pipeline import SelectorPipeline


def test_feature_group_filter_precedes_preprocessing() -> None:
    features = pd.DataFrame(
        {"cheap": [0.0, 1.0, 2.0, 3.0], "expensive": [100.0, 300.0, 200.0, 500.0]}
    )
    performance = pd.DataFrame({"algo": [1.0, 2.0, 3.0, 4.0]})
    pipeline = SelectorPipeline(
        PerformanceModel(estimator=DecisionTreeRegressor()),
        preprocessor=PCA(n_components=1),
        feature_groups={"cheap": {"provides": ["cheap"]}},
    )

    pipeline.fit(features, performance)

    assert pipeline.preprocessor["PCA"].n_features_in_ == 1
    assert all(
        schedule[0] == "cheap" for schedule in pipeline.predict(features).values()
    )


def test_selector_seeds_clones_and_preserves_explicit_random_state() -> None:
    for configured, expected in [(None, 7), (3, 3)]:
        estimator = RandomForestRegressor(random_state=configured)
        selector = PerformanceModel(estimator=estimator, random_state=7)
        model = selector._make_model()
        assert model.random_state == expected
        assert estimator.random_state == configured


def test_pipeline_passes_random_state_to_component_factories() -> None:
    from functools import partial
    from asf.presolving import ASAPv2

    pipeline = SelectorPipeline(
        selector=partial(PerformanceModel, model_class=RandomForestRegressor),
        pre_solving=ASAPv2,
        preprocessor=partial(PCA, svd_solver="randomized"),
        random_state=7,
    )

    assert pipeline.selector.random_state == 7
    assert pipeline.selector._make_model().model_class.random_state == 7
    assert pipeline.pre_solving.random_state == 7
    assert pipeline.preprocessor["PCA"].random_state == 7


def test_selector_preserves_explicit_factory_random_state() -> None:
    from functools import partial

    selector = PerformanceModel(
        model_class=partial(RandomForestRegressor, random_state=3), random_state=7
    )
    assert selector._make_model().random_state == 3
