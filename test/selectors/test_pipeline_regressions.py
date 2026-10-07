import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from asf.selectors.selector_tuner import _seed_component
from asf.selectors.performance_model import PerformanceModel
from asf.selectors.selector_pipeline import SelectorPipeline


def test_feature_group_filter_precedes_preprocessing():
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


def test_seed_component_preserves_explicit_random_state():
    unset = RandomForestRegressor(random_state=None)
    explicit = RandomForestRegressor(random_state=3)

    _seed_component(unset, 7)
    _seed_component(explicit, 7)

    assert unset.random_state == 7
    assert explicit.random_state == 3


def test_seed_component_seeds_model_factory():
    selector = PerformanceModel(model_class=RandomForestRegressor)

    _seed_component(selector, 7)

    assert selector.model_class().model_class.random_state == 7
