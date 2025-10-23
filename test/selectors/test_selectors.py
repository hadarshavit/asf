import shutil

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRanker

from asf.predictors import (
    RegressionMLP,
    XGBoostClassifierWrapper,
    XGBoostRegressorWrapper,
)
from asf.predictors.ranking_mlp import RankingMLP
from asf.predictors.utils.mlp import get_mlp
from asf.selectors import (
    JointRanking,
    MultiClassClassifier,
    PairwiseClassifier,
    PairwiseRegressor,
    PerformanceModel,
    SimpleRanking,
    SurvivalAnalysisSelector,
)
from asf.selectors.collaborative_filtering_selector import (
    CollaborativeFilteringSelector,
)
from asf.selectors.isac import ISAC
from asf.selectors.satzilla import SATzilla
from asf.selectors.selector_pipeline import SelectorPipeline
from asf.selectors.selector_tuner import tune_selector
from asf.selectors.snnap import SNNAP
from asf.selectors.sunny_selector import SunnySelector


@pytest.fixture
def dummy_performance():
    data = np.array(
        [
            [120, 100, 110],
            [140, 150, 130],
            [180, 170, 190],
            [160, 150, 140],
            [250, 240, 260],
            [230, 220, 210],
            [300, 310, 320],
            [280, 290, 270],
            [350, 340, 360],
            [330, 320, 310],
            [400, 390, 410],
            [380, 370, 360],
            [450, 440, 460],
            [430, 420, 410],
            [500, 490, 510],
            [480, 470, 460],
            [550, 540, 560],
            [530, 520, 510],
            [600, 590, 610],
            [580, 570, 560],
        ]
    )
    return pd.DataFrame(data, columns=["algo1", "algo2", "algo3"])


@pytest.fixture
def dummy_features():
    data = np.array(
        [
            [10, 5, 1],
            [20, 10, 2],
            [15, 8, 1.5],
            [25, 12, 2.5],
            [30, 15, 3],
            [35, 18, 3.5],
            [40, 20, 4],
            [45, 22, 4.5],
            [50, 25, 5],
            [55, 28, 5.5],
            [60, 30, 6],
            [65, 32, 6.5],
            [70, 35, 7],
            [75, 38, 7.5],
            [80, 40, 8],
            [85, 42, 8.5],
            [90, 45, 9],
            [95, 48, 9.5],
            [100, 50, 10],
            [105, 52, 10.5],
        ]
    )
    return pd.DataFrame(data, columns=["feature1", "feature2", "feature3"])


def validate_predictions(predictions):
    """
    Validates that predictions have the expected structure:
    - Length of predictions is 20.
    - Each value in predictions is a list.
    - Each list has a length of 1.
    """
    assert len(predictions) == 20, "Predictions length is not 20"
    assert all(isinstance(v, list) for v in predictions.values()), (
        "Not all predictions are lists"
    )
    assert all(len(v) == 1 for v in predictions.values()), (
        "Not all lists in predictions have length 1"
    )
    for pred in predictions.values():
        algo, score = pred[0]
        assert algo in ["algo1", "algo2", "algo3"] or algo is None, (
            "Algorithm name is not valid"
        )
        assert score == 450.0, "Score is not 450.0"


def test_simple_ranking(dummy_performance, dummy_features):
    selector = SimpleRanking(model_class=XGBRanker, budget=450.0)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_joint_ranking(dummy_performance, dummy_features):
    selector = JointRanking(
        model=RankingMLP(
            input_size=3 + 3, epochs=2, model=get_mlp(6, 1, hidden_sizes=[8])
        )
    )
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_survival_analysis(dummy_performance, dummy_features):
    selector = SurvivalAnalysisSelector(budget=450.0)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_collaborative_filtering_selector(dummy_performance, dummy_features):
    # Insert some NaNs into the performance matrix to simulate missing data
    perf = dummy_performance.copy()
    np.random.seed(42)
    nan_mask = np.random.rand(*perf.shape) < 0.2
    perf[nan_mask] = np.nan

    selector = CollaborativeFilteringSelector(
        n_components=3, n_iter=100, lr=0.01, reg=0.1, budget=450.0
    )
    selector.fit(dummy_features, perf)

    # Validate predictions on training set
    predictions = selector.predict(None, None)
    validate_predictions(predictions)

    # Validate predictions with sparse performance matrix
    predictions = selector.predict(None, perf)
    validate_predictions(predictions)

    # Validate cold start predictions (features only)
    predictions = selector.predict(dummy_features, None)
    validate_predictions(predictions)


def test_sunny_selector(dummy_performance, dummy_features):
    budget = 500
    selector = SunnySelector(k=3, use_v2=True, budget=budget)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    assert len(predictions) == len(dummy_features)
    for sched in predictions.values():
        assert isinstance(sched, list)
        assert all(isinstance(x, tuple) and len(x) == 2 for x in sched)
        assert all(isinstance(x[0], str) and isinstance(x[1], float) for x in sched)
        assert all(x[0] in ["algo1", "algo2", "algo3"] for x in sched)
        assert np.isclose(sum(x[1] for x in sched), budget)


def test_isac_selector(dummy_performance, dummy_features):
    selector = ISAC(budget=450.0)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_snnap_selector(dummy_performance, dummy_features):
    selector = SNNAP(k=3, budget=450.0)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_satzilla_selector(dummy_performance, dummy_features):
    selector = SATzilla(budget=450)
    selector.fit(
        dummy_features,
        dummy_performance,
        labels=pd.DataFrame(
            ["SAT"] * (len(dummy_features) // 2)
            + ["UNSAT"] * (len(dummy_features) // 2),
            index=dummy_features.index,
        ),
    )
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_selector_tuner(dummy_performance, dummy_features):
    # Keep runcount_limit and cv low for fast testing
    tuned_pipeline = tune_selector(
        X=dummy_features,
        y=dummy_performance,
        selector_class=[PairwiseClassifier, PairwiseRegressor],
        runcount_limit=2,
        cv=2,
        seed=42,
        output_dir="./smac_test_output",  # Use a test-specific output dir
        smac_scenario_kwargs={},
        budget=450.0,
    )
    assert isinstance(tuned_pipeline, SelectorPipeline)
    # Fit the best pipeline found by the tuner
    tuned_pipeline.fit(dummy_features, dummy_performance)
    predictions = tuned_pipeline.predict(dummy_features)
    validate_predictions(predictions)

    # Clean up SMAC output directory if needed (optional)
    shutil.rmtree("./smac_test_output", ignore_errors=True)


@pytest.mark.parametrize(
    "model_class",
    [
        XGBoostClassifierWrapper,
    ],
)
def test_pairwise_classifier(dummy_performance, dummy_features, model_class):
    classifier = PairwiseClassifier(
        model_class=model_class, use_weights=True, budget=450.0
    )
    classifier.fit(dummy_features, dummy_performance)
    predictions = classifier.predict(dummy_features)
    validate_predictions(predictions)


@pytest.mark.parametrize(
    "model_class",
    [XGBoostClassifierWrapper],
)
def test_multi_class_classifier(dummy_performance, dummy_features, model_class):
    classifier = MultiClassClassifier(model_class=model_class, budget=450.0)
    classifier.fit(dummy_features, dummy_performance)
    predictions = classifier.predict(dummy_features)
    validate_predictions(predictions)


@pytest.mark.parametrize(
    "model_class",
    [XGBoostRegressorWrapper, RegressionMLP],
)
def test_pairwise_regressor(dummy_performance, dummy_features, model_class):
    regressor = PairwiseRegressor(model_class=model_class, budget=450.0)
    regressor.fit(dummy_features, dummy_performance)
    predictions = regressor.predict(dummy_features)
    validate_predictions(predictions)


@pytest.mark.parametrize(
    "model_class",
    [XGBoostRegressorWrapper, RegressionMLP],
)
def test_performance_model(dummy_performance, dummy_features, model_class):
    model = PerformanceModel(model_class=model_class, budget=450.0)
    model.fit(dummy_features, dummy_performance)
    predictions = model.predict(dummy_features)
    validate_predictions(predictions)
