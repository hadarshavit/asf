from ConfigSpace import ConfigurationSpace
import numpy as np
import pytest
import pandas as pd
from asf.predictors import (
    RegressionMLP,
    XGBoostClassifierWrapper,
    XGBoostRegressorWrapper,
)
from asf.selectors import (
    PairwiseClassifier,
    PairwiseRegressor,
    SimpleRanking,
    JointRanking,
    SurvivalAnalysis,
    MultiClassClassifier,
    PerformanceModel,
)
from xgboost import XGBRanker
from asf.selectors.selector_tuner import tune_selector
from asf.selectors.selector_pipeline import SelectorPipeline
import shutil
from asf.selectors.collaborative_filtering_selector import (
    CollaborativeFilteringSelector,
)
from asf.selectors.sunny import SUNNY
from asf.selectors.satzilla import SATzilla
from asf.selectors.isac import ISAC
from asf.selectors.snnap import SNNAP
from asf.selectors.isa import ISA
from asf.selectors.meta_selector import MetaSelector
from asf.selectors.osl_linear import OSLLinearSelector
from asf.selectors.cosine_selector import CosineSelector
from asf.predictors.random_forest import RandomForestRegressorWrapper


@pytest.fixture
def dummy_performance():
    data = np.array(
        [
            [800, 250, 5],
            [200, 800, 130],
            [120, 170, 500],
            [160, 150, 700],
            [2750, 240, 260],
            [230, 220, 210],
            [300, 100, 600],
            [550, 12, 270],
            [350, 3490, 20],
            [100, 320, 310],
            [400, 33, 410],
            [12, 370, 360],
            [6, 440, 460],
            [430, 42, 410],
            [500, 200, 20],
            [480, 470, 100],
            [100, 540, 560],
            [530, 100, 330],
            [25, 250, 610],
            [580, 200, 10],
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
    selector = JointRanking(budget=450.0)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_survival_analysis(dummy_performance, dummy_features):
    selector = SurvivalAnalysis(budget=450.0)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_survival_analysis_schedule(dummy_performance, dummy_features):
    budget = 450.0
    selector = SurvivalAnalysis(budget=budget, use_schedule=True)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)

    assert len(predictions) == len(dummy_features)
    for sched in predictions.values():
        assert isinstance(sched, list)
        assert all(isinstance(x, tuple) and len(x) == 2 for x in sched)
        assert all(
            isinstance(x[0], str) and isinstance(x[1], (float, np.floating))
            for x in sched
        )
        assert np.isclose(sum(x[1] for x in sched), budget)


def test_isa_selector(dummy_performance, dummy_features):
    budget = 450.0
    selector = ISA(k=3, use_k_tuning=True, budget=budget)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)

    assert len(predictions) == len(dummy_features)
    for sched in predictions.values():
        assert isinstance(sched, list)
        assert all(isinstance(x, tuple) and len(x) == 2 for x in sched)
        assert all(
            isinstance(x[0], str) and isinstance(x[1], (float, np.floating, int))
            for x in sched
        )
        assert sum(x[1] for x in sched) == budget


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

    cs, cs_transform = SUNNY.get_configuration_space()
    assert isinstance(cs, ConfigurationSpace)
    SUNNY.get_from_configuration(cs.get_default_configuration(), cs_transform)

    selector = SUNNY(k=3, use_v2=True, use_tsunny=True, budget=budget)

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
    cs, cs_transform = ISAC.get_configuration_space()
    assert isinstance(cs, ConfigurationSpace)
    ISAC.get_from_configuration(cs.get_default_configuration(), cs_transform)

    selector = ISAC(budget=450.0)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_snnap_selector(dummy_performance, dummy_features):
    cs, cs_transform = SNNAP.get_configuration_space()
    assert isinstance(cs, ConfigurationSpace)
    SNNAP.get_from_configuration(cs.get_default_configuration(), cs_transform)

    selector = SNNAP(k=3, budget=450.0)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_satzilla_selector(dummy_performance, dummy_features):
    cs, cs_transform = SATzilla.get_configuration_space()
    assert isinstance(cs, ConfigurationSpace)
    SATzilla.get_from_configuration(cs.get_default_configuration(), cs_transform)

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


def test_meta_selector(dummy_performance, dummy_features):
    budget = 450.0

    base_selectors = [
        SNNAP(k=3, budget=budget),
        SATzilla(budget=budget),
        ISAC(budget=budget),
    ]
    meta_sel = SimpleRanking(model_class=XGBRanker, budget=budget)

    meta = MetaSelector(
        base_selectors=base_selectors, meta_selector=meta_sel, budget=budget, n_folds=2
    )
    meta.fit(dummy_features, dummy_performance)
    predictions = meta.predict(dummy_features)

    validate_predictions(predictions)


def test_meta_selector_rejects_schedule_base(dummy_performance, dummy_features):
    budget = 450.0

    # ISA is a schedule-returning selector; MetaSelector should reject it as a base
    with pytest.raises(ValueError):
        MetaSelector(
            base_selectors=[SNNAP(k=3, budget=budget), ISA(budget=budget)],
            meta_selector=SimpleRanking(model_class=XGBRanker, budget=budget),
            budget=budget,
        )


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
    cs, cs_transform = PairwiseClassifier.get_configuration_space()
    assert isinstance(cs, ConfigurationSpace)
    PairwiseClassifier.get_from_configuration(
        cs.get_default_configuration(), cs_transform
    )

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
    cs, cs_transform = MultiClassClassifier.get_configuration_space()
    assert isinstance(cs, ConfigurationSpace)
    MultiClassClassifier.get_from_configuration(
        cs.get_default_configuration(), cs_transform
    )

    classifier = MultiClassClassifier(model_class=model_class, budget=450.0)
    classifier.fit(dummy_features, dummy_performance)
    predictions = classifier.predict(dummy_features)
    validate_predictions(predictions)


@pytest.mark.parametrize(
    "model_class",
    [XGBoostRegressorWrapper, RegressionMLP],
)
def test_pairwise_regressor(dummy_performance, dummy_features, model_class):
    cs, cs_transform = PairwiseRegressor.get_configuration_space()
    assert isinstance(cs, ConfigurationSpace)
    PairwiseRegressor.get_from_configuration(
        cs.get_default_configuration(), cs_transform
    )

    regressor = PairwiseRegressor(model_class=model_class, budget=450.0)
    regressor.fit(dummy_features, dummy_performance)
    predictions = regressor.predict(dummy_features)
    validate_predictions(predictions)


@pytest.mark.parametrize(
    "model_class",
    [XGBoostRegressorWrapper, RegressionMLP],
)
def test_performance_model(dummy_performance, dummy_features, model_class):
    cs, cs_transform = PerformanceModel.get_configuration_space()
    assert isinstance(cs, ConfigurationSpace)
    PerformanceModel.get_from_configuration(
        cs.get_default_configuration(), cs_transform
    )

    model = PerformanceModel(model_class=model_class, budget=450.0)
    model.fit(dummy_features, dummy_performance)
    predictions = model.predict(dummy_features)
    validate_predictions(predictions)


def test_osl_linear_selector(dummy_performance, dummy_features):
    selector = OSLLinearSelector(
        budget=450.0, reg=1e-3, optimizer_method="L-BFGS-B", maxiter=200
    )
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)

    assert len(predictions) == len(dummy_features)
    for pred in predictions.values():
        assert isinstance(pred, list) and len(pred) == 1
        algo, score = pred[0]
        assert algo in ["algo1", "algo2", "algo3"] or algo is None
        assert isinstance(score, (int, float, np.floating, np.integer))
        assert score >= 0


def test_cosine_selector_default_ridge(dummy_performance, dummy_features):
    # simple algorithm feature matrix matching performance columns
    alg_df = pd.DataFrame(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ],
        index=["algo1", "algo2", "algo3"],
        columns=["af1", "af2"],
    )

    sel = CosineSelector(shared_latent_dim=2, normalize_features=True, budget=450.0)
    sel.fit(dummy_features, dummy_performance, alg_df)
    preds = sel.predict(dummy_features)
    validate_predictions(preds)


def test_cosine_selector_random_forest(dummy_performance, dummy_features):
    alg_df = pd.DataFrame(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ],
        index=["algo1", "algo2", "algo3"],
        columns=["af1", "af2"],
    )

    sel = CosineSelector(
        shared_latent_dim=2,
        normalize_features=True,
        projection_model=RandomForestRegressorWrapper,
        projection_model_kwargs={"n_estimators": 10, "random_state": 42},
        budget=450.0,
    )
    sel.fit(dummy_features, dummy_performance, alg_df)
    preds = sel.predict(dummy_features)
    validate_predictions(preds)
