import csv
import subprocess
from pathlib import Path

import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from asf.cli.cli_train import build_cli_command
from asf.selectors import PairwiseRegressor, SelectorPipeline
from asf.selectors.satzilla import SATzilla
from asf.presolving.asap_v2 import ASAPv2


def _write_dummy_csv(path: Path):
    rows = [["idx", "f1", "f2"], ["inst_1", "1.0", "2.0"], ["inst_2", "3.0", "4.0"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        csv.writer(fh).writerows(rows)


def _write_data(feat_path: Path, perf_path: Path, n_instances: int = 20):
    """Write realistic dummy data with proper structure for validation."""
    feat_data = {
        "idx": [f"inst_{i}" for i in range(n_instances)],
        "f1": [10.0 + i * 5 for i in range(n_instances)],
        "f2": [5.0 + i * 2.5 for i in range(n_instances)],
        "f3": [1.0 + i * 0.5 for i in range(n_instances)],
    }
    feat_df = pd.DataFrame(feat_data).set_index("idx")
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(feat_path)

    perf_data = {
        "idx": [f"inst_{i}" for i in range(n_instances)],
        "algo1": [120 + i * 20 for i in range(n_instances)],
        "algo2": [100 + i * 20 for i in range(n_instances)],
        "algo3": [110 + i * 20 for i in range(n_instances)],
    }
    perf_df = pd.DataFrame(perf_data).set_index("idx")
    perf_path.parent.mkdir(parents=True, exist_ok=True)
    perf_df.to_csv(perf_path)

    return feat_df, perf_df


def test_build_cli_command(tmp_path: Path):
    """Unit test: verify build_cli_command produces correct arguments."""
    feats = tmp_path / "features.csv"
    perf = tmp_path / "performance.csv"
    _write_dummy_csv(feats)
    _write_dummy_csv(perf)

    cmd = build_cli_command(
        selector=[SATzilla, PairwiseRegressor],
        feature_data=feats,
        performance_data=perf,
        destination=tmp_path / "model.pkl",
        model="Ridge",
        tuning=True,
        budget=60,
        maximize=False,
        preprocessors=[StandardScaler(), MinMaxScaler()],
        presolvers=[ASAPv2()],
        presolver_budget=0.2,
    )

    assert isinstance(cmd, list)
    assert "--selectors" in cmd
    assert "SATzilla" in cmd and "PairwiseRegressor" in cmd
    assert "--model" in cmd and "Ridge" in cmd
    assert "--budget" in cmd and "60" in cmd
    assert "--presolver-budget" in cmd and "0.2" in cmd
    assert (
        "--preprocessors" in cmd and "StandardScaler" in cmd and "MinMaxScaler" in cmd
    )
    assert "--presolvers" in cmd and "ASAPv2" in cmd


def _validate_pipeline_extensive(
    pipeline: SelectorPipeline,
    features: pd.DataFrame,
    expected_budget: float,
    expected_selector_types: list,
    expected_preprocessor_types: list = None,
    expected_presolver_type: type = None,
    presolver_budget_fraction: float = 0.0,
    tuning: bool = False,
):
    """Extensive validation of a trained SelectorPipeline."""
    assert isinstance(pipeline, SelectorPipeline)
    assert hasattr(pipeline, "selector") and pipeline.selector is not None
    assert hasattr(pipeline, "predict") and hasattr(pipeline, "fit")

    # Validate budget
    selector_budget = getattr(pipeline.selector, "budget", None)
    assert selector_budget is not None, "Selector missing budget attribute"

    presolver_budget = 0.0
    if pipeline.pre_solving is not None:
        presolver_budget = getattr(pipeline.pre_solving, "budget", 0.0)

    total_budget = selector_budget + presolver_budget
    assert pytest.approx(total_budget, rel=1e-3, abs=1e-6) == expected_budget, (
        f"Total budget mismatch: selector={selector_budget} + presolver={presolver_budget} "
        f"= {total_budget}, expected {expected_budget}"
    )

    # Validate selector type
    selector_type = type(pipeline.selector)
    if tuning:
        assert selector_type in expected_selector_types, (
            f"Selector type mismatch: got {selector_type.__name__}, "
            f"expected one of {[t.__name__ for t in expected_selector_types]}"
        )
    else:
        assert selector_type == expected_selector_types[0], (
            f"Selector type mismatch: got {selector_type.__name__}, "
            f"expected {expected_selector_types[0].__name__}"
        )

    # Validate preprocessor
    assert pipeline.preprocessor is not None, "Expected preprocessor but got None"
    assert isinstance(pipeline.preprocessor, Pipeline), (
        "Preprocessor should be a sklearn Pipeline"
    )

    if expected_preprocessor_types:
        # Skip first step (SimpleImputer) when validating types
        actual_types = [type(step) for _, step in pipeline.preprocessor.steps[1:]]
        if tuning:
            for actual_type in actual_types:
                assert actual_type in expected_preprocessor_types, (
                    f"Unexpected preprocessor {actual_type.__name__} not in candidates. "
                    f"Expected subset of: {[t.__name__ for t in expected_preprocessor_types]}"
                )
        else:
            for expected_type in expected_preprocessor_types:
                assert expected_type in actual_types, (
                    f"Expected preprocessor {expected_type.__name__} not found. "
                    f"Found: {[t.__name__ for t in actual_types]}"
                )
    else:
        assert len(pipeline.preprocessor.steps) == 1, (
            f"Expected only SimpleImputer, found {len(pipeline.preprocessor.steps)} steps"
        )
        assert isinstance(pipeline.preprocessor.steps[0][1], SimpleImputer), (
            f"Expected only SimpleImputer, found {type(pipeline.preprocessor.steps[0][1]).__name__}"
        )

    # Validate presolver
    if expected_presolver_type:
        assert pipeline.pre_solving is not None, "Expected presolver but got None"
        actual_presolver_type = type(pipeline.pre_solving)
        assert actual_presolver_type == expected_presolver_type, (
            f"Presolver type mismatch: got {actual_presolver_type.__name__}, "
            f"expected {expected_presolver_type.__name__}"
        )

        if tuning:
            # During tuning: presolver budget can be any value, just check it's reasonable
            assert 0 <= presolver_budget <= expected_budget, (
                f"Presolver budget {presolver_budget} outside [0, {expected_budget}]"
            )
        else:
            # Non-tuning: presolver budget should equal fraction * total budget
            expected_presolver_budget = expected_budget * float(
                presolver_budget_fraction or 0.0
            )
            assert (
                pytest.approx(presolver_budget, rel=1e-3, abs=1e-6)
                == expected_presolver_budget
            ), (
                f"Presolver budget mismatch: got {presolver_budget}, expected {expected_presolver_budget}"
            )
    else:
        assert pipeline.pre_solving is None, "Unexpected presolver present"

    # Predictions sanity
    predictions = pipeline.predict(features)
    assert isinstance(predictions, dict)
    assert len(predictions) == len(features)
    for inst_id, schedule in predictions.items():
        assert inst_id in features.index
        assert isinstance(schedule, list) and len(schedule) > 0
        total_time = 0.0
        for algo_name, time_alloc in schedule:
            assert isinstance(algo_name, (str, type(None)))
            assert isinstance(time_alloc, (int, float)) and time_alloc >= 0
            total_time += time_alloc
        assert total_time <= expected_budget * 1.01


def _run_cli_and_validate(
    tmp_path: Path,
    selector: list,
    tuning: bool = False,
    preprocessors: list = None,
    presolvers: list = None,
    presolver_budget: float = 0.0,
    budget: int = 450,
):
    """Run CLI subprocess and validate resulting pipeline."""
    feats = tmp_path / "features.csv"
    perf = tmp_path / "performance.csv"
    features_df, _ = _write_data(feats, perf, n_instances=20)

    out_model = tmp_path / "out_model.pkl"

    cmd = build_cli_command(
        selector=selector,
        feature_data=feats,
        performance_data=perf,
        destination=out_model,
        model="Ridge",
        tuning=tuning,
        budget=budget,
        maximize=False,
        preprocessors=preprocessors,
        presolvers=presolvers,
        presolver_budget=presolver_budget,
    )

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    assert result.returncode == 0
    assert out_model.exists()

    pipeline = SelectorPipeline.load(out_model)

    expected_selector_types = [
        type(s) if not isinstance(s, type) else s for s in selector
    ]
    expected_preprocessor_types = (
        [type(p) for p in preprocessors] if preprocessors else None
    )
    expected_presolver_type = type(presolvers[0]) if presolvers else None

    _validate_pipeline_extensive(
        pipeline=pipeline,
        features=features_df,
        expected_budget=budget,
        expected_selector_types=expected_selector_types,
        expected_preprocessor_types=expected_preprocessor_types,
        expected_presolver_type=expected_presolver_type,
        presolver_budget_fraction=presolver_budget,
        tuning=tuning,
    )


def test_cli_basic(tmp_path: Path):
    """Test basic CLI: single selector, no preprocessing or presolving."""
    _run_cli_and_validate(tmp_path, selector=[SATzilla])


def test_cli_with_preprocessing(tmp_path: Path):
    """Test CLI with preprocessing."""
    _run_cli_and_validate(
        tmp_path, selector=[SATzilla], preprocessors=[StandardScaler()]
    )


def test_cli_with_presolver(tmp_path: Path):
    """Test CLI with presolver."""
    _run_cli_and_validate(
        tmp_path,
        selector=[SATzilla],
        presolvers=[ASAPv2()],
        presolver_budget=0.1,
        budget=500,
    )


def test_cli_tuning(tmp_path: Path):
    """Test CLI with tuning: multiple selectors, preprocessing and presolving."""
    _run_cli_and_validate(
        tmp_path,
        selector=[SATzilla, PairwiseRegressor],
        tuning=True,
        preprocessors=[StandardScaler(), MinMaxScaler()],
        presolvers=[ASAPv2()],
        presolver_budget=0.15,
        budget=500,
    )
