from functools import partial

import pandas as pd

from asf.cli import cli_train
from asf import selectors


def test_parser_function_has_expected_arguments():
    parser = cli_train.parser_function()
    args = [a.dest for a in parser._actions]
    for needed in [
        "selectors",
        "model",
        "budget",
        "maximize",
        "feature_data",
        "performance_data",
        "model_path",
    ]:
        assert needed in args


def test_build_cli_command_with_partial_and_direct_model(tmp_path):
    # Build a minimal selector; use an AbstractModelBasedSelector subclass available in package
    # We'll fake a minimal instance exposing attributes used by build_cli_command.
    class FakeSelector(selectors.AbstractModelBasedSelector):
        def __init__(self, model_class, budget=123, maximize=True):
            # model_class can be partial or a class; this mimics real selectors' signature
            self.model_class = model_class
            self.budget = budget
            self.maximize = maximize

        # Unused abstract methods in this test
        def fit(self, X, y):
            return self

        def predict(self, X):
            return {}

        def save(self, path):
            pass

    # two DataFrame inputs saved temporarily to ensure suffix mapping works in build args
    feat = tmp_path / "f.csv"
    perf = tmp_path / "p.csv"
    pd.DataFrame({"a": [1]}).to_csv(feat)
    pd.DataFrame({"A": [0.1]}).to_csv(perf)
    dst = tmp_path / "model.pkl"

    # Case 1: model_class is a partial
    class SomeModel:
        __name__ = "SomeModel"

    # Provide a partial whose first arg is a class so build_cli_command can read __name__
    selector1 = FakeSelector(
        model_class=partial(lambda x: x, SomeModel), budget=5, maximize=False
    )
    cmd1 = cli_train.build_cli_command(selector1, feat, perf, dst)
    # Ensure important flags present and values serialized
    assert "--selectors" in cmd1 and "--model" in cmd1
    assert "--feature-data" in cmd1 and str(feat) in cmd1
    assert "--performance-data" in cmd1 and str(perf) in cmd1
    assert "--model-path" in cmd1 and str(dst) in cmd1

    # Case 2: model_class is a direct class
    selector2 = FakeSelector(model_class=SomeModel, budget=7, maximize=True)
    cmd2 = cli_train.build_cli_command(selector2, feat, perf, dst)
    # Ensure model name appears
    assert "SomeModel" in cmd2
