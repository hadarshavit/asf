#!/usr/bin/env python3
"""CLI entry point for training selectors."""
import ast
import argparse
from pathlib import Path

import pandas as pd

from asf import selectors
from asf.scenario.scenario_metadata import ScenarioMetadata


pandas_read_map = {
    "csv": pd.read_csv,
    "parquet": pd.read_parquet,
    "json": pd.read_json,
    "feather": pd.read_feather,
    "hdf": pd.read_hdf,
    "html": pd.read_html,
    "xml": pd.read_xml,
}


def parser_function() -> argparse.ArgumentParser:
    """Define command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector",
                        choices=selectors.__implemented__,
                        required=True,
                        help="Selector to train")
    parser.add_argument("--model",
                        default="sklearn.ensemble.RandomForestClassifier",
                        help="Model to use for the selector. "
                             "Make sure to specify as a full module path.")
    parser.add_argument("--metadata",
                        required=False,
                        type=str,
                        default=None,
                        help="Metadata for the selector, represented as dictionary.")
    parser.add_argument("--feature-data",
                        type=Path,
                        required=True,
                        help="Path to feature data")
    parser.add_argument("--performance-data",
                        type=Path,
                        required=True,
                        help="Path to performance data")
    parser.add_argument("--model-path",
                        type=Path,
                        required=True,
                        help="Path to save model")
    return parser


def build_cli_command(selector: selectors.AbstractModelBasedSelector,
                   features: Path,
                   performance_data: Path,
                   destination: Path) -> list[str]:
    """Build CLI command from variables for async jobs.

    Args:
        selector: Selector to train
        features: Path to feature data DataFrame
        performance_data: Path to performance data DataFrame
        destination: Path to save model
    """
    return ["python", Path(__file__).absolute(),
            "--selector", selector,
            "--model", str(type(selector.model_class)),
            "--feature-data", str(features),
            "--performance-data", str(performance_data),
            "--model-path", str(destination)]


if __name__ == "__main__":
    parser = parser_function()
    args = parser.parse_args()
    selector_name = args.selector if "selectors." in args.selector else f"selectors.{args.selector}"
    metadata = args.metadata
    if metadata:
        metadata = ScenarioMetadata(**ast.literal_eval(metadata))
    # Parse selector in to variable
    selector: selectors.AbstractModelBasedSelector = eval(selector_name)(type(args.model), metadata)

    # Parse training data into variables
    features = pandas_read_map[args.feature_data.suffix](args.feature_data)
    performance_data = pandas_read_map[args.performance_data.suffix](args.performance_data)
    selector.fit(features, performance_data)

    # Save the model to the specified path
    selector.save(args.model_path)
