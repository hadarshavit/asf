"""
Utilities for loading algorithm features from ASLib scenarios.
"""

import os
import io

import pandas as pd


def _load_arff_bytes(b: bytes) -> pd.DataFrame:
    """
    Load an ARFF file from bytes and return a pandas DataFrame.

    Parameters
    ----------
    b : bytes
        The ARFF file content as bytes.

    Returns
    -------
    pd.DataFrame
        The data from the ARFF file.
    """
    try:
        import arff
    except Exception as e:
        raise ImportError(
            "liac-arff is required to parse .arff files: pip install liac-arff"
        ) from e
    text = (
        b.decode("utf-8", errors="replace")
        if isinstance(b, (bytes, bytearray))
        else str(b)
    )
    obj = arff.load(io.StringIO(text))
    cols = [str(c[0]) for c in obj["attributes"]]
    return pd.DataFrame(obj["data"], columns=cols)  # type: ignore[arg-type]


def get_algorithm_features_from_aslib(scenario_dir: str) -> pd.DataFrame:
    """
    Load algorithm_feature_values.arff from an ASLib scenario directory.

    Returns a numeric DataFrame indexed by algorithm name.

    Assumptions / behavior:
      - scenario_dir must be a directory containing 'algorithm_feature_values.arff'.
      - The ARFF must contain a column naming the algorithm (exactly 'algorithm' or a column
        whose name contains 'algo' or 'name'). Otherwise an error is raised.
      - Non-numeric columns are dropped; if multiple repetitions exist rows are averaged per algorithm.

    Parameters
    ----------
    scenario_dir : str
        Path to the ASLib scenario directory.

    Returns
    -------
    pd.DataFrame
        Algorithm features, indexed by algorithm name.

    Raises
    ------
    FileNotFoundError
        If the scenario directory or ARFF file is not found.
    ValueError
        If no algorithm name column or no numeric features are found.
    """
    if not os.path.isdir(scenario_dir):
        raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")

    path = os.path.join(scenario_dir, "algorithm_feature_values.arff")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Expected 'algorithm_feature_values.arff' in {scenario_dir}"
        )

    with open(path, "rb") as fh:
        df = _load_arff_bytes(fh.read())

    if "algorithm" not in df.columns:
        candidates = [
            c for c in df.columns if "algo" in c.lower() or "name" in c.lower()
        ]
        if not candidates:
            raise ValueError("Could not find an algorithm name column in ARFF")
        df = df.rename(columns={candidates[0]: "algorithm"})

    df = df.drop(columns=[c for c in ("repetition",) if c in df.columns])
    alg_col = df["algorithm"].astype(str)
    df_num = df.select_dtypes(include=["number"]).copy()
    if df_num.empty:
        raise ValueError("No numeric algorithm feature columns found in ARFF")
    df_num.insert(0, "algorithm", alg_col.values)
    df_grp = df_num.groupby("algorithm", as_index=True).mean()
    return df_grp.astype(float)
