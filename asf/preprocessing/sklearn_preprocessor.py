from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from typing import Callable


def get_default_preprocessor(
    categorical_features: list[str] | Callable | None = None,
    numerical_features: list[str] | Callable | None = None,
) -> ColumnTransformer:
    """
    Creates a default preprocessor for handling categorical and numerical features.

    Args:
        categorical_features (list[str] | Callable | None):
            List of categorical feature names or a callable selector. Defaults to selecting object dtype columns.
        numerical_features (list[str] | Callable | None):
            List of numerical feature names or a callable selector. Defaults to selecting numeric dtype columns.

    Returns:
        ColumnTransformer: A transformer that applies preprocessing pipelines to categorical and numerical features.
    """
    if categorical_features is None:
        categorical_features = make_column_selector(dtype_include=object)

    if numerical_features is None:
        numerical_features = make_column_selector(dtype_include="number")

    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                ),
                categorical_features,
            ),
            (
                "cont",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numerical_features,
            ),
        ]
    )

    preprocessor.set_output(transform="pandas")
    return preprocessor
