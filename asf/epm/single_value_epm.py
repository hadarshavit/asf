from functools import partial

from sklearn.base import RegressorMixin
from asf.epm.epm import AbstractEPM

from asf.predictors import SklearnWrapper
from asf.predictors.abstract_predictor import AbstractPredictor
from asf.predictors.random_forest import RandomForestRegressorWrapper


class SingleValueEPM(AbstractEPM):
    """
    The EPM (Empirical Performance Model) class is a wrapper for machine learning models
    that includes preprocessing, normalization, and optional inverse transformation of predictions.

    Attributes:
        predictor_class (type[AbstractPredictor] | type[RegressorMixin]): The class of the predictor to use.
        normalization_class (type[AbstractNormalization]): The normalization class to apply to the target variable.
        transform_back (bool): Whether to apply inverse transformation to predictions.
        features_preprocessing (str | TransformerMixin): Preprocessing pipeline for features.
        predictor_config (dict | None): Configuration for the predictor.
        predictor_kwargs (dict | None): Additional keyword arguments for the predictor.
    """

    def __init__(
        self,
        predictor_class: (
            type[AbstractPredictor] | type[RegressorMixin]
        ) = RandomForestRegressorWrapper,
        **kwargs,
    ):
        """
        Initialize the EPM model.

        Parameters:
            predictor_class (type[AbstractPredictor] | type[RegressorMixin]): The class of the predictor to use.
            predictor_kwargs (dict | None): Additional keyword arguments for the predictor.
        """
        super().__init__(**kwargs)

        if isinstance(predictor_class, type) and issubclass(
            predictor_class, (RegressorMixin)
        ):
            self.model_class = partial(SklearnWrapper, predictor_class)
        else:
            self.model_class = predictor_class

    def _fit(self, X, y, sample_weight):
        self.predictor = self._get_predictor()

        self.predictor.fit(X, y, sample_weight=sample_weight)

        return self

    def _get_predictor(self) -> AbstractPredictor:
        if self.predictor_config is None:
            predictor = self.predictor_class(**self.predictor_kwargs)
        else:
            predictor = self.predictor_class.get_from_configuration(
                self.predictor_config, **self.predictor_kwargs
            )()

        return predictor

    def _predict(self, X):
        y_pred = self.predictor.predict(X)

        return y_pred
