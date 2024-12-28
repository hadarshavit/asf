from asf.selectors.abstract_selector import AbstractSelector
from asf.predictors import SklearnWrapper, AbstractPredictor
from sklearn.base import ClassifierMixin, RegressorMixin
from functools import partial


class AbstractModelBasedSelector(AbstractSelector):
    def __init__(self, model_class, metadata, hierarchical_generator=...):
        super().__init__(metadata, hierarchical_generator)

        if issubclass(model_class, AbstractPredictor):
            self.model_class = model_class
        elif issubclass(model_class, (ClassifierMixin, RegressorMixin)):
            self.model_class = partial(SklearnWrapper, model_class)
        else:
            raise ValueError(
                "model_class must be an instance of AbstractPredictor or ClassifierMixin"
            )
