from __future__ import annotations
from functools import partial
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from asf.utils.g_means import GMeans
from asf.utils.configurable import ConfigurableMixin

try:
    from ConfigSpace import (  # noqa: F401
        ConfigurationSpace,
        Categorical,
        Integer,
        Float,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class GMeansWrapper(ConfigurableMixin):
    PREFIX = "gmeans"

    def __init__(self, **kwargs):
        self.model = GMeans(**kwargs)

    def fit(self, X):
        self.model.fit(X)
        return self

    def predict(self, X):
        return self.model.predict(X)

    @staticmethod
    def _define_hyperparameters(**kwargs):
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        params = [
            Float("min_samples", (0.0001, 0.1), default=0.001, log=True),
            Categorical("significance", [0.15, 0.1, 0.05, 0.025, 0.001], default=0.05),
            Integer("n_init", (1, 10), default=5),
        ]
        return params, [], []

    @classmethod
    def _get_from_clean_configuration(cls, clean_config, **kwargs):
        config = clean_config.copy()
        config.update(kwargs)
        return partial(GMeansWrapper, **config)


class KMeansWrapper(ConfigurableMixin):
    PREFIX = "kmeans"

    def __init__(self, **kwargs):
        self.model = KMeans(**kwargs)

    def fit(self, X):
        self.model.fit(X)
        return self

    def predict(self, X):
        return self.model.predict(X)

    @staticmethod
    def _define_hyperparameters(**kwargs):
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        params = [
            Integer("n_clusters", (2, 20), default=5),
        ]
        return params, [], []

    @classmethod
    def _get_from_clean_configuration(cls, clean_config, **kwargs):
        config = clean_config.copy()
        config.update(kwargs)
        return partial(KMeansWrapper, **config)


class AgglomerativeClusteringWrapper(ConfigurableMixin):
    PREFIX = "agglomerative_clustering"

    def __init__(self, **kwargs):
        self.model = AgglomerativeClustering(**kwargs)

    def fit(self, X):
        self.model.fit(X)
        return self

    def predict(self, X):
        # AgglomerativeClustering does not support predict()
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        raise NotImplementedError("AgglomerativeClustering does not support predict()")

    @staticmethod
    def _define_hyperparameters(**kwargs):
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        params = [
            Integer("n_clusters", (2, 20), default=5),
            Categorical(
                "linkage", ["ward", "complete", "average", "single"], default="ward"
            ),
        ]
        return params, [], []

    @classmethod
    def _get_from_clean_configuration(cls, clean_config, **kwargs):
        config = clean_config.copy()
        config.update(kwargs)
        return partial(AgglomerativeClusteringWrapper, **config)


class DBSCANWrapper(ConfigurableMixin):
    PREFIX = "dbscan"

    def __init__(self, **kwargs):
        self.model = DBSCAN(**kwargs)

    def fit(self, X):
        self.model.fit(X)
        return self

    def predict(self, X):
        # DBSCAN does not support predict()
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        raise NotImplementedError("DBSCAN does not support predict()")

    @staticmethod
    def _define_hyperparameters(**kwargs):
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        params = [
            Float("eps", (0.1, 2.0), default=0.5),
            Integer("min_samples", (2, 10), default=5),
        ]
        return params, [], []

    @classmethod
    def _get_from_clean_configuration(cls, clean_config, **kwargs):
        config = clean_config.copy()
        config.update(kwargs)
        return partial(DBSCANWrapper, **config)
