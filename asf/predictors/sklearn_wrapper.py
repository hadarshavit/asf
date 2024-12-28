from sklearn.base import ClassifierMixin
from asf.selectors.abstract_selector import AbstractSelector


class SklearnWrapper(AbstractSelector):
    """
    A generic wrapper for scikit-learn models.

    This class allows scikit-learn models to be used with the ASF framework.

    Methods
    -------
    fit(X, Y)
        Fit the model to the data.
    predict(X)
        Predict using the model.
    save(file_path)
        Save the model to a file.
    load(file_path)
        Load the model from a file.
    """

    def __init__(self, model_class: ClassifierMixin):
        """
        Initialize the wrapper with a scikit-learn model.

        Parameters
        ----------
        model_class : ClassifierMixin
            An instance of a scikit-learn model.
        """
        self.model_class = model_class()

    def fit(self, X, Y):
        """
        Fit the model to the data.

        Parameters
        ----------
        X : array-like
            Training data.
        Y : array-like
            Target values.
        """
        self.model_class.fit(X, Y)

    def predict(self, X):
        """
        Predict using the model.

        Parameters
        ----------
        X : array-like
            Data to predict on.

        Returns
        -------
        array-like
            Predicted values.
        """
        return self.model_class.predict(X)

    def save(self, file_path: str):
        """
        Save the model to a file.

        Parameters
        ----------
        file_path : str
            Path to the file where the model will be saved.
        """
        import joblib

        joblib.dump(self, file_path)

    def load(self, file_path: str):
        """
        Load the model from a file.

        Parameters
        ----------
        file_path : str
            Path to the file from which the model will be loaded.

        Returns
        -------
        SklearnWrapper
            The loaded model.
        """
        import joblib

        return joblib.load(file_path)
