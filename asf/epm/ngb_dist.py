from functools import partial
from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    from ngboost import NGBRegressor
    from ngboost.distns import LogNormal, Normal, Exponential, RegressionDistn
    from ngboost.scores import LogScore
    from sklearn.tree import DecisionTreeRegressor
    import scipy.stats as sp
    NGBOOST_AVAILABLE = True
except ImportError:
    NGBOOST_AVAILABLE = False

if NGBOOST_AVAILABLE:
    class InverseGaussianLogScore(LogScore):
        def score(self, Y):
            return -self.dist.logpdf(Y)

        def d_score(self, Y):
            D = np.zeros((len(Y), 2))
            # NLL = const - 0.5*log(lam) + 1.5*log(y) + lam*(y-mu)^2/(2*mu^2*y)
            # Since params are log-transformed: mu = exp(params[0]), lam = exp(params[1])
            # Need chain rule: d/d(params[i]) = d/d(param_raw) * param_raw
            
            # d(NLL)/d(mu) = -lam * (y - mu) / mu^3
            # d(NLL)/d(log_mu) = d(NLL)/d(mu) * mu = -lam * (y - mu) / mu^2
            D[:, 0] = -self.lam * (Y - self.mu) / (self.mu ** 2)
            
            # d(NLL)/d(lam) = -0.5/lam + (y - mu)^2 / (2 * mu^2 * y)
            # d(NLL)/d(log_lam) = d(NLL)/d(lam) * lam = -0.5 + lam * (y - mu)^2 / (2 * mu^2 * y)
            D[:, 1] = -0.5 + self.lam * (Y - self.mu) ** 2 / (2 * self.mu ** 2 * Y)
            return D

        def metric(self):
            FI = np.zeros((self.mu.shape[0], 2, 2))
            # Fisher Information for log-transformed parameters
            # FI[log_mu, log_mu] = mu^2 * FI[mu, mu] = mu^2 * lam/mu = mu*lam
            FI[:, 0, 0] = self.mu * self.lam
            # FI[log_lam, log_lam] = lam^2 * FI[lam, lam] = lam^2 * 0.5/lam^2 = 0.5
            FI[:, 1, 1] = 0.5
            return FI

    class InverseGaussian(RegressionDistn):
        n_params = 2
        scores = [InverseGaussianLogScore]

        def __init__(self, params):
            super().__init__(params)
            self.mu = np.exp(params[0])
            self.lam = np.exp(params[1])
            self.dist = sp.invgauss(mu=self.mu/self.lam, scale=self.lam)

        def fit(Y):
            mu_est = np.mean(Y)
            # harmonic mean for lambda
            # 1/lambda = 1/n * sum(1/yi - 1/ybar)
            # Actually MLE for lambda is (1/n sum(1/yi - 1/mu))^-1
            inv_y = 1.0 / Y
            inv_mu = 1.0 / mu_est
            val = np.mean(inv_y - inv_mu)
            lam_est = 1.0 / val
            return np.array([np.log(mu_est), np.log(lam_est)])
        
        @property
        def params(self):
            return {"mu": self.mu, "lambda": self.lam}


try:
    from ConfigSpace import (
        ConfigurationSpace,
        EqualsCondition,
        Float,
        Integer,
    )
    from ConfigSpace.hyperparameters import Hyperparameter

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class NGBDistNet:
    PREFIX = "ngb_distnet"

    def __init__(
        self,
        distribution: str = "lognorm",
        n_estimators: int = 500,
        learning_rate: float = 0.01,
        minibatch_frac: float = 1.0,
        col_sample: float = 1.0,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        early_stopping_rounds: int | None = None,
        early_stopping_tolerance: float = 1e-4,
        input_size=None,
        output_size=None,
        device: str = "cpu",
        **kwargs,
    ):
        if not NGBOOST_AVAILABLE:
            raise ImportError("ngboost is not installed.")

        self.distribution = distribution
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.col_sample = col_sample
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_tolerance = early_stopping_tolerance
        self.device = device
        self.kwargs = kwargs
        
        self.model = None

    def fit(
        self,
        X: pd.DataFrame | pd.Series | list,
        y: pd.Series | list,
    ):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values

        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        X = np.concatenate([[x for i in range(y.shape[1])] for x in X]).astype(
            np.float32, copy=False
        )
        y = y.flatten().astype(np.float32, copy=False)

        # Define base learner with max_depth
        base_learner = DecisionTreeRegressor(
            criterion='friedman_mse', 
            min_samples_split=self.min_samples_split, 
            min_samples_leaf=self.min_samples_leaf, 
            min_weight_fraction_leaf=0.0, 
            max_depth=self.max_depth, 
            splitter='best'
        )

        if self.distribution == "lognorm":
            Dist = LogNormal
        elif self.distribution == "norm":
            Dist = Normal
        elif self.distribution == "exp":
            Dist = Exponential
        elif self.distribution == "invgauss":
            Dist = InverseGaussian
        else:
            raise ValueError(f"Distribution {self.distribution} not supported.")

        self.model = NGBRegressor(
            Dist=Dist,
            Score=LogScore,
            Base=base_learner,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            minibatch_frac=self.minibatch_frac,
            col_sample=self.col_sample,
            verbose=False,
            early_stopping_rounds=self.early_stopping_rounds,
            tol=self.early_stopping_tolerance,
            **self.kwargs
        )

        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame | pd.Series | list) -> np.ndarray:
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        X = np.asarray(X, dtype=np.float32)

        dist = self.model.pred_dist(X)
        
        if self.distribution == "lognorm":
            # LogNormal in ngboost has params 's' and 'scale'
            # XGBDistNet returns [s, scale]
            s = dist.params['s']
            scale = dist.params['scale']
            predictions = np.stack([s, scale], axis=1)
        elif self.distribution == "norm":
            # Normal in ngboost has params 'loc' and 'scale'
            # asf normal_loss expects [mu, sigma] -> [loc, scale]
            loc = dist.params['loc']
            scale = dist.params['scale']
            predictions = np.stack([loc, scale], axis=1)
        elif self.distribution == "exp":
            # Exponential in ngboost has params 'scale'
            # asf exp_loss expects [scale]
            scale = dist.params['scale']
            predictions = scale.reshape(-1, 1)
        elif self.distribution == "invgauss":
            # InverseGaussian has params 'mu' and 'lambda'
            # asf invgauss_loss expects [nu, lam] where mu = nu * lam
            # so nu = mu / lam
            mu = dist.params['mu']
            lam = dist.params['lambda']
            nu = mu / lam
            predictions = np.stack([nu, lam], axis=1)
        else:
            raise ValueError(f"Distribution {self.distribution} not supported.")
        
        return predictions

    def save(self, path: str):
        import pickle
        save_dict = {
            'model': self.model,
            'distribution': self.distribution,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'minibatch_frac': self.minibatch_frac,
            'col_sample': self.col_sample,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'early_stopping_rounds': self.early_stopping_rounds,
            'early_stopping_tolerance': self.early_stopping_tolerance,
            'device': self.device,
            'kwargs': self.kwargs,
        }
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)

    @classmethod
    def load(cls, path: str):
        """Load the NGBDistNet model from a file.
        
        Parameters
        ----------
        path : str
            The file path from which the model will be loaded.
            
        Returns
        -------
        NGBDistNet
            Loaded model instance.
        """
        import pickle
        with open(path, "rb") as f:
            save_dict = pickle.load(f)
        
        # Create instance with saved parameters
        instance = cls(
            distribution=save_dict['distribution'],
            n_estimators=save_dict['n_estimators'],
            learning_rate=save_dict['learning_rate'],
            minibatch_frac=save_dict['minibatch_frac'],
            col_sample=save_dict['col_sample'],
            max_depth=save_dict['max_depth'],
            min_samples_split=save_dict['min_samples_split'],
            min_samples_leaf=save_dict['min_samples_leaf'],
            early_stopping_rounds=save_dict.get('early_stopping_rounds'),
            early_stopping_tolerance=save_dict.get('early_stopping_tolerance', 1e-4),
            device=save_dict.get('device', 'cpu'),
            **save_dict.get('kwargs', {})
        )
        
        # Restore the trained model
        instance.model = save_dict['model']
        
        return instance
    
    def load_old_format(self, path: str):
        """Load model saved in old format (pickled model only).
        
        Parameters
        ----------
        path : str
            The file path from which the model will be loaded.
        """
        import pickle
        with open(path, "rb") as f:
            self.model = pickle.load(f)

    @staticmethod
    def get_configuration_space(
        cs: ConfigurationSpace | None = None,
        pre_prefix: str = "",
        parent_param: Hyperparameter | None = None,
        parent_value: str | None = None,
    ) -> ConfigurationSpace:
        if cs is None:
            cs = ConfigurationSpace(name="NGBRegressor")

        if pre_prefix != "":
            prefix = f"{pre_prefix}:{NGBDistNet.PREFIX}"
        else:
            prefix = NGBDistNet.PREFIX

        n_estimators = Integer(
            f"{prefix}:n_estimators",
            (100, 2000),
            log=True,
            default=500,
        )
        learning_rate = Float(
            f"{prefix}:learning_rate",
            (0.001, 0.5),
            log=True,
            default=0.01,
        )
        minibatch_frac = Float(
            f"{prefix}:minibatch_frac",
            (0.1, 1.0),
            log=False,
            default=1.0,
        )
        col_sample = Float(
            f"{prefix}:col_sample",
            (0.1, 1.0),
            log=False,
            default=1.0,
        )
        max_depth = Integer(
            f"{prefix}:max_depth",
            (1, 10),
            log=False,
            default=3,
        )
        min_samples_split = Integer(
            f"{prefix}:min_samples_split",
            (2, 20),
            log=False,
            default=2,
        )
        min_samples_leaf = Integer(
            f"{prefix}:min_samples_leaf",
            (1, 20),
            log=False,
            default=1,
        )

        params = [
            n_estimators,
            learning_rate,
            minibatch_frac,
            col_sample,
            max_depth,
            min_samples_split,
            min_samples_leaf,
        ]
        
        if parent_param is not None:
            conditions = [
                EqualsCondition(
                    child=param,
                    parent=parent_param,
                    value=parent_value,
                )
                for param in params
            ]
        else:
            conditions = []

        cs.add(params + conditions)

        return cs

    @staticmethod
    def get_from_configuration(
        configuration: dict[str, Any],
        pre_prefix: str = "",
        **kwargs,
    ) -> Callable[..., "NGBDistNet"]:
        if pre_prefix != "":
            prefix = f"{pre_prefix}:{NGBDistNet.PREFIX}"
        else:
            prefix = NGBDistNet.PREFIX

        ngb_params = {
            "n_estimators": configuration[f"{prefix}:n_estimators"],
            "learning_rate": configuration[f"{prefix}:learning_rate"],
            "minibatch_frac": configuration[f"{prefix}:minibatch_frac"],
            "col_sample": configuration[f"{prefix}:col_sample"],
            "max_depth": configuration[f"{prefix}:max_depth"],
            "min_samples_split": configuration[f"{prefix}:min_samples_split"],
            "min_samples_leaf": configuration[f"{prefix}:min_samples_leaf"],
            **kwargs,
        }

        return partial(NGBDistNet, **ngb_params)
