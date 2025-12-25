from __future__ import annotations

import re
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from asf.selectors.abstract_selector import AbstractSelector
from asf.utils.configurable import ConfigurableMixin

try:
    import torch
    import torch.nn as nn
    import torch.utils.data as Data

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ConfigSpace import (  # noqa: F401
        Categorical,
        ConfigurationSpace,
        Float,
        Integer,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


if TORCH_AVAILABLE:

    class _ASLLMRecommendationModel(nn.Module):
        num_algorithms: int
        num_user_features: int
        alpha: float
        beta: float
        """
        Neural network model implementing the AS-LLM architecture.

        Architecture:
        - User (instance) features → MLP → user embedding
        - Algorithm features (provided) → frozen embedding
        - Algorithm index → learnable embedding → LSTM → encoding
        - Fusion: α * LSTM_encoding + β * algorithm_features
        - Fused features → MLP → algorithm embedding
        - Cosine similarity between user and algorithm embeddings
        - Concat(user_emb, algo_emb, cosine_sim) → MLP → binary classification
        """

        def __init__(
            self,
            num_algorithms: int,
            num_user_features: int,
            algorithm_features: torch.Tensor,
            embed_size: int = 50,
            num_hiddens: int = 50,
            num_layers: int = 2,
            output_dim: int = 10,
            alpha: float = 0.9,
            beta: float = 0.1,
        ):
            super().__init__()
            self.num_algorithms = num_algorithms  # type: ignore[attr-defined]
            self.num_user_features = num_user_features  # type: ignore[attr-defined]
            self.alpha = alpha  # type: ignore[attr-defined]
            self.beta = beta  # type: ignore[attr-defined]

            # User (instance) feature MLP: input → 50 → output_dim
            self.user_mlp = nn.Sequential(
                nn.Linear(num_user_features, 50),
                nn.Linear(50, output_dim),
            )

            # Frozen algorithm features embedding (from self.algorithm_features)
            self.llm_embedding = nn.Embedding.from_pretrained(
                algorithm_features.float(), freeze=True
            )

            # Learnable algorithm embedding
            self.item_embedding = nn.Embedding(num_algorithms, embed_size)

            # LSTM for processing algorithm embeddings
            self.algorithm_lstm = nn.LSTM(
                input_size=embed_size,
                hidden_size=num_hiddens,
                num_layers=num_layers,
                bidirectional=False,
            )

            # Algorithm feature MLP: 2*num_hiddens (or algo_feat_dim) → 50 → output_dim
            # After fusion, dimension is 2*num_hiddens (from LSTM concat)
            self.algorithm_mlp = nn.Sequential(
                nn.Linear(2 * num_hiddens, 50),
                nn.ReLU(),
                nn.Linear(50, output_dim),
            )

            # Cosine similarity
            self.cosine_similarity = nn.CosineSimilarity(dim=1)

            # Final MLP: concat(user_emb, algo_emb, cosine_sim) → output
            self.concat_mlp = nn.Sequential(
                nn.Linear(output_dim + output_dim + 1, 10),
                nn.Linear(10, 10),
                nn.ReLU(),
                nn.Linear(10, 2),  # Binary classification: match or not
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.

            Parameters
            ----------
            inputs : torch.Tensor
                Shape: (batch, num_user_features + num_algorithms + 1)
                First num_user_features columns: instance features
                Next num_algorithms columns: one-hot algorithm encoding
                Last column: algorithm index

            Returns
            -------
            torch.Tensor
                Shape: (batch, 2) - binary classification logits
            """
            # Extract user features
            user_features = inputs[:, : self.num_user_features]
            user_vector = self.user_mlp(user_features)

            # Extract algorithm one-hot and index
            algo_onehot = inputs[:, self.num_user_features : -1]
            algo_idx = inputs[:, -1].long()

            # Algorithm embedding through LSTM
            # algo_onehot shape: (batch, num_algorithms) - used as sequence input
            item_embed = self.item_embedding(algo_onehot.permute(1, 0).long())
            # item_embed shape: (num_algorithms, batch, embed_size)

            lstm_out, _ = self.algorithm_lstm(item_embed)
            # Concatenate first and last hidden states
            encoding = torch.cat((lstm_out[0], lstm_out[-1]), dim=-1)
            # encoding shape: (batch, 2*num_hiddens)

            # Get frozen algorithm features for this algorithm
            llm_embed = self.llm_embedding(algo_idx)
            # llm_embed shape: (batch, algo_feat_dim)

            # Fusion: alpha * encoding + beta * llm_embed
            # Need to project llm_embed to match encoding dimension
            # For simplicity, we'll pad or truncate if dimensions don't match
            if llm_embed.shape[1] != encoding.shape[1]:
                # Project llm_embed to encoding dimension
                if not hasattr(self, "_llm_proj"):
                    self._llm_proj = nn.Linear(
                        llm_embed.shape[1], encoding.shape[1]
                    ).to(encoding.device)
                llm_embed = self._llm_proj(llm_embed)

            alg_feature = self.alpha * encoding + self.beta * llm_embed

            # Algorithm MLP
            item_vector = self.algorithm_mlp(alg_feature)

            # Cosine similarity
            similarity = self.cosine_similarity(user_vector, item_vector)

            # Concatenate and final MLP
            concat_input = torch.cat(
                (user_vector, item_vector, similarity.unsqueeze(1)), dim=1
            )
            output = self.concat_mlp(concat_input)

            return output
else:

    class _ASLLMRecommendationModel:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            pass

        def train(self):
            pass

        def eval(self):
            pass

        def parameters(self):
            return []


class CosineSelector(ConfigurableMixin, AbstractSelector):
    """
    Algorithm selector based on the AS-LLM architecture.

    Uses cosine similarity in a learned latent space to match instances to algorithms.
    This implementation follows the AS-LLM paper (arXiv:2311.13184) architecture:
    - Instance features → MLP → instance embedding
    - Algorithm index → Embedding → LSTM → fused with algorithm_features → algorithm embedding
    - Cosine similarity + MLP for final compatibility prediction

    Attributes
    ----------
    normalize_features : bool
        If True, standardize instance features.
    embed_size : int
        Dimensionality of algorithm embeddings.
    num_hiddens : int
        Hidden units in LSTM.
    num_layers : int
        Number of LSTM layers.
    alpha : float
        Weight for learned LSTM features in fusion.
    beta : float
        Weight for algorithm_features in fusion.
    lr : float
        Learning rate for training.
    num_epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    device : str
        Device for PyTorch ('cuda' or 'cpu').
    """

    PREFIX = "cosine"
    RETURN_TYPE = "single"

    # Type hints for attributes set in __init__
    normalize_features: bool
    embed_size: int
    num_hiddens: int
    num_layers: int
    alpha: float
    beta: float
    lr: float
    num_epochs: int
    batch_size: int
    random_state: int
    _device: torch.device
    _model: nn.Module | None
    _scaler: StandardScaler | None
    _imputer: SimpleImputer | None
    _alg_feats: pd.DataFrame | None
    algorithms: list[str]

    def __init__(
        self,
        normalize_features: bool = True,
        embed_size: int = 50,
        num_hiddens: int = 50,
        num_layers: int = 2,
        alpha: float = 0.9,
        beta: float = 0.1,
        lr: float = 0.001,
        num_epochs: int = 100,
        batch_size: int = 128,
        device: str | None = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the CosineSelector (AS-LLM architecture).

        Parameters
        ----------
        normalize_features : bool, default=True
            If True, standardize instance features.
        embed_size : int, default=50
            Dimensionality of algorithm embeddings.
        num_hiddens : int, default=50
            Hidden units in LSTM.
        num_layers : int, default=2
            Number of LSTM layers.
        alpha : float, default=0.9
            Weight for learned LSTM features in fusion.
        beta : float, default=0.1
            Weight for algorithm_features in fusion.
        lr : float, default=0.001
            Learning rate for training.
        num_epochs : int, default=100
            Number of training epochs.
        batch_size : int, default=128
            Batch size for training.
        device : str, default=None
            Device for PyTorch ('cuda', 'cpu'). If None, auto-detect.
        random_state : int, default=42
            Random seed for reproducibility.
        **kwargs : Any
            Additional keyword arguments.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "CosineSelector requires PyTorch. Install with: pip install torch"
            )

        super().__init__(**kwargs)
        self.normalize_features = bool(normalize_features)
        self.embed_size = int(embed_size)
        self.num_hiddens = int(num_hiddens)
        self.num_layers = int(num_layers)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lr = float(lr)
        self.num_epochs = int(num_epochs)
        self.batch_size = int(batch_size)
        self.random_state = int(random_state)

        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._model: nn.Module | None = None
        self._scaler: StandardScaler | None = None
        self._imputer: SimpleImputer | None = None
        self._alg_feats: pd.DataFrame | None = None

    def _norm(self, s: str) -> str:
        """Minimal string normalization for matching algorithm identifiers."""
        s = str(s).lower().strip()
        return re.sub(r"[\W_]+", "", s)

    def _generate_training_data(
        self,
        features: np.ndarray,
        performance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate pairwise training data.

        For each instance, create (instance, algorithm) pairs with label 1 if
        the algorithm is best for that instance, 0 otherwise.
        """
        num_instances = features.shape[0]
        num_algorithms = performance.shape[1]

        training_data = []
        training_labels = []

        for i in range(num_instances):
            best_algo = int(np.argmin(performance[i]))
            for j in range(num_algorithms):
                # One-hot encoding for algorithm
                alg_embed = [0] * num_algorithms
                alg_embed[j] = 1
                # Concatenate: instance features + one-hot + algorithm index
                sample = np.append(np.append(features[i], alg_embed), [j])
                training_data.append(sample)
                training_labels.append(1 if j == best_algo else 0)

        return np.array(training_data), np.array(training_labels)

    def _fit(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        """
        Fit the AS-LLM selector.

        Parameters
        ----------
        features : pd.DataFrame
            Instance feature matrix (rows = instances).
        performance : pd.DataFrame
            Performance matrix (rows = instances, columns = algorithms).
        **kwargs : Any
            Additional keyword arguments.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Get algorithm features
        alg_df = getattr(self, "algorithm_features", None)
        if alg_df is None or not isinstance(alg_df, pd.DataFrame):
            raise ValueError(
                "Set selector.algorithm_features (pd.DataFrame indexed by algorithm names) before fit()"
            )

        self.algorithms = [str(a) for a in performance.columns]
        alg_df.index = alg_df.index.astype(str)

        # Map algorithm features to performance columns
        norm_to_orig = {self._norm(n): n for n in alg_df.index}
        mapped = []
        missing = []
        for a in self.algorithms:
            na = self._norm(a)
            orig = norm_to_orig.get(na)
            if orig is None:
                missing.append(a)
            else:
                mapped.append(orig)
        if missing:
            avail = list(alg_df.index)[:10]
            raise ValueError(
                f"Algorithm feature rows do not match performance columns. Missing: {missing}. Available sample: {avail}"
            )

        alg_df = alg_df.loc[mapped].copy()
        alg_df.index = [str(a) for a in self.algorithms]
        alg_df = alg_df.select_dtypes(include=[np.number]).astype(float)
        self._alg_feats = alg_df

        # Preprocess features
        X = features.fillna(0.0).to_numpy(dtype=float)
        self._imputer = SimpleImputer()
        X = self._imputer.fit_transform(X)

        if self.normalize_features:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        # Performance matrix
        Y_perf = performance.loc[features.index, self.algorithms].to_numpy(dtype=float)
        col_mean = np.nanmean(Y_perf, axis=0)
        inds = np.where(np.isnan(Y_perf))
        if inds[0].size:
            Y_perf[inds] = np.take(col_mean, inds[1])

        # Generate training data
        training_data, training_labels = self._generate_training_data(X, Y_perf)

        # Convert to tensors
        X_train = torch.tensor(training_data, dtype=torch.float32)
        y_train = torch.tensor(training_labels, dtype=torch.long)
        alg_features_tensor = torch.tensor(alg_df.values, dtype=torch.float32)

        # Create data loader
        train_set = Data.TensorDataset(X_train, y_train)
        train_loader = Data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True
        )

        # Create model
        num_user_features = X.shape[1]
        num_algorithms = len(self.algorithms)

        self._model = _ASLLMRecommendationModel(  # type: ignore[attr-defined]
            num_algorithms=num_algorithms,
            num_user_features=num_user_features,
            algorithm_features=alg_features_tensor,
            embed_size=self.embed_size,
            num_hiddens=self.num_hiddens,
            num_layers=self.num_layers,
            alpha=self.alpha,
            beta=self.beta,
        ).to(self._device)

        # Training
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        self._model.train()
        for epoch in range(self.num_epochs):
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self._device)
                batch_y = batch_y.to(self._device)

                optimizer.zero_grad()
                outputs = self._model(batch_X)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()

        self._num_user_features = num_user_features

    def _predict(
        self,
        features: pd.DataFrame | None,
        performance: pd.DataFrame | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Predict the best algorithm for each query instance.
        """
        if self._model is None:
            raise ValueError("fit() must be called before predict()")

        if features is None:
            raise ValueError("CosineSelector requires features for prediction.")

        # Preprocess features
        X = features.fillna(0.0).to_numpy(dtype=float)
        if self._imputer is not None:
            X = self._imputer.transform(X)
        if self.normalize_features and self._scaler is not None:
            X = self._scaler.transform(X)

        self._model.eval()
        num_algorithms = len(self.algorithms)

        out: dict[str, list[tuple[str, float]]] = {}

        with torch.no_grad():
            for i, inst in enumerate(features.index):
                inst_features = X[i]
                best_score = -float("inf")
                best_algo = None

                for j in range(num_algorithms):
                    # Create input: instance features + one-hot + algo index
                    alg_embed = [0] * num_algorithms
                    alg_embed[j] = 1
                    sample = np.append(np.append(inst_features, alg_embed), [j])
                    sample_tensor = (
                        torch.tensor(sample, dtype=torch.float32)
                        .unsqueeze(0)
                        .to(self._device)
                    )

                    output = self._model(sample_tensor)
                    # Get probability of being a match (class 1)
                    probs = torch.softmax(output, dim=1)
                    score = probs[0, 1].item()

                    if score > best_score:
                        best_score = score
                        best_algo = self.algorithms[j]

                out[str(inst)] = [(best_algo, float(self.budget or 0))]  # type: ignore[assignment]

        return out

    # save and load are inherited from AbstractSelector/ConfigurableMixin

    @staticmethod
    def _define_hyperparameters(
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """
        Define hyperparameters for CosineSelector.
        """
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        normalize_features_param = Categorical(
            name="normalize_features",
            items=[True, False],
            default=True,
        )

        embed_size_param = Integer(
            name="embed_size",
            bounds=(10, 100),
            default=50,
        )

        num_hiddens_param = Integer(
            name="num_hiddens",
            bounds=(10, 100),
            default=50,
        )

        num_layers_param = Integer(
            name="num_layers",
            bounds=(1, 4),
            default=2,
        )

        alpha_param = Float(
            name="alpha",
            bounds=(0.0, 1.0),
            default=0.9,
        )

        beta_param = Float(
            name="beta",
            bounds=(0.0, 1.0),
            default=0.1,
        )

        lr_param = Float(
            name="lr",
            bounds=(1e-5, 1e-2),
            log=True,
            default=0.001,
        )

        num_epochs_param = Integer(
            name="num_epochs",
            bounds=(10, 200),
            default=100,
        )

        batch_size_param = Integer(
            name="batch_size",
            bounds=(32, 256),
            default=128,
        )

        params = [
            normalize_features_param,
            embed_size_param,
            num_hiddens_param,
            num_layers_param,
            alpha_param,
            beta_param,
            lr_param,
            num_epochs_param,
            batch_size_param,
        ]

        return params, [], []

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs: Any,
    ) -> partial[CosineSelector]:
        """
        Create a partial function from a clean configuration.
        """
        config = clean_config.copy()
        config.update(kwargs)
        return partial(CosineSelector, **config)
