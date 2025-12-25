"""
GroupKFoldShuffle: K-Fold cross-validator with group-based splitting and shuffling.
"""

import numpy as np
from typing import Iterator, Any
from sklearn.model_selection._split import _BaseKFold


class GroupKFoldShuffle(_BaseKFold):
    """
    K-Fold cross-validator with group-based splitting and shuffling.

    This class ensures that the same group is not represented in both
    training and testing sets, while allowing for shuffling of groups.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = False,
        random_state: int | None = None,
    ):
        """
        Initialize the GroupKFoldShuffle.

        Parameters
        ----------
        n_splits : int, default=5
            Number of folds. Must be at least 2.
        shuffle : bool, default=False
            Whether to shuffle the groups before splitting into batches.
        random_state : int or None, default=None
            When shuffle is True, random_state affects the ordering of the
            indices, which controls the randomness of each fold.
        """
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(
        self,
        X: Any,
        y: Any = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train_idx : ndarray
            The training set indices for that split.
        test_idx : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")

        # Find the unique groups in the dataset.
        unique_groups = np.unique(groups)

        # Shuffle the unique groups if shuffle is true.
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            unique_groups = rng.permutation(unique_groups)

        # Split the shuffled groups into n_splits.
        split_groups = np.array_split(unique_groups, self.n_splits)

        # For each split, determine the train and test indices.
        for test_group_ids in split_groups:
            test_mask = np.isin(groups, test_group_ids)
            train_mask = ~test_mask

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            yield train_idx, test_idx
