"""Backward-compatible name for sklearn's shuffled group cross-validator."""

from sklearn.model_selection import GroupKFold


class GroupKFoldShuffle(GroupKFold):
    """Compatibility wrapper around sklearn's public GroupKFold API."""

    pass
