import warnings
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import anderson
from sklearn.utils import check_random_state
from sklearn.metrics import pairwise_distances


class GMeans:
    """
    G-Means clustering algorithm.

    Automatically determines the number of clusters by recursively splitting clusters
    and testing for Gaussianity using the Anderson-Darling test.

    Args:
        min_samples (float): Minimum fraction or count of samples per cluster.
        significance (float): Significance level for the Anderson-Darling test.
            Must be one of [0.15, 0.1, 0.05, 0.025, 0.001].
        n_init (int): Number of initializations for the recursive splitting.
        n_init_kmeans (int): Number of initializations for KMeans during splitting.
        n_init_final (int): Number of initializations for the final KMeans fit.
        random_state (int or None): Random seed for reproducibility.
    """

    def __init__(
        self,
        min_samples=0.001,
        significance=0.05,
        n_init=5,
        n_init_kmeans=5,
        n_init_final=5,
        random_state=None,
    ):
        self._min_samples = min_samples
        self._n_init = n_init
        self._n_init_kmeans = n_init_kmeans
        self._n_init_final = n_init_final
        self._random_state = check_random_state(random_state)
        self._kmeans = None
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

        allowed_significance = [0.15, 0.1, 0.05, 0.025, 0.001]
        if significance not in allowed_significance:
            raise ValueError(
                f"Invalid significance value: {significance}. Must be one of {allowed_significance}."
            )
        self._significance = allowed_significance.index(significance)

    def fit(self, X):
        """
        Fit G-Means clustering to the data.

        Args:
            X (np.ndarray): Data matrix (n_samples, n_features).

        Returns:
            self
        """
        if self._min_samples < 1.0:
            self._min_samples = X.shape[0] * self._min_samples

        self.inertia_ = np.inf
        self._k = 3
        self._kmeans = None

        for _ in range(self._n_init):
            seed_main = self._random_state.randint(0, 2**31 - 1)
            kmeans = KMeans(n_clusters=1, n_init=1, random_state=seed_main).fit(X)
            queue = [0]

            while queue:
                center_idx = queue.pop()
                X_ = X[kmeans.labels_ == center_idx]
                if np.size(X_, axis=0) <= 2:
                    continue

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    seed_tmp = self._random_state.randint(0, 2**31 - 1)
                    tmp_kmeans = KMeans(
                        n_clusters=2,
                        n_init=self._n_init_kmeans,
                        random_state=seed_tmp,
                    ).fit(X_)

                child_one, child_two = tmp_kmeans.cluster_centers_
                v = child_two - child_one
                tmp_labels = tmp_kmeans.predict(X)
                unique = np.unique(tmp_labels[kmeans.labels_ == center_idx])

                if np.linalg.norm(v, ord=2) <= 0.0 or unique.size < 2:
                    continue

                # Project data onto the vector between the two centroids
                y = np.inner(v, X_) / np.linalg.norm(v, ord=2)
                mean = np.mean(y)
                std = np.std(y)
                if std == 0:
                    continue
                y = (y - mean) / std
                A2, critical, _ = anderson(y)

                if A2 > critical[self._significance]:
                    kmeans.cluster_centers_ = np.delete(
                        kmeans.cluster_centers_, center_idx, axis=0
                    )
                    kmeans.cluster_centers_ = np.vstack(
                        [kmeans.cluster_centers_, child_one, child_two]
                    )
                    offset = np.size(kmeans.cluster_centers_, axis=0) - 2

                    del_idx = kmeans.labels_ > center_idx
                    ins_idx = kmeans.labels_ == center_idx
                    kmeans.labels_[del_idx] -= 1
                    kmeans.labels_[ins_idx] = tmp_labels[ins_idx] + offset

                    queue.extend([offset, offset + 1])

            if kmeans.inertia_ < self.inertia_:
                self.inertia_ = kmeans.inertia_
                self._k = np.size(kmeans.cluster_centers_, axis=0)

            seed_final = self._random_state.randint(0, 2**31 - 1)
            candidate_kmeans = KMeans(
                n_clusters=self._k,
                n_init=self._n_init_final,
                random_state=seed_final,
            ).fit(X)
            # accept candidate only if it improves inertia
            if candidate_kmeans.inertia_ < self.inertia_:
                self.inertia_ = candidate_kmeans.inertia_
                self._kmeans = candidate_kmeans
                self.cluster_centers_ = candidate_kmeans.cluster_centers_
                self.labels_ = candidate_kmeans.labels_

        # If no candidate was accepted (edge case), fit a final kmeans with fallback _k
        if self._kmeans is None:
            seed_final = self._random_state.randint(0, 2**31 - 1)
            self._kmeans = KMeans(
                n_clusters=self._k,
                n_init=self._n_init_final,
                random_state=seed_final,
            ).fit(X)
            self.inertia_ = self._kmeans.inertia_
            self.cluster_centers_ = self._kmeans.cluster_centers_
            self.labels_ = self._kmeans.labels_

        # Ensure minimum cluster size by redistributing small clusters
        self._redistribute(X)
        return self

    def _redistribute(self, X):
        redistribute = {
            label: center for label, center in enumerate(self._kmeans.cluster_centers_)
        }

        while redistribute:
            label, center = redistribute.popitem()
            X_ = X[self._kmeans.labels_ == label]

            if np.size(X_, axis=0) >= self._min_samples:
                continue

            if self._kmeans.cluster_centers_.shape[0] <= 1:
                break

            distances = pairwise_distances(
                X_, self._kmeans.cluster_centers_, metric="euclidean"
            )
            assignments = np.argpartition(distances, 1, axis=1)[:, 1]

            self._kmeans.labels_[self._kmeans.labels_ == label] = assignments
            self._kmeans.cluster_centers_ = np.delete(
                self._kmeans.cluster_centers_, label, axis=0
            )
            self._kmeans.labels_[self._kmeans.labels_ > label] -= 1
            self.labels_ = self._kmeans.labels_
            self.cluster_centers_ = self._kmeans.cluster_centers_

            centroids = np.zeros(self.cluster_centers_.shape)
            for lab, center in enumerate(self.cluster_centers_):
                X_assigned = X[self.labels_ == lab]
                if X_assigned.shape[0] > 0:
                    centroids[lab] = np.mean(X_assigned, axis=0)

            self.cluster_centers_ = centroids
            self._kmeans.cluster_centers_ = self.cluster_centers_

    def predict(self, X):
        """
        Predict cluster labels for new data.

        Args:
            X (np.ndarray): Data matrix (n_samples, n_features).

        Returns:
            np.ndarray: Cluster labels.
        """
        if self._kmeans is None:
            raise RuntimeError("GMeans instance is not fitted yet.")
        return self._kmeans.predict(X)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
