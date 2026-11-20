"""Novelty detector class, in its own module so saved detectors load from any script."""
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

K = 5


class KNNNovelty:
    """Mean distance to the k nearest training points, in standardized feature space."""

    def __init__(self, k=K):
        self.k = k

    def fit(self, X):
        self.scaler = StandardScaler().fit(X)
        self.nn = NearestNeighbors(n_neighbors=self.k).fit(self.scaler.transform(X))
        return self

    def score(self, X):
        d, _ = self.nn.kneighbors(self.scaler.transform(X))
        return d.mean(1)
