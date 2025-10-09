"""Spectral clustering implementation."""

import numpy as np
from sklearn.cluster import SpectralClustering as SKSpectralClustering
from ..base import BaseClusterer


class SpectralClustering(BaseClusterer):
    """Spectral clustering algorithm."""

    def __init__(
        self,
        n_clusters: int = 3,
        affinity: str = 'rbf',
        n_neighbors: int = 10,
        random_state: int = 42,
        normalize: bool = True
    ):
        """
        Initialize Spectral clusterer.
        
        Args:
            n_clusters: Số lượng clusters
            affinity: Cách tính affinity matrix ('rbf', 'nearest_neighbors')
            n_neighbors: Số neighbors cho kNN graph
            random_state: Random seed
            normalize: Chuẩn hóa dữ liệu
        """

        super().__init__(name="Spectral", normalize=normalize)
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.random_state = random_state



    def fit(self, X: np.ndarray) -> 'SpectralClustering':
        """Huấn luyện Spectral model."""

        X_processed = self.preprocess(X, fit=True)

        self.model = SKSpectralClustering(
            n_clusters=self.n_clusters,
            affinity=self.affinity,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state
        )

        self.labels_ = self.model.fit_predict(X_processed)
        self.n_clusters_ = self.n_clusters

        return self
    


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Spectral clustering không có predict method.
        Gán label dựa trên điểm gần nhất trong mỗi cluster.
        """

        if self.model is None:
            raise ValueError("Model chưa được train. Gọi fit() trước.")
        
        X_processed = self.preprocess(X, fit=False)

        # Tính centroid của mỗi cluster từ training data
        X_train = self.scaler.inverse_transform(X_processed) if self.normalize else X_processed
        X_train_processed = self.preprocess(X_train, fit=False)


        centroids = np.array(
            [
                X_train_processed[self.labels_ == i].mean(axis=0)
                for i in range(self.n_clusters_)
            ]
        )

        # Gán label dựa trên centroid gần nhất
        from scipy.spatial.distance import cdist
        distances = cdist(X_processed, centroids)
        return np.argmin(distances, axis=1)