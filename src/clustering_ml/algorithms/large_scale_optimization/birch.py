"""BIRCH clustering implementation."""


import numpy as np
from sklearn.cluster import Birch
from ..base import BaseClusterer


class BIRCHClustering(BaseClusterer):
    """BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) algorithm."""

    def __init__(
        self,
        n_clusters: int = 3,
        threshold: float = 0.5,
        branching_factor: int = 50,
        normalize: bool = True
    ):
        """
        Initialize BIRCH clusterer.
        
        Args:
            n_clusters: Số lượng clusters
            threshold: Threshold cho subcluster radius
            branching_factor: Maximum số subclusters trong mỗi node
            normalize: Chuẩn hóa dữ liệu
        """

        super().__init__(name="BIRCH", normalize=normalize)
        self.n_clusters = n_clusters
        self.threshold = threshold
        self.branching_factor = branching_factor


    def fit(self, X: np.ndarray) -> 'BIRCHClustering':
        """Huấn luyện BIRCH model."""
        
        X_processed = self.preprocess(X, fit=True)

        self.model = Birch(
            n_clusters=self.n_clusters,
            threshold=self.threshold,
            branching_factor=self.branching_factor
        )

        self.labels_ = self.model.fit_predict(X_processed)
        self.n_clusters_ = self.n_clusters
        self.subcluster_centers_ = self.model.subcluster_centers_

        return self
    

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán cluster cho dữ liệu mới."""

        if self.model is None:
            raise ValueError("Model chưa được train. Gọi fit() trước.")
        
        X_processed = self.preprocess(X, fit=False)
        return self.model.predict(X_processed)