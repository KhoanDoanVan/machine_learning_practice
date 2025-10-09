"""Hierarchical clustering implementation."""


import numpy as np
from sklearn.cluster import AgglomerativeClustering
from ..base import BaseClusterer



class HierarchicalClustering(BaseClusterer):
    """Hierarchical (Agglomerative) clustering algorithm."""

    def __init__(
        self, 
        n_clusters: int = 3, 
        linkage: str = 'ward', 
        affinity: str = 'euclidean', 
        normalize: bool = True
    ):

        """
        Initialize Hierarchical clusterer.
        
        Args:
            n_clusters: Số lượng clusters
            linkage: Phương pháp liên kết ('ward', 'complete', 'average', 'single')
            affinity: Metric để tính khoảng cách
            normalize: Chuẩn hóa dữ liệu
        """
        
        super().__init__(name="Hierarchical", normalize=normalize)
        self.n_clusters_ = n_clusters
        self.linkage = linkage
        self.affinity = affinity


    
    def fit(self, X: np.ndarray) -> 'HierarchicalClustering':
        """Huấn luyện Hierarchical model."""

        X_processed = self.preprocess(X, fit=True)

        self.model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.linkage,
            affinity=self.affinity
        )

        self.labels_ = self.model.fit_predict(X_processed)
        self.n_clusters_ = self.n_clusters

        return self
    


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Hierarchical clustering không có predict method.
        Gán label dựa trên điểm gần nhất trong mỗi cluster.
        """

        if self.model is None:
            raise ValueError("Model chưa được train. Gọi fit() trước.")
        
        X_processed = self.preprocess(X, fit=False)

        # Tính centroid của mỗi cluster
        X_train = self.preprocess(
            self.scaler.inverse_transform(X_processed) if self.normalize else X_processed,
            fit=True
        )

        centroids = np.array(
            [
                X_train[self.labels_ == i].mean(axis=0)
                for i in range(self.n_clusters_)
            ]
        )

        # Gán label dựa trên centroid gần nhất
        from scipy.spatial.distance import cdist
        distances = cdist(X_processed, self.preprocess(centroids, fit=False))
        return np.argmin(distances, axis=1)