"""K-Means clustering implementation"""


import numpy as np
from sklearn.cluster import KMeans
from .base import BaseClusterer



class KMeansClustering(BaseClusterer):

    def __init__(
        self, 
        n_clusters: int = 3, 
        max_iter: int = 300, 
        n_init: int = 10, 
        random_state: int = 42, 
        normalize: bool = True
    ):
        """
        Initialize K-Means clusterer.
        
        Args:
            n_clusters: Số lượng clusters
            max_iter: Số lần lặp tối đa
            n_init: Số lần chạy với các centroid khởi tạo khác nhau
            random_state: Random seed
            normalize: Chuẩn hóa dữ liệu
        """

        super().__init__(name="K-Means", normalize=normalize)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

    

    def fit(self, X: np.ndarray) -> 'KMeansClustering':
        """Huấn luyện K-Means model."""

        X_processed = self.preprocess(X, fit=True)

        self.model = KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=self.random_state
        )
        
        self.labels_ = self.model.fit_predict(X_processed)
        self.n_clusters_ = self.n_clusters
        self.cluster_centers_ = self.model.cluster_centers_
        self.inertia_ = self.model.inertia_

        return self
    


    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán cluster cho dữ liệu mới."""
        if self.model is None:
            raise ValueError("Model chưa được train. Gọi fit() trước.")
        
        X_processed = self.preprocess(X, fit=False)
        return self.model.predict(X_processed)