"""DBSCAN clustering implementation."""


import numpy as np
from sklearn.cluster import DBSCAN
from ..base import BaseClusterer


class DBSCANClustering(BaseClusterer):

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = 'euclidean',
        normalize: bool = True
    ):
        """
        Initialize DBSCAN clusterer.
        
        Args:
            eps: Khoảng cách tối đa giữa 2 điểm để được coi là neighbors
            min_samples: Số điểm tối thiểu trong neighborhood để là core point
            metric: Metric để tính khoảng cách
            normalize: Chuẩn hóa dữ liệu
        """

        super().__init__(name="DBSCAN", normalize=normalize)
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    
    def fit(self, X: np.ndarray) -> 'DBSCANClustering':
        """Huấn luyện DBSCAN model."""
        
        X_processed = self.preprocess(X, fit=True)

        self.model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric
        )

        self.labels_ = self.model.fit_predict(X_processed)
        # DBSCAN có thể tạo ra noise points (label = -1)
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)

        return self
    

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        DBSCAN không có phương thức predict trực tiếp.
        Gán label dựa trên core samples gần nhất.
        """

        if self.model is None:
            raise ValueError("Model chưa được train. Gọi fit() trước.")
        
        X_processed = self.preprocess(X, fit=False)

        # Sử dụng core samples để dự đoán
        core_sample_mask = np.zeros_like(self.labels_, dtype=bool)
        core_sample_mask[self.model.core_sample_indices_] = True

        # Đơn giản hóa: gán label của điểm gần nhất
        from scipy.spatial.distance import cdist
        X_train = self.preprocess(self.model.components_, fit=False)
        distances = cdist(X_processed, X_train)
        nearest = np.argmin(distances, axis=1)

        return self.labels_[self.model.core_sample_indices_][nearest]
