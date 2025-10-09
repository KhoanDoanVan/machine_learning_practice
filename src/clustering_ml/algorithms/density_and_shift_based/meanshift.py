"""Mean Shift clustering implementation."""


import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from ..base import BaseClusterer


class MeanShiftClustering(BaseClusterer):
    """Mean Shift clustering algorithm."""


    def __init__(
        self, 
        bandwidth: float = None,
        bin_seeding: bool = False,
        min_bin_freq: int = 1,
        normalize: bool = True
    ):
        """
        Initialize Mean Shift clusterer.
        
        Args:
            bandwidth: Kernel bandwidth. Nếu None, sẽ tự động ước lượng
            bin_seeding: Sử dụng bin seeding để tăng tốc
            min_bin_freq: Số điểm tối thiểu trong bin để là seed
            normalize: Chuẩn hóa dữ liệu
        """
        
        super().__init__(name="Mean Shift", normalize=normalize)
        self.bandwidth = bandwidth
        self.bin_seeding = bin_seeding
        self.min_bin_freq = min_bin_freq


    
    def fit(self, X: np.ndarray) -> 'MeanShiftClustering':
        """Huấn luyện Mean Shift model."""
        X_processed = self.preprocess(X, fit=True)

        # Tự động ước lượng bandwidth nếu không được cung cấp
        if self.bandwidth is None:
            self.bandwidth = estimate_bandwidth(X_processed, quantile=0.2)
        
        self.model = MeanShift(
            bandwidth=self.bandwidth,
            bin_seeding=self.bin_seeding,
            min_bin_freq=self.min_bin_freq
        )

        self.labels_ = self.model.fit_predict(X_processed)
        self.n_clusters_ = len(np.unique(self.labels_))
        self.cluster_centers_ = self.model.cluster_centers_

        return self
    


    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán cluster cho dữ liệu mới."""
        if self.model is None:
            raise ValueError("Model chưa được train. Gọi fit() trước.")
        
        X_processed = self.preprocess(X, fit=False)
        return self.model.predict(X_processed)