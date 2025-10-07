"""Gaussian Mixture Model clustering implementation."""

import numpy as np
from sklearn.mixture import GaussianMixture
from .base import BaseClusterer



class GaussianMixtureClustering(BaseClusterer):
    """Gaussian Mixture Model (GMM) clustering algorithm."""

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = 'full',
        max_iter: int = 100,
        random_state: int = 42,
        normalize: bool = True
    ):
        """
        Initialize GMM clusterer.
        
        Args:
            n_components: Số lượng Gaussian components
            covariance_type: Loại covariance matrix ('full', 'tied', 'diag', 'spherical')
            max_iter: Số lần lặp tối đa
            random_state: Random seed
            normalize: Chuẩn hóa dữ liệu
        """

        super().__init__(name="Gaussian Mixture", normalize=normalize)
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.random_state = random_state


    
    def fit(self, X: np.ndarray) -> 'GaussianMixtureClustering':
        """Huấn luyện GMM model."""

        X_processed = self.preprocess(X, fit=True)

        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            random_state=self.random_state
        )  

        self.model.fit(X_processed)
        self.labels_ = self.model.predict(X_processed)
        self.n_clusters_ = self.model.means_
        self.means_ = self.model.means_
        self.covariances_ = self.model.covariances_

        return self


    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán cluster cho dữ liệu mới."""

        if self.model is None:
            raise ValueError("Model chưa được train. Gọi fit() trước.")
        
        X_processed = self.preprocess(X, fit=False)
        return self.model.predict(X_processed)
    


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán xác suất thuộc mỗi cluster."""

        if self.model is None:
            raise ValueError("Model chưa được train. Gọi fit() trước.")
        
        X_processed = self.preprocess(X, fit=False)
        return self.model.predict_proba(X_processed)
