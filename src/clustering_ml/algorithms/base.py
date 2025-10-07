"""Base Class"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np
from sklearn.preprocessing import StandardScaler


class BaseClusterer(ABC):
    """Abstract base class cho clustering algorithms."""

    def __init__(self, name: str, normalize: bool = True):
        """
        Initialize base clusterer.
        
        Args:
            name: Tên thuật toán
            normalize: Có chuẩn hóa dữ liệu không
        """

        self.name = name
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        self.model = None
        self.labels_ = None
        self.n_clusters_ = None


    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseClusterer':
        """Huấn luyện model."""
        pass


    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán cluster cho dữ liệu mới."""
        pass
    

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Huấn luyện và dự đoán."""
        self.fit(X)
        return self.labels_
    

    def preprocess(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Tiền xử lý dữ liệu."""
        if self.normalize:
            if fit:
                return self.scaler.fit_transform(X)
            return self.scaler.transform(X)
        return X
    

    def get_params(self) -> Dict[str, Any]:
        """Lấy parameters của model."""
        return {
            'name': self.name,
            'n_clusters': self.n_clusters_,
            'normalize': self.normalize
        }
    

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
