"""Dataset generation utilities."""

import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
from typing import Tuple, Dict


class DataGenerator:
    """Generator cho các dataset clustering."""
    
    @staticmethod
    def generate_blobs(
        n_samples: int = 300, 
        n_features: int = 2, 
        centers: int = 3,
        cluster_std: float = 1.0,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tạo dataset với các blob clusters.
        
        Features:
            - feature_0, feature_1, ...: Các tọa độ trong không gian
        
        Returns:
            X: Feature matrix (n_samples, n_features)
            y: True labels (n_samples,)
        """
        return make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=centers,
            cluster_std=cluster_std,
            random_state=random_state
        )
    


    @staticmethod
    def generate_moons(
        n_samples: int = 300, 
        noise: float = 0.1,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tạo dataset với 2 moon shapes.
        
        Features:
            - x_coordinate: Tọa độ x
            - y_coordinate: Tọa độ y
        
        Returns:
            X: Feature matrix (n_samples, 2)
            y: True labels (n_samples,)
        """
        return make_moons(
            n_samples=n_samples,
            noise=noise,
            random_state=random_state
        )
    

    @staticmethod
    def generate_circles(
        n_samples: int = 300, 
        noise: float = 0.05,
        factor: float = 0.5, 
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tạo dataset với 2 concentric circles.
        
        Features:
            - x_coordinate: Tọa độ x
            - y_coordinate: Tọa độ y
        
        Returns:
            X: Feature matrix (n_samples, 2)
            y: True labels (n_samples,)
        """
        return make_circles(
            n_samples=n_samples,
            noise=noise,
            factor=factor,
            random_state=random_state
        )
    

    @staticmethod
    def generate_anisotropic(
        n_samples: int = 300, 
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tạo dataset với anisotropic blobs.
        
        Features:
            - feature_0: Tọa độ x sau transformation
            - feature_1: Tọa độ y sau transformation
        
        Returns:
            X: Feature matrix (n_samples, 2)
            y: True labels (n_samples,)
        """
        X, y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X = np.dot(X, transformation)
        return X, y
    


    @staticmethod
    def generate_varied_density(random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tạo dataset với varied density clusters.
        
        Features:
            - feature_0: Tọa độ x
            - feature_1: Tọa độ y
        
        Returns:
            X: Feature matrix (n_samples, 2)
            y: True labels (n_samples,)
        """
        X, y = make_blobs(
            n_samples=[100, 500, 100],
            cluster_std=[1.0, 2.5, 0.5],
            random_state=random_state
        )
        return X, y
    

    @staticmethod
    def get_feature_descriptions() -> Dict[str, str]:
        """
        Lấy mô tả các features trong datasets.
        
        Returns:
            Dictionary mapping feature names to descriptions
        """
        return {
            'blobs': {
                'feature_0': 'X coordinate in 2D/3D space',
                'feature_1': 'Y coordinate in 2D/3D space',
                'feature_n': 'Additional dimensions for high-dimensional clustering'
            },
            'moons': {
                'x_coordinate': 'X position on the moon shape',
                'y_coordinate': 'Y position on the moon shape'
            },
            'circles': {
                'x_coordinate': 'X position on concentric circles',
                'y_coordinate': 'Y position on concentric circles'
            },
            'anisotropic': {
                'feature_0': 'Transformed X coordinate',
                'feature_1': 'Transformed Y coordinate'
            },
            'varied_density': {
                'feature_0': 'X coordinate with varied cluster densities',
                'feature_1': 'Y coordinate with varied cluster densities'
            }
        }