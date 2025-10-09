"""Data loading utilities."""

import numpy as np
from typing import Dict, Tuple
from .generator import DataGenerator


class DataLoader:
    """Loader cho datasets clustering."""
    
    def __init__(self):
        self.generator = DataGenerator()


    def load_all_datasets(self, random_state: int = 42) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Load tất cả các demo datasets.
        
        Returns:
            Dictionary với tên dataset và (X, y_true)
        """
        datasets = {
            'Blobs': self.generator.generate_blobs(random_state=random_state),
            'Moons': self.generator.generate_moons(random_state=random_state),
            'Circles': self.generator.generate_circles(random_state=random_state),
            'Anisotropic': self.generator.generate_anisotropic(random_state=random_state),
            'Varied Density': self.generator.generate_varied_density(random_state=random_state)
        }
        return datasets
    

    def get_dataset(self, name: str, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Load một dataset cụ thể."""
        datasets = self.load_all_datasets(random_state)
        if name not in datasets:
            raise ValueError(f"Dataset '{name}' không tồn tại. Chọn từ: {list(datasets.keys())}")
        return datasets[name]
    
print("✓ Cấu trúc project và các thuật toán clustering đã được định nghĩa")