

import numpy as np
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score, homogeneity_score,
    completeness_score, v_measure_score
)
from typing import Dict, Optional



class ClusteringMetrics:
    """Tính toán các metrics đánh giá clustering."""
    
    @staticmethod
    def compute_internal_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Tính các internal metrics (không cần ground truth).
        
        Args:
            X: Feature matrix
            labels: Predicted cluster labels
            
        Returns:
            Dictionary chứa các metrics
        """
        # Kiểm tra số clusters hợp lệ
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters < 2:
            return {
                'silhouette': 0.0,
                'davies_bouldin': float('inf'),
                'calinski_harabasz': 0.0
            }
        
        try:
            silhouette = silhouette_score(X, labels)
        except:
            silhouette = 0.0
            
        try:
            davies_bouldin = davies_bouldin_score(X, labels)
        except:
            davies_bouldin = float('inf')
            
        try:
            calinski = calinski_harabasz_score(X, labels)
        except:
            calinski = 0.0
        
        return {
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski_harabasz': calinski
        }
    

    @staticmethod
    def compute_external_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Tính các external metrics (cần ground truth).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary chứa các metrics
        """
        return {
            'adjusted_rand_index': adjusted_rand_score(y_true, y_pred),
            'normalized_mutual_info': normalized_mutual_info_score(y_true, y_pred),
            'homogeneity': homogeneity_score(y_true, y_pred),
            'completeness': completeness_score(y_true, y_pred),
            'v_measure': v_measure_score(y_true, y_pred)
        }
    

    @staticmethod
    def compute_all_metrics(X: np.ndarray, y_pred: np.ndarray, 
                          y_true: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Tính tất cả các metrics.
        
        Args:
            X: Feature matrix
            y_pred: Predicted labels
            y_true: True labels (optional)
            
        Returns:
            Dictionary chứa tất cả metrics
        """
        metrics = ClusteringMetrics.compute_internal_metrics(X, y_pred)
        
        if y_true is not None:
            external = ClusteringMetrics.compute_external_metrics(y_true, y_pred)
            metrics.update(external)
        
        return metrics
    

    @staticmethod
    def format_metrics(metrics: Dict[str, float]) -> str:
        """Format metrics thành string dễ đọc."""
        lines = ["Clustering Metrics:"]
        lines.append("-" * 50)
        
        # Internal metrics
        if 'silhouette' in metrics:
            lines.append("Internal Metrics (không cần ground truth):")
            lines.append(f"  Silhouette Score: {metrics['silhouette']:.4f} (higher is better, range: [-1, 1])")
            lines.append(f"  Davies-Bouldin Index: {metrics['davies_bouldin']:.4f} (lower is better)")
            lines.append(f"  Calinski-Harabasz Index: {metrics['calinski_harabasz']:.2f} (higher is better)")
            lines.append("")
        
        # External metrics
        if 'adjusted_rand_index' in metrics:
            lines.append("External Metrics (cần ground truth):")
            lines.append(f"  Adjusted Rand Index: {metrics['adjusted_rand_index']:.4f} (range: [-1, 1])")
            lines.append(f"  Normalized Mutual Info: {metrics['normalized_mutual_info']:.4f} (range: [0, 1])")
            lines.append(f"  Homogeneity: {metrics['homogeneity']:.4f} (range: [0, 1])")
            lines.append(f"  Completeness: {metrics['completeness']:.4f} (range: [0, 1])")
            lines.append(f"  V-Measure: {metrics['v_measure']:.4f} (range: [0, 1])")
        
        return "\n".join(lines)
    


# Example usage
if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    
    # Generate sample data
    X, y_true = make_blobs(n_samples=300, centers=3, random_state=42)
    
    # Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    y_pred = kmeans.fit_predict(X)
    
    # Compute metrics
    metrics = ClusteringMetrics.compute_all_metrics(X, y_pred, y_true)
    
    # Print formatted metrics
    print(ClusteringMetrics.format_metrics(metrics))