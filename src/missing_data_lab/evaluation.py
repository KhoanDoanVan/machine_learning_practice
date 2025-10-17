"""
evaluation.py
So sánh kết quả của các chiến lược imputation bằng metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score


class ImputationEvaluator:
    """Đánh giá và so sánh các chiến lược imputation."""
    
    def __init__(self, original_df: pd.DataFrame, missing_df: pd.DataFrame):
        """
        Khởi tạo evaluator.
        
        Args:
            original_df: DataFrame gốc (không có missing)
            missing_df: DataFrame có missing values
        """
        self.original_df = original_df.copy()
        self.missing_df = missing_df.copy()

    
    def evaluate_numeric(self, imputed_df: pd.DataFrame, columns: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Đánh giá imputation cho numeric columns.
        
        Args:
            imputed_df: DataFrame sau khi impute
            columns: Danh sách columns cần đánh giá (None = tất cả numeric)
        
        Returns:
            Dictionary chứa metrics cho mỗi column
        """

        if columns is None:
            columns = self.original_df.select_dtypes(include=[np.number]).columns.tolist()


        results = {}

        for col in columns:
            if col not in self.original_df.columns or col not in imputed_df.columns:
                continue

            # Tìm các vị trí bị missing
            missing_mask = self.missing_df[col].isnull()

            if missing_mask.sum() == 0:
                continue

            # Lấy giá trị thật và giá trị imputed tại các vị trí missing
            true_values = self.original_df.loc[missing_mask, col]
            imputed_values = imputed_df.loc[missing_mask, col]

            # Bỏ qua các giá trị vẫn còn NaN sau imputation
            valid_mask = ~imputed_values.isnull()
            true_values = true_values[valid_mask]
            imputed_values = imputed_values[valid_mask]

            if len(true_values) == 0:
                continue

            # Tính accuracy
            accuracy = accuracy_score(true_values, imputed_values)

            results[col] = {
                'Accuracy': accuracy,
                'Accuracy_%': accuracy * 100,
                'n_imputed': len(true_values)
            }

        return results
    

    def compare_strategies(self, imputed_dfs: Dict[str, pd.DataFrame],data_type: str = 'numeric') -> pd.DataFrame:
        """
        So sánh nhiều chiến lược cùng lúc.
        
        Args:
            imputed_dfs: Dictionary {strategy_name: imputed_df}
            data_type: 'numeric' hoặc 'categorical'
        
        Returns:
            DataFrame tổng hợp kết quả so sánh
        """

        all_results = []

        for strategy_name, imputed_df in imputed_dfs.items():
            if data_type == 'numeric':
                results = self.evaluate_numeric(imputed_df)
                
                for col, metrics in results.items():
                    all_results.append({
                        'Strategy': strategy_name,
                        'Column': col,
                        'MAE': metrics['MAE'],
                        'RMSE': metrics['RMSE'],
                        'MAE_%': metrics['MAE_%'],
                        'RMSE_%': metrics['RMSE_%'],
                        'N_Imputed': metrics['n_imputed']
                    })

            elif data_type == 'categorical':
                results = self.evaluate_categorical(imputed_df)
                
                for col, metrics in results.items():
                    all_results.append({
                        'Strategy': strategy_name,
                        'Column': col,
                        'Accuracy': metrics['Accuracy'],
                        'Accuracy_%': metrics['Accuracy_%'],
                        'N_Imputed': metrics['n_imputed']
                    })

        if len(all_results) == 0:
            return pd.DataFrame()
        

        comparison_df = pd.DataFrame(all_results)
        return comparison_df
    


    def get_best_strategy(self, comparison_df: pd.DataFrame, metric: str = 'MAE') -> Dict[str, str]:
        """
        Tìm strategy tốt nhất cho mỗi column.
        
        Args:
            comparison_df: DataFrame từ compare_strategies()
            metric: Metric để đánh giá ('MAE', 'RMSE', 'Accuracy')
        
        Returns:
            Dictionary {column: best_strategy}
        """


        