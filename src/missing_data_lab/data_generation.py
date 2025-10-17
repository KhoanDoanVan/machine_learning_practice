"""
data_generator.py
Sinh dữ liệu giả lập với các loại missing patterns khác nhau.

Missing patterns:
- MCAR (Missing Completely At Random): missing ngẫu nhiên hoàn toàn
- MAR (Missing At Random): missing phụ thuộc vào các biến quan sát được
- MNAR (Missing Not At Random): missing phụ thuộc vào chính giá trị bị thiếu
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class DataGenerator:
    """Sinh dữ liệu giả lập cho việc học xử lý missing data."""
    
    def __init__(self, random_state=42):
        """
        Khởi tạo data generator.
        
        Args:
            random_state: Seed cho random number generator
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_numeric_data(self, n_samples=1000, missing_pattern='MCAR', missing_rate=0.2):
        """
        Sinh dữ liệu numeric (dữ liệu sinh viên).
        
        Args:
            n_samples: Số lượng mẫu
            missing_pattern: 'MCAR', 'MAR', hoặc 'MNAR'
            missing_rate: Tỷ lệ missing (0-1)
        
        Returns:
            DataFrame chứa dữ liệu với missing values
        """
        # Sinh dữ liệu gốc
        data = {
            'student_id': range(1, n_samples + 1),
            'age': np.random.normal(20, 2, n_samples).clip(18, 25),
            'score': np.random.normal(75, 10, n_samples).clip(0, 100),
            'height': np.random.normal(170, 10, n_samples).clip(150, 190),
            'weight': np.random.normal(65, 12, n_samples).clip(45, 100),
            'study_hours': np.random.gamma(2, 2, n_samples).clip(0, 20)
        }
        df = pd.DataFrame(data)
        
        # Tạo missing pattern
        if missing_pattern == 'MCAR':
            # Missing hoàn toàn ngẫu nhiên
            for col in ['score', 'height', 'weight', 'study_hours']:
                mask = np.random.random(n_samples) < missing_rate
                df.loc[mask, col] = np.nan
                
        elif missing_pattern == 'MAR':
            # Missing phụ thuộc vào age
            # Sinh viên trẻ hơn có xu hướng không báo cáo cân nặng
            age_threshold = df['age'].quantile(0.3)
            young_mask = df['age'] < age_threshold
            young_indices = df[young_mask].index
            n_missing = int(len(young_indices) * missing_rate * 2)
            missing_indices = np.random.choice(young_indices, n_missing, replace=False)
            df.loc[missing_indices, 'weight'] = np.nan
            
            # Score cao có xu hướng không báo cáo giờ học (không muốn lộ bí quyết)
            score_threshold = df['score'].quantile(0.7)
            high_score_mask = df['score'] > score_threshold
            high_score_indices = df[high_score_mask].index
            n_missing = int(len(high_score_indices) * missing_rate * 1.5)
            missing_indices = np.random.choice(high_score_indices, n_missing, replace=False)
            df.loc[missing_indices, 'study_hours'] = np.nan
            
        elif missing_pattern == 'MNAR':
            # Missing phụ thuộc vào chính giá trị đó
            # Điểm thấp có xu hướng không muốn báo cáo
            score_threshold = df['score'].quantile(0.3)
            low_score_mask = df['score'] < score_threshold
            low_score_indices = df[low_score_mask].index
            n_missing = int(len(low_score_indices) * missing_rate * 2)
            missing_indices = np.random.choice(low_score_indices, n_missing, replace=False)
            df.loc[missing_indices, 'score'] = np.nan
            
            # Cân nặng quá cao/thấp có xu hướng không muốn báo cáo
            weight_outliers = (df['weight'] < df['weight'].quantile(0.15)) | \
                            (df['weight'] > df['weight'].quantile(0.85))
            outlier_indices = df[weight_outliers].index
            n_missing = int(len(outlier_indices) * missing_rate * 1.5)
            missing_indices = np.random.choice(outlier_indices, 
                                             min(n_missing, len(outlier_indices)), 
                                             replace=False)
            df.loc[missing_indices, 'weight'] = np.nan
        
        return df
    
    def generate_categorical_data(self, n_samples=1000, missing_pattern='MCAR', missing_rate=0.2):
        """
        Sinh dữ liệu categorical (survey data).
        
        Args:
            n_samples: Số lượng mẫu
            missing_pattern: 'MCAR', 'MAR', hoặc 'MNAR'
            missing_rate: Tỷ lệ missing (0-1)
        
        Returns:
            DataFrame chứa dữ liệu categorical với missing values
        """
        # Sinh dữ liệu gốc
        data = {
            'respondent_id': range(1, n_samples + 1),
            'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.48, 0.48, 0.04]),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                         n_samples, p=[0.3, 0.45, 0.2, 0.05]),
            'income_level': np.random.choice(['Low', 'Medium', 'High'], 
                                            n_samples, p=[0.3, 0.5, 0.2]),
            'preference': np.random.choice(['A', 'B', 'C', 'D'], n_samples)
        }
        df = pd.DataFrame(data)
        
        # Tạo missing pattern
        if missing_pattern == 'MCAR':
            for col in ['education', 'income_level', 'preference']:
                mask = np.random.random(n_samples) < missing_rate
                df.loc[mask, col] = np.nan
                
        elif missing_pattern == 'MAR':
            # Thu nhập thấp có xu hướng không muốn trả lời về education
            low_income_mask = df['income_level'] == 'Low'
            low_income_indices = df[low_income_mask].index
            n_missing = int(len(low_income_indices) * missing_rate * 2)
            missing_indices = np.random.choice(low_income_indices, n_missing, replace=False)
            df.loc[missing_indices, 'education'] = np.nan
            
        elif missing_pattern == 'MNAR':
            # Thu nhập cao có xu hướng không muốn tiết lộ thu nhập
            high_income_mask = df['income_level'] == 'High'
            high_income_indices = df[high_income_mask].index
            n_missing = int(len(high_income_indices) * missing_rate * 2.5)
            missing_indices = np.random.choice(high_income_indices, 
                                             min(n_missing, len(high_income_indices)), 
                                             replace=False)
            df.loc[missing_indices, 'income_level'] = np.nan
        
        return df
    
    def generate_timeseries_data(self, n_days=365, missing_pattern='MCAR', missing_rate=0.15):
        """
        Sinh dữ liệu time-series (dữ liệu cảm biến).
        
        Args:
            n_days: Số ngày dữ liệu
            missing_pattern: 'MCAR', 'MAR', hoặc 'MNAR'
            missing_rate: Tỷ lệ missing (0-1)
        
        Returns:
            DataFrame chứa dữ liệu time-series với missing values
        """
        # Sinh dữ liệu gốc với seasonal pattern
        start_date = datetime(2024, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n_days)]
        
        # Temperature có seasonal pattern
        t = np.arange(n_days)
        seasonal = 10 * np.sin(2 * np.pi * t / 365)  # Yearly cycle
        temperature = 20 + seasonal + np.random.normal(0, 2, n_days)
        
        # Humidity có correlation với temperature
        humidity = 60 - 0.5 * seasonal + np.random.normal(0, 5, n_days)
        humidity = humidity.clip(30, 90)
        
        # Pressure tương đối ổn định
        pressure = np.random.normal(1013, 5, n_days)
        
        data = {
            'timestamp': dates,
            'temperature': temperature,
            'humidity': humidity,
            'pressure': pressure
        }
        df = pd.DataFrame(data)
        
        # Tạo missing pattern
        if missing_pattern == 'MCAR':
            for col in ['temperature', 'humidity', 'pressure']:
                mask = np.random.random(n_days) < missing_rate
                df.loc[mask, col] = np.nan
                
        elif missing_pattern == 'MAR':
            # Khi nhiệt độ cao, cảm biến humidity dễ hỏng
            high_temp_mask = df['temperature'] > df['temperature'].quantile(0.75)
            high_temp_indices = df[high_temp_mask].index
            n_missing = int(len(high_temp_indices) * missing_rate * 2)
            missing_indices = np.random.choice(high_temp_indices, n_missing, replace=False)
            df.loc[missing_indices, 'humidity'] = np.nan
            
        elif missing_pattern == 'MNAR':
            # Giá trị nhiệt độ cực đoan (quá cao/thấp) có thể bị mất
            temp_outliers = (df['temperature'] < df['temperature'].quantile(0.1)) | \
                          (df['temperature'] > df['temperature'].quantile(0.9))
            outlier_indices = df[temp_outliers].index
            n_missing = int(len(outlier_indices) * missing_rate * 2)
            missing_indices = np.random.choice(outlier_indices, 
                                             min(n_missing, len(outlier_indices)), 
                                             replace=False)
            df.loc[missing_indices, 'temperature'] = np.nan
        
        return df


if __name__ == "__main__":
    # Test code
    generator = DataGenerator()
    
    print("=== NUMERIC DATA (MCAR) ===")
    df_numeric = generator.generate_numeric_data(n_samples=100, missing_pattern='MCAR')
    print(df_numeric.head())
    print(f"\nMissing counts:\n{df_numeric.isnull().sum()}")
    
    print("\n=== CATEGORICAL DATA (MAR) ===")
    df_cat = generator.generate_categorical_data(n_samples=100, missing_pattern='MAR')
    print(df_cat.head())
    print(f"\nMissing counts:\n{df_cat.isnull().sum()}")
    
    print("\n=== TIME-SERIES DATA (MNAR) ===")
    df_ts = generator.generate_timeseries_data(n_days=30, missing_pattern='MNAR')
    print(df_ts.head())
    print(f"\nMissing counts:\n{df_ts.isnull().sum()}")