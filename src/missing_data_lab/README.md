# Missing Data Lab 🔬

**Interactive Python project để học và thực hành các kỹ thuật xử lý dữ liệu bị thiếu (missing data)**

## 📋 Mô tả

Missing Data Lab là một công cụ học tập tương tác giúp bạn:
- Hiểu các loại missing data patterns (MCAR, MAR, MNAR)
- Thực hành các chiến lược xử lý missing data
- So sánh hiệu quả của từng chiến lược
- Visualize ảnh hưởng của imputation lên dữ liệu

## 🎯 Tính năng chính

### 1. Sinh dữ liệu giả lập
- **Numeric data**: Dữ liệu sinh viên (tuổi, điểm, chiều cao, cân nặng)
- **Categorical data**: Dữ liệu khảo sát (giới tính, vùng, trình độ)
- **Time-series data**: Dữ liệu cảm biến (nhiệt độ, độ ẩm, áp suất)

### 2. Missing patterns
- **MCAR** (Missing Completely At Random): Missing hoàn toàn ngẫu nhiên
- **MAR** (Missing At Random): Missing phụ thuộc vào biến quan sát được
- **MNAR** (Missing Not At Random): Missing phụ thuộc vào chính giá trị bị thiếu

### 3. Các chiến lược xử lý
- **drop**: Xóa hàng có missing
- **mean**: Điền trung bình (numeric)
- **median**: Điền trung vị (numeric)
- **mode**: Điền mode (categorical)
- **constant**: Điền giá trị cố định
- **ffill/bfill**: Forward/Backward fill (time-series)
- **knn**: K-Nearest Neighbors imputation
- **iterative**: MICE (Multiple Imputation by Chained Equations)

### 4. Đánh giá và visualization
- MAE, RMSE cho dữ liệu numeric
- Accuracy cho dữ liệu categorical
- So sánh distribution trước và sau imputation
- Missing pattern visualization
- Strategy comparison charts

## 📦 Cài đặt

### Yêu cầu
- Python 3.11+
- pip

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Cách sử dụng

### Quick Start

```bash
python main.py
```

Chương trình sẽ hiển thị menu tương tác:

```
================================================================================
                         MISSING DATA LAB
================================================================================

📚 CHỌN LOẠI DỮ LIỆU:
  1. Numeric Data (Student Data)
  2. Categorical Data (Survey Data)
  3. Time-Series Data (Sensor Data)

🔧 CÁC CHỨC NĂNG:
  4. Xem thông tin missing data
  5. Visualize missing patterns
  6. Thử một strategy cụ thể
  7. So sánh tất cả strategies
  8. Tìm hiểu về missing patterns
  9. Tìm hiểu về strategies
  0. Thoát
```

### Ví dụ workflow

1. **Chọn loại dữ liệu** (ví dụ: option 1 - Numeric Data)
2. **Chọn missing pattern** (MCAR/MAR/MNAR)
3. **Xem thông tin missing** (option 4)
4. **Visualize patterns** (option 5)
5. **So sánh strategies** (option 7)

### Sử dụng từng module riêng

#### Data Generator
```python
from data_generator import DataGenerator

generator = DataGenerator()
df = generator.generate_numeric_data(
    n_samples=1000, 
    missing_pattern='MAR', 
    missing_rate=0.2
)
```

#### Imputation Strategy
```python
from imputation_strategies import StrategyFactory

strategy = StrategyFactory.create_strategy('knn', n_neighbors=5)
df_imputed = strategy.fit_transform(df)
```

#### Evaluation
```python
from evaluation import ImputationEvaluator

evaluator = ImputationEvaluator(df_original, df_missing)
results = evaluator.evaluate_numeric(df_imputed)
```

#### Visualization
```python
from visualization import MissingDataVisualizer

viz = MissingDataVisualizer()
viz.plot_missing_barplot(df)
viz.plot_distribution_comparison(df_original, df_imputed)
```

## 📊 Cấu trúc project

```
missing_data_lab/
│
├── data_generator.py          # Sinh dữ liệu với missing patterns
├── imputation_strategies.py   # Các chiến lược xử lý missing data
├── evaluation.py              # Đánh giá và so sánh strategies
├── visualization.py           # Visualization tools
├── main.py                    # Interactive CLI interface
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

## 🎓 Kiến thức học được

### Khi nào dùng strategy nào?

| Strategy | Best for | Pros | Cons |
|----------|----------|------|------|
| **drop** | MCAR, missing rate <5% | Không tạo bias | Mất data |
| **mean/median** | MCAR, numeric | Nhanh, đơn giản | Giảm variance |
| **mode** | MCAR, categorical | Phù hợp categorical | Tăng class phổ biến |
| **ffill/bfill** | Time-series | Bảo toàn temporal | Giả định không đổi |
| **knn** | MAR | Học từ neighbors | Chậm với big data |
| **iterative** | MAR, high accuracy | Sophisticated nhất | Phức tạp, chậm |

### Missing Pattern Decision Tree

```
Dữ liệu bị thiếu có liên quan đến:
├─ Không liên quan gì? → MCAR
│   └─ Dùng: mean/median/mode
├─ Các biến khác? → MAR
│   └─ Dùng: KNN, Iterative
└─ Chính nó? → MNAR
    └─ Cần domain knowledge
```

## 📖 Learning Tips

1. **Bắt đầu với MCAR**: Dễ hiểu nhất, thử tất cả strategies
2. **So sánh MAR vs MCAR**: Xem KNN/Iterative tốt hơn mean/median như thế nào
3. **Thử MNAR**: Hiểu tại sao đây là case khó nhất
4. **Experiment**: Thay đổi missing_rate và xem ảnh hưởng
5. **Visualize**: Luôn xem distribution trước và sau imputation

## 🔍 Ví dụ output

### Comparison Summary
```
================================================================================
TỔNG KẾT SO SÁNH CÁC CHIẾN LƯỢC (NUMERIC)
================================================================================

📊 Column: score
--------------------------------------------------------------------------------
Strategy        MAE          RMSE         MAE_%        RMSE_%       N_Imputed 
--------------------------------------------------------------------------------
mean            8.2341       10.5632      10.98        14.08        100       
median          8.1892       10.4521      10.92        13.94        100       
knn             6.7231       8.9234       8.96         11.90        100       
iterative       6.4521       8.6743       8.60         11.57        100       

✅ Best strategy (MAE): iterative
✅ Best strategy (RMSE): iterative
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Add more strategies
- Improve documentation

## 📝 License

This project is for educational purposes.

## 👨‍💻 Author

Created for learning and practicing missing data handling techniques in Python.

---

**Happy Learning! 🎓**