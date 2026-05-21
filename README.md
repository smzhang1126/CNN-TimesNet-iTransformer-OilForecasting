# CNN-TimesNet-iTransformer-OilForecasting

A hybrid deep learning framework for monthly oil production forecasting based on CNN, TimesNet, iTransformer, XGBoost feature selection, and Whale Optimization Algorithm (WOA).

本项目实现了一种面向油井月产油量预测的混合深度学习模型。模型融合了 CNN 的局部时序特征提取能力、TimesNet 的周期模式建模能力以及 iTransformer 的变量维注意力建模能力，并结合 XGBoost 进行单井特征选择，使用 WOA 对关键超参数进行优化。

---

## 1. Overview

Oil production forecasting is an important task in oilfield development and production management. Monthly oil production is affected by formation pressure, bottom-hole pressure, casing pressure, tubing pressure, back pressure, seasonal factors, and other production-related variables. Traditional forecasting methods may have difficulty capturing nonlinear temporal dependencies, periodic variations, and cross-variable interactions simultaneously.

To address this issue, this repository provides an end-to-end forecasting pipeline:

1. Data preprocessing
2. Per-well feature selection using XGBoost
3. Sliding-window dataset construction
4. CNN-TimesNet-iTransformer model training
5. Hyperparameter optimization using WOA
6. Model evaluation and visualization

---

## 2. Main Contributions

The implemented framework contains the following components:

- **CNN branch**  
  Extracts local temporal patterns from multivariate monthly production data.

- **TimesNet branch**  
  Uses FFT-based period detection and period-aware convolution to capture periodic production characteristics.

- **iTransformer branch**  
  Treats variables as tokens and models inter-variable relationships using inverted self-attention.

- **Feature selection**  
  XGBoost is used to select the most important features for each well separately.

- **Hyperparameter optimization**  
  Whale Optimization Algorithm is used to search learning rate, hidden dimension, feed-forward dimension, dropout, attention heads, CNN kernel size, and batch size.

- **Log-space modeling**  
  Monthly oil production is transformed by `log1p`, and predictions are restored by `expm1`.

---

## 3. Repository Structure

```text
CNN-TimesNet-iTransformer-OilForecasting/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── src/
│   ├── model.py                  # CNN-TimesNet-iTransformer model and WOA
│   ├── data_loader.py            # Sliding-window DataLoader
│   ├── preprocess.py             # Data preprocessing
│   ├── top_features_selector.py  # XGBoost feature selection
│   ├── train.py                  # Training pipeline
│   └── evaluate.py               # Evaluation and visualization
│
├── examples/
│   ├── demo_oil_production.csv
│   ├── demo_top_features_per_well.csv
│   └── quick_test.py
│
├── scripts/
│   ├── run_demo.sh
│   └── run_demo.bat
│
└── results/
    └── .gitkeep