# CNN-TimesNet-iTransformer-OilForecasting

A hybrid deep learning framework for monthly oil production forecasting based on CNN, TimesNet, iTransformer, XGBoost feature selection, and Whale Optimization Algorithm (WOA).

本项目实现了一种面向油井月产量预测的混合深度学习模型。模型融合了 CNN 的局部时序特征提取能力、TimesNet 的周期模式建模能力以及 iTransformer 的变量维度注意力建模能力，并结合 XGBoost 进行单井特征选择，使用 WOA 对关键超参数进行优化。

---

## 1. Overview

Oil production forecasting is an important task in oilfield development and production management.  
Accurate forecasting helps with production planning, reservoir evaluation, and operational optimization.

This project proposes a hybrid forecasting framework that integrates:

- **CNN** for local temporal feature extraction
- **TimesNet** for periodic pattern modeling
- **iTransformer** for multivariate time-series representation
- **XGBoost** for feature selection
- **Whale Optimization Algorithm (WOA)** for hyperparameter optimization

The framework is designed for **monthly oil production forecasting**.

---

## 2. Features

- Hybrid deep learning architecture for oil production forecasting
- Feature selection using XGBoost
- Hyperparameter optimization using WOA
- Supports multivariate time-series input
- Modular and extensible code structure
- Suitable for single-well monthly production prediction tasks

---

## 3. Installation and Usage

### 3.1 Environment Requirements

Recommended environment:

- Python >= 3.8
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib

It is recommended to use a virtual environment such as `conda` or `venv`.

### 3.2 Clone the Repository

```bash
git clone https://github.com/smzhang1126/CNN-TimesNet-iTransformer-OilForecasting.git
cd CNN-TimesNet-iTransformer-OilForecasting
```
---

## 4. Environment Requirements

The code was developed and tested using Python. The recommended environment is:

- Python >= 3.8
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib

---

## 5. Installation

### 5.1 Clone the Repository

```bash
git clone https://github.com/smzhang1126/CNN-TimesNet-iTransformer-OilForecasting.git
cd CNN-TimesNet-iTransformer-OilForecasting
```
### 5.2 Install Dependencies

```bash
pip install -r requirements.txt
pip install numpy pandas scikit-learn xgboost matplotlib pyyaml torch
```
---

## 6. Data Description

The model is designed for monthly oil production forecasting. The input data should be organized as a time-series table, where each row represents one monthly record.

A typical input file may contain columns such as:

```text
date, oil_production, water_cut, liquid_production, pressure, injection_rate, ...
```
---

## 7. Quick Test

### 7.1 Run the quick test

A quick-test script is provided to verify whether the repository can run correctly.
```bash
python quick_test.py
```
The quick test will:

- Load the example dataset from `data/example_data.csv`
- Perform basic preprocessing
- Build the forecasting model
- Run a short training or inference process
- Output basic evaluation metrics
  
### 7.2 Expected output

After running the script, the terminal should print training loss and evaluation metrics such as:
```bash
Loading example data...
Building CNN-TimesNet-iTransformer quick-test model...
Running quick test...
Epoch [1/10], Loss: ...
Epoch [2/10], Loss: ...
...
Quick test completed successfully.
MAE:  ...
RMSE: ...
R2:   ...
Prediction results saved to: results/quick_test_predictions.csv
```
The exact numerical values may vary depending on the computing environment and package versions.

---

## 8. Running with User Data

To run the workflow using user-provided data, prepare a CSV file with the same column names as the example dataset:

```text
date,well_id,oil_production,liquid_production,water_cut,formation_pressure,bottom_hole_pressure,casing_pressure,tubing_pressure,back_pressure,injection_rate
```

Then replace:
```text
oil_production
```

with the user dataset, or modify the data path in `quick_test.py`:
```bash
DATA_PATH = "data/example_data.csv"
```

The target variable is:
```text
oil_production
```

The default input features are:
```text
liquid_production
water_cut
formation_pressure
bottom_hole_pressure
casing_pressure
tubing_pressure
back_pressure
injection_rate
```

After running the quick test, the following file will be generated:
```text
results/quick_test_predictions.csv
```
The `results/` directory is created automatically if it does not already exist.

---
