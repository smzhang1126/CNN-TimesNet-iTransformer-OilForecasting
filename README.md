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

---

### 3.2 Clone the Repository

```bash
git clone https://github.com/smzhang1126/CNN-TimesNet-iTransformer-OilForecasting.git
cd CNN-TimesNet-iTransformer-OilForecasting

```bash
git clone https://github.com/smzhang1126/CNN-TimesNet-iTransformer-OilForecasting.git
cd CNN-TimesNet-iTransformer-OilForecasting
