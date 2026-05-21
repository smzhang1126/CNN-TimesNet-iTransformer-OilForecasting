# -*- codeing = utf-8 -*-
# @time ： 2025/12/10 12:44
# @author : likun
# @file : top_selectorv2.py
# @software : PyCharm
# -*- coding: utf-8 -*-
"""
top_features_selector.py

功能：
- 遍历预处理后的单井 CSV 文件（*_preprocessed.csv）
- 对每口井使用 XGBoost 进行特征重要性分析
- 输出两份结果：
    1) top_features_per_well.csv
       格式：well_id, feature_1, feature_2, ...
    2) top_features_detailed.csv
       格式：well_id, rank, feature, importance_score

说明：
- 自动排除目标变量（月产油量、月产油量_log）
- 自动排除常见非特征列（date, dataset, well, 井号）
- 若样本过少或训练失败，则返回空列表
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")


# =========================
# 默认参数
# =========================
DEFAULT_DATA_DIR = "preprocessed_data"
DEFAULT_TOP_N = 5
DEFAULT_OUT_SUM = "top_features_per_well.csv"
DEFAULT_OUT_DETAILED = "top_features_detailed.csv"


def _select_top_for_single_well(
    df,
    top_n=DEFAULT_TOP_N,
    target_col_pref=("月产油量_log", "月产油量")
):
    """
    对单井 DataFrame 进行 XGBoost 特征重要性评估并返回 top_n 特征。

    返回：
        [(feature, score), ...]
    """
    if df is None or len(df) == 0:
        return []

    df = df.copy()

    if "date" in df.columns:
        try:
            df = df.sort_values("date").reset_index(drop=True)
        except Exception:
            pass

    # 选择目标列
    target_col = None
    for t in target_col_pref:
        if t in df.columns:
            target_col = t
            break

    if target_col is None:
        return []

    exclude_cols = {
        "date",
        "dataset",
        "well",
        "well_id",
        "井号",
        "井名",
        "月产油量",
        "月产油量_log",
        target_col,
    }

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    numeric_feature_cols = []

    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            if df[c].nunique(dropna=True) > 1:
                numeric_feature_cols.append(c)

    if len(numeric_feature_cols) == 0:
        return []

    X = df[numeric_feature_cols].copy()
    y = df[target_col].copy()

    valid_idx = ~(X.isna().any(axis=1) | y.isna())

    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    if len(X) < 5:
        return []

    try:
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )

        xgb_model.fit(X, y)

        importances = xgb_model.feature_importances_

    except Exception as e:
        print(f"[WARN] XGBoost 训练失败，跳过该井：{e}")
        return []

    if importances is None or len(importances) == 0:
        return []

    indices = np.argsort(importances)[::-1][:top_n]

    top_features = [numeric_feature_cols[i] for i in indices]
    top_scores = [float(importances[i]) for i in indices]

    return list(zip(top_features, top_scores))


def select_top_features_per_well(
    data_dir=DEFAULT_DATA_DIR,
    top_n=DEFAULT_TOP_N,
    output_path=DEFAULT_OUT_SUM,
    detailed_output=DEFAULT_OUT_DETAILED
):
    """
    遍历目录下所有 *_preprocessed.csv 文件，为每口井选择 Top-N 特征。
    """
    pattern = os.path.join(data_dir, "*_preprocessed.csv")
    files = glob.glob(pattern)

    wells_map = {}
    all_rows_sum = []
    all_rows_detail = []

    if len(files) == 0:
        print(f"[WARN] 目录 {data_dir} 下未找到 *_preprocessed.csv 文件")
        return wells_map

    files = sorted(files)

    for fp in files:
        well_name = os.path.basename(fp).replace("_preprocessed.csv", "")

        try:
            df = pd.read_csv(fp, encoding="utf-8-sig")
        except Exception as e:
            print(f"[WARN] 读取失败：{fp}，原因：{e}")
            continue

        top_list = _select_top_for_single_well(df, top_n=top_n)

        wells_map[well_name] = top_list

        # 汇总表：well_id, feature_1, feature_2, ...
        row_sum = {"well_id": well_name}

        for i, (feat, score) in enumerate(top_list, start=1):
            row_sum[f"feature_{i}"] = feat

        all_rows_sum.append(row_sum)

        # 详细表：well_id, rank, feature, importance_score
        for rank, (feat, score) in enumerate(top_list, start=1):
            all_rows_detail.append({
                "well_id": well_name,
                "rank": rank,
                "feature": feat,
                "importance_score": score
            })

    if len(all_rows_sum) > 0:
        df_sum = pd.DataFrame(all_rows_sum)
        df_sum.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"[OK] Top-N 特征汇总已输出：{output_path}")

    if len(all_rows_detail) > 0:
        df_detail = pd.DataFrame(all_rows_detail)
        df_detail.to_csv(detailed_output, index=False, encoding="utf-8-sig")
        print(f"[OK] Top-N 特征详细得分已输出：{detailed_output}")

    return wells_map


def main():
    print("开始按井计算 Top-N 特征...")

    wells_map = select_top_features_per_well(
        data_dir=DEFAULT_DATA_DIR,
        top_n=DEFAULT_TOP_N,
        output_path=DEFAULT_OUT_SUM,
        detailed_output=DEFAULT_OUT_DETAILED
    )

    print("完成。")

    return wells_map


if __name__ == "__main__":
    main()