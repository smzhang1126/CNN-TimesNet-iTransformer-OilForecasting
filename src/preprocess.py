# -*- coding: utf-8 -*-
"""
preprocess.py

数据预处理模块。

功能：
1. 读取 init_data.xlsx 中每个 sheet，每个 sheet 视为一口井；
2. 解析日期；
3. 填充缺失值；
4. 构造时间特征：
   - month_idx
   - month_days
   - month_sin
   - month_cos
5. 对目标月产油量做 log1p 变换，生成 月产油量_log；
6. 保存每口井的预处理 CSV：
   preprocessed_data/{well}_preprocessed.csv
7. 保存合并 CSV：
   preprocessed_data.csv

注意：
- 特征标准化不在 preprocess.py 中做；
- 标准化放在 data_loader.py 中，并且只用训练集统计量，避免数据泄漏。
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")


# =========================
# 默认配置
# =========================
DEFAULT_OUTPUT_DIR = "preprocessed_data"
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VALID_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15


# =========================
# 工具函数
# =========================
def parse_date(df):
    """
    将原始表格中的第一列重命名为 date，并构造时间特征。
    """
    df = df.copy()

    if len(df.columns) == 0:
        raise ValueError("输入表为空，无法解析日期。")

    # 如果没有 date 列，则默认第一列为日期列
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 如果日期解析失败，则用索引构造日期字符串
    if df["date"].isna().all():
        print("[WARN] 日期列全部解析失败，将使用行号作为 date。")
        df["date"] = pd.date_range(start="2000-01-01", periods=len(df), freq="MS")
    else:
        df["date"] = df["date"].fillna(method="ffill").fillna(method="bfill")

    df = df.sort_values("date").reset_index(drop=True)

    # 月份索引，从 1 开始
    year_min = df["date"].dt.year.min()
    df["month_idx"] = (df["date"].dt.year - year_min) * 12 + df["date"].dt.month

    # 当月天数
    df["month_days"] = df["date"].dt.days_in_month

    # 周期特征
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)

    return df


def fill_missing(df):
    """
    缺失值填充：
    1. 前向填充；
    2. 后向填充；
    3. 数值列用均值填充。
    """
    df = df.copy()

    df = df.fillna(method="ffill")
    df = df.fillna(method="bfill")

    num_cols = df.select_dtypes(include=[np.number]).columns

    for c in num_cols:
        if df[c].isna().any():
            mean_val = df[c].mean()
            if np.isnan(mean_val):
                mean_val = 0.0
            df[c] = df[c].fillna(mean_val)

    return df


def make_one_hot_encoder():
    """
    兼容不同版本 sklearn 的 OneHotEncoder。
    """
    try:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    return encoder


def get_candidate_feature_cols(df_all):
    """
    获取候选特征列。
    """
    base_candidates = [
        "地层压力",
        "井底压力",
        "套压",
        "油压",
        "回压",
        "month_idx",
        "month_days",
        "month_sin",
        "month_cos",
    ]

    candidate_cols = [c for c in base_candidates if c in df_all.columns]

    # 加入井号 One-Hot 列
    ohe_cols = [c for c in df_all.columns if str(c).startswith("well_")]
    candidate_cols += [c for c in ohe_cols if c not in candidate_cols]

    # 只保留数值列
    candidate_cols = [
        c for c in candidate_cols
        if c in df_all.columns and pd.api.types.is_numeric_dtype(df_all[c])
    ]

    return candidate_cols


# =========================
# 主预处理函数
# =========================
def process_all_sheets(
    input_excel_path,
    output_dir=DEFAULT_OUTPUT_DIR,
    train_ratio=DEFAULT_TRAIN_RATIO,
    valid_ratio=DEFAULT_VALID_RATIO,
    test_ratio=DEFAULT_TEST_RATIO
):
    """
    读取 Excel 所有 sheet 并进行预处理。
    """
    if not os.path.exists(input_excel_path):
        raise FileNotFoundError(f"找不到原始 Excel 文件: {input_excel_path}")

    os.makedirs(output_dir, exist_ok=True)

    xls = pd.ExcelFile(input_excel_path)
    sheet_names = xls.sheet_names

    if len(sheet_names) == 0:
        raise ValueError("Excel 中没有任何 sheet。")

    all_dfs = []

    for well in sheet_names:
        print(f"正在处理井：{well}")

        df = pd.read_excel(xls, sheet_name=well)

        df = parse_date(df)
        df = fill_missing(df)

        df["well"] = str(well)

        # 目标 log1p 变换
        if "月产油量" in df.columns:
            df["月产油量"] = pd.to_numeric(df["月产油量"], errors="coerce")
            df["月产油量"] = df["月产油量"].fillna(method="ffill").fillna(method="bfill")
            df["月产油量"] = df["月产油量"].fillna(0.0)
            df["月产油量"] = df["月产油量"].clip(lower=0.0)

            df["月产油量_log"] = np.log1p(df["月产油量"])
        else:
            raise ValueError(f"井 {well} 中找不到目标列：月产油量")

        all_dfs.append(df)

    # 合并所有井
    df_all = pd.concat(all_dfs, ignore_index=True)

    # 井号 One-Hot 编码
    try:
        encoder = make_one_hot_encoder()
        well_ohe = encoder.fit_transform(df_all[["well"]])
        well_cols = [f"well_{w}" for w in encoder.categories_[0]]

        well_ohe_df = pd.DataFrame(
            well_ohe,
            columns=well_cols,
            index=df_all.index
        )

        df_all = pd.concat([df_all, well_ohe_df], axis=1)

    except Exception as e:
        print(f"[WARN] 井号 One-Hot 编码失败，将跳过 One-Hot 特征。原因：{e}")

    candidate_feature_cols = get_candidate_feature_cols(df_all)

    print("候选特征列：")
    for c in candidate_feature_cols:
        print(f"  - {c}")

    # 按井保存
    for well in sheet_names:
        df_well = df_all[df_all["well"].astype(str) == str(well)].copy()
        df_well = df_well.sort_values("date").reset_index(drop=True)

        n = len(df_well)

        train_end = int(n * train_ratio)
        valid_end = train_end + int(n * valid_ratio)

        df_well["dataset"] = "test"

        if train_end > 0:
            df_well.loc[:train_end - 1, "dataset"] = "train"

        if valid_end > train_end:
            df_well.loc[train_end:valid_end - 1, "dataset"] = "valid"

        out_cols = ["date", "well", "月产油量", "月产油量_log", "dataset"]

        for c in candidate_feature_cols:
            if c in df_well.columns and c not in out_cols:
                out_cols.append(c)

        out_path = os.path.join(output_dir, f"{well}_preprocessed.csv")
        df_well[out_cols].to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"已保存井 {well} 的预处理文件：{out_path}，样本数：{n}")

    combined_out_path = os.path.join(output_dir, "all_wells_preprocessed.csv")
    df_all.to_csv(combined_out_path, index=False, encoding="utf-8-sig")

    print(f"已保存合并预处理文件：{combined_out_path}")

    return {
        "per_well_dir": output_dir,
        "combined_csv": combined_out_path,
        "feature_candidates": candidate_feature_cols
    }


def preprocess_data(
    input_excel_path="init_data.xlsx",
    combined_csv_path=None,
    output_dir=DEFAULT_OUTPUT_DIR
):
    """
    对外统一接口，供 train_v2.py 调用。

    参数：
        input_excel_path:
            原始 Excel 文件路径。

        combined_csv_path:
            合并 CSV 输出路径，例如 preprocessed_data.csv。
            如果为 None，则只保存到 preprocessed_data/all_wells_preprocessed.csv。

        output_dir:
            单井预处理文件输出目录。
    """
    res = process_all_sheets(
        input_excel_path=input_excel_path,
        output_dir=output_dir
    )

    if combined_csv_path is not None:
        try:
            df_all = pd.read_csv(res["combined_csv"], encoding="utf-8-sig")
            df_all.to_csv(combined_csv_path, index=False, encoding="utf-8-sig")
            print(f"合并 CSV 另存为：{combined_csv_path}")
            res["combined_csv"] = combined_csv_path
        except Exception as e:
            print(f"[WARN] 保存合并 CSV 到指定路径失败：{e}")

    return res


if __name__ == "__main__":
    preprocess_data(
        input_excel_path="init_data.xlsx",
        combined_csv_path="preprocessed_data.csv",
        output_dir=DEFAULT_OUTPUT_DIR
    )