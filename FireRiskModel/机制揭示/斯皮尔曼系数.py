#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称:
    XGBoost 预测变量斯皮尔曼系数计算与三角形热图绘制

功能简介:
    1. 按 XGBoost 建模脚本相同的数据来源，读取全部 face_* 与 mix_* 的 Basic CSV。
    2. 仅提取 XGBoost 建模使用的预测变量，不包含目标变量。
    3. 不再区分 Interface 与 Intermix，直接合并全部样本计算预测变量之间的 Spearman 相关系数。
    4. 输出一个总的预测变量 Spearman 三角形热图，不导出任何 CSV。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable


plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11

ANNOTATION_FONT_SIZE = 14

OUTPUT_DIR = Path(r"B:\WUI\Picture")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRIC_CONFIGS = [
    {
        "name": "Fire frequency",
        "slug": "fire_frequency",
        "effect_dir": Path(r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num"),
        "basic_dir": Path(r"I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic"),
    },
    {
        "name": "Normalized burn area",
        "slug": "normalized_burn_area",
        "effect_dir": Path(r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area"),
        "basic_dir": Path(r"I:\Processing data\渔网分类建模\Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic"),
    },
]

FEATURE_COLUMNS = ["Cropland_Area_km2", "Pasture_Band2_Mean", "GUB_area_km2", "GDP", "PopD", "Roadsd"]
FEATURE_LABELS = {
    "GDP": "GDP",
    "PopD": "Population density",
    "Roadsd": "Road density",
    "Cropland_Area_km2": "Cropland",
    "GUB_area_km2": "Built environment",
    "Pasture_Band2_Mean": "Pasture",
}
HEATMAP_OUTPUT = OUTPUT_DIR / "XGBoost_predictors_spearman_triangle_heatmap_All_WUI.png"


def normalize_grid_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def extract_effect_category_id(csv_path: Path) -> str:
    stem = csv_path.stem
    prefix = "HFI_effect_"
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected effect filename: {csv_path.name}")
    return stem[len(prefix):]


def discover_matched_files(metric_config: dict[str, object]) -> list[tuple[str, str, str, Path]]:
    effect_dir = Path(metric_config["effect_dir"])
    basic_dir = Path(metric_config["basic_dir"])

    effect_files = {extract_effect_category_id(path): path for path in sorted(effect_dir.glob("HFI_effect_*.csv"))}
    basic_files = {path.stem: path for path in sorted(basic_dir.glob("*.csv"))}
    matched_ids = sorted(set(effect_files).intersection(basic_files))

    rows: list[tuple[str, str, str, Path]] = []
    for category_id in matched_ids:
        rows.append(
            (
                str(metric_config["name"]),
                str(metric_config["slug"]),
                category_id,
                basic_files[category_id],
            )
        )
    return rows


def read_basic_predictors(csv_path: Path) -> pd.DataFrame:
    required_columns = ["GRID_ID", "Year"] + FEATURE_COLUMNS
    df = pd.read_csv(csv_path, usecols=required_columns, low_memory=False)
    df["GRID_ID"] = normalize_grid_id(df["GRID_ID"])
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(np.float32)
    df = df.dropna(subset=["GRID_ID", "Year"] + FEATURE_COLUMNS).copy()
    return df.groupby(["GRID_ID", "Year"], as_index=False).agg(**{column: (column, "mean") for column in FEATURE_COLUMNS})


def build_predictor_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for metric_config in METRIC_CONFIGS:
        matched_files = discover_matched_files(metric_config)
        for metric_name, metric_slug, category_id, basic_path in matched_files:
            del metric_slug
            basic_df = read_basic_predictors(basic_path)
            if basic_df.empty:
                summary_rows.append(
                    {
                        "metric_name": metric_name,
                        "category_id": category_id,
                        "rows_loaded": 0,
                    }
                )
                continue

            basic_df["metric_name"] = metric_name
            basic_df["category_id"] = category_id
            dataset_parts.append(basic_df)
            summary_rows.append(
                {
                    "metric_name": metric_name,
                    "category_id": category_id,
                    "rows_loaded": len(basic_df),
                }
            )

    if not dataset_parts:
        raise ValueError("No predictor samples were loaded from the matched Basic CSV files.")

    dataset = pd.concat(dataset_parts, ignore_index=True)
    dataset = dataset.drop_duplicates(
        subset=["category_id", "GRID_ID", "Year"] + FEATURE_COLUMNS
    ).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows)
    return dataset, summary_df


def compute_spearman_matrix(dataset: pd.DataFrame) -> pd.DataFrame:
    subset = dataset[FEATURE_COLUMNS].copy()
    if subset.empty:
        raise ValueError("No samples available for all WUI.")
    return subset.corr(method="spearman")


def plot_triangle_heatmap(ax: plt.Axes, corr_df: pd.DataFrame) -> None:
    label_order = [FEATURE_LABELS[column] for column in corr_df.columns]
    mask = np.triu(np.ones(corr_df.shape, dtype=bool), k=1)
    data = corr_df.to_numpy(dtype=float)
    masked_data = np.ma.array(data, mask=mask)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="white")

    image = ax.imshow(masked_data, cmap=cmap, vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(label_order)))
    ax.set_yticks(np.arange(len(label_order)))
    ax.set_xticklabels(label_order, rotation=90, ha="center", va="top")
    ax.set_yticklabels(label_order)

    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            if mask[row, col]:
                continue
            value = data[row, col]
            text_color = "white" if abs(value) >= 0.5 else "black"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=ANNOTATION_FONT_SIZE)
            ax.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="lightgray",
                    linewidth=0.8,
                )
            )

    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def save_predictor_triangle_heatmap(dataset: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.4))

    corr_df = compute_spearman_matrix(dataset)
    image = plot_triangle_heatmap(ax, corr_df)

    ax.set_anchor("E")
    divider = make_axes_locatable(ax)
    colorbar_ax = divider.append_axes("right", size="5%", pad=0.3)
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    colorbar.set_label("Spearman correlation")
    fig.subplots_adjust(left=0.16, right=0.91, bottom=0.36, top=0.91)
    fig.savefig(HEATMAP_OUTPUT, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    dataset, summary_df = build_predictor_dataset()
    del summary_df
    save_predictor_triangle_heatmap(dataset)
    print(f"Saved heatmap: {HEATMAP_OUTPUT}")


if __name__ == "__main__":
    main()
