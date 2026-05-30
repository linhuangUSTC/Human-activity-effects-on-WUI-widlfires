#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称: 将 GDP / PopD / Roadsd 分成 20 组并统计组内 HFI 影响

功能:
    1. 读取 Fire frequency 和 Fire area 的 HFI effect CSV，并保留 2003-2022 年的全部匹配样本。
    2. 读取对应的 Basic CSV，通过 category_id + Year + GRID_ID 进行匹配。
    3. 对每个指标和每个参数，先将 all WUI 的全部样本拉通，再按参数值从低到高划分为 20 个等样本组。
    4. 计算每组对应的 HFI effect 均值、标准误和参数范围。
    5. 绘制柱状图加误差棒，展示不同参数分组下的 HFI effect。

输入:
    - B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num
    - B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area
    - I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic
    - I:\Processing data\渔网分类建模\Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic

输出:
    - B:\WUI\Picture\Grouped_Human Footprint_effect_Fire_frequency_by_GDP_2003_2022.png
    - B:\WUI\Picture\Grouped_Human Footprint_effect_Fire_frequency_by_PopD_2003_2022.png
    - B:\WUI\Picture\Grouped_Human Footprint_effect_Fire_frequency_by_Roadsd_2003_2022.png
    - B:\WUI\Picture\Grouped_Human Footprint_effect_Fire_area_by_GDP_2003_2022.png
    - B:\WUI\Picture\Grouped_Human Footprint_effect_Fire_area_by_PopD_2003_2022.png
    - B:\WUI\Picture\Grouped_Human Footprint_effect_Fire_area_by_Roadsd_2003_2022.png
    - B:\WUI\Picture\Grouped_Human Footprint_effect_2003_2022_group_summary.csv（保留横轴分组范围与每组样本量）
    - B:\WUI\Picture\Grouped_Human Footprint_effect_2003_2022_spearman_coefficients.png
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.stats import spearmanr


OUTPUT_DIR = Path(r"B:\WUI\Picture")

YEAR_START = 2003
YEAR_END = 2022
YEAR_TAG = f"{YEAR_START}_{YEAR_END}"
YEAR_DESC = f"{YEAR_START}-{YEAR_END}"
N_GROUPS = 20
CHUNK_SIZE = 250_000
KEY_COLUMNS = ["Year", "GRID_ID"]
PARAM_COLUMNS = ["GDP", "PopD", "Roadsd"]
EFFECT_COLUMN = "diff_contribution"

PARAMETER_CONFIGS = [
    {
        "column": "GDP",
        "label": "GDP",
        "slug": "GDP",
        "color": "#d95f0e",
    },
    {
        "column": "PopD",
        "label": "Population density",
        "slug": "PopD",
        "color": "#1b9e77",
    },
    {
        "column": "Roadsd",
        "label": "Road density",
        "slug": "Roadsd",
        "color": "#386cb0",
    },
]

DATASETS = [
    {
        "metric_slug": "Fire_frequency",
        "effect_dir": Path(r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num"),
        "basic_dir": Path(r"I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic"),
        "effect_suffix": "fire_num_with_WUI_offset",
        "effect_label": "Fire frequency",
    },
    {
        "metric_slug": "Fire_area",
        "effect_dir": Path(r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area"),
        "basic_dir": Path(r"I:\Processing data\渔网分类建模\Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic"),
        "effect_suffix": "fire_area",
        "effect_label": "Fire area",
    },
]


plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 12


def extract_category_id(path: str, *, is_effect: bool, suffix: str) -> str:
    stem = Path(path).stem
    if is_effect:
        pattern = rf"HFI_effect_((face|mix)_[A-D]_[A-Z]+)_{suffix}$"
    else:
        pattern = rf"((face|mix)_[A-D]_[A-Z]+)_{suffix}$"
    match = re.search(pattern, stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unrecognized file name pattern: {stem}")
    return match.group(1)


def normalize_grid_id(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.round().astype("Int64")


def load_basic_lookup(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=KEY_COLUMNS + PARAM_COLUMNS, low_memory=False)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["GRID_ID"] = normalize_grid_id(df["GRID_ID"])
    for column in PARAM_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df[df["Year"].between(YEAR_START, YEAR_END, inclusive="both")]
    df = df.dropna(subset=KEY_COLUMNS + PARAM_COLUMNS)
    if df.empty:
        return df

    deduped = (
        df.groupby(KEY_COLUMNS, as_index=False)[PARAM_COLUMNS]
        .mean()
        .reset_index(drop=True)
    )
    deduped["Year"] = deduped["Year"].astype(np.int64)
    deduped["GRID_ID"] = deduped["GRID_ID"].astype(np.int64)
    return deduped


def load_and_match_all_samples(dataset: dict[str, object]) -> pd.DataFrame:
    effect_dir = Path(str(dataset["effect_dir"]))
    basic_dir = Path(str(dataset["basic_dir"]))
    effect_suffix = str(dataset["effect_suffix"])
    effect_files = sorted(glob.glob(str(effect_dir / "HFI_effect_*.csv")))
    basic_files = sorted(glob.glob(str(basic_dir / "*.csv")))
    if not effect_files:
        raise FileNotFoundError(f"No effect CSV files found in: {effect_dir}")
    if not basic_files:
        raise FileNotFoundError(f"No basic CSV files found in: {basic_dir}")

    effect_map: dict[str, str] = {}
    basic_map: dict[str, str] = {}

    for path in effect_files:
        effect_map[extract_category_id(path, is_effect=True, suffix=effect_suffix)] = path
    for path in basic_files:
        try:
            basic_map[extract_category_id(path, is_effect=False, suffix=effect_suffix)] = path
        except ValueError:
            continue

    matched_categories = sorted(set(effect_map).intersection(basic_map))
    if not matched_categories:
        raise ValueError("No matched category IDs found between effect and basic CSVs.")

    parts: list[pd.DataFrame] = []
    print(f"{dataset['effect_label']}: matched category files = {len(matched_categories)}")

    for idx, category_id in enumerate(matched_categories, start=1):
        effect_csv = effect_map[category_id]
        basic_csv = basic_map[category_id]
        basic_lookup = load_basic_lookup(basic_csv)
        if basic_lookup.empty:
            print(
                f"  [{idx}/{len(matched_categories)}] {category_id}: "
                f"no basic rows for Years={YEAR_DESC}"
            )
            continue

        matched_count = 0
        for chunk in pd.read_csv(
            effect_csv,
            usecols=KEY_COLUMNS + [EFFECT_COLUMN, "HFI"],
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            chunk["Year"] = pd.to_numeric(chunk["Year"], errors="coerce").astype("Int64")
            chunk["GRID_ID"] = normalize_grid_id(chunk["GRID_ID"])
            chunk[EFFECT_COLUMN] = pd.to_numeric(chunk[EFFECT_COLUMN], errors="coerce")
            chunk["HFI"] = pd.to_numeric(chunk["HFI"], errors="coerce")
            chunk = chunk[
                chunk["Year"].between(YEAR_START, YEAR_END, inclusive="both")
            ].dropna(subset=KEY_COLUMNS + [EFFECT_COLUMN])
            if chunk.empty:
                continue

            chunk["Year"] = chunk["Year"].astype(np.int64)
            chunk["GRID_ID"] = chunk["GRID_ID"].astype(np.int64)
            merged = chunk.merge(basic_lookup, how="inner", on=KEY_COLUMNS, validate="many_to_one")
            if merged.empty:
                continue

            merged = merged[KEY_COLUMNS + [EFFECT_COLUMN, "HFI"] + PARAM_COLUMNS].copy()
            merged["Year"] = merged["Year"].astype(np.int16)
            merged["GRID_ID"] = merged["GRID_ID"].astype(np.int64)
            for column in [EFFECT_COLUMN, "HFI"] + PARAM_COLUMNS:
                merged[column] = merged[column].astype(np.float32)
            parts.append(merged)
            matched_count += len(merged)

        print(f"  [{idx}/{len(matched_categories)}] {category_id}: matched rows = {matched_count:,}")

    if not parts:
        raise ValueError(f"No matched sample rows found for Years={YEAR_DESC}.")

    matched = pd.concat(parts, ignore_index=True)
    matched = matched.dropna(subset=[EFFECT_COLUMN] + PARAM_COLUMNS)
    print(
        f"{dataset['effect_label']}: total matched rows for Years={YEAR_DESC}: "
        f"{len(matched):,}; unique Years = {matched['Year'].nunique():,}; "
        f"unique GRID_ID = {matched['GRID_ID'].nunique():,}"
    )
    return matched


def assign_equal_count_groups_by_parameter(
    df: pd.DataFrame,
    parameter_col: str,
    n_groups: int,
) -> pd.DataFrame:
    grouped = df.copy()
    values = grouped[parameter_col].to_numpy(dtype=float)
    if not np.isfinite(values).any():
        raise ValueError(f"{parameter_col} contains no valid numeric data for binning.")
    if np.nanmin(values) == np.nanmax(values):
        raise ValueError(f"{parameter_col} is constant; cannot create equal-count bins.")
    if len(grouped) < n_groups:
        raise ValueError(f"Only {len(grouped)} rows available; cannot split into {n_groups} groups.")

    ranked = grouped[parameter_col].rank(method="first", ascending=True)
    raw_bins = pd.qcut(ranked, q=n_groups, labels=False)
    if raw_bins.isna().any():
        raise ValueError(f"Some {parameter_col} values could not be assigned to equal-count bins.")

    grouped["parameter_group"] = (raw_bins.astype(int) + 1).astype(int)
    group_ranges = (
        grouped.groupby("parameter_group", as_index=False)[parameter_col]
        .agg(param_bin_lower="min", param_bin_upper="max")
    )
    grouped = grouped.merge(group_ranges, how="left", on="parameter_group", validate="many_to_one")
    return grouped.sort_values(["parameter_group", parameter_col], ascending=[True, True]).reset_index(drop=True)


def summarize_parameter_groups(df: pd.DataFrame, parameter_col: str) -> pd.DataFrame:
    summary = (
        df.groupby("parameter_group", as_index=False)
        .agg(
            n_samples=(EFFECT_COLUMN, "size"),
            param_bin_lower=("param_bin_lower", "min"),
            param_bin_upper=("param_bin_upper", "max"),
            min_GDP=("GDP", "min"),
            mean_GDP=("GDP", "mean"),
            max_GDP=("GDP", "max"),
            min_PopD=("PopD", "min"),
            mean_PopD=("PopD", "mean"),
            max_PopD=("PopD", "max"),
            min_Roadsd=("Roadsd", "min"),
            mean_Roadsd=("Roadsd", "mean"),
            max_Roadsd=("Roadsd", "max"),
            mean_parameter=(parameter_col, "mean"),
            median_parameter=(parameter_col, "median"),
            mean_hfi_effect=(EFFECT_COLUMN, "mean"),
            median_hfi_effect=(EFFECT_COLUMN, "median"),
            sd_hfi_effect=(EFFECT_COLUMN, "std"),
            mean_HFI=("HFI", "mean"),
        )
        .sort_values("parameter_group")
        .reset_index(drop=True)
    )
    summary["se_hfi_effect"] = summary["sd_hfi_effect"] / np.sqrt(summary["n_samples"].clip(lower=1))
    summary["se_hfi_effect"] = summary["se_hfi_effect"].fillna(0.0)
    summary["group_label"] = summary["parameter_group"].apply(lambda x: f"G{x:02d}")
    return summary


def _format_p_value(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def _format_value(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    abs_value = abs(value)
    if abs_value >= 1000:
        return f"{value:.0f}"
    if abs_value >= 10:
        return f"{value:.1f}"
    if abs_value >= 0.1:
        return f"{value:.2f}"
    if abs_value >= 0.01:
        return f"{value:.3f}"
    return f"{value:.2e}"


def _format_range_label(lower: float, upper: float) -> str:
    return f"[{_format_value(lower)}, {_format_value(upper)}]"


def compute_group_spearman(
    group_summary: pd.DataFrame,
    dataset: dict[str, object],
    parameter_config: dict[str, str],
) -> pd.DataFrame:
    rho, p_value = spearmanr(group_summary["mean_parameter"], group_summary["mean_hfi_effect"])
    return pd.DataFrame(
        [
            {
                "year_range": YEAR_DESC,
                "n_groups": len(group_summary),
                "metric_slug": dataset["metric_slug"],
                "effect_label": dataset["effect_label"],
                "parameter": parameter_config["slug"],
                "spearman_rho": rho,
                "p_value": p_value,
            }
        ]
    )


def plot_parameter_group_chart(
    group_summary: pd.DataFrame,
    dataset: dict[str, object],
    parameter_config: dict[str, str],
) -> Path:
    summary = group_summary.sort_values("parameter_group").reset_index(drop=True)
    positions = np.arange(len(summary), dtype=float)
    range_labels = [
        _format_range_label(row["param_bin_lower"], row["param_bin_upper"])
        for _, row in summary.iterrows()
    ]
    rho, p_value = spearmanr(summary["mean_parameter"], summary["mean_hfi_effect"])

    fig, ax = plt.subplots(figsize=(12.5, 7.8), facecolor="white")
    ax.bar(
        positions,
        summary["mean_hfi_effect"],
        width=0.72,
        color=parameter_config["color"],
        alpha=0.60,
        edgecolor="#666666",
        linewidth=0.9,
        yerr=summary["se_hfi_effect"],
        ecolor="#404040",
        capsize=3.5,
        zorder=3,
    )
    ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", zorder=2)

    ax.text(
        0.97,
        0.97,
        f"Spearman rho = {rho:.3f}\np = {_format_p_value(p_value)}\nn = {len(summary)} groups",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#b8b8b8",
            "alpha": 0.95,
        },
    )

    ax.set_xlabel(f"{parameter_config['label']} range")
    ax.set_ylabel(f"Mean Human Footprint effect on {dataset['effect_label']}")
    ax.set_title(
        f"Years {YEAR_DESC} pooled: 20 equal-count groups by {parameter_config['label']}\n"
        f"Human Footprint effect on {dataset['effect_label']} across {parameter_config['label']} groups",
        fontsize=14,
        pad=12,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(range_labels, rotation=-90)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_path = OUTPUT_DIR / (
        f"Grouped_Human Footprint_effect_{dataset['metric_slug']}_by_{parameter_config['slug']}_{YEAR_TAG}.png"
    )
    fig.subplots_adjust(bottom=0.34, top=0.90, left=0.10, right=0.98)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def _spearman_parameter_label(parameter: str) -> str:
    return {
        "GDP": "GDP",
        "PopD": "Population density",
        "Roadsd": "Road density",
    }[parameter]


def _significance_marker(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "ns"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def plot_spearman_heatmap(all_spearman: pd.DataFrame) -> Path:
    plot_df = all_spearman.copy()
    plot_df["parameter_label"] = plot_df["parameter"].map(_spearman_parameter_label)
    parameter_order = ["GDP", "Population density", "Road density"]
    metric_order = ["Fire_frequency", "Fire_area"]
    metric_labels = {
        "Fire_frequency": "Human Footprint effect on Fire frequency",
        "Fire_area": "Human Footprint effect on Fire area",
    }

    rho_matrix = np.full((len(metric_order), len(parameter_order)), np.nan, dtype=float)
    annotation_matrix: list[list[str]] = []

    for row_idx, metric_slug in enumerate(metric_order):
        subset = (
            plot_df.loc[plot_df["metric_slug"] == metric_slug]
            .set_index("parameter_label")
            .reindex(parameter_order)
        )
        row_annotations: list[str] = []
        for col_idx, parameter_label in enumerate(parameter_order):
            row = subset.loc[parameter_label]
            rho = float(row["spearman_rho"])
            p_value = float(row["p_value"])
            rho_matrix[row_idx, col_idx] = rho
            row_annotations.append(f"{rho:.3f}\n{_significance_marker(p_value)}")
        annotation_matrix.append(row_annotations)

    fig, ax = plt.subplots(figsize=(8.8, 5.8), facecolor="white")
    if np.nanmax(rho_matrix) > 0:
        norm = Normalize(vmin=-1.0, vmax=1.0)
        cmap_name = "RdBu_r"
        cbar_ticks = np.linspace(-1.0, 1.0, 5)
    else:
        norm = Normalize(vmin=-1.0, vmax=0.0)
        cmap_name = "Blues_r"
        cbar_ticks = np.linspace(-1.0, 0.0, 6)
    image = ax.imshow(rho_matrix, cmap=cmap_name, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(parameter_order)))
    ax.set_xticklabels(parameter_order)
    ax.set_yticks(np.arange(len(metric_order)))
    ax.set_yticklabels([metric_labels[item] for item in metric_order])
    ax.set_title(
        "Spearman correlation heatmap of grouped Human Footprint effect with GDP, population density and road density\n"
        f"Years {YEAR_DESC} pooled, 20 equal-count groups",
        fontsize=14,
        pad=12,
    )

    ax.set_xticks(np.arange(-0.5, len(parameter_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(metric_order), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row_idx in range(len(metric_order)):
        for col_idx in range(len(parameter_order)):
            rho = rho_matrix[row_idx, col_idx]
            text_color = "white" if abs(rho) >= 0.55 else "black"
            ax.text(
                col_idx,
                row_idx,
                annotation_matrix[row_idx][col_idx],
                ha="center",
                va="center",
                fontsize=11,
                color=text_color,
                fontweight="bold",
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman rho")
    cbar.set_ticks(cbar_ticks)
    cbar.outline.set_visible(False)

    ax.text(
        1.0,
        -0.13,
        "*** p < 0.001, ** p < 0.01, * p < 0.05, ns not significant",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#444444",
    )

    output_path = OUTPUT_DIR / f"Grouped_Human Footprint_effect_{YEAR_TAG}_spearman_coefficients.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def save_group_summary_csv(all_group_summaries: pd.DataFrame) -> Path:
    output_path = OUTPUT_DIR / f"Grouped_Human Footprint_effect_{YEAR_TAG}_group_summary.csv"
    export_df = all_group_summaries[
        [
            "metric_slug",
            "effect_label",
            "grouping_parameter",
            "parameter_group",
            "group_label",
            "n_samples",
            "param_bin_lower",
            "param_bin_upper",
        ]
    ].copy()
    export_df["x_axis_range"] = export_df.apply(
        lambda row: _format_range_label(row["param_bin_lower"], row["param_bin_upper"]),
        axis=1,
    )
    export_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_spearman: list[pd.DataFrame] = []
    all_group_summaries: list[pd.DataFrame] = []

    for dataset in DATASETS:
        print(f"\n{'=' * 80}")
        print(f"Processing dataset: {dataset['effect_label']}")
        matched_samples = load_and_match_all_samples(dataset)

        for parameter_config in PARAMETER_CONFIGS:
            parameter_col = parameter_config["column"]
            print(f"  Grouping by {parameter_col} ...")
            grouped = assign_equal_count_groups_by_parameter(matched_samples, parameter_col, N_GROUPS)
            summary = summarize_parameter_groups(grouped, parameter_col)
            summary.insert(0, "grouping_parameter", parameter_config["slug"])
            summary.insert(0, "effect_label", dataset["effect_label"])
            summary.insert(0, "metric_slug", dataset["metric_slug"])
            all_group_summaries.append(summary.copy())
            all_spearman.append(compute_group_spearman(summary, dataset, parameter_config))
            output_path = plot_parameter_group_chart(summary, dataset, parameter_config)
            print(f"  Plot saved to: {output_path}")

    group_summary_csv_path = save_group_summary_csv(pd.concat(all_group_summaries, ignore_index=True))
    heatmap_path = plot_spearman_heatmap(pd.concat(all_spearman, ignore_index=True))
    print(f"\nGroup summary CSV saved to: {group_summary_csv_path}")
    print(f"\nSpearman heatmap saved to: {heatmap_path}")


if __name__ == "__main__":
    main()

