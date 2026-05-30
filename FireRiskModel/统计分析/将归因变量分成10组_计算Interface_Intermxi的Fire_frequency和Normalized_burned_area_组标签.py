#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称:
    将多种归因参数分成 10 组，并将 Fire frequency 与 Normalized burned area 合并为 1x2 柱状图

功能:
    1. 直接从 Fire frequency 和 Normalized burned area 的 Basic CSV 读取 2003-2022 年样本。
    2. 保留 interface 与 intermix 两种 WUI 范围，不再输出 overall。
    3. 对每个指标和每个参数，先将 Interface + Intermix 合并样本按参数值从低到高划分为
       10 个等样本组，再分别映射回两种 WUI 范围，确保两类 WUI 的同编号组使用完全一致的参数范围。
    4. 计算每个分组的 Fire frequency / Normalized burned area 均值和标准误。
    5. 对每个参数输出一张 1x2 图：左侧 Fire frequency，右侧 Normalized burned area。

输入:
    - I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic
    - I:\Processing data\渔网分类建模\Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic

输出:
    - B:\WUI\Picture\Grouped_Fire_metrics_by_{parameter}_interface_intermix_2003_2022_10groups_1x2_barplot.png
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, ScalarFormatter


OUTPUT_DIR = Path(r"B:\WUI\Picture")

YEAR_START = 2003
YEAR_END = 2022
YEAR_TAG = f"{YEAR_START}_{YEAR_END}"
YEAR_DESC = f"{YEAR_START}-{YEAR_END}"
N_GROUPS = 10
GROUP_TAG = f"{N_GROUPS}groups"
CHUNK_SIZE = 250_000
KEY_COLUMNS = ["Year", "GRID_ID"]
RESPONSE_COLUMN = "metric_value"
PARAMETER_COLUMNS = ["GDP", "PopD", "Roadsd", "GUB_area_km2", "Cropland_Area_km2", "Pasture_Band2_Mean"]

PARAMETER_CONFIGS = [
    {"column": "GDP", "label": "GDP", "slug": "GDP"},
    {"column": "PopD", "label": "Population density", "slug": "PopD"},
    {"column": "Roadsd", "label": "Road density", "slug": "Roadsd"},
    {"column": "GUB_area_km2", "label": "Built environment", "slug": "Built_environment"},
    {"column": "Cropland_Area_km2", "label": "Cropland", "slug": "Cropland"},
    {"column": "Pasture_Band2_Mean", "label": "Pasture", "slug": "Pasture"},
]

WUI_SCOPE_CONFIGS = [
    {"slug": "interface", "label": "Interface WUI", "mode": "face", "color": "#2b8cbe"},
    {"slug": "intermix", "label": "Intermix WUI", "mode": "mix", "color": "#d95f0e"},
]

DATASETS = [
    {
        "metric_slug": "Fire_frequency",
        "basic_dir": Path(r"I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic"),
        "basic_suffix": "fire_num_with_WUI_offset",
        "target_column": "Fire_frequency",
        "metric_label": "Fire frequency",
    },
    {
        "metric_slug": "Normalized_burned_area",
        "basic_dir": Path(r"I:\Processing data\渔网分类建模\Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic"),
        "basic_suffix": "fire_area",
        "target_column": "Normalized_fire_area",
        "metric_label": "Normalized burned area",
    },
]


plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 12

AXIS_LABEL_FONT_SIZE = 28
TICK_LABEL_FONT_SIZE = 20
SUBPLOT_TITLE_FONT_SIZE = 30
FIGURE_WIDTH = 24.0
FIGURE_HEIGHT = 8.075
Y_MAJOR_TICK_BINS = 4


def extract_category_id(path: str, suffix: str) -> str:
    stem = Path(path).stem
    pattern = rf"((face|mix)_[A-D]_[A-Z]+)_{suffix}$"
    match = re.search(pattern, stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unrecognized file name pattern: {stem}")
    return match.group(1)


def category_mode(category_id: str) -> str:
    return category_id.split("_", 1)[0].lower()


def normalize_grid_id(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.round().astype("Int64")


def load_basic_dataset(dataset: dict[str, object]) -> pd.DataFrame:
    basic_dir = Path(str(dataset["basic_dir"]))
    basic_suffix = str(dataset["basic_suffix"])
    target_column = str(dataset["target_column"])

    basic_files = sorted(glob.glob(str(basic_dir / "*.csv")))
    if not basic_files:
        raise FileNotFoundError(f"No basic CSV files found in: {basic_dir}")

    parts: list[pd.DataFrame] = []
    for csv_path in basic_files:
        try:
            category_id = extract_category_id(csv_path, basic_suffix)
        except ValueError:
            continue

        wui_mode = category_mode(category_id)
        for chunk in pd.read_csv(
            csv_path,
            usecols=KEY_COLUMNS + PARAMETER_COLUMNS + [target_column],
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            chunk["Year"] = pd.to_numeric(chunk["Year"], errors="coerce").astype("Int64")
            chunk["GRID_ID"] = normalize_grid_id(chunk["GRID_ID"])
            chunk[target_column] = pd.to_numeric(chunk[target_column], errors="coerce")
            for parameter_col in PARAMETER_COLUMNS:
                chunk[parameter_col] = pd.to_numeric(chunk[parameter_col], errors="coerce")

            chunk = chunk[
                chunk["Year"].between(YEAR_START, YEAR_END, inclusive="both")
            ].dropna(subset=KEY_COLUMNS + [target_column])
            chunk = chunk.dropna(subset=PARAMETER_COLUMNS, how="all")
            if chunk.empty:
                continue

            kept = chunk[PARAMETER_COLUMNS + [target_column]].copy()
            for parameter_col in PARAMETER_COLUMNS:
                kept[parameter_col] = kept[parameter_col].astype(np.float32)
            kept[RESPONSE_COLUMN] = kept.pop(target_column).astype(np.float32)
            kept["metric_slug"] = str(dataset["metric_slug"])
            kept["metric_label"] = str(dataset["metric_label"])
            kept["wui_mode"] = wui_mode
            parts.append(kept)

    if not parts:
        raise ValueError(f"No valid sample rows found for Years={YEAR_DESC} in {basic_dir}.")

    matched = pd.concat(parts, ignore_index=True)
    matched = matched.dropna(subset=[RESPONSE_COLUMN])
    matched = matched.dropna(subset=PARAMETER_COLUMNS, how="all")
    return matched


def assign_equal_count_groups(df: pd.DataFrame, parameter_col: str, n_groups: int) -> pd.DataFrame:
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
        .sort_values("parameter_group")
        .reset_index(drop=True)
    )
    grouped = grouped.merge(group_ranges, how="left", on="parameter_group", validate="many_to_one")
    return grouped


def extract_group_ranges(grouped: pd.DataFrame) -> pd.DataFrame:
    return (
        grouped[["parameter_group", "param_bin_lower", "param_bin_upper"]]
        .drop_duplicates(subset=["parameter_group"])
        .sort_values("parameter_group")
        .reset_index(drop=True)
    )


def summarize_groups(
    df: pd.DataFrame,
    parameter_col: str,
    group_ranges: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if group_ranges is None:
        group_ranges = (
            df.groupby("parameter_group", as_index=False)
            .agg(
                param_bin_lower=("param_bin_lower", "min"),
                param_bin_upper=("param_bin_upper", "max"),
            )
            .sort_values("parameter_group")
            .reset_index(drop=True)
        )
    else:
        group_ranges = group_ranges.copy().sort_values("parameter_group").reset_index(drop=True)

    summary = (
        df.groupby("parameter_group", as_index=False)
        .agg(
            n_samples=(RESPONSE_COLUMN, "size"),
            mean_parameter=(parameter_col, "mean"),
            mean_metric_value=(RESPONSE_COLUMN, "mean"),
            sd_metric_value=(RESPONSE_COLUMN, "std"),
        )
        .sort_values("parameter_group")
        .reset_index(drop=True)
    )

    full_groups = pd.DataFrame({"parameter_group": np.arange(1, N_GROUPS + 1, dtype=int)})
    full_groups = full_groups.merge(group_ranges, how="left", on="parameter_group")
    completed = full_groups.merge(summary, how="left", on="parameter_group")
    completed["n_samples"] = completed["n_samples"].fillna(0).astype(int)
    completed["se_metric_value"] = completed["sd_metric_value"] / np.sqrt(completed["n_samples"].clip(lower=1))
    completed["se_metric_value"] = completed["se_metric_value"].fillna(0.0)
    completed["group_label"] = completed["parameter_group"].apply(lambda x: f"G{x:02d}")
    return completed


def build_x_group_labels() -> list[str]:
    return [f"G{i:02d}" for i in range(1, N_GROUPS + 1)]


def draw_metric_interface_intermix_bar_chart(
    ax,
    scope_summaries: dict[str, pd.DataFrame],
    dataset: dict[str, str | Path],
    parameter_config: dict[str, str],
) -> None:
    positions = np.arange(1, N_GROUPS + 1, dtype=float)
    x_labels = build_x_group_labels()
    bar_width = 0.32
    offsets = {
        "interface": -0.18,
        "intermix": 0.18,
    }

    for scope_config in WUI_SCOPE_CONFIGS:
        subset = scope_summaries[str(scope_config["slug"])].sort_values("parameter_group").reset_index(drop=True)
        x = positions + offsets[str(scope_config["slug"])]
        ax.bar(
            x,
            subset["mean_metric_value"],
            width=bar_width,
            color=str(scope_config["color"]),
            alpha=0.72,
            edgecolor=str(scope_config["color"]),
            linewidth=1.0,
            yerr=subset["se_metric_value"],
            error_kw={
                "ecolor": str(scope_config["color"]),
                "elinewidth": 1.15,
                "capsize": 3.0,
            },
            label=str(scope_config["label"]),
            zorder=4,
        )

    ax.axhline(0.0, color="#7f7f7f", linewidth=1.0, linestyle="--", zorder=1)
    for pos in positions:
        ax.axvline(pos, color="#eeeeee", linewidth=0.8, zorder=0)

    ax.set_title(str(dataset["metric_label"]), fontsize=SUBPLOT_TITLE_FONT_SIZE, pad=10)
    ax.set_xlim(0.45, N_GROUPS + 0.55)
    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels, fontsize=TICK_LABEL_FONT_SIZE, rotation=0, ha="center", va="center")
    ax.set_xlabel(str(parameter_config["label"]), fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(str(dataset["metric_label"]), fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis="x", pad=10)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.38)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    formatter = ScalarFormatter(useMathText=False)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=Y_MAJOR_TICK_BINS))
    ax.yaxis.get_offset_text().set_fontsize(AXIS_LABEL_FONT_SIZE)


def plot_parameter_metrics_1x2(
    metric_payloads: dict[str, dict[str, object]],
    parameter_config: dict[str, str],
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), facecolor="white")

    for ax, dataset in zip(axes, DATASETS):
        payload = metric_payloads.get(str(dataset["metric_slug"]))
        if payload is None:
            ax.axis("off")
            continue
        draw_metric_interface_intermix_bar_chart(
            ax,
            payload["scope_summaries"],
            payload["dataset"],
            parameter_config,
        )

    fig.subplots_adjust(bottom=0.34, top=0.88, left=0.075, right=0.99, wspace=0.28)
    output_path = OUTPUT_DIR / (
        f"Grouped_Fire_metrics_by_{parameter_config['slug']}_interface_intermix_{YEAR_TAG}_{GROUP_TAG}_1x2_barplot.png"
    )
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    matched_parts: list[pd.DataFrame] = []
    for dataset in DATASETS:
        matched_parts.append(load_basic_dataset(dataset))

    combined = pd.concat(matched_parts, ignore_index=True)
    metric_payloads_by_parameter: dict[str, dict[str, dict[str, object]]] = {
        str(parameter_config["slug"]): {} for parameter_config in PARAMETER_CONFIGS
    }

    for dataset in DATASETS:
        metric_df = combined.loc[combined["metric_slug"] == dataset["metric_slug"]].copy()
        if metric_df.empty:
            continue

        for parameter_config in PARAMETER_CONFIGS:
            parameter_col = str(parameter_config["column"])
            metric_param_df = metric_df.dropna(subset=[parameter_col, RESPONSE_COLUMN]).copy()
            if metric_param_df.empty:
                continue
            grouped_all = assign_equal_count_groups(metric_param_df, parameter_col, N_GROUPS)
            shared_group_ranges = extract_group_ranges(grouped_all)
            scope_summaries: dict[str, pd.DataFrame] = {}
            for scope_config in WUI_SCOPE_CONFIGS:
                scoped_grouped = grouped_all.loc[grouped_all["wui_mode"] == scope_config["mode"]].copy()
                if scoped_grouped.empty:
                    continue
                summary = summarize_groups(scoped_grouped, parameter_col, shared_group_ranges)
                scope_summaries[str(scope_config["slug"])] = summary

            if len(scope_summaries) != len(WUI_SCOPE_CONFIGS):
                continue

            metric_payloads_by_parameter[str(parameter_config["slug"])][str(dataset["metric_slug"])] = {
                "dataset": dataset,
                "scope_summaries": scope_summaries,
            }

    for parameter_config in PARAMETER_CONFIGS:
        metric_payloads = metric_payloads_by_parameter[str(parameter_config["slug"])]
        if len(metric_payloads) != len(DATASETS):
            continue
        plot_parameter_metrics_1x2(metric_payloads, parameter_config)


if __name__ == "__main__":
    main()
