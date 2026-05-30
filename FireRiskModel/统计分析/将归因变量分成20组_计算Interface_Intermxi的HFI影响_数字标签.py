#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称:
    将多种归因参数分成 20 组并同时绘制 Fire frequency 与 Normalized burned area 的 HFI 影响柱状图

功能:
    1. 读取 Fire frequency 和 Normalized burned area 的 HFI effect CSV，并保留 2003-2022 年全部匹配样本。
    2. 读取对应的 Basic CSV，通过 category_id + Year + GRID_ID 进行匹配。
    3. 保留 interface 与 intermix 两种 WUI 范围，不再输出 overall。
    4. 对每个指标和每个参数，先将 Interface + Intermix 合并样本按参数值从低到高划分为
       20 个等样本组，再分别映射回两种 WUI 范围，确保两类 WUI 的同编号组使用完全一致的参数范围。
    5. 计算每个分组的 HFI effect 均值和标准误。
    6. 对每个参数 × 指标输出一张柱状图，在同一张图中比较 Interface WUI 与 Intermix WUI。

输入:
    - B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num
    - B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area
    - I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic
    - I:\Processing data\渔网分类建模\Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic

输出:
    - B:\WUI\Picture\Grouped_Human Footprint_effect_Fire_frequency_by_GDP_interface_intermix_2003_2022_barplot.png
    - B:\WUI\Picture\Grouped_Human Footprint_effect_Fire_frequency_by_PopD_interface_intermix_2003_2022_barplot.png
    - B:\WUI\Picture\Grouped_Human Footprint_effect_Fire_frequency_by_Roadsd_interface_intermix_2003_2022_barplot.png
    - B:\WUI\Picture\Grouped_Human Footprint_effect_Fire_frequency_by_Built_environment_interface_intermix_2003_2022_barplot.png
    - B:\WUI\Picture\Grouped_Human Footprint_effect_Fire_frequency_by_Cropland_interface_intermix_2003_2022_barplot.png
    - B:\WUI\Picture\Grouped_Human Footprint_effect_Fire_frequency_by_Pasture_interface_intermix_2003_2022_barplot.png
    - B:\WUI\Picture\Grouped_Human Footprint_effect_Normalized_burn_area_by_GDP_interface_intermix_2003_2022_barplot.png
    - B:\WUI\Picture\Grouped_Human Footprint_effect_Normalized_burn_area_by_PopD_interface_intermix_2003_2022_barplot.png
    - B:\WUI\Picture\Grouped_Human Footprint_effect_Normalized_burn_area_by_Roadsd_interface_intermix_2003_2022_barplot.png
    - B:\WUI\Picture\Grouped_Human Footprint_effect_Normalized_burn_area_by_Built_environment_interface_intermix_2003_2022_barplot.png
    - B:\WUI\Picture\Grouped_Human Footprint_effect_Normalized_burn_area_by_Cropland_interface_intermix_2003_2022_barplot.png
    - B:\WUI\Picture\Grouped_Human Footprint_effect_Normalized_burn_area_by_Pasture_interface_intermix_2003_2022_barplot.png
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
N_GROUPS = 20
CHUNK_SIZE = 250_000
KEY_COLUMNS = ["Year", "GRID_ID"]
EFFECT_COLUMN = "diff_contribution"
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
        "effect_dir": Path(r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num"),
        "basic_dir": Path(r"I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic"),
        "effect_suffix": "fire_num_with_WUI_offset",
        "effect_label": "Fire frequency",
    },
    {
        "metric_slug": "Normalized_burn_area",
        "effect_dir": Path(r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area"),
        "basic_dir": Path(r"I:\Processing data\渔网分类建模\Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic"),
        "effect_suffix": "fire_area",
        "effect_label": "Normalized burned area",
    },
]


plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 12

AXIS_LABEL_FONT_SIZE = 31
TICK_LABEL_FONT_SIZE = 22
FIGURE_WIDTH = 19.8 * 1.3
FIGURE_HEIGHT = 7.6 * 0.8
Y_MAJOR_TICK_BINS = 4


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


def category_mode(category_id: str) -> str:
    return category_id.split("_", 1)[0].lower()


def normalize_grid_id(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.round().astype("Int64")


def load_basic_lookup(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=KEY_COLUMNS + PARAMETER_COLUMNS, low_memory=False)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["GRID_ID"] = normalize_grid_id(df["GRID_ID"])
    for parameter_col in PARAMETER_COLUMNS:
        df[parameter_col] = pd.to_numeric(df[parameter_col], errors="coerce")

    df = df[df["Year"].between(YEAR_START, YEAR_END, inclusive="both")]
    df = df.dropna(subset=KEY_COLUMNS)
    df = df.dropna(subset=PARAMETER_COLUMNS, how="all")
    if df.empty:
        return df

    deduped = (
        df.groupby(KEY_COLUMNS, as_index=False)[PARAMETER_COLUMNS]
        .mean()
        .reset_index(drop=True)
    )
    deduped["Year"] = deduped["Year"].astype(np.int64)
    deduped["GRID_ID"] = deduped["GRID_ID"].astype(np.int64)
    return deduped


def load_and_match_dataset(dataset: dict[str, object]) -> pd.DataFrame:
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
    for idx, category_id in enumerate(matched_categories, start=1):
        effect_csv = effect_map[category_id]
        basic_csv = basic_map[category_id]
        wui_mode = category_mode(category_id)
        basic_lookup = load_basic_lookup(basic_csv)
        if basic_lookup.empty:
            continue

        matched_count = 0
        for chunk in pd.read_csv(
            effect_csv,
            usecols=KEY_COLUMNS + [EFFECT_COLUMN],
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            chunk["Year"] = pd.to_numeric(chunk["Year"], errors="coerce").astype("Int64")
            chunk["GRID_ID"] = normalize_grid_id(chunk["GRID_ID"])
            chunk[EFFECT_COLUMN] = pd.to_numeric(chunk[EFFECT_COLUMN], errors="coerce")
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

            kept = merged[PARAMETER_COLUMNS + [EFFECT_COLUMN]].copy()
            for parameter_col in PARAMETER_COLUMNS:
                kept[parameter_col] = kept[parameter_col].astype(np.float32)
            kept[EFFECT_COLUMN] = kept[EFFECT_COLUMN].astype(np.float32)
            kept["metric_slug"] = str(dataset["metric_slug"])
            kept["effect_label"] = str(dataset["effect_label"])
            kept["wui_mode"] = wui_mode
            parts.append(kept)
            matched_count += len(kept)

    if not parts:
        raise ValueError(f"No matched sample rows found for Years={YEAR_DESC}.")

    matched = pd.concat(parts, ignore_index=True)
    matched = matched.dropna(subset=[EFFECT_COLUMN])
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
            n_samples=(EFFECT_COLUMN, "size"),
            mean_parameter=(parameter_col, "mean"),
            mean_hfi_effect=(EFFECT_COLUMN, "mean"),
            sd_hfi_effect=(EFFECT_COLUMN, "std"),
        )
        .sort_values("parameter_group")
        .reset_index(drop=True)
    )

    full_groups = pd.DataFrame({"parameter_group": np.arange(1, N_GROUPS + 1, dtype=int)})
    full_groups = full_groups.merge(group_ranges, how="left", on="parameter_group")
    completed = full_groups.merge(summary, how="left", on="parameter_group")
    completed["n_samples"] = completed["n_samples"].fillna(0).astype(int)
    completed["se_hfi_effect"] = completed["sd_hfi_effect"] / np.sqrt(completed["n_samples"].clip(lower=1))
    completed["se_hfi_effect"] = completed["se_hfi_effect"].fillna(0.0)
    completed["group_label"] = completed["parameter_group"].apply(lambda x: f"G{x:02d}")
    return completed


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


def build_x_range_labels(scope_summaries: dict[str, pd.DataFrame]) -> list[str]:
    if not scope_summaries:
        return []
    combined_ranges = next(iter(scope_summaries.values())).sort_values("parameter_group")
    return [
        _format_range_label(row["param_bin_lower"], row["param_bin_upper"])
        for _, row in combined_ranges.iterrows()
    ]


def plot_metric_interface_intermix_bar_chart(
    scope_summaries: dict[str, pd.DataFrame],
    dataset: dict[str, str | Path],
    parameter_config: dict[str, str],
) -> Path:
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), facecolor="white")

    positions = np.arange(1, N_GROUPS + 1, dtype=float)
    x_labels = build_x_range_labels(scope_summaries)
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
            subset["mean_hfi_effect"],
            width=bar_width,
            color=str(scope_config["color"]),
            alpha=0.72,
            edgecolor=str(scope_config["color"]),
            linewidth=1.0,
            yerr=subset["se_hfi_effect"],
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

    ax.set_xlim(0.45, N_GROUPS + 0.55)
    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels, fontsize=TICK_LABEL_FONT_SIZE, rotation=90, ha="center", va="top")
    ax.set_xlabel(str(parameter_config["label"]), fontsize=AXIS_LABEL_FONT_SIZE)
    if str(dataset["metric_slug"]) == "Fire_frequency":
        y_label = "Human Footprint effect\non fire frequency"
    else:
        y_label = "Human Footprint effect\non normalized burned area"
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONT_SIZE)
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

    fig.subplots_adjust(bottom=0.36, top=0.94, left=0.12, right=0.985)
    output_path = OUTPUT_DIR / (
        f"Grouped_Human Footprint_effect_{dataset['metric_slug']}_by_{parameter_config['slug']}_interface_intermix_{YEAR_TAG}_barplot.png"
    )
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    matched_parts: list[pd.DataFrame] = []
    for dataset in DATASETS:
        matched_parts.append(load_and_match_dataset(dataset))

    combined = pd.concat(matched_parts, ignore_index=True)
    for dataset in DATASETS:
        metric_df = combined.loc[combined["metric_slug"] == dataset["metric_slug"]].copy()
        if metric_df.empty:
            continue
        for parameter_config in PARAMETER_CONFIGS:
            parameter_col = str(parameter_config["column"])
            metric_param_df = metric_df.dropna(subset=[parameter_col, EFFECT_COLUMN]).copy()
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

            plot_metric_interface_intermix_bar_chart(scope_summaries, dataset, parameter_config)


if __name__ == "__main__":
    main()
