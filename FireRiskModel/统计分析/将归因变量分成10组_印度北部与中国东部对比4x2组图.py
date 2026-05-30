#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称:
    将多种归因参数分成 10 组，并对比印度北部与中国东部的 HFI 影响 4x2 组合图

功能:
    1. 使用空间分布图脚本中的两个局部放大范围：
       - Northern India: lon 74-91, lat 19-31
       - East-Central China: lon 105-122, lat 24-36
    2. 从 WUI.gdb 的 WUI 渔网图层中提取两个区域内的 GRID_ID，并按 WUI 年份映射到 2003-2022 年。
    3. 读取 Fire frequency 和 Normalized burned area 的 HFI effect CSV。
    4. 读取对应 Basic CSV，并按 Year + GRID_ID 匹配 Cropland、Pasture、GDP、Population density。
    5. 对每个机制变量在印度+中国两个区域的匹配样本中共同分成 10 组。
    6. 输出一张 4x2 组图：
       - 4 行: Cropland、Pasture、GDP、Population density
       - 2 列: Fire frequency、Normalized burned area
       - 每个子图内用两种颜色柱子对比 Northern India 与 East-Central China 的均值 + 标准误。

输出:
    - B:\WUI\Picture\Grouped_Human Footprint_effect_by_attribution_Northern_India_vs_East_Central_China_2003_2022_10groups_4x2.png
    - B:\WUI\Picture\Grouped_Human activity_effect_group_ranges_Northern_India_vs_East_Central_China_2003_2022_10groups.xlsx
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import geopandas as gpd
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

WUI_GDB_PATH = Path(r"I:\Processing data\gdb集合\WUI.gdb")
WUI_YEARS = [2005, 2010, 2015, 2020]
WUI_LAYER_TEMPLATE = "Grid_Clip_{wui_year}_WUI"
WUI_YEAR_MAPPING = {
    2005: list(range(2003, 2008)),
    2010: list(range(2008, 2013)),
    2015: list(range(2013, 2018)),
    2020: list(range(2018, 2023)),
}

OUTPUT_FIGURE_PATH = OUTPUT_DIR / (
    f"Grouped_Human Footprint_effect_by_attribution_Northern_India_vs_East_Central_China_"
    f"{YEAR_TAG}_{GROUP_TAG}_4x2.png"
)
OUTPUT_TABLE_PATH = OUTPUT_DIR / (
    f"Grouped_Human activity_effect_group_ranges_Northern_India_vs_East_Central_China_"
    f"{YEAR_TAG}_{GROUP_TAG}.xlsx"
)

CHUNK_SIZE = 250_000
KEY_COLUMNS = ["Year", "GRID_ID"]
EFFECT_COLUMN = "diff_contribution"
PARAMETER_COLUMNS = ["Cropland_Area_km2", "Pasture_Band2_Mean", "GDP", "PopD"]

PARAMETER_CONFIGS = [
    {"column": "Cropland_Area_km2", "label": "Cropland", "slug": "Cropland"},
    {"column": "Pasture_Band2_Mean", "label": "Pasture", "slug": "Pasture"},
    {"column": "GDP", "label": "GDP", "slug": "GDP"},
    {"column": "PopD", "label": "Population density", "slug": "PopD"},
]

REGIONS = [
    {
        "slug": "northern_india",
        "label": "Northern India",
        "lon_min": 74.0,
        "lon_max": 91.0,
        "lat_min": 19.0,
        "lat_max": 31.0,
        "color": "#d73027",
    },
    {
        "slug": "east_central_china",
        "label": "East-Central China",
        "lon_min": 105.0,
        "lon_max": 122.0,
        "lat_min": 24.0,
        "lat_max": 36.0,
        "color": "#1e90ff",
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
        "metric_slug": "Normalized_burn_area",
        "effect_dir": Path(r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area"),
        "basic_dir": Path(r"I:\Processing data\渔网分类建模\Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic"),
        "effect_suffix": "fire_area",
        "effect_label": "Normalized burned area",
    },
]


plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 16

AXIS_LABEL_FONT_SIZE = 28
TICK_LABEL_FONT_SIZE = 25
LEGEND_FONT_SIZE = 24
FIGURE_TITLE_FONT_SIZE = 32
FIGURE_WIDTH = 22.0
FIGURE_HEIGHT = 24.0
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


def normalize_grid_id(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.round().astype("Int64")


def normalize_year(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.round().astype("Int64")


def read_region_grid_ids(region: dict[str, object], wui_year: int) -> pd.DataFrame:
    layer = WUI_LAYER_TEMPLATE.format(wui_year=wui_year)
    bbox = (
        float(region["lon_min"]),
        float(region["lat_min"]),
        float(region["lon_max"]),
        float(region["lat_max"]),
    )

    def crop_to_region(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if gdf.empty:
            return gdf
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        return gdf.cx[
            float(region["lon_min"]):float(region["lon_max"]),
            float(region["lat_min"]):float(region["lat_max"]),
        ].copy()

    def read_full_layer() -> gpd.GeoDataFrame:
        return gpd.read_file(
            WUI_GDB_PATH,
            layer=layer,
            columns=["GRID_ID"],
            engine="pyogrio",
        )

    try:
        fishnet = gpd.read_file(
            WUI_GDB_PATH,
            layer=layer,
            bbox=bbox,
            columns=["GRID_ID"],
            engine="pyogrio",
        )
    except Exception:
        fishnet = read_full_layer()

    if fishnet.empty:
        print(f"bbox read returned empty for {region['label']} / {layer}; reading full layer as fallback...")
        fishnet = read_full_layer()

    if fishnet.empty:
        return pd.DataFrame(columns=["GRID_ID", "wui_year", "region_slug", "region_label"])

    fishnet = crop_to_region(fishnet)
    if fishnet.empty:
        print(f"bbox read produced no cells after lon/lat crop for {region['label']} / {layer}; reading full layer as fallback...")
        fishnet = crop_to_region(read_full_layer())
    if fishnet.empty:
        return pd.DataFrame(columns=["GRID_ID", "wui_year", "region_slug", "region_label"])

    grid_ids = normalize_grid_id(fishnet["GRID_ID"]).dropna().astype(np.int64).drop_duplicates()
    return pd.DataFrame(
        {
            "GRID_ID": grid_ids.to_numpy(dtype=np.int64),
            "wui_year": int(wui_year),
            "region_slug": str(region["slug"]),
            "region_label": str(region["label"]),
        }
    )


def build_region_year_lookup() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for region in REGIONS:
        for wui_year in WUI_YEARS:
            region_grid = read_region_grid_ids(region, wui_year)
            if region_grid.empty:
                print(f"Warning: no WUI grid found for {region['label']} in WUI year {wui_year}.")
                continue
            expanded_parts: list[pd.DataFrame] = []
            for year in WUI_YEAR_MAPPING[wui_year]:
                year_df = region_grid[["GRID_ID", "region_slug", "region_label"]].copy()
                year_df["Year"] = int(year)
                year_df["wui_year"] = int(wui_year)
                expanded_parts.append(year_df)
            parts.append(pd.concat(expanded_parts, ignore_index=True))

    if not parts:
        raise ValueError("No region GRID_ID-Year lookup rows were created.")

    lookup = pd.concat(parts, ignore_index=True)
    lookup["GRID_ID"] = normalize_grid_id(lookup["GRID_ID"]).astype(np.int64)
    lookup["Year"] = normalize_year(lookup["Year"]).astype(np.int64)
    lookup = (
        lookup.sort_values(["Year", "GRID_ID", "region_slug"])
        .drop_duplicates(subset=["Year", "GRID_ID"], keep="first")
        .reset_index(drop=True)
    )
    print(
        "Region lookup loaded: "
        f"rows={len(lookup):,}, "
        f"unique GRID_ID={lookup['GRID_ID'].nunique():,}, "
        f"years={lookup['Year'].nunique()}"
    )
    return lookup[["Year", "GRID_ID", "wui_year", "region_slug", "region_label"]]


def load_basic_lookup(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=KEY_COLUMNS + PARAMETER_COLUMNS, low_memory=False)
    df["Year"] = normalize_year(df["Year"])
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


def load_and_match_dataset(dataset: dict[str, object], region_lookup: pd.DataFrame) -> pd.DataFrame:
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
    region_lookup = region_lookup[["Year", "GRID_ID", "region_slug", "region_label"]].copy()
    for category_id in matched_categories:
        effect_csv = effect_map[category_id]
        basic_csv = basic_map[category_id]
        basic_lookup = load_basic_lookup(basic_csv)
        if basic_lookup.empty:
            continue

        for chunk in pd.read_csv(
            effect_csv,
            usecols=KEY_COLUMNS + [EFFECT_COLUMN],
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            chunk["Year"] = normalize_year(chunk["Year"])
            chunk["GRID_ID"] = normalize_grid_id(chunk["GRID_ID"])
            chunk[EFFECT_COLUMN] = pd.to_numeric(chunk[EFFECT_COLUMN], errors="coerce")
            chunk = chunk[
                chunk["Year"].between(YEAR_START, YEAR_END, inclusive="both")
            ].dropna(subset=KEY_COLUMNS + [EFFECT_COLUMN])
            if chunk.empty:
                continue

            chunk["Year"] = chunk["Year"].astype(np.int64)
            chunk["GRID_ID"] = chunk["GRID_ID"].astype(np.int64)
            chunk = chunk.merge(region_lookup, how="inner", on=KEY_COLUMNS, validate="many_to_one")
            if chunk.empty:
                continue

            merged = chunk.merge(basic_lookup, how="inner", on=KEY_COLUMNS, validate="many_to_one")
            if merged.empty:
                continue

            kept = merged[PARAMETER_COLUMNS + [EFFECT_COLUMN, "region_slug", "region_label"]].copy()
            for parameter_col in PARAMETER_COLUMNS:
                kept[parameter_col] = kept[parameter_col].astype(np.float32)
            kept[EFFECT_COLUMN] = kept[EFFECT_COLUMN].astype(np.float32)
            kept["metric_slug"] = str(dataset["metric_slug"])
            kept["effect_label"] = str(dataset["effect_label"])
            kept["category_id"] = category_id
            parts.append(kept)

    if not parts:
        raise ValueError(f"No matched regional sample rows found for {dataset['effect_label']}, Years={YEAR_DESC}.")

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
    group_ranges: pd.DataFrame,
) -> pd.DataFrame:
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


def build_x_range_labels() -> list[str]:
    return [f"G{i:02d}" for i in range(1, N_GROUPS + 1)]


def build_group_range_table(
    *,
    parameter_config: dict[str, str],
    dataset: dict[str, str | Path],
    group_ranges: pd.DataFrame,
) -> pd.DataFrame:
    table = group_ranges.copy().sort_values("parameter_group").reset_index(drop=True)
    table["group_label"] = table["parameter_group"].apply(lambda x: f"G{x:02d}")
    table["x_axis_range"] = table.apply(
        lambda row: _format_range_label(row["param_bin_lower"], row["param_bin_upper"]),
        axis=1,
    )
    table.insert(0, "metric_slug", str(dataset["metric_slug"]))
    table.insert(0, "metric_label", str(dataset["effect_label"]))
    table.insert(0, "parameter_slug", str(parameter_config["slug"]))
    table.insert(0, "parameter_label", str(parameter_config["label"]))
    return table


def build_summary_table(
    region_summaries: dict[str, pd.DataFrame],
    *,
    parameter_config: dict[str, str],
    dataset: dict[str, str | Path],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for region_config in REGIONS:
        region_slug = str(region_config["slug"])
        if region_slug not in region_summaries:
            continue
        table = region_summaries[region_slug].copy()
        table.insert(0, "region_label", str(region_config["label"]))
        table.insert(0, "region_slug", region_slug)
        table.insert(0, "metric_slug", str(dataset["metric_slug"]))
        table.insert(0, "metric_label", str(dataset["effect_label"]))
        table.insert(0, "parameter_slug", str(parameter_config["slug"]))
        table.insert(0, "parameter_label", str(parameter_config["label"]))
        parts.append(table)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def build_metric_payloads(
    combined: pd.DataFrame,
) -> tuple[dict[str, dict[str, dict[str, object]]], list[pd.DataFrame], list[pd.DataFrame]]:
    metric_payloads_by_parameter: dict[str, dict[str, dict[str, object]]] = {
        str(parameter_config["slug"]): {} for parameter_config in PARAMETER_CONFIGS
    }
    group_range_tables: list[pd.DataFrame] = []
    summary_tables: list[pd.DataFrame] = []

    for dataset in DATASETS:
        metric_df = combined.loc[combined["metric_slug"] == dataset["metric_slug"]].copy()
        if metric_df.empty:
            continue

        for parameter_config in PARAMETER_CONFIGS:
            parameter_col = str(parameter_config["column"])
            metric_param_df = metric_df.dropna(subset=[parameter_col, EFFECT_COLUMN]).copy()
            if metric_param_df.empty:
                continue
            try:
                grouped_all = assign_equal_count_groups(metric_param_df, parameter_col, N_GROUPS)
            except ValueError as exc:
                print(f"Skip {dataset['effect_label']}, {parameter_config['label']}: {exc}")
                continue

            shared_group_ranges = extract_group_ranges(grouped_all)
            region_summaries: dict[str, pd.DataFrame] = {}
            for region_config in REGIONS:
                region_slug = str(region_config["slug"])
                region_grouped = grouped_all.loc[grouped_all["region_slug"] == region_slug].copy()
                if region_grouped.empty:
                    continue
                region_summaries[region_slug] = summarize_groups(
                    region_grouped,
                    parameter_col,
                    shared_group_ranges,
                )

            if len(region_summaries) != len(REGIONS):
                continue

            metric_payloads_by_parameter[str(parameter_config["slug"])][str(dataset["metric_slug"])] = {
                "dataset": dataset,
                "region_summaries": region_summaries,
                "group_ranges": shared_group_ranges.copy(),
            }
            group_range_tables.append(
                build_group_range_table(
                    parameter_config=parameter_config,
                    dataset=dataset,
                    group_ranges=shared_group_ranges,
                )
            )
            summary_tables.append(
                build_summary_table(
                    region_summaries,
                    parameter_config=parameter_config,
                    dataset=dataset,
                )
            )

    return metric_payloads_by_parameter, group_range_tables, summary_tables


def draw_metric_region_bar_chart(
    ax,
    region_summaries: dict[str, pd.DataFrame],
    dataset: dict[str, str | Path],
    parameter_config: dict[str, str],
    *,
    show_legend: bool,
) -> None:
    positions = np.arange(1, N_GROUPS + 1, dtype=float)
    x_labels = build_x_range_labels()
    bar_width = 0.32
    offsets = {
        str(REGIONS[0]["slug"]): -0.18,
        str(REGIONS[1]["slug"]): 0.18,
    }

    for region_config in REGIONS:
        region_slug = str(region_config["slug"])
        subset = region_summaries[region_slug].sort_values("parameter_group").reset_index(drop=True)
        x = positions + offsets[region_slug]
        ax.bar(
            x,
            subset["mean_hfi_effect"],
            width=bar_width,
            color=str(region_config["color"]),
            alpha=0.72,
            edgecolor=str(region_config["color"]),
            linewidth=0.9,
            yerr=subset["se_hfi_effect"],
            error_kw={
                "ecolor": str(region_config["color"]),
                "elinewidth": 0.9,
                "capsize": 2.5,
            },
            label=str(region_config["label"]),
            zorder=4,
        )

    ax.axhline(0.0, color="#7f7f7f", linewidth=0.9, linestyle="--", zorder=1)
    for pos in positions:
        ax.axvline(pos, color="#eeeeee", linewidth=0.65, zorder=0)

    ax.set_xlim(0.45, N_GROUPS + 0.55)
    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels, fontsize=TICK_LABEL_FONT_SIZE, rotation=0, ha="center", va="center")
    ax.set_xlabel(str(parameter_config["label"]), fontsize=AXIS_LABEL_FONT_SIZE, labelpad=10)
    ax.set_ylabel("")
    ax.set_title("")
    ax.tick_params(axis="x", pad=13)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.38)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    formatter = ScalarFormatter(useMathText=False)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=Y_MAJOR_TICK_BINS))
    ax.yaxis.get_offset_text().set_fontsize(TICK_LABEL_FONT_SIZE)

    if show_legend:
        ax.legend(
            loc="lower left",
            ncol=1,
            fontsize=LEGEND_FONT_SIZE,
            frameon=False,
            handlelength=1.4,
            labelspacing=0.25,
            borderaxespad=0.35,
        )


def metric_y_axis_label(dataset: dict[str, str | Path]) -> str:
    if str(dataset["metric_slug"]) == "Fire_frequency":
        return "Human activity effects on fire frequency"
    return "Human activity effects on normalized burned area"


def add_shared_y_axis_labels(fig, axes: np.ndarray) -> None:
    for col_idx, dataset in enumerate(DATASETS):
        top_box = axes[0, col_idx].get_position()
        bottom_box = axes[-1, col_idx].get_position()
        x = top_box.x0 - 0.040
        y = (top_box.y1 + bottom_box.y0) / 2.0
        fig.text(
            x,
            y,
            metric_y_axis_label(dataset),
            ha="center",
            va="center",
            rotation=90,
            fontsize=AXIS_LABEL_FONT_SIZE,
            color="#222222",
        )


def plot_group_4x2(metric_payloads_by_parameter: dict[str, dict[str, dict[str, object]]]) -> Path | None:
    valid_parameter_configs = [
        parameter_config
        for parameter_config in PARAMETER_CONFIGS
        if len(metric_payloads_by_parameter[str(parameter_config["slug"])]) == len(DATASETS)
    ]
    if not valid_parameter_configs:
        print("Skip regional comparison plot: no complete 4x2 payload.")
        return None

    fig, axes = plt.subplots(
        len(valid_parameter_configs),
        2,
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
        facecolor="white",
        squeeze=False,
    )

    for row_idx, parameter_config in enumerate(valid_parameter_configs):
        metric_payloads = metric_payloads_by_parameter[str(parameter_config["slug"])]
        for col_idx, dataset in enumerate(DATASETS):
            ax = axes[row_idx, col_idx]
            payload = metric_payloads.get(str(dataset["metric_slug"]))
            if payload is None:
                ax.axis("off")
                continue
            draw_metric_region_bar_chart(
                ax,
                payload["region_summaries"],
                payload["dataset"],
                parameter_config,
                show_legend=(row_idx == 0 and col_idx == 0),
            )

    fig.subplots_adjust(bottom=0.075, top=0.985, left=0.100, right=0.99, wspace=0.16, hspace=0.34)
    fig.text(
        0.545,
        1.005,
        "Northern India vs East-Central China",
        ha="center",
        va="bottom",
        fontsize=FIGURE_TITLE_FONT_SIZE,
        color="#222222",
    )
    add_shared_y_axis_labels(fig, axes)
    fig.savefig(OUTPUT_FIGURE_PATH, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved figure to: {OUTPUT_FIGURE_PATH}")
    return OUTPUT_FIGURE_PATH


def save_output_tables(group_range_tables: list[pd.DataFrame], summary_tables: list[pd.DataFrame]) -> None:
    with pd.ExcelWriter(OUTPUT_TABLE_PATH) as writer:
        if group_range_tables:
            pd.concat(group_range_tables, ignore_index=True).to_excel(
                writer,
                sheet_name="group_ranges",
                index=False,
            )
        if summary_tables:
            pd.concat(summary_tables, ignore_index=True).to_excel(
                writer,
                sheet_name="regional_summary",
                index=False,
            )
    print(f"Saved table to: {OUTPUT_TABLE_PATH}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    region_lookup = build_region_year_lookup()
    matched_parts: list[pd.DataFrame] = []
    for dataset in DATASETS:
        print(f"\nLoading and matching dataset: {dataset['effect_label']}")
        matched_parts.append(load_and_match_dataset(dataset, region_lookup))

    combined = pd.concat(matched_parts, ignore_index=True)
    print(
        "Matched regional samples: "
        f"rows={len(combined):,}, "
        f"metrics={combined['metric_slug'].nunique()}, "
        f"regions={combined['region_slug'].nunique()}"
    )

    payloads, group_range_tables, summary_tables = build_metric_payloads(combined)
    plot_group_4x2(payloads)
    save_output_tables(group_range_tables, summary_tables)


if __name__ == "__main__":
    main()
