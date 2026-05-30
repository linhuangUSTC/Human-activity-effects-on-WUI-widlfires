#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称: 大洲火灾统计综合绘图程序

功能简介:
    1. 在同一程序中完成 Fire number、Burned area、Fire frequency per WUI
       和 Fire area density 的大洲统计绘图，不再调用其他脚本。
    2. 从 Grid_Clip 年度 WUI 数据中提取 Fire number 和 Burned area,
       并生成大洲柱状图、年际趋势图和 Total 合并趋势图。
    3. 从分类建模 Basic 数据中提取 Fire_frequency 和 Normalized_fire_area,
       通过 GRID_ID 与年度 WUI 数据匹配大洲信息，生成大洲柱状图。
    4. 当各大洲数值差异过大时，自动在 y 轴上设置断点，使图像更易读。

输出图片:
    - B:\WUI\Picture\continent_fire_number_statistics.jpg
    - B:\WUI\Picture\continent_burned_area_statistics.jpg
    - B:\WUI\Picture\Interface_fire_number_annual_trend.jpg
    - B:\WUI\Picture\Intermix_fire_number_annual_trend.jpg
    - B:\WUI\Picture\All_fire_number_annual_trend.jpg
    - B:\WUI\Picture\Merged_Total_fire_number_annual_trend.jpg
    - B:\WUI\Picture\Interface_burned_area_annual_trend.jpg
    - B:\WUI\Picture\Intermix_burned_area_annual_trend.jpg
    - B:\WUI\Picture\All_burned_area_annual_trend.jpg
    - B:\WUI\Picture\Merged_Total_burned_area_annual_trend.jpg
    - B:\WUI\Picture\continent_fire_frequency_per_wui_statistics.jpg
    - B:\WUI\Picture\continent_fire_area_density_statistics.jpg
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 12


OUTPUT_DIR = Path(r"B:\WUI\Picture")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRIDCLIP_YEARS = [2005, 2010, 2015, 2020]
GRIDCLIP_TEMPLATE = r"I:\Processing data\Grid_Clip_{year}_WUI.csv"

FIRE_NUM_BASIC_DIR = Path(
    r"I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic"
)
FIRE_AREA_BASIC_DIR = Path(
    r"I:\Processing data\渔网分类建模\Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic"
)

CONTINENT_MAPPING = {
    1: "Africa",
    2: "Asia",
    3: "Oceania",
    4: "North America",
    5: "Oceania",
    6: "South America",
    8: "Europe",
}

BAR_COLORS = {
    "All": "#3498db",
    "Intermix": "#e74c3c",
    "Interface": "#ff9800",
}

FIRE_YEAR_TO_WUI_YEAR: dict[int, int] = {}
for wui_year in GRIDCLIP_YEARS:
    for offset in range(-2, 3):
        FIRE_YEAR_TO_WUI_YEAR[wui_year + offset] = wui_year

GRIDCLIP_STATS_CONFIG = [
    {
        "name": "Fire Number",
        "interface_field": lambda year: f"Interface_Fire_num_{year}",
        "intermix_field": lambda year: f"Intermix_Fire_num_{year}",
        "all_field": lambda year: f"All_Fire_num_{year}",
        "ylabel": "Number of Fires",
        "filename_suffix": "fire_number",
    },
    {
        "name": "Burned Area",
        "interface_field": lambda year: f"Interface_Fire_area_{year}",
        "intermix_field": lambda year: f"Intermix_Fire_area_{year}",
        "all_field": lambda year: f"All_Fire_area_{year}",
        "ylabel": "Burned Area (km²)",
        "filename_suffix": "burned_area",
    },
]

BASIC_STATS_CONFIG = [
    {
        "name": "Fire Frequency per WUI",
        "metric_column": "Fire_frequency",
        "ylabel": "Fire Frequency per WUI (count/km²)",
        "filename_suffix": "fire_frequency_per_wui",
    },
    {
        "name": "Fire Area Density",
        "metric_column": "Normalized_fire_area",
        "ylabel": "Fire Area Density (km²/km²)",
        "filename_suffix": "fire_area_density",
    },
]


def format_continent_labels(continents: list[int]) -> list[str]:
    labels: list[str] = []
    for code in continents:
        name = CONTINENT_MAPPING.get(code, f"Unknown({code})")
        if name == "South America":
            labels.append("South\nAmerica")
        elif name == "North America":
            labels.append("North\nAmerica")
        else:
            labels.append(name)
    return labels


def normalize_grid_id(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.round().astype("Int64").astype("string")


def detect_y_break(values: np.ndarray) -> tuple[float, float, float] | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr >= 0]
    if arr.size < 6:
        return None

    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return None

    threshold = q3 + 1.5 * iqr
    non_outliers = arr[arr <= threshold]
    outliers = arr[arr > threshold]
    if non_outliers.size == 0 or outliers.size == 0:
        return None

    low_max = float(non_outliers.max()) * 1.12
    high_min = float(outliers.min()) * 0.92
    ymax = float(arr.max()) * 1.06

    if low_max <= 0 or high_min <= low_max * 1.15 or ymax <= high_min:
        return None
    return low_max, high_min, ymax


def create_plot_axes(
    figsize: tuple[float, float], y_values: np.ndarray
) -> tuple[plt.Figure, plt.Axes, plt.Axes | None, tuple[float, float, float] | None]:
    break_info = detect_y_break(y_values)
    if break_info is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax, None, None

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 3.2], "hspace": 0.05},
    )
    return fig, ax_bottom, ax_top, break_info


def finalize_layout(fig: plt.Figure, has_break: bool) -> None:
    if has_break:
        fig.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.1, hspace=0.06)
    else:
        fig.tight_layout()


def apply_break_format(ax_bottom: plt.Axes, ax_top: plt.Axes, break_info: tuple[float, float, float]) -> None:
    low_max, high_min, ymax = break_info
    ax_bottom.set_ylim(0, low_max)
    ax_top.set_ylim(high_min, ymax)

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_bottom.xaxis.tick_bottom()

    d = 0.012
    kwargs = dict(transform=ax_top.transAxes, color="black", clip_on=False, linewidth=1.2)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)


def load_gridclip_data(year: int) -> pd.DataFrame | None:
    print(f"Loading Grid_Clip data for year {year}...")
    csv_path = GRIDCLIP_TEMPLATE.format(year=year)
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        df = df[df["Continent"] != 0].copy()
        df.loc[df["Continent"] == 5, "Continent"] = 3
        print(f"Loaded {len(df):,} records for year {year}.")
        return df
    except Exception as exc:
        print(f"Error loading {csv_path}: {exc}")
        return None


def extract_gridclip_annual_data(
    df: pd.DataFrame, base_year: int, stat_config: dict[str, object]
) -> pd.DataFrame | None:
    data_years = list(range(base_year - 2, base_year + 3))
    annual_parts: list[pd.DataFrame] = []

    for data_year in data_years:
        interface_field = stat_config["interface_field"](data_year)
        intermix_field = stat_config["intermix_field"](data_year)
        all_field = stat_config["all_field"](data_year)

        if (
            interface_field not in df.columns
            or intermix_field not in df.columns
            or all_field not in df.columns
        ):
            print(f"Fields not found for year {data_year} in base year {base_year}")
            continue

        continent_stats = (
            df.groupby("Continent")
            .agg(
                Interface=(interface_field, "sum"),
                Intermix=(intermix_field, "sum"),
                All=(all_field, "sum"),
            )
            .reset_index()
        )
        continent_stats["Year"] = data_year
        annual_parts.append(continent_stats)

    if not annual_parts:
        return None
    return pd.concat(annual_parts, ignore_index=True)


def plot_gridclip_continent_bars(all_annual_data: pd.DataFrame, stat_config: dict[str, object]) -> None:
    avg_stats = (
        all_annual_data.groupby("Continent")[["All", "Intermix", "Interface"]]
        .mean()
        .reset_index()
        .sort_values("Continent")
        .reset_index(drop=True)
    )

    continents = avg_stats["Continent"].tolist()
    index = np.arange(len(continents))
    bar_width = 0.25

    all_values = avg_stats["All"].to_numpy(dtype=float)
    intermix_values = avg_stats["Intermix"].to_numpy(dtype=float)
    interface_values = avg_stats["Interface"].to_numpy(dtype=float)
    y_values = np.concatenate([all_values, intermix_values, interface_values])

    fig, ax_bottom, ax_top, break_info = create_plot_axes((12, 8), y_values)
    axes = [ax_bottom] if ax_top is None else [ax_top, ax_bottom]

    for ax in axes:
        ax.bar(index, all_values, bar_width, color=BAR_COLORS["All"], alpha=0.85, label="Total")
        ax.bar(
            index + bar_width,
            intermix_values,
            bar_width,
            color=BAR_COLORS["Intermix"],
            alpha=0.85,
            label="Intermix",
        )
        ax.bar(
            index + 2 * bar_width,
            interface_values,
            bar_width,
            color=BAR_COLORS["Interface"],
            alpha=0.85,
            label="Interface",
        )
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    if ax_top is not None and break_info is not None:
        apply_break_format(ax_bottom, ax_top, break_info)
        ax_top.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=14)
    else:
        ax_bottom.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=14)

    ax_bottom.set_xlabel("Continent")
    ax_bottom.set_ylabel(str(stat_config["ylabel"]))
    ax_bottom.set_xticks(index + bar_width)
    ax_bottom.set_xticklabels(format_continent_labels(continents), rotation=0, ha="center", fontsize=14)
    ax_bottom.set_xlim(-0.3, len(continents) - 0.3)

    finalize_layout(fig, ax_top is not None)
    output_filename = f"continent_{stat_config['filename_suffix']}_statistics.jpg"
    output_path = OUTPUT_DIR / output_filename
    fig.savefig(output_path, dpi=1200, bbox_inches="tight", format="jpg")
    plt.close(fig)
    print(f"Chart saved to: {output_path}")


def plot_gridclip_annual_trend(
    all_annual_data: pd.DataFrame, stat_config: dict[str, object], wui_type: str
) -> None:
    data_type = str(stat_config["name"])
    all_years = sorted(all_annual_data["Year"].unique())
    continents = sorted(all_annual_data["Continent"].unique())
    continent_colors = plt.cm.tab10(np.linspace(0, 1, len(continents)))

    y_values: list[float] = []
    for continent in continents:
        continent_data = all_annual_data[all_annual_data["Continent"] == continent]
        y_values.extend(continent_data[wui_type].dropna().tolist())

    fig, ax_bottom, ax_top, break_info = create_plot_axes((12, 8), np.asarray(y_values))
    axes = [ax_bottom] if ax_top is None else [ax_top, ax_bottom]

    lines = []
    labels = []

    for continent, color in zip(continents, continent_colors):
        continent_data = all_annual_data[
            (all_annual_data["Continent"] == continent) & (all_annual_data["Year"].isin(all_years))
        ].sort_values("Year")

        for ax in axes:
            line, = ax.plot(
                continent_data["Year"],
                continent_data[wui_type],
                marker="o",
                markersize=6,
                linewidth=2,
                color=color,
                label=CONTINENT_MAPPING.get(continent, f"Unknown({continent})"),
            )
        lines.append(line)
        labels.append(CONTINENT_MAPPING.get(continent, f"Unknown({continent})"))

    for ax in axes:
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.9)

    if ax_top is not None and break_info is not None:
        apply_break_format(ax_bottom, ax_top, break_info)
        fig.legend(lines, labels, loc="upper right", bbox_to_anchor=(0.94, 0.97), ncol=3, fontsize=14)
    else:
        ax_bottom.legend(loc="upper right", fontsize=14, ncol=3)

    ax_bottom.set_xlabel("Year")
    ax_bottom.set_ylabel(f"{data_type} - {wui_type}")
    ax_bottom.set_xticks(all_years)
    ax_bottom.set_xticklabels(all_years)

    finalize_layout(fig, ax_top is not None)
    output_filename = f"{wui_type}_{stat_config['filename_suffix']}_annual_trend.jpg"
    output_path = OUTPUT_DIR / output_filename
    fig.savefig(output_path, dpi=1200, bbox_inches="tight", format="jpg")
    plt.close(fig)
    print(f"Annual trend chart saved to: {output_path}")


def plot_gridclip_merged_total_trends(all_annual_data: pd.DataFrame, stat_config: dict[str, object]) -> None:
    data_type = str(stat_config["name"])
    all_years = sorted(all_annual_data["Year"].unique())

    wui_types = ["Interface", "Intermix", "All"]
    colors = {"Interface": "#ff9800", "Intermix": "#e74c3c", "All": "#3498db"}
    markers = {"Interface": "o", "Intermix": "s", "All": "^"}
    labels = {"Interface": "Interface", "Intermix": "Intermix", "All": "All"}

    series: dict[str, pd.DataFrame] = {}
    all_values: list[float] = []

    for wui_type in wui_types:
        total_data = (
            all_annual_data[all_annual_data["Year"].isin(all_years)]
            .groupby("Year")
            .agg(**{wui_type: (wui_type, "sum")})
            .reset_index()
            .sort_values("Year")
        )
        series[wui_type] = total_data
        all_values.extend(total_data[wui_type].tolist())

    fig, ax_bottom, ax_top, break_info = create_plot_axes((12, 8), np.asarray(all_values))
    axes = [ax_bottom] if ax_top is None else [ax_top, ax_bottom]

    lines = []
    line_labels = []
    for wui_type in wui_types:
        total_data = series[wui_type]
        for ax in axes:
            line, = ax.plot(
                total_data["Year"],
                total_data[wui_type],
                marker=markers[wui_type],
                markersize=6,
                linewidth=2,
                color=colors[wui_type],
                label=labels[wui_type],
            )
        lines.append(line)
        line_labels.append(labels[wui_type])

    for ax in axes:
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.9)

    if ax_top is not None and break_info is not None:
        apply_break_format(ax_bottom, ax_top, break_info)
        fig.legend(lines, line_labels, loc="upper right", bbox_to_anchor=(0.98, 0.97), fontsize=14)
    else:
        ax_bottom.legend(lines, line_labels, loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=14)

    ax_bottom.set_xlabel("Year")
    ax_bottom.set_ylabel(f"Total {data_type}")
    ax_bottom.set_xticks(all_years)
    ax_bottom.set_xticklabels(all_years)

    finalize_layout(fig, ax_top is not None)
    output_filename = f"Merged_Total_{stat_config['filename_suffix']}_annual_trend.jpg"
    output_path = OUTPUT_DIR / output_filename
    fig.savefig(output_path, dpi=1200, bbox_inches="tight", format="jpg")
    plt.close(fig)
    print(f"Merged total trend chart saved to: {output_path}")


def parse_category_metadata(csv_path: Path) -> tuple[str, str]:
    stem = csv_path.stem
    if stem.endswith("_fire_num_with_WUI_offset"):
        category_id = stem[: -len("_fire_num_with_WUI_offset")]
    elif stem.endswith("_fire_area"):
        category_id = stem[: -len("_fire_area")]
    else:
        category_id = stem

    mode = category_id.split("_", 1)[0]
    if mode == "face":
        return category_id, "Interface"
    if mode == "mix":
        return category_id, "Intermix"
    raise ValueError(f"Unknown WUI mode parsed from file name: {csv_path.name}")


def load_basic_metric_directory(directory: Path, metric_column: str) -> pd.DataFrame:
    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {directory}")

    parts: list[pd.DataFrame] = []
    for csv_path in csv_files:
        category_id, wui_type = parse_category_metadata(csv_path)
        df = pd.read_csv(csv_path, usecols=["GRID_ID", "Year", metric_column], low_memory=False)
        df["GRID_ID"] = normalize_grid_id(df["GRID_ID"])
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        df[metric_column] = pd.to_numeric(df[metric_column], errors="coerce")
        df = df.dropna(subset=["GRID_ID", "Year", metric_column])
        if df.empty:
            continue

        df = (
            df.groupby(["GRID_ID", "Year"], as_index=False)[metric_column]
            .mean()
            .assign(category_id=category_id, WUI_Type=wui_type)
        )
        parts.append(df)

    if not parts:
        raise ValueError(f"No valid {metric_column} rows found in: {directory}")
    return pd.concat(parts, ignore_index=True)


def load_basic_continent_lookup() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for wui_year in GRIDCLIP_YEARS:
        csv_path = GRIDCLIP_TEMPLATE.format(year=wui_year)
        df = pd.read_csv(csv_path, usecols=["GRID_ID", "Continent"], low_memory=False)
        df["GRID_ID"] = normalize_grid_id(df["GRID_ID"])
        df["Continent"] = pd.to_numeric(df["Continent"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["GRID_ID", "Continent"])
        df = df[df["Continent"] != 0].copy()
        df.loc[df["Continent"] == 5, "Continent"] = 3
        df = df.drop_duplicates(subset=["GRID_ID"])
        df["WUI_Year"] = wui_year
        parts.append(df)

    return pd.concat(parts, ignore_index=True).drop_duplicates(subset=["WUI_Year", "GRID_ID"])


def build_basic_aligned_dataset() -> pd.DataFrame:
    fire_freq_df = load_basic_metric_directory(FIRE_NUM_BASIC_DIR, "Fire_frequency")
    norm_area_df = load_basic_metric_directory(FIRE_AREA_BASIC_DIR, "Normalized_fire_area")

    merged = fire_freq_df.merge(
        norm_area_df,
        how="outer",
        on=["category_id", "WUI_Type", "GRID_ID", "Year"],
    )
    merged["WUI_Year"] = merged["Year"].map(FIRE_YEAR_TO_WUI_YEAR).astype("Int64")
    merged = merged.dropna(subset=["GRID_ID", "Year", "WUI_Year"])

    merged = merged.merge(load_basic_continent_lookup(), how="left", on=["WUI_Year", "GRID_ID"])
    merged = merged.dropna(subset=["Continent"])
    merged["Continent"] = merged["Continent"].astype(int)

    print(
        "Matched Basic datasets: "
        f"{merged['GRID_ID'].nunique():,} GRID_ID, {len(merged):,} GRID_ID-Year rows."
    )
    return merged


def aggregate_basic_metric(data: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    subset = data.dropna(subset=[metric_column]).copy()
    if subset.empty:
        raise ValueError(f"No valid data found for metric: {metric_column}")

    by_type = (
        subset.groupby(["Year", "Continent", "WUI_Type"], as_index=False)[metric_column]
        .mean()
        .pivot_table(
            index=["Year", "Continent"],
            columns="WUI_Type",
            values=metric_column,
            aggfunc="mean",
        )
        .reset_index()
    )
    by_type.columns.name = None

    total_mean = (
        subset.groupby(["Year", "Continent"], as_index=False)[metric_column]
        .mean()
        .rename(columns={metric_column: "All"})
    )
    annual_data = total_mean.merge(by_type, how="left", on=["Year", "Continent"])
    return annual_data.sort_values(["Year", "Continent"]).reset_index(drop=True)


def plot_basic_metric_bars(all_annual_data: pd.DataFrame, stat_config: dict[str, object]) -> None:
    avg_stats = (
        all_annual_data.groupby("Continent")[["All", "Intermix", "Interface"]]
        .mean()
        .reset_index()
        .sort_values("Continent")
        .reset_index(drop=True)
    )

    continents = avg_stats["Continent"].tolist()
    index = np.arange(len(continents))
    bar_width = 0.25

    all_values = avg_stats["All"].fillna(0).to_numpy(dtype=float)
    intermix_values = avg_stats["Intermix"].fillna(0).to_numpy(dtype=float)
    interface_values = avg_stats["Interface"].fillna(0).to_numpy(dtype=float)
    y_values = np.concatenate([all_values, intermix_values, interface_values])

    fig, ax_bottom, ax_top, break_info = create_plot_axes((12, 8), y_values)
    axes = [ax_bottom] if ax_top is None else [ax_top, ax_bottom]

    for ax in axes:
        ax.bar(index, all_values, bar_width, color=BAR_COLORS["All"], alpha=0.85, label="All")
        ax.bar(
            index + bar_width,
            intermix_values,
            bar_width,
            color=BAR_COLORS["Intermix"],
            alpha=0.85,
            label="Intermix",
        )
        ax.bar(
            index + 2 * bar_width,
            interface_values,
            bar_width,
            color=BAR_COLORS["Interface"],
            alpha=0.85,
            label="Interface",
        )
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    if ax_top is not None and break_info is not None:
        apply_break_format(ax_bottom, ax_top, break_info)
        ax_top.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=14)
    else:
        ax_bottom.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=14)

    ax_bottom.set_xlabel("Continent")
    ax_bottom.set_ylabel(str(stat_config["ylabel"]))
    ax_bottom.set_xticks(index + bar_width)
    ax_bottom.set_xticklabels(format_continent_labels(continents), rotation=0, ha="center", fontsize=14)
    ax_bottom.set_xlim(-0.3, len(continents) - 0.3)

    finalize_layout(fig, ax_top is not None)
    output_filename = f"continent_{stat_config['filename_suffix']}_statistics.jpg"
    output_path = OUTPUT_DIR / output_filename
    fig.savefig(output_path, dpi=1200, bbox_inches="tight", format="jpg")
    plt.close(fig)
    print(f"Chart saved to: {output_path}")


def process_gridclip_metrics() -> None:
    print("=== Processing Fire Number and Burned Area ===")
    year_data_dict: dict[int, pd.DataFrame] = {}

    for year in GRIDCLIP_YEARS:
        data = load_gridclip_data(year)
        if data is not None:
            year_data_dict[year] = data

    if not year_data_dict:
        raise RuntimeError("Failed to load any Grid_Clip data.")

    annual_data_dict: dict[str, pd.DataFrame] = {}

    for stat_config in GRIDCLIP_STATS_CONFIG:
        print(f"\nProcessing {stat_config['name']} statistics...")
        all_annual_data: list[pd.DataFrame] = []

        for base_year, data in year_data_dict.items():
            annual_data = extract_gridclip_annual_data(data, base_year, stat_config)
            if annual_data is not None:
                all_annual_data.append(annual_data)

        if not all_annual_data:
            print(f"No annual data found for {stat_config['name']}")
            continue

        merged_annual_data = pd.concat(all_annual_data, ignore_index=True)
        merged_annual_data = merged_annual_data.drop_duplicates(
            subset=["Year", "Continent"], keep="first"
        )
        annual_data_dict[str(stat_config["name"])] = merged_annual_data
        plot_gridclip_continent_bars(merged_annual_data, stat_config)

    print("\n=== Plotting Fire Number and Burned Area Annual Trends ===")
    for stat_config in GRIDCLIP_STATS_CONFIG:
        data_key = str(stat_config["name"])
        if data_key not in annual_data_dict:
            continue

        all_annual_data = annual_data_dict[data_key]
        print(f"\nProcessing {stat_config['name']} annual trend...")
        for wui_type in ["Interface", "Intermix", "All"]:
            print(f"Plotting {wui_type} {stat_config['name']} annual trend...")
            plot_gridclip_annual_trend(all_annual_data, stat_config, wui_type)

        print(f"Plotting merged Total {stat_config['name']} annual trend...")
        plot_gridclip_merged_total_trends(all_annual_data, stat_config)


def process_basic_metrics() -> None:
    print("\n=== Processing Fire Frequency and Fire Area Density ===")
    aligned_data = build_basic_aligned_dataset()

    for stat_config in BASIC_STATS_CONFIG:
        print(f"\nProcessing {stat_config['name']} statistics...")
        annual_data = aggregate_basic_metric(aligned_data, str(stat_config["metric_column"]))
        plot_basic_metric_bars(annual_data, stat_config)


def main() -> None:
    print("=== Continent Fire Statistics Combined Program ===")
    process_gridclip_metrics()
    process_basic_metrics()
    print("\nAll statistics processing completed!")


if __name__ == "__main__":
    main()
