#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称:
    统计印度北部与中国东部两个区域的机制变量均值并绘图

功能:
    1. 沿用局部放大图中的两个区域范围：
       - Northern India: lon 74-91, lat 19-31
       - East-Central China: lon 105-122, lat 24-36
    2. 从 WUI.gdb 提取两个区域内的 GRID_ID，并按 WUI 年份映射到 2003-2022 年。
    3. 从 Fire num 的 Basic CSV 中读取 Human Footprint、Cropland、Pasture、GDP、Population density。
    4. 根据 Year + GRID_ID 匹配两个区域，统计各区域 5 个变量的均值。
    5. 输出一个 1x5 组图，每个子图展示一个变量在两个区域中的均值。

输出:
    - B:\WUI\Picture\Regional_mechanism_variable_means_Northern_India_vs_East_Central_China_2003_2022.png
    - B:\WUI\Picture\Regional_mechanism_variable_means_Northern_India_vs_East_Central_China_2003_2022.xlsx
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter


OUTPUT_DIR = Path(r"B:\WUI\Picture")
YEAR_START = 2003
YEAR_END = 2022
YEAR_TAG = f"{YEAR_START}_{YEAR_END}"

WUI_GDB_PATH = Path(r"I:\Processing data\gdb集合\WUI.gdb")
WUI_YEARS = [2005, 2010, 2015, 2020]
WUI_LAYER_TEMPLATE = "Grid_Clip_{wui_year}_WUI"
WUI_YEAR_MAPPING = {
    2005: list(range(2003, 2008)),
    2010: list(range(2008, 2013)),
    2015: list(range(2013, 2018)),
    2020: list(range(2018, 2023)),
}

BASIC_INPUT_DIR = Path(
    r"I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic"
)

OUTPUT_FIGURE_PATH = OUTPUT_DIR / (
    f"Regional_mechanism_variable_means_Northern_India_vs_East_Central_China_{YEAR_TAG}.png"
)
OUTPUT_TABLE_PATH = OUTPUT_DIR / (
    f"Regional_mechanism_variable_means_Northern_India_vs_East_Central_China_{YEAR_TAG}.xlsx"
)

KEY_COLUMNS = ["Year", "GRID_ID"]
PARAMETER_COLUMNS = ["HFI", "Cropland_Area_km2", "Pasture_Band2_Mean", "GDP", "PopD"]
PARAMETER_CONFIGS = [
    {"column": "HFI", "label": "Human Footprint"},
    {"column": "Cropland_Area_km2", "label": "Cropland"},
    {"column": "Pasture_Band2_Mean", "label": "Pasture"},
    {"column": "GDP", "label": "GDP"},
    {"column": "PopD", "label": "Population density"},
]

REGIONS = [
    {
        "slug": "northern_india",
        "label": "Northern India",
        "lon_min": 74.0,
        "lon_max": 91.0,
        "lat_min": 19.0,
        "lat_max": 31.0,
        "color": "#ff9994",
    },
    {
        "slug": "east_central_china",
        "label": "East-Central China",
        "lon_min": 105.0,
        "lon_max": 122.0,
        "lat_min": 24.0,
        "lat_max": 36.0,
        "color": "#88c4ff",
    },
]
REGION_X_LABELS = ["Region 1", "Region 2"]


plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 15

TITLE_FONT_SIZE = 30
AXIS_LABEL_FONT_SIZE = 24
TICK_FONT_SIZE = 24
VALUE_FONT_SIZE = 26
PANEL_LABEL_FONT_SIZE = 30
PANEL_LABELS = ["a", "b", "c", "d", "e"]


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
    summary_rows: list[dict[str, object]] = []
    for region in REGIONS:
        for wui_year in WUI_YEARS:
            region_grid = read_region_grid_ids(region, wui_year)
            grid_count = int(region_grid["GRID_ID"].nunique()) if not region_grid.empty else 0
            summary_rows.append(
                {
                    "region_label": str(region["label"]),
                    "wui_year": int(wui_year),
                    "grid_count": grid_count,
                }
            )
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
    lookup.attrs["summary"] = pd.DataFrame(summary_rows)
    print(
        "Region lookup loaded: "
        f"rows={len(lookup):,}, "
        f"unique GRID_ID={lookup['GRID_ID'].nunique():,}, "
        f"years={lookup['Year'].nunique()}"
    )
    return lookup[["Year", "GRID_ID", "wui_year", "region_slug", "region_label"]]


def is_basic_csv(csv_path: Path) -> bool:
    try:
        header = pd.read_csv(csv_path, nrows=0)
    except Exception:
        return False
    required_columns = set(KEY_COLUMNS + PARAMETER_COLUMNS)
    return required_columns.issubset(set(header.columns))


def load_region_basic_samples(region_lookup: pd.DataFrame) -> pd.DataFrame:
    if not BASIC_INPUT_DIR.exists():
        raise FileNotFoundError(f"Basic input directory not found: {BASIC_INPUT_DIR}")

    basic_files = [path for path in sorted(BASIC_INPUT_DIR.glob("*.csv")) if is_basic_csv(path)]
    if not basic_files:
        raise FileNotFoundError(f"No valid Basic CSV files found in: {BASIC_INPUT_DIR}")

    parts: list[pd.DataFrame] = []
    for idx, csv_path in enumerate(basic_files, start=1):
        df = pd.read_csv(csv_path, usecols=KEY_COLUMNS + PARAMETER_COLUMNS, low_memory=False)
        df["Year"] = normalize_year(df["Year"])
        df["GRID_ID"] = normalize_grid_id(df["GRID_ID"])
        for parameter_col in PARAMETER_COLUMNS:
            df[parameter_col] = pd.to_numeric(df[parameter_col], errors="coerce")
        df = df[
            df["Year"].between(YEAR_START, YEAR_END, inclusive="both")
        ].dropna(subset=KEY_COLUMNS)
        if df.empty:
            continue

        df["Year"] = df["Year"].astype(np.int64)
        df["GRID_ID"] = df["GRID_ID"].astype(np.int64)
        matched = df.merge(region_lookup, how="inner", on=KEY_COLUMNS, validate="many_to_one")
        if matched.empty:
            continue
        parts.append(matched[["region_slug", "region_label", "Year", "GRID_ID"] + PARAMETER_COLUMNS])
        print(f"[{idx}/{len(basic_files)}] matched rows from {csv_path.name}: {len(matched):,}")

    if not parts:
        raise ValueError("No Basic rows matched the two regional GRID_ID-Year lookup tables.")

    samples = pd.concat(parts, ignore_index=True)
    samples = samples.dropna(subset=PARAMETER_COLUMNS, how="all")
    samples = (
        samples.groupby(["region_slug", "region_label", "Year", "GRID_ID"], as_index=False)[PARAMETER_COLUMNS]
        .mean()
        .reset_index(drop=True)
    )
    print(
        "Regional Basic samples loaded: "
        f"rows={len(samples):,}, "
        f"unique GRID_ID={samples['GRID_ID'].nunique():,}, "
        f"regions={samples['region_label'].nunique()}"
    )
    return samples


def summarize_region_means(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for region in REGIONS:
        region_df = samples.loc[samples["region_slug"] == region["slug"]].copy()
        for parameter_config in PARAMETER_CONFIGS:
            column = str(parameter_config["column"])
            values = pd.to_numeric(region_df[column], errors="coerce").dropna()
            rows.append(
                {
                    "region_slug": str(region["slug"]),
                    "region_label": str(region["label"]),
                    "parameter_column": column,
                    "parameter_label": str(parameter_config["label"]),
                    "n_samples": int(values.size),
                    "mean": float(values.mean()) if not values.empty else np.nan,
                    "sd": float(values.std(ddof=1)) if values.size > 1 else np.nan,
                    "se": float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _format_bar_label(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    abs_value = abs(value)
    if abs_value >= 1000:
        return f"{value:.2e}"
    if abs_value >= 10:
        return f"{value:.1f}"
    if abs_value >= 1:
        return f"{value:.2f}"
    if abs_value >= 0.01:
        return f"{value:.3f}"
    return f"{value:.2e}"


def plot_region_means(summary_df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(12.8, 14.0), facecolor="white")
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.0])
    axes = [
        fig.add_subplot(gs[0, 1:3]),
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
        fig.add_subplot(gs[2, 0:2]),
        fig.add_subplot(gs[2, 2:4]),
    ]
    region_labels = [str(region["label"]) for region in REGIONS]
    colors = [str(region["color"]) for region in REGIONS]

    for ax, parameter_config in zip(axes, PARAMETER_CONFIGS):
        parameter_label = str(parameter_config["label"])
        plot_df = (
            summary_df.loc[summary_df["parameter_label"] == parameter_label]
            .set_index("region_label")
            .loc[region_labels]
            .reset_index()
        )
        means = plot_df["mean"].to_numpy(dtype=float)
        x = np.array([-0.1, 0.1], dtype=float)
        bars = ax.bar(
            x,
            means,
            width=0.08,
            color=colors,
            alpha=0.82,
            edgecolor=colors,
            linewidth=1.0,
        )
        ax.set_ylabel(parameter_label, fontsize=AXIS_LABEL_FONT_SIZE, labelpad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(REGION_X_LABELS, fontsize=TICK_FONT_SIZE)
        ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        formatter = ScalarFormatter(useMathText=False)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 3))
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.get_offset_text().set_fontsize(TICK_FONT_SIZE)

        finite_means = means[np.isfinite(means)]
        ymax = np.nanmax(finite_means) if len(finite_means) else 1.0
        ymin = np.nanmin(finite_means) if len(finite_means) else 0.0
        span = ymax - ymin if ymax != ymin else max(abs(ymax), 1.0)
        ax.set_ylim(bottom=min(0.0, ymin - span * 0.08), top=ymax + span * 0.20)
        for bar, value in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                float(value) + span * 0.04,
                _format_bar_label(float(value)),
                ha="center",
                va="bottom",
                fontsize=VALUE_FONT_SIZE,
                color="#222222",
            )

    fig.subplots_adjust(left=0.100, right=0.985, top=0.975, bottom=0.065, wspace=0.60, hspace=0.3)
    for ax, panel_label in zip(axes, PANEL_LABELS):
        bbox = ax.get_position()
        fig.text(
            bbox.x0 - 0.025,
            bbox.y1 + 0.012,
            panel_label,
            ha="left",
            va="bottom",
            fontsize=PANEL_LABEL_FONT_SIZE,
            fontweight="bold",
        )
    fig.savefig(OUTPUT_FIGURE_PATH, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved figure to: {OUTPUT_FIGURE_PATH}")


def save_tables(summary_df: pd.DataFrame, samples: pd.DataFrame, region_lookup: pd.DataFrame) -> None:
    with pd.ExcelWriter(OUTPUT_TABLE_PATH) as writer:
        summary_df.to_excel(writer, sheet_name="regional_means", index=False)
        region_lookup.attrs.get("summary", pd.DataFrame()).to_excel(writer, sheet_name="region_grid_counts", index=False)
        samples.groupby(["region_label"], as_index=False).agg(
            n_grid_year_rows=("GRID_ID", "size"),
            n_unique_grid=("GRID_ID", "nunique"),
            n_years=("Year", "nunique"),
        ).to_excel(writer, sheet_name="sample_counts", index=False)
    print(f"Saved table to: {OUTPUT_TABLE_PATH}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    region_lookup = build_region_year_lookup()
    samples = load_region_basic_samples(region_lookup)
    summary_df = summarize_region_means(samples)
    plot_region_means(summary_df)
    save_tables(summary_df, samples, region_lookup)


if __name__ == "__main__":
    main()
