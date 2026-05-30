#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称: Fire frequency 与 Normalized burned area 双变量空间分布图

功能简介:
    1. 从两个分类建模 CSV 目录中分别读取 Fire_frequency 与 Normalized_fire_area。
    2. 仅保留 GRID_ID、Year 和对应指标列，并在全局范围内合并。
    3. 根据 Year 选择对应的 WUI 渔网图层:
       - 2003-2007 -> Grid_Clip_2005_WUI
       - 2008-2012 -> Grid_Clip_2010_WUI
       - 2013-2017 -> Grid_Clip_2015_WUI
       - 2018-2022 -> Grid_Clip_2020_WUI
    4. 逐年将 Fire frequency 和 Normalized burned area 分别分成 3 个等级，
       再组合成 3x3 双变量颜色矩阵绘制全球空间分布图。
    5. 底图、渔网和版式尽量保持与「Spatial_distribution_主导气候带类型.py」一致。

输入数据:
    - Fire frequency CSV:
      I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic
    - Normalized burned area CSV:
      I:\Processing data\渔网分类建模\Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic
    - WUI 渔网:
      I:\Processing data\gdb集合\WUI.gdb
    - 大陆底图:
      I:\Data_huanglin\map\world continent boundary\continent.shp

输出:
    - B:\WUI\Picture\Bivariate_FireFrequency_NormalizedFireArea_YYYY.png
"""

from __future__ import annotations

import glob
import os
from typing import Iterable

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置字体
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

FONT_SCALE = 1.2


def _fs(size: float) -> float:
    return size * FONT_SCALE

# =========================
# 路径与年份配置
# =========================
FIRE_FREQUENCY_SOURCE_DIR = (
    r"I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic"
)
NORMALIZED_FIRE_AREA_SOURCE_DIR = (
    r"I:\Processing data\渔网分类建模\Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic"
)

WUI_GDB_PATH = r"I:\Processing data\gdb集合\WUI.gdb"
WUI_YEARS = [2005, 2010, 2015, 2020]
WUI_LAYER_TEMPLATE = "Grid_Clip_{wui_year}_WUI"

CONTINENT_SHP_PATH = r"I:\Data_huanglin\map\world continent boundary\continent.shp"

OUTPUT_DIR = r"B:\WUI\Picture"
OUTPUT_FILENAME_TEMPLATE = "Bivariate_FireFrequency_NormalizedFireArea_{year}.png"
OUTPUT_DPI = 1200

FIRE_YEAR_START = int(os.getenv("BIVAR_FIRE_YEAR_START", "2003"))
FIRE_YEAR_END = int(os.getenv("BIVAR_FIRE_YEAR_END", "2022"))

# 读取渔网时按 GRID_ID 过滤，避免一次性读整层
MAX_IDS_FOR_WHERE = 50000
WHERE_CHUNK_SIZE = 3000

# =========================
# 绘图样式
# =========================
BASEMAP_FACE = "#ffffff"
BASEMAP_EDGE = "#000000"
NO_DATA_COLOR = "#e3dddd"

# 3x3 双变量配色:
# 横轴 = Fire frequency, 纵轴 = Normalized burned area
# 设计原则:
#   - Fire frequency 越高越偏黄
#   - Normalized burned area 越高越偏红
#   - 二者同时较高时显示橙色/橙红色，避免和纯黄、纯红混淆
BIVARIATE_COLORS = {
    (1, 1): "#FFF7EC",
    (2, 1): "#FEE08B",
    (3, 1): "#E6AB02",
    (1, 2): "#FDBB84",
    (2, 2): "#F28E2B",
    (3, 2): "#D95F0E",
    (1, 3): "#EF6548",
    (2, 3): "#D7301F",
    (3, 3): "#7F0000",
}

def _chunks(items: list[int], chunk_size: int) -> Iterable[list[int]]:
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
    for name in candidates:
        if name in columns:
            return name
    return None


def read_metric_from_csv_dir(
    source_dir: str,
    value_candidates: list[str],
    output_col: str,
) -> pd.DataFrame:
    csv_files = sorted(glob.glob(os.path.join(source_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {source_dir}")

    parts: list[pd.DataFrame] = []
    print(f"Reading metric '{output_col}' from {len(csv_files)} CSV file(s): {source_dir}")

    for csv_path in csv_files:
        header = pd.read_csv(csv_path, nrows=0)
        cols = header.columns.tolist()

        grid_col = _pick_column(cols, ["GRID_ID", "grid_id", "Grid_ID"])
        year_col = _pick_column(cols, ["Year", "YEAR", "year"])
        value_col = _pick_column(cols, value_candidates)

        missing = [k for k, v in {"GRID_ID": grid_col, "Year": year_col, output_col: value_col}.items() if v is None]
        if missing:
            raise ValueError(f"Missing required column(s) {missing} in: {csv_path}")

        df = pd.read_csv(
            csv_path,
            usecols=[grid_col, year_col, value_col],
            dtype={grid_col: "string"},
            low_memory=False,
        )
        df = df.rename(columns={grid_col: "GRID_ID", year_col: "Year", value_col: output_col})

        df["GRID_ID"] = pd.to_numeric(df["GRID_ID"], errors="coerce").astype("Int64")
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        df[output_col] = pd.to_numeric(df[output_col], errors="coerce")
        df = df.dropna(subset=["GRID_ID", "Year"])
        df = df[df["Year"].between(FIRE_YEAR_START, FIRE_YEAR_END)]
        if df.empty:
            continue

        parts.append(df[["GRID_ID", "Year", output_col]])

    if not parts:
        raise ValueError(f"No valid records loaded from: {source_dir}")

    combined = pd.concat(parts, ignore_index=True)
    combined = (
        combined.groupby(["Year", "GRID_ID"], as_index=False)
        .agg(**{output_col: (output_col, "mean")})
        .dropna(subset=["GRID_ID", "Year"])
    )

    print(
        f"Loaded '{output_col}': "
        f"years={combined['Year'].nunique()}, rows={len(combined)}, unique GRID_ID={combined['GRID_ID'].nunique()}"
    )
    return combined


def load_bivariate_metrics() -> pd.DataFrame:
    fire_frequency = read_metric_from_csv_dir(
        source_dir=FIRE_FREQUENCY_SOURCE_DIR,
        value_candidates=["Fire_frequency", "fire_frequency", "Fire Frequency"],
        output_col="Fire_frequency",
    )
    normalized_fire_area = read_metric_from_csv_dir(
        source_dir=NORMALIZED_FIRE_AREA_SOURCE_DIR,
        value_candidates=["Normalized_fire_area", "normalized_fire_area", "Normalized burn area", "Normalized burned area"],
        output_col="Normalized_fire_area",
    )

    merged = pd.merge(
        fire_frequency,
        normalized_fire_area,
        on=["Year", "GRID_ID"],
        how="outer",
    )
    merged = merged.sort_values(["Year", "GRID_ID"]).reset_index(drop=True)

    print(
        "Merged metrics: "
        f"years={merged['Year'].nunique()}, rows={len(merged)}, "
        f"rows_with_both={((merged['Fire_frequency'].notna()) & (merged['Normalized_fire_area'].notna())).sum()}"
    )
    return merged


def read_wui_fishnet_filtered(
    gdb_path: str,
    layer: str,
    grid_ids: list[int],
    max_ids_for_where: int = MAX_IDS_FOR_WHERE,
    chunk_size: int = WHERE_CHUNK_SIZE,
) -> gpd.GeoDataFrame:
    if not grid_ids:
        print(f"No GRID_ID provided, reading full WUI fishnet layer: {layer}")
        return gpd.read_file(gdb_path, layer=layer, columns=["GRID_ID"], engine="pyogrio")

    if len(grid_ids) > max_ids_for_where:
        print(
            f"GRID_ID count = {len(grid_ids)} > {max_ids_for_where}, "
            f"reading full WUI fishnet layer instead: {layer}"
        )
        return gpd.read_file(gdb_path, layer=layer, columns=["GRID_ID"], engine="pyogrio")

    print(f"Reading WUI fishnet by GRID_ID filter (layer={layer}, n={len(grid_ids)}) ...")
    current_chunk_size = max(1, int(chunk_size))
    while True:
        try:
            gdfs: list[gpd.GeoDataFrame] = []
            for chunk in _chunks(grid_ids, current_chunk_size):
                where = f"GRID_ID IN ({','.join(map(str, chunk))})"
                gdfs.append(
                    gpd.read_file(
                        gdb_path,
                        layer=layer,
                        where=where,
                        columns=["GRID_ID"],
                        engine="pyogrio",
                    )
                )

            fishnet = pd.concat(gdfs, ignore_index=True)
            if isinstance(fishnet, gpd.GeoDataFrame):
                return fishnet
            return gpd.GeoDataFrame(fishnet, geometry="geometry", crs=gdfs[0].crs)
        except ValueError as e:
            msg = str(e)
            if "Invalid SQL query" in msg and current_chunk_size > 1:
                new_chunk_size = max(1, current_chunk_size // 2)
                if new_chunk_size == current_chunk_size:
                    raise
                print(
                    f"Invalid SQL query with WHERE_CHUNK_SIZE={current_chunk_size}, "
                    f"retrying with {new_chunk_size}..."
                )
                current_chunk_size = new_chunk_size
                continue
            raise


def wui_year_for_fire_year(fire_year: int) -> int:
    if fire_year < FIRE_YEAR_START or fire_year > FIRE_YEAR_END:
        raise ValueError(f"Fire year {fire_year} out of range [{FIRE_YEAR_START}, {FIRE_YEAR_END}]")

    base_wui_year = min(WUI_YEARS)
    wui_year = base_wui_year + 5 * ((fire_year - FIRE_YEAR_START) // 5)
    if wui_year not in WUI_YEARS:
        raise ValueError(f"No WUI layer configured for fire year {fire_year} (computed WUI year {wui_year}).")
    return wui_year


def classify_three_classes(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    classes = pd.Series(pd.NA, index=values.index, dtype="Int64")
    valid = values.dropna()
    if valid.empty:
        return classes

    q1 = valid.quantile(1 / 3)
    q2 = valid.quantile(2 / 3)

    if pd.notna(q1) and pd.notna(q2) and q1 < q2:
        bins = pd.cut(
            valid,
            bins=[-np.inf, q1, q2, np.inf],
            labels=[1, 2, 3],
            include_lowest=True,
        )
        classes.loc[valid.index] = bins.astype("Int64")
        return classes

    dense_rank = valid.rank(method="dense")
    max_rank = int(dense_rank.max())
    if max_rank <= 1:
        classes.loc[valid.index] = 1
        return classes

    scaled = np.ceil(dense_rank / max_rank * 3).astype(int).clip(1, 3)
    classes.loc[valid.index] = pd.Series(scaled, index=valid.index).astype("Int64")
    return classes


def choose_display_exponent(values: pd.Series, classes: pd.Series) -> int:
    numeric_values = pd.to_numeric(values, errors="coerce")
    medians: list[float] = []
    for class_id in (1, 2, 3):
        subset = numeric_values[classes == class_id].dropna()
        if not subset.empty:
            medians.append(float(subset.median()))
    if not medians:
        return 0
    max_abs = max(abs(v) for v in medians)
    if max_abs < 1e-12:
        return 0
    return int(np.floor(np.log10(max_abs)))


def format_class_value(value: float | int | None, exponent: int) -> str:
    if value is None or pd.isna(value):
        return "-"
    raw_value = float(value)
    if abs(raw_value) < 5e-13:
        return "0.0"
    scaled = raw_value / (10.0 ** exponent)
    rounded = round(scaled, 1)
    if abs(rounded) < 0.05:
        rounded = 0.0
    return f"{rounded:.1f}"


def get_class_representative_values(values: pd.Series, classes: pd.Series) -> tuple[list[str], str]:
    exponent = choose_display_exponent(values, classes)
    labels: list[str] = []
    numeric_values = pd.to_numeric(values, errors="coerce")
    for class_id in (1, 2, 3):
        subset = numeric_values[classes == class_id].dropna()
        if subset.empty:
            labels.append("-")
        else:
            labels.append(format_class_value(subset.median(), exponent))
    return labels, f"1e{exponent:+d}"


def add_bivariate_legend(
    ax: plt.Axes,
    fire_frequency_labels: list[str],
    normalized_fire_area_labels: list[str],
    fire_frequency_scale_text: str,
    normalized_fire_area_scale_text: str,
) -> None:
    # Legend layout is controlled entirely in lon/lat coordinates.
    legend_fontsize = _fs(14.7)
    grid_left_lon = -154.0
    grid_bottom_lat = -34.0
    cell_width_lon = 14.0
    cell_height_lat = 14.0

    top_scale_lon = -162.0
    top_scale_lat = 11.5
    x_axis_title_lon = -133.0
    x_axis_title_lat = -43.5
    x_axis_scale_lon = -106.0
    x_axis_scale_lat = -35.0
    y_axis_title_lon = -171.0
    y_axis_title_lat = -13.0
    x_tick_lat = -35.0
    y_tick_lon = -158.0

    no_data_left_lon = -175.0
    no_data_bottom_lat = -55.8
    no_data_width_lon = 12.0
    no_data_height_lat = 5.5
    no_data_label_lon = -158.0
    no_data_label_lat = -53.6

    for x_class in range(1, 4):
        for y_class in range(1, 4):
            ax.add_patch(
                mpatches.Rectangle(
                    (
                        grid_left_lon + (x_class - 1) * cell_width_lon,
                        grid_bottom_lat + (y_class - 1) * cell_height_lat,
                    ),
                    cell_width_lon,
                    cell_height_lat,
                    facecolor=BIVARIATE_COLORS[(x_class, y_class)],
                    edgecolor="white",
                    linewidth=0.6,
                    zorder=21,
                )
            )

    ax.text(
        top_scale_lon,
        top_scale_lat,
        normalized_fire_area_scale_text,
        ha="center",
        va="bottom",
        fontsize=legend_fontsize,
        zorder=22,
    )
    ax.text(
        x_axis_title_lon,
        x_axis_title_lat,
        "Fire frequency",
        ha="center",
        va="top",
        fontsize=legend_fontsize,
        zorder=22,
    )
    ax.text(
        x_axis_scale_lon,
        x_axis_scale_lat,
        fire_frequency_scale_text,
        ha="left",
        va="top",
        fontsize=legend_fontsize,
        zorder=22,
    )
    ax.text(
        y_axis_title_lon,
        y_axis_title_lat,
        "Normalized burned area",
        ha="right",
        va="center",
        rotation=90,
        fontsize=legend_fontsize,
        zorder=22,
    )

    for idx, label in enumerate(fire_frequency_labels):
        ax.text(
            grid_left_lon + (idx + 0.5) * cell_width_lon,
            x_tick_lat,
            label,
            ha="center",
            va="top",
            fontsize=legend_fontsize,
            zorder=22,
        )
    for idx, label in enumerate(normalized_fire_area_labels):
        ax.text(
            y_tick_lon,
            grid_bottom_lat + (idx + 0.5) * cell_height_lat,
            label,
            ha="right",
            va="center",
            fontsize=legend_fontsize,
            zorder=22,
        )

    ax.add_patch(
        mpatches.Rectangle(
            (no_data_left_lon, no_data_bottom_lat),
            no_data_width_lon,
            no_data_height_lat,
            facecolor=NO_DATA_COLOR,
            edgecolor="none",
            zorder=21,
        )
    )
    ax.text(
        no_data_label_lon,
        no_data_label_lat,
        "Fire frequency and normalized burned area = 0",
        ha="left",
        va="center",
        fontsize=legend_fontsize,
        zorder=22,
    )


def plot_bivariate_map(
    *,
    continent: gpd.GeoDataFrame,
    fishnet: gpd.GeoDataFrame,
    fire_year: int,
    fire_frequency_labels: list[str],
    normalized_fire_area_labels: list[str],
    fire_frequency_scale_text: str,
    normalized_fire_area_scale_text: str,
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(15, 10))

    if continent is not None:
        continent.plot(ax=ax, color=BASEMAP_FACE, edgecolor=BASEMAP_EDGE, linewidth=0.5)

    ax.set_facecolor("white")
    ax.set_xlim([-180, 180])
    ax.set_ylim([-60, 75])

    if not fishnet.empty:
        zero_zero_mask = (
            fishnet["Fire_frequency"].fillna(np.nan).eq(0) &
            fishnet["Normalized_fire_area"].fillna(np.nan).eq(0)
        )
        no_data_mask = fishnet["bivariate_key"].isna() | zero_zero_mask
        fishnet_no_data = fishnet[no_data_mask]
        fishnet_has_data = fishnet[~no_data_mask]

        if not fishnet_no_data.empty:
            fishnet_no_data.plot(ax=ax, color=NO_DATA_COLOR, edgecolor="none", linewidth=0, alpha=1.0)

        for key, color in BIVARIATE_COLORS.items():
            subset = fishnet_has_data[fishnet_has_data["bivariate_key"] == key]
            if not subset.empty:
                subset.plot(ax=ax, color=color, edgecolor="none", linewidth=0, alpha=0.9)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.text(0.02, 0.92, f"{fire_year}", transform=ax.transAxes, fontsize=20, verticalalignment="top")

    add_bivariate_legend(
        ax,
        fire_frequency_labels,
        normalized_fire_area_labels,
        fire_frequency_scale_text,
        normalized_fire_area_scale_text,
    )
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.055, top=0.995)
    plt.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {output_path}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading global map data...")
    continent = gpd.read_file(CONTINENT_SHP_PATH)
    continent = continent.to_crs("EPSG:4326")
    print(f"Loaded continent layer: {len(continent)} features")

    metric_all = load_bivariate_metrics()
    metric_all["GRID_ID"] = metric_all["GRID_ID"].astype(int)
    metric_all["Year"] = metric_all["Year"].astype(int)

    available_years = sorted(metric_all["Year"].unique().tolist())
    print(f"Available years: {available_years}")

    fishnet_cache: dict[int, gpd.GeoDataFrame] = {}
    for wui_year in WUI_YEARS:
        year_group = [y for y in available_years if wui_year_for_fire_year(y) == wui_year]
        if not year_group:
            continue

        needed_ids = sorted(
            set(
                metric_all.loc[metric_all["Year"].isin(year_group), "GRID_ID"]
                .dropna()
                .astype(int)
                .tolist()
            )
        )

        layer = WUI_LAYER_TEMPLATE.format(wui_year=wui_year)
        fishnet_base = read_wui_fishnet_filtered(WUI_GDB_PATH, layer, needed_ids)
        if "GRID_ID" not in fishnet_base.columns:
            raise ValueError(f"GRID_ID field not found in WUI fishnet layer: {layer}")

        fishnet_base["GRID_ID"] = pd.to_numeric(fishnet_base["GRID_ID"], errors="coerce")
        fishnet_base = fishnet_base.dropna(subset=["GRID_ID"])
        fishnet_base["GRID_ID"] = fishnet_base["GRID_ID"].astype(int)
        fishnet_base = fishnet_base.to_crs(epsg=4326)
        fishnet_cache[wui_year] = fishnet_base
        print(f"Cached WUI layer {layer}: {len(fishnet_base)} features")

    for fire_year in available_years:
        print(f"\n=== Processing year {fire_year} ===")
        year_metric = metric_all.loc[
            metric_all["Year"] == fire_year,
            ["GRID_ID", "Fire_frequency", "Normalized_fire_area"],
        ].copy()

        year_metric["Fire_frequency_class"] = classify_three_classes(year_metric["Fire_frequency"])
        year_metric["Normalized_fire_area_class"] = classify_three_classes(year_metric["Normalized_fire_area"])
        fire_frequency_labels, fire_frequency_scale_text = get_class_representative_values(
            year_metric["Fire_frequency"],
            year_metric["Fire_frequency_class"],
        )
        normalized_fire_area_labels, normalized_fire_area_scale_text = get_class_representative_values(
            year_metric["Normalized_fire_area"],
            year_metric["Normalized_fire_area_class"],
        )

        valid_class = (
            year_metric["Fire_frequency_class"].notna() &
            year_metric["Normalized_fire_area_class"].notna()
        )
        year_metric["bivariate_key"] = pd.Series([pd.NA] * len(year_metric), dtype="object")
        year_metric.loc[valid_class, "bivariate_key"] = pd.Series(
            list(
                zip(
                    year_metric.loc[valid_class, "Fire_frequency_class"].astype(int),
                    year_metric.loc[valid_class, "Normalized_fire_area_class"].astype(int),
                )
            ),
            index=year_metric.index[valid_class],
            dtype="object",
        )

        wui_year = wui_year_for_fire_year(fire_year)
        fishnet_base = fishnet_cache[wui_year]
        year_ids = set(year_metric["GRID_ID"].astype(int).tolist())
        fishnet_year = fishnet_base[fishnet_base["GRID_ID"].isin(year_ids)].copy()
        fishnet_year = fishnet_year.merge(year_metric, on="GRID_ID", how="left")

        output_path = os.path.join(
            OUTPUT_DIR,
            OUTPUT_FILENAME_TEMPLATE.format(year=fire_year),
        )
        plot_bivariate_map(
            continent=continent,
            fishnet=fishnet_year,
            fire_year=fire_year,
            fire_frequency_labels=fire_frequency_labels,
            normalized_fire_area_labels=normalized_fire_area_labels,
            fire_frequency_scale_text=fire_frequency_scale_text,
            normalized_fire_area_scale_text=normalized_fire_area_scale_text,
            output_path=output_path,
        )

    print("\n=== All years completed ===")


if __name__ == "__main__":
    main()
