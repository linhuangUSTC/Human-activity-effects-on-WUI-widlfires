#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
程序名称: 总火灾频率与面积密度空间分布绘制程序（2003-2022均值）
功能简介: 本程序在 Spatial_distribution_world_fire_frequency_normalized_burned_area_total.py
         的基础上，将 2003-2022 全部年份的总火灾频率与总火灾面积密度先计算到年尺度，
         再对每个 GRID_ID 按可用年份求均值，最终仅输出两张图：
         1. Fire frequency 的 2003-2022 多年均值空间分布图
         2. Normalized burned area 的 2003-2022 多年均值空间分布图

处理逻辑:
    1. 分别读取 2005/2010/2015/2020 四套 Interface 和 Intermix 渔网图层。
    2. 对每个 WUI 图层所覆盖的 5 个年份，按 GRID_ID 合并 Interface 与 Intermix：
       - 总 Fire_num = Interface + Intermix
       - 总 Fire_area = Interface + Intermix
       - 总 WUI_area = Interface + Intermix
    3. 先计算每个 GRID_ID 在每个年份的：
       - Total_Fire_Frequency = Total_Fire_num / Total_WUI_area
       - Total_Fire_Area_Density = Total_Fire_area / Total_WUI_area
    4. 再对每个 GRID_ID 跨 2003-2022 全部可用年份求均值；
       若某格网不足 20 年，则仅按其可用年份求均值。
    5. Fire frequency 的颜色映射仍使用 log10(1 + x)；
       colorbar 仍显示原始值。
    6. Normalized burned area 的颜色映射与原脚本保持一致。

输入数据:
    - 全球地图边界数据: I:/Data_huanglin/map/world continent boundary/continent.shp
    - Interface WUI Fire Risk数据: I:/Processing data/gdb集合/Interface_WUI_Fire_Risk.gdb
    - Intermix WUI Fire Risk数据: I:/Processing data/gdb集合/Intermix_WUI_Fire_Risk.gdb

输出文件:
    - Fire frequency均值分布图:
      B:/WUI/Picture/Fire_frequency_mean_2003_2022.jpg
    - Normalized burned area均值分布图:
      B:/WUI/Picture/Normalized_burned_area_mean_2003_2022.jpg
"""

from __future__ import annotations

import os

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, FuncFormatter

# 设置英文显示，使用Arial字体
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

# 定义数据路径
continent_shp_path = r"I:\Data_huanglin\map\world continent boundary\continent.shp"

interface_config = {
    "type": "Interface",
    "gdb_path": r"I:\Processing data\gdb集合\Interface_WUI_Fire_Risk.gdb",
    "years": [2005, 2010, 2015, 2020],
    "layer_prefix": "Grid_Clip_",
}

intermix_config = {
    "type": "Intermix",
    "gdb_path": r"I:\Processing data\gdb集合\Intermix_WUI_Fire_Risk.gdb",
    "years": [2005, 2010, 2015, 2020],
    "layer_prefix": "Grid_Clip_",
}

output_dir = r"B:\WUI\Picture"
os.makedirs(output_dir, exist_ok=True)

OUTPUT_FIRE_FREQUENCY = os.path.join(output_dir, "Fire_frequency_mean_2003_2022.jpg")
OUTPUT_FIRE_AREA_DENSITY = os.path.join(output_dir, "Normalized_burned_area_mean_2003_2022.jpg")

FIG_SIZE = (15, 5.8)
MAP_AX_POSITION = [0.045, 0.13, 0.82, 0.795]
CBAR_AX_POSITION = [0.895, 0.13, 0.018, 0.795]


def log10p_transform(values: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return np.log10(1.0 + values)


def inverse_log10p_transform(values: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return np.power(10.0, values) - 1.0


def format_original_fire_frequency(value: float, _position: int) -> str:
    original_value = inverse_log10p_transform(value)
    return f"{original_value:.1f}"


def find_fire_area_field(columns: list[str], data_year: int) -> str | None:
    candidates = [
        str(data_year),
        f"Fire_area_{data_year}",
        f"过火面积{data_year}",
    ]
    for field in candidates:
        if field in columns:
            return field
    return None


def ensure_grid_id(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "GRID_ID" not in frame.columns:
        frame = frame.copy()
        frame["GRID_ID"] = frame.geometry.apply(lambda x: x.centroid.wkt)
    return frame


def aggregate_total_grid_for_wui_year(
    *,
    wui_year: int,
    interface_fishnet: gpd.GeoDataFrame,
    intermix_fishnet: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    total_fishnet = gpd.GeoDataFrame(
        pd.concat([interface_fishnet, intermix_fishnet], ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )

    data_years = list(range(wui_year - 2, wui_year + 3))
    total_columns = total_fishnet.columns.tolist()

    agg_fields: dict[str, str] = {
        "geometry": "first",
        "WUI_area": "sum",
    }
    fire_area_field_by_year: dict[int, str] = {}

    for data_year in data_years:
        fire_num_field = f"Fire_num_{data_year}"
        if fire_num_field in total_columns:
            agg_fields[fire_num_field] = "sum"

        fire_area_field = find_fire_area_field(total_columns, data_year)
        if fire_area_field is not None:
            fire_area_field_by_year[data_year] = fire_area_field
            agg_fields[fire_area_field] = "sum"

    total_grid_data = total_fishnet.groupby("GRID_ID").agg(agg_fields).reset_index()
    total_grid_gdf = gpd.GeoDataFrame(total_grid_data, geometry="geometry", crs="EPSG:4326")
    total_grid_gdf["GRID_ID"] = pd.to_numeric(total_grid_gdf["GRID_ID"], errors="coerce")
    total_grid_gdf = total_grid_gdf.dropna(subset=["GRID_ID"]).copy()
    total_grid_gdf["GRID_ID"] = total_grid_gdf["GRID_ID"].astype(int)

    records: list[dict[str, float | int]] = []
    valid_wui_mask = total_grid_gdf["WUI_area"] > 0

    for data_year in data_years:
        fire_num_field = f"Fire_num_{data_year}"
        if fire_num_field not in total_grid_gdf.columns:
            continue

        freq = pd.Series(np.nan, index=total_grid_gdf.index, dtype="float64")
        freq.loc[valid_wui_mask] = (
            total_grid_gdf.loc[valid_wui_mask, fire_num_field] /
            total_grid_gdf.loc[valid_wui_mask, "WUI_area"]
        )

        if data_year not in fire_area_field_by_year:
            continue

        fire_area_field = fire_area_field_by_year[data_year]
        density = pd.Series(np.nan, index=total_grid_gdf.index, dtype="float64")
        density.loc[valid_wui_mask] = (
            total_grid_gdf.loc[valid_wui_mask, fire_area_field] /
            total_grid_gdf.loc[valid_wui_mask, "WUI_area"]
        )
        density = density.clip(upper=1)

        year_df = pd.DataFrame(
            {
                "GRID_ID": total_grid_gdf["GRID_ID"].astype(int),
                "Year": data_year,
                "Total_Fire_Frequency": freq,
                "Total_Fire_Area_Density": density,
            }
        )
        year_df = year_df.dropna(
            subset=["Total_Fire_Frequency", "Total_Fire_Area_Density"],
            how="all",
        )
        if not year_df.empty:
            records.extend(year_df.to_dict("records"))

    annual_metrics = pd.DataFrame.from_records(records)
    return total_grid_gdf[["GRID_ID", "geometry"]].copy(), annual_metrics


def load_global_map() -> gpd.GeoDataFrame:
    print(f"Loading global map data: {continent_shp_path}")
    continent = gpd.read_file(continent_shp_path)
    continent = continent.to_crs("EPSG:4326")
    print(f"Successfully loaded global map with {len(continent)} features")
    return continent


def load_annual_totals() -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    fishnet_parts: list[gpd.GeoDataFrame] = []
    annual_parts: list[pd.DataFrame] = []

    for year in interface_config["years"]:
        print(f"\n{'=' * 80}")
        print(f"Loading total fire inputs for WUI year {year}")
        print(f"{'=' * 80}")

        interface_layer = f"{interface_config['layer_prefix']}{year}_{interface_config['type'].lower()}_WUI"
        intermix_layer = f"{intermix_config['layer_prefix']}{year}_{intermix_config['type'].lower()}_WUI"

        interface_fishnet = gpd.read_file(interface_config["gdb_path"], layer=interface_layer).to_crs("EPSG:4326")
        intermix_fishnet = gpd.read_file(intermix_config["gdb_path"], layer=intermix_layer).to_crs("EPSG:4326")
        interface_fishnet = ensure_grid_id(interface_fishnet)
        intermix_fishnet = ensure_grid_id(intermix_fishnet)

        print(
            f"Loaded {interface_layer}: {len(interface_fishnet)} rows; "
            f"{intermix_layer}: {len(intermix_fishnet)} rows"
        )

        fishnet_base, annual_metric = aggregate_total_grid_for_wui_year(
            wui_year=year,
            interface_fishnet=interface_fishnet,
            intermix_fishnet=intermix_fishnet,
        )

        fishnet_parts.append(fishnet_base)
        if not annual_metric.empty:
            annual_parts.append(annual_metric)
            print(
                f"Generated annual metric rows for WUI year {year}: "
                f"{len(annual_metric)}"
            )
        else:
            print(f"No annual metric rows generated for WUI year {year}")

    if not fishnet_parts:
        raise ValueError("No fishnet features were loaded.")
    if not annual_parts:
        raise ValueError("No annual fire metrics were generated.")

    fishnet_all = pd.concat(fishnet_parts, ignore_index=True)
    if not isinstance(fishnet_all, gpd.GeoDataFrame):
        fishnet_all = gpd.GeoDataFrame(fishnet_all, geometry="geometry", crs="EPSG:4326")
    fishnet_all = fishnet_all.drop_duplicates(subset=["GRID_ID"], keep="first").copy()
    fishnet_all["GRID_ID"] = pd.to_numeric(fishnet_all["GRID_ID"], errors="coerce")
    fishnet_all = fishnet_all.dropna(subset=["GRID_ID"]).copy()
    fishnet_all["GRID_ID"] = fishnet_all["GRID_ID"].astype(int)

    annual_all = pd.concat(annual_parts, ignore_index=True)
    annual_all["GRID_ID"] = pd.to_numeric(annual_all["GRID_ID"], errors="coerce")
    annual_all["Year"] = pd.to_numeric(annual_all["Year"], errors="coerce")
    annual_all = annual_all.dropna(subset=["GRID_ID", "Year"]).copy()
    annual_all["GRID_ID"] = annual_all["GRID_ID"].astype(int)
    annual_all["Year"] = annual_all["Year"].astype(int)

    print(f"Combined fishnet features after GRID_ID deduplication: {len(fishnet_all)}")
    print(
        "Combined annual metric rows: "
        f"{len(annual_all)}, years={annual_all['Year'].nunique()}, "
        f"unique GRID_ID={annual_all['GRID_ID'].nunique()}"
    )
    return fishnet_all, annual_all


def compute_mean_totals(annual_metrics: pd.DataFrame) -> pd.DataFrame:
    mean_metric = (
        annual_metrics.groupby("GRID_ID", as_index=False)
        .agg(
            Total_Fire_Frequency=("Total_Fire_Frequency", "mean"),
            Total_Fire_Area_Density=("Total_Fire_Area_Density", "mean"),
            available_years=("Year", "nunique"),
        )
    )
    print(
        "Computed mean totals: "
        f"rows={len(mean_metric)}, "
        f"GRID_ID with <20 years={int((mean_metric['available_years'] < 20).sum())}"
    )
    return mean_metric


def build_fire_frequency_colormap() -> tuple[mcolors.ListedColormap, FuncFormatter]:
    full_cmap = plt.get_cmap("YlOrRd", 10)
    colors = [full_cmap(i) for i in range(4, 10)]
    cmap = mcolors.ListedColormap(colors)
    formatter = FuncFormatter(format_original_fire_frequency)
    return cmap, formatter


def build_normalized_burned_area_colormap() -> tuple[mcolors.ListedColormap, FormatStrFormatter]:
    full_cmap = plt.get_cmap("YlOrRd", 10)
    colors = [full_cmap(i) for i in range(4, 10)]
    cmap = mcolors.ListedColormap(colors)
    formatter = FormatStrFormatter("%.2f")
    return cmap, formatter


def plot_fire_frequency_mean(continent: gpd.GeoDataFrame, fishnet_mean: gpd.GeoDataFrame) -> None:
    valid_freq = fishnet_mean["Total_Fire_Frequency"].dropna()
    positive_freq = valid_freq[valid_freq > 0]
    if positive_freq.empty:
        raise ValueError("No positive mean Fire frequency values found.")

    cmap_fire_freq, formatter_fire_freq = build_fire_frequency_colormap()
    transformed_min = log10p_transform(positive_freq.min())
    transformed_max = log10p_transform(positive_freq.max())
    boundaries = np.linspace(transformed_min, transformed_max, 7)
    norm_fire_freq = mcolors.BoundaryNorm(boundaries, cmap_fire_freq.N)
    sm_fire_freq = plt.cm.ScalarMappable(cmap=cmap_fire_freq, norm=norm_fire_freq)
    sm_fire_freq.set_array([])

    fishnet_plot = fishnet_mean.copy()
    fishnet_plot["Total_Fire_Frequency"] = (
        fishnet_plot["Total_Fire_Frequency"].replace(0, np.nan).fillna(np.nan)
    )
    positive_mask = fishnet_plot["Total_Fire_Frequency"].gt(0)
    fishnet_plot["Total_Fire_Frequency_Log10P"] = np.nan
    fishnet_plot.loc[positive_mask, "Total_Fire_Frequency_Log10P"] = log10p_transform(
        fishnet_plot.loc[positive_mask, "Total_Fire_Frequency"]
    )

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    continent.plot(ax=ax, color="#f0f0f0", edgecolor="#d0d0d0", linewidth=0.5)
    fishnet_plot.plot(
        ax=ax,
        column="Total_Fire_Frequency_Log10P",
        cmap=cmap_fire_freq,
        norm=norm_fire_freq,
        edgecolor="none",
        linewidth=0,
    )

    cbar_ax = fig.add_axes(CBAR_AX_POSITION)
    cbar = fig.colorbar(sm_fire_freq, cax=cbar_ax, orientation="vertical", label="Fire frequency")
    cbar.ax.tick_params(direction="in")
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.yaxis.label.set_size(16)
    cbar.ax.yaxis.set_major_formatter(formatter_fire_freq)

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
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 75)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_position(MAP_AX_POSITION)
    ax.set_aspect("equal", adjustable="box")

    fig.savefig(OUTPUT_FIRE_FREQUENCY, dpi=1200, format="jpg")
    plt.close(fig)
    print(f"Mean Fire frequency figure saved to: {OUTPUT_FIRE_FREQUENCY}")


def plot_fire_area_density_mean(continent: gpd.GeoDataFrame, fishnet_mean: gpd.GeoDataFrame) -> None:
    valid_density = fishnet_mean["Total_Fire_Area_Density"].dropna()
    if valid_density.empty:
        raise ValueError("No valid mean Normalized burned area values found.")

    cmap_fire_density, formatter_fire_density = build_normalized_burned_area_colormap()
    boundaries = np.linspace(valid_density.min(), valid_density.max(), 7)
    norm_fire_density = mcolors.BoundaryNorm(boundaries, cmap_fire_density.N)
    sm_fire_density = plt.cm.ScalarMappable(cmap=cmap_fire_density, norm=norm_fire_density)
    sm_fire_density.set_array([])

    fishnet_plot = fishnet_mean.copy()
    fishnet_plot["Total_Fire_Area_Density"] = (
        fishnet_plot["Total_Fire_Area_Density"].replace(0, np.nan).fillna(np.nan)
    )

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    continent.plot(ax=ax, color="#f0f0f0", edgecolor="#d0d0d0", linewidth=0.5)
    fishnet_plot.plot(
        ax=ax,
        column="Total_Fire_Area_Density",
        cmap=cmap_fire_density,
        norm=norm_fire_density,
        edgecolor="none",
        linewidth=0,
    )

    cbar_ax = fig.add_axes(CBAR_AX_POSITION)
    cbar = fig.colorbar(
        sm_fire_density,
        cax=cbar_ax,
        orientation="vertical",
        label="Normalized burned area",
    )
    cbar.ax.tick_params(direction="in")
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.yaxis.label.set_size(16)
    cbar.ax.yaxis.set_major_formatter(formatter_fire_density)

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
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 75)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_position(MAP_AX_POSITION)
    ax.set_aspect("equal", adjustable="box")

    fig.savefig(OUTPUT_FIRE_AREA_DENSITY, dpi=1200, format="jpg")
    plt.close(fig)
    print(f"Mean Normalized burned area figure saved to: {OUTPUT_FIRE_AREA_DENSITY}")


def main() -> None:
    continent = load_global_map()
    fishnet_all, annual_all = load_annual_totals()
    mean_metric = compute_mean_totals(annual_all)

    fishnet_mean = fishnet_all.merge(mean_metric, on="GRID_ID", how="inner")
    print(f"Fishnet features retained for mean mapping: {len(fishnet_mean)}")

    plot_fire_frequency_mean(continent, fishnet_mean)
    plot_fire_area_density_mean(continent, fishnet_mean)

    print("\n20-year mean total fire maps completed.")


if __name__ == "__main__":
    main()
