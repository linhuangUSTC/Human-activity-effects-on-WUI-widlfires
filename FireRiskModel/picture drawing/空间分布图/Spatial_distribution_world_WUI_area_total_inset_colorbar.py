#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
程序名称: WUI总面积空间分布绘制程序（内嵌色标样式版）
功能简介: 绘制全球范围内2005、2010、2015、2020年的WUI总面积空间分布图

输入文件：
    - 大陆边界数据:I:/Data_huanglin/map/world continent boundary/continent.shp
    - Interface WUI数据:I:/Processing data/gdb集合/Interface_WUI_Fire_Risk.gdb
    - Intermix WUI数据:I:/Processing data/gdb集合/Intermix_WUI_Fire_Risk.gdb

输出文件：
    - 空间分布图:B:/WUI/Picture/Total_WUI_area_YYYY.jpg

主要功能：
    1. 计算每个渔网栅格的Interface和Intermix WUI面积总和
    2. 获取所有渔网栅格的并集，确保完整绘制
    3. 保持原有展示内容不变，仅调整图形样式
    4. 将横向colorbar嵌入地图左下角，仅将colorbar标签显示改为百分比
    5. 输出图片尺寸、坐标范围和图框内部样式与“Spatial_distribution_主导WUI类型.py”一致
"""

from __future__ import annotations

import os

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

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

MAP_LON_MIN = -180.0
MAP_LON_MAX = 180.0
MAP_LAT_MIN = -60.0
MAP_LAT_MAX = 75.0
COLORBAR_LEFT_SHIFT_DEGREES = 20.0
COLORBAR_UP_SHIFT_DEGREES = 10.0
COLORBAR_PERCENT_DENOMINATOR = 85.93


def load_world_continent() -> gpd.GeoDataFrame:
    print(f"Loading global map data: {continent_shp_path}")
    continent = gpd.read_file(continent_shp_path)
    continent = continent.to_crs("EPSG:4326")
    print(f"Successfully loaded global map with {len(continent)} features")
    print(f"Map data CRS: {continent.crs}")
    return continent


def build_total_wui_gdf(year: int) -> gpd.GeoDataFrame:
    interface_layer = f"{interface_config['layer_prefix']}{year}_{interface_config['type'].lower()}_WUI"
    intermix_layer = f"{intermix_config['layer_prefix']}{year}_{intermix_config['type'].lower()}_WUI"

    print(f"Loading Interface fishnet data layer: {interface_layer}")
    interface_fishnet = gpd.read_file(interface_config["gdb_path"], layer=interface_layer)
    print(f"Successfully loaded Interface fishnet data with {len(interface_fishnet)} features")

    print(f"Loading Intermix fishnet data layer: {intermix_layer}")
    intermix_fishnet = gpd.read_file(intermix_config["gdb_path"], layer=intermix_layer)
    print(f"Successfully loaded Intermix fishnet data with {len(intermix_fishnet)} features")

    print("Unifying coordinate systems...")
    interface_fishnet = interface_fishnet.to_crs("EPSG:4326")
    intermix_fishnet = intermix_fishnet.to_crs("EPSG:4326")
    print(f"Interface fishnet CRS: {interface_fishnet.crs}")
    print(f"Intermix fishnet CRS: {intermix_fishnet.crs}")

    print("Merging fishnet grids and calculating total WUI area...")
    total_fishnet = gpd.GeoDataFrame(pd.concat([interface_fishnet, intermix_fishnet], ignore_index=True))

    if "GRID_ID" not in total_fishnet.columns:
        total_fishnet["GRID_ID"] = total_fishnet.geometry.apply(lambda geom: geom.centroid.wkt)

    total_wui_area = (
        total_fishnet.groupby("GRID_ID")
        .agg(
            geometry=("geometry", "first"),
            WUI_area=("WUI_area", "sum"),
        )
        .reset_index()
    )
    total_fishnet_gdf = gpd.GeoDataFrame(total_wui_area, geometry="geometry", crs="EPSG:4326")
    print(f"Created total WUI fishnet with {len(total_fishnet_gdf)} features")
    return total_fishnet_gdf


def draw_inset_colorbar(fig: plt.Figure, ax: plt.Axes, scalar_mappable) -> None:
    left_anchor = max(0.01, 0.085 - COLORBAR_LEFT_SHIFT_DEGREES / (MAP_LON_MAX - MAP_LON_MIN))
    bottom_anchor = 0.11 + COLORBAR_UP_SHIFT_DEGREES / (MAP_LAT_MAX - MAP_LAT_MIN)
    cax = inset_axes(
        ax,
        width="25%",
        height="4.0%",
        loc="lower left",
        bbox_to_anchor=(left_anchor, bottom_anchor, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar = fig.colorbar(scalar_mappable, cax=cax, orientation="horizontal")
    vmin, vmax = scalar_mappable.get_clim()
    tick_values = np.linspace(vmin, vmax, 6)
    cbar.set_ticks(tick_values)
    cbar.ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda value, _pos: f"{int(round(value / COLORBAR_PERCENT_DENOMINATOR * 100.0))}"
        )
    )
    cbar.ax.minorticks_off()
    cbar.set_label("WUI area proportion (%)", fontsize=18, labelpad=10)
    cbar.ax.xaxis.set_label_position("bottom")
    cbar.ax.tick_params(axis="x", which="major", direction="in", labelsize=17, pad=3, length=6)
    cbar.outline.set_visible(False)


def draw_total_wui_map(continent: gpd.GeoDataFrame, total_fishnet_gdf: gpd.GeoDataFrame, year: int) -> None:
    fig, ax = plt.subplots(figsize=(15, 10))

    print("Drawing global map base layer...")
    continent.plot(ax=ax, color="white", edgecolor="#d0d0d0", linewidth=0.5)
    ax.set_facecolor("white")

    print("Configuring fishnet color mapping...")
    wui_area_min = total_fishnet_gdf["WUI_area"].min()
    wui_area_max = total_fishnet_gdf["WUI_area"].max()
    print(f"Total WUI_area range: {wui_area_min} to {wui_area_max}")

    cmap = plt.get_cmap("viridis_r", 10)
    if wui_area_min == wui_area_max:
        boundaries = np.linspace(wui_area_min - 0.5, wui_area_max + 0.5, 11)
    else:
        boundaries = np.linspace(wui_area_min, wui_area_max, 11)
    norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    sm.set_clim(boundaries[0], boundaries[-1])

    print("Drawing total fishnet data...")
    total_fishnet_gdf.plot(
        ax=ax,
        column="WUI_area",
        cmap=cmap,
        norm=norm,
        edgecolor="none",
        linewidth=0,
        alpha=1.0,
    )

    print("Creating inset colorbar...")
    draw_inset_colorbar(fig, ax, sm)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xlim([MAP_LON_MIN, MAP_LON_MAX])
    ax.set_ylim([MAP_LAT_MIN, MAP_LAT_MAX])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    output_filename = f"Total_WUI_area_{year}.jpg"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=1200, bbox_inches="tight", format="jpg")
    print(f"Figure successfully saved to: {output_path}")
    plt.close(fig)


def main() -> None:
    continent = load_world_continent()

    for year in interface_config["years"]:
        print(f"\n{'=' * 50}")
        print(f"Processing Total WUI data for year {year}")
        print(f"{'=' * 50}")

        try:
            total_fishnet_gdf = build_total_wui_gdf(year)
            draw_total_wui_map(continent, total_fishnet_gdf, year)
        except Exception as exc:
            print(f"Error processing Total WUI data for year {year}: {exc}")
            continue

    print("\nAll processing completed!")


if __name__ == "__main__":
    main()
