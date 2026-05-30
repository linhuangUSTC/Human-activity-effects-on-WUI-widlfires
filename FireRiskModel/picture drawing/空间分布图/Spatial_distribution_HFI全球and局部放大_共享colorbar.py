#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称: HFI 全球与局部放大图（共享全球色标，多年均值）

功能简介:
    1) 读取 I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic
       文件夹下全部分类 CSV。
    2) 提取 GRID_ID、Year、HFI，并保留 2003-2022 年。
    3) 先按 Year + GRID_ID 聚合 HFI，再按 GRID_ID 计算 2003-2022 多年均值；
       若某格网不足 20 年，则仅对其可用年份求均值。
    4) 根据火灾年份与 WUI 年份对应关系，合并 2005 / 2010 / 2015 / 2020 四套 WUI 渔网图层。
    5) 输出 3 张图：
       - 1 张全球多年均值图（带 colorbar）
       - 2 张局部放大多年均值图（不单独绘制 colorbar）
    6) 局部图与全球图共用同一个 colormap 范围。

输入:
    - I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic
    - I:\Processing data\gdb集合\WUI.gdb
    - I:\Data_huanglin\map\world continent boundary\continent.shp

输出:
    - B:\WUI\Picture\Spatial_distribution_Human Footprint_global_2003_2022_mean.png
    - B:\WUI\Picture\Spatial_distribution_Human Footprint_local_zoom_2003_2022_mean_a.png
    - B:\WUI\Picture\Spatial_distribution_Human Footprint_local_zoom_2003_2022_mean_b.png
"""

from __future__ import annotations

import glob
import os
from typing import Iterable

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter


plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False


SOURCE_DIR = r"I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic"
WUI_GDB_PATH = r"I:\Processing data\gdb集合\WUI.gdb"
WUI_YEARS = [2005, 2010, 2015, 2020]
WUI_LAYER_TEMPLATE = "Grid_Clip_{wui_year}_WUI"
CONTINENT_SHP_PATH = r"I:\Data_huanglin\map\world continent boundary\continent.shp"

FIRE_YEAR_START = 2003
FIRE_YEAR_END = 2022

OUTPUT_DIR = r"B:\WUI\Picture"
GLOBAL_OUTPUT_FILENAME = "Spatial_distribution_Human Footprint_global_2003_2022_mean.png"
LOCAL_OUTPUT_FILENAME_TEMPLATE = "Spatial_distribution_Human Footprint_local_zoom_2003_2022_mean_{region_suffix}.png"
OUTPUT_DPI = 600

MAX_IDS_FOR_WHERE = 50000
WHERE_CHUNK_SIZE = 3000
CHUNK_SIZE = 250000

BASEMAP_FACE = "#ffffff"
BASEMAP_EDGE = "#000000"
NO_DATA_COLOR = "#d9d9d9"
PANEL_FACE = "#ffffff"
CMAP_NAME = "YlGnBu"

REGIONS = [
    {
        "label": "A",
        "name": "Region A: Northern India",
        "lon_min": 74.0,
        "lon_max": 91.0,
        "lat_min": 19.0,
        "lat_max": 31.0,
        "box_color": "#d73027",
    },
    {
        "label": "B",
        "name": "Region B: East-Central China",
        "lon_min": 105.0,
        "lon_max": 122.0,
        "lat_min": 24.0,
        "lat_max": 36.0,
        "box_color": "#4575b4",
    },
]


def _chunks(items: list[int], chunk_size: int) -> Iterable[list[int]]:
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
    for name in candidates:
        if name in columns:
            return name
    return None


def load_hfi_data(source_dir: str, year_start: int, year_end: int) -> pd.DataFrame:
    csv_files = sorted(glob.glob(os.path.join(source_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {source_dir}")

    print(f"Reading {len(csv_files)} category CSV file(s) from: {source_dir}")
    parts: list[pd.DataFrame] = []

    for csv_path in csv_files:
        header = pd.read_csv(csv_path, nrows=0)
        cols = header.columns.tolist()

        grid_col = _pick_column(cols, ["GRID_ID", "grid_id", "Grid_ID"])
        year_col = _pick_column(cols, ["Year", "YEAR", "year"])
        missing = [name for name, value in {"GRID_ID": grid_col, "Year": year_col}.items() if value is None]
        if missing or "HFI" not in cols:
            required = missing.copy()
            if "HFI" not in cols:
                required.append("HFI")
            raise ValueError(f"Missing required column(s) {required} in: {csv_path}")

        for chunk in pd.read_csv(
            csv_path,
            usecols=[grid_col, year_col, "HFI"],
            dtype={grid_col: "string"},
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            chunk = chunk.rename(columns={grid_col: "GRID_ID", year_col: "Year"})
            chunk["Year"] = pd.to_numeric(chunk["Year"], errors="coerce").astype("Int64")
            chunk["GRID_ID"] = pd.to_numeric(chunk["GRID_ID"], errors="coerce").astype("Int64")
            chunk["HFI"] = pd.to_numeric(chunk["HFI"], errors="coerce")
            chunk = chunk[chunk["Year"].between(year_start, year_end)]
            chunk = chunk.dropna(subset=["GRID_ID"])
            if chunk.empty:
                continue

            chunk["GRID_ID"] = chunk["GRID_ID"].astype(int)
            parts.append(chunk[["GRID_ID", "Year", "HFI"]].copy())

    if not parts:
        raise ValueError(f"No records found for Years in [{year_start}, {year_end}] in: {source_dir}")

    combined = pd.concat(parts, ignore_index=True)
    combined = (
        combined.groupby(["Year", "GRID_ID"], as_index=False)
        .agg(HFI=("HFI", "mean"))
        .dropna(subset=["Year", "GRID_ID"])
    )

    print(
        "Loaded HFI data: "
        f"years={combined['Year'].nunique()}, "
        f"rows={len(combined)}, "
        f"unique GRID_ID={combined['GRID_ID'].nunique()}"
    )
    return combined


def compute_mean_hfi(hfi_all: pd.DataFrame) -> pd.DataFrame:
    mean_hfi = (
        hfi_all.groupby("GRID_ID", as_index=False)
        .agg(
            HFI=("HFI", "mean"),
            available_years=("Year", "nunique"),
        )
    )
    print(
        "Computed mean HFI: "
        f"rows={len(mean_hfi)}, "
        f"GRID_ID with <20 years={int((mean_hfi['available_years'] < (FIRE_YEAR_END - FIRE_YEAR_START + 1)).sum())}"
    )
    return mean_hfi


def read_wui_fishnet_filtered(
    gdb_path: str,
    layer: str,
    grid_ids: list[int],
    max_ids_for_where: int = MAX_IDS_FOR_WHERE,
    chunk_size: int = WHERE_CHUNK_SIZE,
) -> gpd.GeoDataFrame:
    if not grid_ids:
        print("No GRID_ID provided, reading full WUI fishnet layer...")
        return gpd.read_file(gdb_path, layer=layer, columns=["GRID_ID"], engine="pyogrio")

    if len(grid_ids) > max_ids_for_where:
        print(f"GRID_ID count = {len(grid_ids)} > {max_ids_for_where}, reading full WUI fishnet layer instead...")
        return gpd.read_file(gdb_path, layer=layer, columns=["GRID_ID"], engine="pyogrio")

    print(f"Reading WUI fishnet by GRID_ID filter (n={len(grid_ids)}) ...")
    current_chunk_size = max(1, int(chunk_size))
    while True:
        try:
            gdfs: list[gpd.GeoDataFrame] = []
            for chunk in _chunks(grid_ids, current_chunk_size):
                where = f"GRID_ID IN ({','.join(map(str, chunk))})"
                gdfs.append(
                    gpd.read_file(gdb_path, layer=layer, where=where, columns=["GRID_ID"], engine="pyogrio")
                )

            fishnet = pd.concat(gdfs, ignore_index=True)
            if isinstance(fishnet, gpd.GeoDataFrame):
                return fishnet
            return gpd.GeoDataFrame(fishnet, geometry="geometry", crs=gdfs[0].crs)
        except ValueError as exc:
            if "Invalid SQL query" in str(exc) and current_chunk_size > 1:
                current_chunk_size = max(1, current_chunk_size // 2)
                continue
            raise


def wui_year_for_fire_year(fire_year: int) -> int:
    if fire_year < FIRE_YEAR_START or fire_year > FIRE_YEAR_END:
        raise ValueError(f"Fire Year {fire_year} out of range [{FIRE_YEAR_START}, {FIRE_YEAR_END}].")

    base_wui_year = min(WUI_YEARS)
    wui_year = base_wui_year + 5 * ((fire_year - FIRE_YEAR_START) // 5)
    if wui_year not in WUI_YEARS:
        raise ValueError(f"No WUI layer configured for fire year {fire_year} (computed WUI year {wui_year}).")
    return wui_year


def load_combined_wui_fishnet(hfi_all: pd.DataFrame) -> gpd.GeoDataFrame:
    available_years = sorted(hfi_all["Year"].dropna().astype(int).unique().tolist())
    fishnet_parts: list[gpd.GeoDataFrame] = []

    for wui_year in WUI_YEARS:
        year_group = [year for year in available_years if wui_year_for_fire_year(year) == wui_year]
        if not year_group:
            continue

        needed_ids = sorted(
            set(
                hfi_all.loc[hfi_all["Year"].isin(year_group), "GRID_ID"]
                .dropna()
                .astype(int)
                .tolist()
            )
        )
        layer = WUI_LAYER_TEMPLATE.format(wui_year=wui_year)
        print(f"Loading WUI layer {layer} for years {year_group[0]}-{year_group[-1]} ...")
        fishnet_base = read_wui_fishnet_filtered(WUI_GDB_PATH, layer, needed_ids)
        if "GRID_ID" not in fishnet_base.columns:
            raise ValueError(f"GRID_ID field not found in WUI fishnet layer: {layer}")

        fishnet_base["GRID_ID"] = pd.to_numeric(fishnet_base["GRID_ID"], errors="coerce")
        fishnet_base = fishnet_base.dropna(subset=["GRID_ID"]).copy()
        fishnet_base["GRID_ID"] = fishnet_base["GRID_ID"].astype(int)
        fishnet_base = fishnet_base.to_crs("EPSG:4326")
        fishnet_parts.append(fishnet_base)
        print(f"Loaded {layer}: {len(fishnet_base)} features")

    if not fishnet_parts:
        raise ValueError("No WUI fishnet features were loaded for the available years.")

    fishnet_all = pd.concat(fishnet_parts, ignore_index=True)
    if not isinstance(fishnet_all, gpd.GeoDataFrame):
        fishnet_all = gpd.GeoDataFrame(fishnet_all, geometry="geometry", crs=fishnet_parts[0].crs)
    fishnet_all = fishnet_all.drop_duplicates(subset=["GRID_ID"], keep="first").copy()
    print(f"Combined WUI fishnet features after GRID_ID deduplication: {len(fishnet_all)}")
    return fishnet_all


def build_robust_norm(values: pd.Series) -> mcolors.Normalize:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    if valid.empty:
        return mcolors.Normalize(vmin=0.0, vmax=1.0)

    vmin = float(valid.min())
    vmax = float(valid.quantile(0.98))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(valid.max())
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    if vmin >= 0:
        vmin = 0.0
    return mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)


def subset_region(gdf: gpd.GeoDataFrame, region: dict[str, float]) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    return gdf.cx[region["lon_min"]:region["lon_max"], region["lat_min"]:region["lat_max"]].copy()


def split_hfi_layers(
    fishnet: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    has_value = fishnet["HFI"].notna()
    return fishnet[~has_value], fishnet[has_value].copy()


def _plot_single_map(
    *,
    ax: plt.Axes,
    continent: gpd.GeoDataFrame,
    fishnet: gpd.GeoDataFrame,
    norm: mcolors.Normalize,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    show_axis_labels: bool,
    continent_linewidth: float,
) -> None:
    fishnet_no_data, fishnet_data = split_hfi_layers(fishnet)

    continent.plot(ax=ax, color=BASEMAP_FACE, edgecolor=BASEMAP_EDGE, linewidth=continent_linewidth)
    ax.set_facecolor(PANEL_FACE)

    if not fishnet_no_data.empty:
        fishnet_no_data.plot(ax=ax, color=NO_DATA_COLOR, edgecolor="none", linewidth=0)
    if not fishnet_data.empty:
        plot_col = "HFI_plot"
        fishnet_data = fishnet_data.copy()
        fishnet_data[plot_col] = fishnet_data["HFI"].clip(lower=norm.vmin, upper=norm.vmax)
        fishnet_data.plot(
            ax=ax,
            column=plot_col,
            cmap=plt.get_cmap(CMAP_NAME),
            norm=norm,
            edgecolor="none",
            linewidth=0,
        )

    continent.boundary.plot(ax=ax, color=BASEMAP_EDGE, linewidth=continent_linewidth)
    if show_axis_labels:
        ax.set_xlabel("Longitude", fontsize=16)
        ax.set_ylabel("Latitude", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(
            axis="both",
            which="both",
            length=0,
            labelbottom=False,
            labelleft=False,
            labeltop=False,
            labelright=False,
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(False)


def plot_global_mean_hfi_map(
    *,
    continent: gpd.GeoDataFrame,
    fishnet: gpd.GeoDataFrame,
    norm: mcolors.Normalize,
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(15, 10), facecolor="white")

    _plot_single_map(
        ax=ax,
        continent=continent,
        fishnet=fishnet,
        norm=norm,
        xlim=(-180, 180),
        ylim=(-60, 75),
        show_axis_labels=False,
        continent_linewidth=0.25,
    )

    for region in REGIONS:
        rect = Rectangle(
            (float(region["lon_min"]), float(region["lat_min"])),
            float(region["lon_max"]) - float(region["lon_min"]),
            float(region["lat_max"]) - float(region["lat_min"]),
            fill=False,
            edgecolor=str(region["box_color"]),
            linewidth=1.1,
            linestyle="-",
            zorder=10,
        )
        ax.add_patch(rect)

    sm = ScalarMappable(norm=norm, cmap=plt.get_cmap(CMAP_NAME))
    sm.set_array([])
    cbar = plt.colorbar(
        sm,
        ax=ax,
        shrink=0.72,
        aspect=35,
        fraction=0.05,
        pad=0.08,
        orientation="horizontal",
        label="Human Footprint",
    )
    formatter = ScalarFormatter(useMathText=False)
    formatter.set_scientific(False)
    formatter.set_useOffset(False)
    cbar.formatter = formatter
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=18, direction="in")
    cbar.ax.xaxis.label.set_size(20)
    cbar.ax.xaxis.get_offset_text().set_fontsize(18)

    fig.tight_layout()
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_local_mean_hfi_map(
    *,
    continent: gpd.GeoDataFrame,
    fishnet: gpd.GeoDataFrame,
    norm: mcolors.Normalize,
    region: dict[str, float | str],
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 7.2), facecolor="white")

    continent_region = subset_region(continent, region)
    fishnet_region = subset_region(fishnet, region)
    if fishnet_region is None or fishnet_region.empty:
        fishnet_region = fishnet.iloc[0:0].copy()

    _plot_single_map(
        ax=ax,
        continent=continent_region,
        fishnet=fishnet_region,
        norm=norm,
        xlim=(float(region["lon_min"]), float(region["lon_max"])),
        ylim=(float(region["lat_min"]), float(region["lat_max"])),
        show_axis_labels=False,
        continent_linewidth=0.35,
    )

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.96)
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading HFI from: {SOURCE_DIR}")
    hfi_all = load_hfi_data(SOURCE_DIR, FIRE_YEAR_START, FIRE_YEAR_END)

    print(f"Loading world base map: {CONTINENT_SHP_PATH}")
    continent = gpd.read_file(CONTINENT_SHP_PATH).to_crs("EPSG:4326")
    print(f"Base map loaded: {len(continent)} feature(s), CRS={continent.crs}")

    fishnet_all = load_combined_wui_fishnet(hfi_all)
    mean_hfi = compute_mean_hfi(hfi_all)
    mean_ids = set(mean_hfi["GRID_ID"].astype(int).tolist())
    fishnet_mean = fishnet_all[fishnet_all["GRID_ID"].isin(mean_ids)].copy()
    fishnet_mean = fishnet_mean.merge(mean_hfi, on="GRID_ID", how="left")

    _, fishnet_data = split_hfi_layers(fishnet_mean)
    norm = build_robust_norm(fishnet_data["HFI"]) if not fishnet_data.empty else mcolors.Normalize(vmin=0.0, vmax=1.0)

    global_output_path = os.path.join(OUTPUT_DIR, GLOBAL_OUTPUT_FILENAME)
    plot_global_mean_hfi_map(
        continent=continent,
        fishnet=fishnet_mean,
        norm=norm,
        output_path=global_output_path,
    )
    print(f"Figure saved to: {global_output_path}")

    for region in REGIONS:
        output_path = os.path.join(
            OUTPUT_DIR,
            LOCAL_OUTPUT_FILENAME_TEMPLATE.format(region_suffix=str(region["label"]).lower()),
        )
        plot_local_mean_hfi_map(
            continent=continent,
            fishnet=fishnet_mean,
            norm=norm,
            region=region,
            output_path=output_path,
        )
        print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
