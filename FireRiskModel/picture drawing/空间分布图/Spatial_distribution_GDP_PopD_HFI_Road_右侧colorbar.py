#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称: GDP / HFI / PopD / Roadsd 全球 WUI 渔网空间分布图

功能简介:
    1) 读取 I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic
       文件夹下所有分类 CSV
    2) 仅保留 2005 年的 GRID_ID、GDP、HFI、PopD、Roadsd
    3) 按 GRID_ID 在全局范围内合并并聚合
    4) 读取 2005 年对应的 WUI 渔网图层 Grid_Clip_2005_WUI
    5) 分别绘制 GDP、HFI、PopD、Roadsd 的全球空间分布图，各自输出为单独图片

输入:
    - 分类建模 CSV:
      I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic
    - WUI 渔网:
      I:\Processing data\gdb集合\WUI.gdb
    - 大陆底图:
      I:\Data_huanglin\map\world continent boundary\continent.shp

输出:
    - B:\WUI\Picture\Spatial_distribution_GDP_2005.png
    - B:\WUI\Picture\Spatial_distribution_Human Footprint_2005.png
    - B:\WUI\Picture\Spatial_distribution_PopD_2005.png
    - B:\WUI\Picture\Spatial_distribution_Roadsd_2005.png
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
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False


SOURCE_DIR = r"I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic"
WUI_GDB_PATH = r"I:\Processing data\gdb集合\WUI.gdb"
WUI_LAYER = "Grid_Clip_2005_WUI"
CONTINENT_SHP_PATH = r"I:\Data_huanglin\map\world continent boundary\continent.shp"

TARGET_YEAR = 2005
OUTPUT_DIR = r"B:\WUI\Picture"
OUTPUT_DPI = 600

MAX_IDS_FOR_WHERE = 50000
WHERE_CHUNK_SIZE = 3000
CHUNK_SIZE = 250000

BASEMAP_FACE = "#ffffff"
BASEMAP_EDGE = "#000000"
NO_DATA_COLOR = "#d9d9d9"
PANEL_FACE = "#ffffff"

VARIABLE_CONFIGS = [
    {
        "column": "GDP",
        "title": "GDP",
        "cmap_name": "YlOrRd",
        "filename": f"Spatial_distribution_GDP_{TARGET_YEAR}.png",
    },
    {
        "column": "HFI",
        "title": "Human Footprint",
        "cmap_name": "YlGnBu",
        "filename": f"Spatial_distribution_Human Footprint_{TARGET_YEAR}.png",
    },
    {
        "column": "PopD",
        "title": "Population density",
        "cmap_name": "OrRd",
        "filename": f"Spatial_distribution_PopD_{TARGET_YEAR}.png",
    },
    {
        "column": "Roadsd",
        "title": "Road density",
        "cmap_name": "PuBuGn",
        "filename": f"Spatial_distribution_Roadsd_{TARGET_YEAR}.png",
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


def load_metrics_for_target_year(source_dir: str, target_year: int) -> pd.DataFrame:
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
        required_metrics = ["GDP", "HFI", "PopD", "Roadsd"]
        missing_metrics = [col for col in required_metrics if col not in cols]
        missing = [name for name, value in {"GRID_ID": grid_col, "Year": year_col}.items() if value is None]
        if missing or missing_metrics:
            raise ValueError(
                f"Missing required column(s) {missing + missing_metrics} in: {csv_path}"
            )

        for chunk in pd.read_csv(
            csv_path,
            usecols=[grid_col, year_col, "GDP", "HFI", "PopD", "Roadsd"],
            dtype={grid_col: "string"},
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            chunk = chunk.rename(columns={grid_col: "GRID_ID", year_col: "Year"})
            chunk["Year"] = pd.to_numeric(chunk["Year"], errors="coerce")
            chunk["GRID_ID"] = pd.to_numeric(chunk["GRID_ID"], errors="coerce")
            chunk = chunk[chunk["Year"] == target_year]
            if chunk.empty:
                continue

            for col in ["GDP", "HFI", "PopD", "Roadsd"]:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

            chunk = chunk.dropna(subset=["GRID_ID"])
            if chunk.empty:
                continue

            chunk["GRID_ID"] = chunk["GRID_ID"].astype(int)
            parts.append(chunk[["GRID_ID", "GDP", "HFI", "PopD", "Roadsd"]].copy())

    if not parts:
        raise ValueError(f"No records found for target year {target_year} in: {source_dir}")

    combined = pd.concat(parts, ignore_index=True)
    combined = (
        combined.groupby("GRID_ID", as_index=False)
        .agg(
            GDP=("GDP", "mean"),
            HFI=("HFI", "mean"),
            PopD=("PopD", "mean"),
            Roadsd=("Roadsd", "mean"),
        )
        .sort_values("GRID_ID")
        .reset_index(drop=True)
    )

    print(
        "Loaded target-year metrics: "
        f"year={target_year}, rows={len(combined)}, "
        f"unique GRID_ID={combined['GRID_ID'].nunique()}"
    )
    return combined


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
        except ValueError as exc:
            msg = str(exc)
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


def plot_variable_map(
    *,
    ax: plt.Axes,
    continent: gpd.GeoDataFrame,
    fishnet: gpd.GeoDataFrame,
    value_col: str,
    title: str,
    cmap_name: str,
) -> None:
    continent.plot(ax=ax, color=BASEMAP_FACE, edgecolor=BASEMAP_EDGE, linewidth=0.25)
    ax.set_facecolor(PANEL_FACE)

    has_value = fishnet[value_col].notna()
    fishnet_no_data = fishnet[~has_value]
    fishnet_data = fishnet[has_value].copy()

    if not fishnet_no_data.empty:
        fishnet_no_data.plot(ax=ax, color=NO_DATA_COLOR, edgecolor="none", linewidth=0)

    if not fishnet_data.empty:
        norm = build_robust_norm(fishnet_data[value_col])
        clipped_col = f"{value_col}_plot"
        fishnet_data[clipped_col] = fishnet_data[value_col].clip(lower=norm.vmin, upper=norm.vmax)
        fishnet_data.plot(
            ax=ax,
            column=clipped_col,
            cmap=plt.get_cmap(cmap_name),
            norm=norm,
            edgecolor="none",
            linewidth=0,
        )

        sm = ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap_name))
        sm.set_array([])
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes("right", size="3%", pad=0.18)
        cbar = plt.colorbar(
            sm,
            cax=cbar_ax,
            orientation="vertical",
        )
        cbar.set_label(title, fontsize=14, fontweight="bold", labelpad=8)
        formatter = ScalarFormatter(useMathText=value_col != "HFI")
        if value_col == "HFI":
            formatter.set_scientific(False)
            formatter.set_useOffset(False)
        else:
            formatter.set_powerlimits((0, 0))
        cbar.formatter = formatter
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=9, direction="in")
    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=13)

    continent.boundary.plot(ax=ax, color=BASEMAP_EDGE, linewidth=0.25)
    ax.set_xlabel("Longitude", fontsize=16)
    ax.set_ylabel("Latitude", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 75)
    ax.grid(False)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading 2005 metrics from: {SOURCE_DIR}")
    metrics_df = load_metrics_for_target_year(SOURCE_DIR, TARGET_YEAR)

    print(f"Loading world base map: {CONTINENT_SHP_PATH}")
    continent = gpd.read_file(CONTINENT_SHP_PATH).to_crs("EPSG:4326")
    print(f"Base map loaded: {len(continent)} feature(s), CRS={continent.crs}")

    grid_ids = metrics_df["GRID_ID"].astype(int).drop_duplicates().sort_values().tolist()
    print(f"Loading WUI fishnet layer: {WUI_LAYER}")
    fishnet = read_wui_fishnet_filtered(WUI_GDB_PATH, WUI_LAYER, grid_ids)
    if "GRID_ID" not in fishnet.columns:
        raise ValueError(f"GRID_ID field not found in WUI fishnet layer: {WUI_LAYER}")

    fishnet["GRID_ID"] = pd.to_numeric(fishnet["GRID_ID"], errors="coerce")
    fishnet = fishnet.dropna(subset=["GRID_ID"]).copy()
    fishnet["GRID_ID"] = fishnet["GRID_ID"].astype(int)
    fishnet = fishnet.to_crs("EPSG:4326")
    fishnet = fishnet.merge(metrics_df, on="GRID_ID", how="left")

    for config in VARIABLE_CONFIGS:
        fig, ax = plt.subplots(figsize=(15, 10), facecolor="white")
        plot_variable_map(
            ax=ax,
            continent=continent,
            fishnet=fishnet,
            value_col=config["column"],
            title=config["title"],
            cmap_name=config["cmap_name"],
        )
        fig.tight_layout()

        output_path = os.path.join(OUTPUT_DIR, config["filename"])
        fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()


