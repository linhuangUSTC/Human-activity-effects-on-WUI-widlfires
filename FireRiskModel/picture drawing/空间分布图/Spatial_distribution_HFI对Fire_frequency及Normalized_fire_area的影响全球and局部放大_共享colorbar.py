#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称: Human activity对Fire frequency及Normalized burned area的影响全球与局部放大图（共享全球色标，多年均值）

功能简介:
    1) 分别读取 HFI 对 Fire frequency 与 Normalized burned area 影响的 effect CSV
    2) 使用与全球图脚本相同的数据读取、WUI 渔网匹配与显著性着色逻辑
    3) 对每个 GRID_ID，基于 2003-2022 全部可用年份计算 delta_contribution 的均值；
       若某格网不足 20 年，则仅对其可用年份求均值。
    4) 显著性沿用当前脚本口径：只要某格网在任一可用年份中 delta_sig == TRUE，
       则多年均值图中该格网记为显著；否则记为不显著。
    5) 每个指标输出 3 张图：
       - 1 张全球多年均值图（带 colorbar）
       - 2 张局部放大多年均值图（不单独绘制 colorbar）

输入:
    - B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num
    - B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area
    - I:\Processing data\gdb集合\WUI.gdb
    - I:\Data_huanglin\map\world continent boundary\continent.shp

输出:
    - B:\WUI\Picture\Human Footprint对Fire_frequency的影响全球分布图_共享色标_2003_2022_mean.jpg
    - B:\WUI\Picture\Human Footprint对Fire_frequency的影响局部放大_共享全球色标_2003_2022_mean_a.jpg
    - B:\WUI\Picture\Human Footprint对Fire_frequency的影响局部放大_共享全球色标_2003_2022_mean_b.jpg
    - B:\WUI\Picture\Human Footprint对Normalized_burn_area的影响全球分布图_共享色标_2003_2022_mean.jpg
    - B:\WUI\Picture\Human Footprint对Normalized_burn_area的影响局部放大_共享全球色标_2003_2022_mean_a.jpg
    - B:\WUI\Picture\Human Footprint对Normalized_burn_area的影响局部放大_共享全球色标_2003_2022_mean_b.jpg
"""

from __future__ import annotations

import glob
import math
import os
from typing import Iterable

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle


plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False


FIRE_YEAR_START = 2003
FIRE_YEAR_END = 2022

WUI_GDB_PATH = r"I:\Processing data\gdb集合\WUI.gdb"
WUI_YEARS = [2005, 2010, 2015, 2020]
WUI_LAYER_TEMPLATE = "Grid_Clip_{wui_year}_WUI"
CONTINENT_SHP_PATH = r"I:\Data_huanglin\map\world continent boundary\continent.shp"

OUTPUT_DIR = r"B:\WUI\Picture"
OUTPUT_DPI = 2400

# 以0为分界：负值用5级蓝色，正值用5级红色，使用离散色阶而非连续色带
CMAP_NAME = "BlueRedDiscrete"
NEGATIVE_CMAP_COLORS = ["#08306B", "#08519C", "#2171B5", "#6BAED6", "#C6DBEF"]
POSITIVE_CMAP_COLORS = ["#FEE0D2", "#FC9272", "#FB6A4A", "#DE2D26", "#A50F15"]
CMAP_COLORS = NEGATIVE_CMAP_COLORS + POSITIVE_CMAP_COLORS
NO_DATA_COLOR = "#d9d9d9"
NON_SIG_COLOR = "#bdbdbd"
BASEMAP_FACE = "#ffffff"
BASEMAP_EDGE = "#000000"

MAX_IDS_FOR_WHERE = 50000
WHERE_CHUNK_SIZE = 3000

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
        "box_color": "#1e90ff",
    },
]

DATASETS = [
    {
        "metric_title": "Fire frequency",
        "effect_source": r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num",
        "global_output_filename": "Human Footprint对Fire_frequency的影响全球分布图_共享色标_2003_2022_mean.jpg",
        "local_output_filename_template": "Human Footprint对Fire_frequency的影响局部放大_共享全球色标_2003_2022_mean_{region_suffix}.jpg",
        "colorbar_label": "Human activity effects on fire frequency",
    },
    {
        "metric_title": "Normalized burned area",
        "effect_source": r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area",
        "global_output_filename": "Human Footprint对Normalized_burn_area的影响全球分布图_共享色标_2003_2022_mean.jpg",
        "local_output_filename_template": "Human Footprint对Normalized_burn_area的影响局部放大_共享全球色标_2003_2022_mean_{region_suffix}.jpg",
        "colorbar_label": "Human activity effects on normalized burned area",
    },
]


def _chunks(items: list[int], chunk_size: int) -> Iterable[list[int]]:
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def _resolve_effect_files(effect_source: str) -> list[str]:
    if os.path.isfile(effect_source):
        return [effect_source]
    if os.path.isdir(effect_source):
        return sorted(glob.glob(os.path.join(effect_source, "HFI_effect_*.csv")))
    return sorted(glob.glob(effect_source))


def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
    for name in candidates:
        if name in columns:
            return name
    return None


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    values = series.astype(str).str.strip().str.upper()
    return values.isin({"TRUE", "T", "1", "YES", "Y"})


def load_effect_data(effect_source: str, year_start: int, year_end: int) -> pd.DataFrame:
    files = _resolve_effect_files(effect_source)
    if not files:
        raise FileNotFoundError(f"No effect CSV found by: {effect_source}")

    print(f"Found {len(files)} effect CSV file(s).")
    parts: list[pd.DataFrame] = []

    for csv_path in files:
        header = pd.read_csv(csv_path, nrows=0)
        cols = header.columns.tolist()

        grid_col = _pick_column(cols, ["GRID_ID", "grid_id", "Grid_ID"])
        year_col = _pick_column(cols, ["Year", "YEAR", "year", "Fire_Year", "fire_year", "年份"])
        sig_col = _pick_column(cols, ["delta_sig", "DELTA_SIG", "Delta_Sig"])
        contrib_col = _pick_column(cols, ["delta_contribution", "diff_contribution", "delta"])

        missing = [k for k, v in {"GRID_ID": grid_col, "Year": year_col, "delta_sig": sig_col}.items() if v is None]
        if missing:
            raise ValueError(f"Missing required column(s) {missing} in: {csv_path}")
        if contrib_col is None:
            raise ValueError(f"Missing contribution column in: {csv_path}")

        df = pd.read_csv(
            csv_path,
            usecols=[grid_col, year_col, contrib_col, sig_col],
            low_memory=False,
            dtype={grid_col: "string"},
        )
        df = df.rename(
            columns={
                grid_col: "GRID_ID",
                year_col: "Year",
                contrib_col: "delta_contribution",
                sig_col: "delta_sig",
            }
        )
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        df = df[df["Year"].between(year_start, year_end)]
        if df.empty:
            continue

        df["GRID_ID"] = pd.to_numeric(df["GRID_ID"], errors="coerce").astype("Int64")
        df["delta_contribution"] = pd.to_numeric(df["delta_contribution"], errors="coerce")
        df["delta_sig"] = _coerce_bool_series(df["delta_sig"])
        df = df.dropna(subset=["GRID_ID"])
        parts.append(df[["GRID_ID", "Year", "delta_contribution", "delta_sig"]])

    if not parts:
        raise ValueError(f"No records found for Year in [{year_start}, {year_end}].")

    combined = pd.concat(parts, ignore_index=True)
    combined = (
        combined.groupby(["Year", "GRID_ID"], as_index=False)
        .agg(
            delta_contribution=("delta_contribution", "mean"),
            delta_sig=("delta_sig", "max"),
        )
        .dropna(subset=["Year", "GRID_ID"])
    )

    print(
        "Effect data loaded: "
        f"years={combined['Year'].nunique()}, "
        f"rows={len(combined)}, "
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


def compute_mean_effect(effect_all: pd.DataFrame) -> pd.DataFrame:
    mean_effect = (
        effect_all.groupby("GRID_ID", as_index=False)
        .agg(
            delta_contribution=("delta_contribution", "mean"),
            delta_sig=("delta_sig", "max"),
            available_years=("Year", "nunique"),
        )
    )
    print(
        "Computed mean effect: "
        f"rows={len(mean_effect)}, "
        f"GRID_ID with <20 years={int((mean_effect['available_years'] < (FIRE_YEAR_END - FIRE_YEAR_START + 1)).sum())}"
    )
    return mean_effect


def load_combined_wui_fishnet(effect_all: pd.DataFrame) -> gpd.GeoDataFrame:
    available_years = sorted(effect_all["Year"].dropna().astype(int).unique().tolist())
    fishnet_parts: list[gpd.GeoDataFrame] = []

    for wui_year in WUI_YEARS:
        year_group = [year for year in available_years if wui_year_for_fire_year(year) == wui_year]
        if not year_group:
            continue

        needed_ids = sorted(
            set(
                effect_all.loc[effect_all["Year"].isin(year_group), "GRID_ID"]
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


def _subset_region(gdf: gpd.GeoDataFrame, region: dict[str, float]) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    return gdf.cx[
        region["lon_min"]:region["lon_max"],
        region["lat_min"]:region["lat_max"],
    ].copy()


def build_global_norm(fishnet_year: gpd.GeoDataFrame) -> tuple[mcolors.BoundaryNorm, list[float], float]:
    has_value = fishnet_year["delta_contribution"].notna()
    fishnet_data = fishnet_year[has_value]
    if fishnet_data.empty:
        abs_max = 1
    else:
        values = fishnet_data["delta_contribution"].astype(float)
        abs_max = max(abs(values.min()), abs(values.max()))
        if pd.isna(abs_max) or abs_max == 0:
            abs_max = 1

    step = abs_max / 5
    boundaries = [(-abs_max + i * step) for i in range(6)] + [(i * step) for i in range(1, 6)]
    norm = mcolors.BoundaryNorm(boundaries, len(CMAP_COLORS), clip=True)
    return norm, boundaries, abs_max


def split_fishnet_layers(
    fishnet_year: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    has_value = fishnet_year["delta_contribution"].notna()
    fishnet_no_data = fishnet_year[~has_value]
    fishnet_data = fishnet_year[has_value]
    sig_mask = fishnet_data["delta_sig"].fillna(False).astype(bool) if not fishnet_data.empty else pd.Series(dtype=bool)
    fishnet_sig = fishnet_data[sig_mask] if not fishnet_data.empty else fishnet_data
    fishnet_non_sig = fishnet_data[~sig_mask] if not fishnet_data.empty else fishnet_data
    return fishnet_no_data, fishnet_non_sig, fishnet_sig


def _plot_single_map(
    *,
    ax: plt.Axes,
    continent: gpd.GeoDataFrame,
    fishnet_year: gpd.GeoDataFrame,
    norm: mcolors.BoundaryNorm,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    show_axis_labels: bool,
    continent_linewidth: float,
) -> None:
    cmap = mcolors.ListedColormap(CMAP_COLORS, name=CMAP_NAME)
    fishnet_no_data, fishnet_non_sig, fishnet_sig = split_fishnet_layers(fishnet_year)

    continent.plot(ax=ax, color=BASEMAP_FACE, edgecolor=BASEMAP_EDGE, linewidth=continent_linewidth)

    if not fishnet_no_data.empty:
        fishnet_no_data.plot(ax=ax, color=NO_DATA_COLOR, edgecolor="none", linewidth=0)
    if not fishnet_non_sig.empty:
        fishnet_non_sig.plot(ax=ax, color=NON_SIG_COLOR, edgecolor="none", linewidth=0)
    if not fishnet_sig.empty:
        fishnet_sig.plot(
            ax=ax,
            column="delta_contribution",
            cmap=cmap,
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

def plot_global_mean_map(
    *,
    continent: gpd.GeoDataFrame,
    fishnet_mean: gpd.GeoDataFrame,
    norm: mcolors.BoundaryNorm,
    boundaries: list[float],
    abs_max: float,
    dataset: dict[str, str],
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(15, 10), facecolor="white")

    _plot_single_map(
        ax=ax,
        continent=continent,
        fishnet_year=fishnet_mean,
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
            linewidth=1.0,
            linestyle="-",
            zorder=10,
        )
        ax.add_patch(rect)

    cmap = mcolors.ListedColormap(CMAP_COLORS, name=CMAP_NAME)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(
        sm,
        ax=ax,
        shrink=0.72,
        aspect=35,
        fraction=0.05,
        pad=0.08,
        orientation="horizontal",
        label=dataset["colorbar_label"],
        drawedges=True,
    )
    exponent = int(math.floor(math.log10(abs_max))) if abs_max > 0 else 0
    scale = 10 ** exponent
    cbar.set_ticks(boundaries)
    cbar.ax.set_xticklabels([f"{value / scale:.2f}" for value in boundaries])
    cbar.ax.text(
        1.01,
        0.5,
        f"1e{exponent}",
        transform=cbar.ax.transAxes,
        ha="left",
        va="center",
        fontsize=18,
    )
    cbar.ax.tick_params(direction="in")
    cbar.ax.tick_params(labelsize=18)
    cbar.ax.xaxis.label.set_size(20)

    plt.tight_layout()
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight", format="jpg")
    plt.close(fig)
    print(f"Global mean figure saved to: {output_path}")


def plot_local_mean_zoom_map(
    *,
    continent: gpd.GeoDataFrame,
    fishnet_mean: gpd.GeoDataFrame,
    norm: mcolors.BoundaryNorm,
    region: dict[str, float | str],
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 7.2), facecolor="white")
    continent_region = _subset_region(continent, region)
    fishnet_region = _subset_region(fishnet_mean, region)
    if fishnet_region is None or fishnet_region.empty:
        fishnet_region = fishnet_mean.iloc[0:0].copy()

    _plot_single_map(
        ax=ax,
        continent=continent_region,
        fishnet_year=fishnet_region,
        norm=norm,
        xlim=(float(region["lon_min"]), float(region["lon_max"])),
        ylim=(float(region["lat_min"]), float(region["lat_max"])),
        show_axis_labels=False,
        continent_linewidth=0.35,
    )

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.96)
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight", format="jpg")
    plt.close(fig)
    print(f"Local mean figure saved to: {output_path}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading world base map: {CONTINENT_SHP_PATH}")
    continent = gpd.read_file(CONTINENT_SHP_PATH).to_crs("EPSG:4326")
    print(f"Base map loaded: {len(continent)} feature(s), CRS={continent.crs}")

    for dataset in DATASETS:
        print(f"\n{'=' * 80}")
        print(f"Processing dataset: {dataset['metric_title']}")
        print(f"Loading effect CSV(s) from: {dataset['effect_source']}")

        effect_all = load_effect_data(dataset["effect_source"], FIRE_YEAR_START, FIRE_YEAR_END)
        effect_all["Year"] = effect_all["Year"].astype(int)
        effect_all["GRID_ID"] = effect_all["GRID_ID"].astype(int)
        fishnet_all = load_combined_wui_fishnet(effect_all)
        mean_effect = compute_mean_effect(effect_all)
        mean_ids = set(mean_effect["GRID_ID"].astype(int).tolist())
        fishnet_mean = fishnet_all[fishnet_all["GRID_ID"].isin(mean_ids)].copy()
        fishnet_mean = fishnet_mean.merge(mean_effect, on="GRID_ID", how="left")
        norm, boundaries, abs_max = build_global_norm(fishnet_mean)

        global_output_path = os.path.join(OUTPUT_DIR, str(dataset["global_output_filename"]))
        plot_global_mean_map(
            continent=continent,
            fishnet_mean=fishnet_mean,
            norm=norm,
            boundaries=boundaries,
            abs_max=abs_max,
            dataset=dataset,
            output_path=global_output_path,
        )

        for region in REGIONS:
            local_output_path = os.path.join(
                OUTPUT_DIR,
                str(dataset["local_output_filename_template"]).format(
                    region_suffix=str(region["label"]).lower()
                ),
            )
            plot_local_mean_zoom_map(
                continent=continent,
                fishnet_mean=fishnet_mean,
                norm=norm,
                region=region,
                output_path=local_output_path,
            )


if __name__ == "__main__":
    main()


