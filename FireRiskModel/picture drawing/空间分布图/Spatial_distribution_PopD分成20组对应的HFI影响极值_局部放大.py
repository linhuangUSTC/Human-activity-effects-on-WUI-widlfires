#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称: Population density 第 12 组与第 20 组对应 HFI 影响的全球与局部空间分布图

功能简介:
    1) 直接读取「将GDP_人口密度_道路密度分成20组_计算组内的HFI影响_柱状图.py」
       输出的 Grouped_Human Footprint_effect_2003_2022_group_summary.csv
    2) 读取 2005、2010、2015、2020 四个年份的 Population density 数据以及同年份 HFI 影响数据
    3) 将当年 Population density 落在第 12 组与第 20 组范围内的 WUI 格网筛出
    4) 全球图保持 PopD 分组的红/蓝栅格分布，并叠加 4 个局部放大范围框
    5) 局部图仅绘制这两组对应格网的 HFI 对 Fire frequency 的影响值，每个区域单独输出一张图

输入:
    - B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\归因分析\与GDP等参数的分组关联性分析\20年所有数据，含未通过显著性检验数据\将GDP_人口密度_道路密度分成20组_计算组内的HFI影响\斯皮尔曼系数\使用20组均值计算的\Grouped_Human Footprint_effect_2003_2022_group_summary.csv
    - I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic
    - B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num
    - I:\Processing data\gdb集合\WUI.gdb
    - I:\Data_huanglin\map\world continent boundary\continent.shp

输出:
    - B:\WUI\Picture\Spatial_distribution_PopD_group12_group20_Fire_frequency_2005.png
    - B:\WUI\Picture\Spatial_distribution_PopD_group12_group20_Fire_frequency_2010.png
    - B:\WUI\Picture\Spatial_distribution_PopD_group12_group20_Fire_frequency_2015.png
    - B:\WUI\Picture\Spatial_distribution_PopD_group12_group20_Fire_frequency_2020.png
    - B:\WUI\Picture\Spatial_distribution_PopD_group12_group20_Human Footprint_effect_Fire_frequency_2005_A.png
    - B:\WUI\Picture\Spatial_distribution_PopD_group12_group20_Human Footprint_effect_Fire_frequency_2005_B.png
    - 其余年份同名模式输出
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch, Rectangle


plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False


TARGET_YEARS = [2005, 2010, 2015, 2020]
TARGET_GROUPS = [12, 20]
CHUNK_SIZE = 250000
MAX_IDS_FOR_WHERE = 50000
WHERE_CHUNK_SIZE = 3000
POPD_COLUMN = "PopD"

WUI_GDB_PATH = r"I:\Processing data\gdb集合\WUI.gdb"
WUI_LAYER_TEMPLATE = "Grid_Clip_{wui_year}_WUI"
CONTINENT_SHP_PATH = r"I:\Data_huanglin\map\world continent boundary\continent.shp"
OUTPUT_DIR = Path(r"B:\WUI\Picture")
OUTPUT_DPI = 600
GROUP_SUMMARY_CSV = Path(
    r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\归因分析\与GDP等参数的分组关联性分析\20年所有数据，含未通过显著性检验数据\将GDP_人口密度_道路密度分成20组_计算组内的HFI影响\斯皮尔曼系数\使用20组均值计算的\Grouped_Human Footprint_effect_2003_2022_group_summary.csv"
)

BASEMAP_FACE = "#ffffff"
BASEMAP_EDGE = "#000000"
EMPTY_TEXT_COLOR = "#666666"
GROUP_COLORS = {
    12: "#d73027",
    20: "#4575b4",
}
CMAP_NAME = "RdBuYellow"
CMAP_COLORS = ["#0D4C93", "#3578bb", "#f8f1c2ed", "#ca1d31", "#7E0C0C"]
NON_SIG_COLOR = "#bdbdbd"
NO_DATA_COLOR = "#d9d9d9"

REGIONS = [
    {
        "label": "A",
        "slug": "India",
        "lon_min": 68.0,
        "lon_max": 93.0,
        "lat_min": 8.0,
        "lat_max": 33.0,
        "box_color": "#4c78a8",
    },
    {
        "label": "B",
        "slug": "Africa",
        "lon_min": 15.0,
        "lon_max": 40.0,
        "lat_min": -10.0,
        "lat_max": 15.0,
        "box_color": "#f58518",
    },
]

DATASET = {
    "csv_metric_slug": "Fire_frequency",
    "output_slug": "Fire_frequency",
    "title_label": "Fire frequency",
    "basic_dir": Path(r"I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic"),
    "effect_source": Path(r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num"),
}


def _chunks(items: list[int], chunk_size: int) -> Iterable[list[int]]:
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
    for name in candidates:
        if name in columns:
            return name
    return None


def _resolve_effect_files(effect_source: Path) -> list[str]:
    if effect_source.is_file():
        return [str(effect_source)]
    if effect_source.is_dir():
        return sorted(glob.glob(str(effect_source / "HFI_effect_*.csv")))
    return sorted(glob.glob(str(effect_source)))


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    values = series.astype(str).str.strip().str.upper()
    return values.isin({"TRUE", "T", "1", "YES", "Y"})


def load_group_ranges_from_csv() -> pd.DataFrame:
    if not GROUP_SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Group summary CSV not found: {GROUP_SUMMARY_CSV}")

    df = pd.read_csv(
        GROUP_SUMMARY_CSV,
        usecols=[
            "metric_slug",
            "grouping_parameter",
            "parameter_group",
            "param_bin_lower",
            "param_bin_upper",
            "group_label",
            "x_axis_range",
        ],
        low_memory=False,
    )
    df["parameter_group"] = pd.to_numeric(df["parameter_group"], errors="coerce")
    df["param_bin_lower"] = pd.to_numeric(df["param_bin_lower"], errors="coerce")
    df["param_bin_upper"] = pd.to_numeric(df["param_bin_upper"], errors="coerce")
    df = df.dropna(subset=["parameter_group", "param_bin_lower", "param_bin_upper"]).copy()
    df["parameter_group"] = df["parameter_group"].astype(int)

    subset = df[
        (df["metric_slug"] == DATASET["csv_metric_slug"])
        & (df["grouping_parameter"] == "PopD")
        & (df["parameter_group"].isin(TARGET_GROUPS))
    ].copy()
    if subset.empty:
        raise ValueError(
            f"No PopD group ranges found in {GROUP_SUMMARY_CSV.name} for metric_slug={DATASET['csv_metric_slug']}."
        )
    return subset.sort_values("parameter_group").reset_index(drop=True)


def load_target_year_popd(basic_dir: Path, target_years: list[int]) -> pd.DataFrame:
    csv_files = sorted(glob.glob(str(basic_dir / "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {basic_dir}")

    target_year_set = set(int(year) for year in target_years)
    parts: list[pd.DataFrame] = []
    print(f"Reading target-year PopD from: {basic_dir}")

    for csv_path in csv_files:
        header = pd.read_csv(csv_path, nrows=0)
        cols = header.columns.tolist()
        grid_col = _pick_column(cols, ["GRID_ID", "grid_id", "Grid_ID"])
        year_col = _pick_column(cols, ["Year", "YEAR", "year"])
        missing = [name for name, value in {"GRID_ID": grid_col, "Year": year_col}.items() if value is None]
        if missing or POPD_COLUMN not in cols:
            required = missing.copy()
            if POPD_COLUMN not in cols:
                required.append(POPD_COLUMN)
            raise ValueError(f"Missing required column(s) {required} in: {csv_path}")

        for chunk in pd.read_csv(
            csv_path,
            usecols=[grid_col, year_col, POPD_COLUMN],
            dtype={grid_col: "string"},
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            chunk = chunk.rename(columns={grid_col: "GRID_ID", year_col: "Year"})
            chunk["Year"] = pd.to_numeric(chunk["Year"], errors="coerce")
            chunk["GRID_ID"] = pd.to_numeric(chunk["GRID_ID"], errors="coerce")
            chunk[POPD_COLUMN] = pd.to_numeric(chunk[POPD_COLUMN], errors="coerce")
            chunk = chunk[chunk["Year"].isin(target_year_set)]
            chunk = chunk.dropna(subset=["Year", "GRID_ID", POPD_COLUMN])
            if chunk.empty:
                continue

            chunk["Year"] = chunk["Year"].astype(int)
            chunk["GRID_ID"] = chunk["GRID_ID"].round().astype(int)
            parts.append(chunk[["Year", "GRID_ID", POPD_COLUMN]].copy())

    if not parts:
        raise ValueError(f"No PopD records found for target years {target_years} in: {basic_dir}")

    combined = pd.concat(parts, ignore_index=True)
    combined = (
        combined.groupby(["Year", "GRID_ID"], as_index=False)[POPD_COLUMN]
        .mean()
        .sort_values(["Year", "GRID_ID"])
        .reset_index(drop=True)
    )
    print(
        f"Loaded target-year PopD rows: {len(combined):,}; "
        f"years = {sorted(combined['Year'].unique().tolist())}; "
        f"unique GRID_ID = {combined['GRID_ID'].nunique():,}"
    )
    return combined


def load_target_year_effects(effect_source: Path, target_years: list[int]) -> pd.DataFrame:
    csv_files = _resolve_effect_files(effect_source)
    if not csv_files:
        raise FileNotFoundError(f"No Human Footprint effect CSV files found in: {effect_source}")

    target_year_set = set(int(year) for year in target_years)
    parts: list[pd.DataFrame] = []
    print(f"Reading target-year Human Footprint effects from: {effect_source}")

    for csv_path in csv_files:
        header = pd.read_csv(csv_path, nrows=0)
        cols = header.columns.tolist()
        grid_col = _pick_column(cols, ["GRID_ID", "grid_id", "Grid_ID"])
        year_col = _pick_column(cols, ["Year", "YEAR", "year", "Fire_Year", "fire_year", "年份"])
        sig_col = _pick_column(cols, ["delta_sig", "DELTA_SIG", "Delta_Sig"])
        contrib_col = _pick_column(cols, ["delta_contribution", "diff_contribution", "delta"])
        missing = [name for name, value in {"GRID_ID": grid_col, "Year": year_col, "delta_sig": sig_col}.items() if value is None]
        if missing or contrib_col is None:
            required = missing.copy()
            if contrib_col is None:
                required.append("delta_contribution/diff_contribution")
            raise ValueError(f"Missing required column(s) {required} in: {csv_path}")

        for chunk in pd.read_csv(
            csv_path,
            usecols=[grid_col, year_col, contrib_col, sig_col],
            dtype={grid_col: "string"},
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            chunk = chunk.rename(
                columns={
                    grid_col: "GRID_ID",
                    year_col: "Year",
                    contrib_col: "delta_contribution",
                    sig_col: "delta_sig",
                }
            )
            chunk["Year"] = pd.to_numeric(chunk["Year"], errors="coerce")
            chunk["GRID_ID"] = pd.to_numeric(chunk["GRID_ID"], errors="coerce")
            chunk["delta_contribution"] = pd.to_numeric(chunk["delta_contribution"], errors="coerce")
            chunk["delta_sig"] = _coerce_bool_series(chunk["delta_sig"])
            chunk = chunk[chunk["Year"].isin(target_year_set)]
            chunk = chunk.dropna(subset=["Year", "GRID_ID"])
            if chunk.empty:
                continue

            chunk["Year"] = chunk["Year"].astype(int)
            chunk["GRID_ID"] = chunk["GRID_ID"].round().astype(int)
            parts.append(chunk[["Year", "GRID_ID", "delta_contribution", "delta_sig"]].copy())

    if not parts:
        raise ValueError(f"No Human Footprint effect records found for target years {target_years} in: {effect_source}")

    combined = pd.concat(parts, ignore_index=True)
    combined = (
        combined.groupby(["Year", "GRID_ID"], as_index=False)
        .agg(
            delta_contribution=("delta_contribution", "mean"),
            delta_sig=("delta_sig", "max"),
        )
        .sort_values(["Year", "GRID_ID"])
        .reset_index(drop=True)
    )
    print(
        f"Loaded target-year Human Footprint effect rows: {len(combined):,}; "
        f"years = {sorted(combined['Year'].unique().tolist())}; "
        f"unique GRID_ID = {combined['GRID_ID'].nunique():,}"
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
        return gpd.GeoDataFrame({"GRID_ID": []}, geometry=[], crs="EPSG:4326")

    if len(grid_ids) > max_ids_for_where:
        fishnet = gpd.read_file(gdb_path, layer=layer, columns=["GRID_ID"], engine="pyogrio")
        fishnet["GRID_ID"] = pd.to_numeric(fishnet["GRID_ID"], errors="coerce")
        fishnet = fishnet.dropna(subset=["GRID_ID"]).copy()
        fishnet["GRID_ID"] = fishnet["GRID_ID"].astype(int)
        return fishnet[fishnet["GRID_ID"].isin(grid_ids)].copy()

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
            if "Invalid SQL query" in str(exc) and current_chunk_size > 1:
                current_chunk_size = max(1, current_chunk_size // 2)
                continue
            raise


def select_grid_ids_for_year(year_popd_df: pd.DataFrame, year: int, lower: float, upper: float) -> list[int]:
    selected = year_popd_df[
        (year_popd_df["Year"] == year)
        & year_popd_df[POPD_COLUMN].between(lower, upper, inclusive="both")
    ].copy()
    return (
        selected["GRID_ID"]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )


def subset_region(gdf: gpd.GeoDataFrame, region: dict[str, object]) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty:
        return gdf
    return gdf.cx[
        float(region["lon_min"]):float(region["lon_max"]),
        float(region["lat_min"]):float(region["lat_max"]),
    ].copy()


def build_effect_norm(fishnet_year: gpd.GeoDataFrame) -> mcolors.Normalize:
    has_value = fishnet_year["delta_contribution"].notna()
    fishnet_data = fishnet_year[has_value]
    if fishnet_data.empty:
        return mcolors.Normalize(vmin=-1, vmax=1)

    values = fishnet_data["delta_contribution"].astype(float)
    abs_max = max(abs(values.min()), abs(values.max()))
    if pd.isna(abs_max) or abs_max == 0:
        return mcolors.Normalize(vmin=-1, vmax=1)
    return mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)


def plot_group_year_map(
    *,
    continent: gpd.GeoDataFrame,
    fishnet_by_group: dict[int, gpd.GeoDataFrame],
    year: int,
    range_info_by_group: dict[int, dict[str, object]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(15, 10), facecolor="white")

    continent.plot(ax=ax, color=BASEMAP_FACE, edgecolor=BASEMAP_EDGE, linewidth=0.25)
    cell_counts: dict[int, int] = {}
    legend_handles: list[Patch] = []
    for group_id in sorted(TARGET_GROUPS, reverse=True):
        fishnet = fishnet_by_group.get(group_id)
        if fishnet is not None and not fishnet.empty:
            fishnet.plot(
                ax=ax,
                color=GROUP_COLORS[group_id],
                edgecolor="none",
                linewidth=0,
                alpha=0.90,
            )
            cell_counts[group_id] = len(fishnet)
        else:
            cell_counts[group_id] = 0

        legend_handles.append(
            Patch(
                facecolor=GROUP_COLORS[group_id],
                edgecolor="none",
                label=(
                    f"G{group_id:02d}: {range_info_by_group[group_id]['x_axis_range']}   "
                    f"n = {cell_counts[group_id]:,}"
                ),
            )
        )

    if not any(cell_counts.values()):
        ax.text(
            0.5,
            0.5,
            "No selected cells",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color=EMPTY_TEXT_COLOR,
        )

    continent.boundary.plot(ax=ax, color=BASEMAP_EDGE, linewidth=0.25)
    for region in REGIONS:
        rect = Rectangle(
            (float(region["lon_min"]), float(region["lat_min"])),
            float(region["lon_max"]) - float(region["lon_min"]),
            float(region["lat_max"]) - float(region["lat_min"]),
            fill=False,
            edgecolor=str(region["box_color"]),
            linewidth=1.2,
            linestyle="-",
            zorder=10,
        )
        ax.add_patch(rect)
        ax.text(
            float(region["lon_min"]) + 0.4,
            float(region["lat_max"]) - 0.4,
            str(region["label"]),
            color=str(region["box_color"]),
            fontsize=10.5,
            fontweight="bold",
            ha="left",
            va="top",
            zorder=11,
        )
    ax.set_title(
        f"Population density groups G12 and G20 for {DATASET['title_label']} ({year})",
        fontsize=16,
        pad=10,
    )
    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 75)
    ax.grid(False)
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=False,
        fontsize=11,
    )
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_local_effect_region_map(
    *,
    continent: gpd.GeoDataFrame,
    selected_fishnet: gpd.GeoDataFrame,
    year: int,
    region: dict[str, object],
    norm: mcolors.Normalize,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(7.8, 7.8), facecolor="white")
    ax = fig.add_axes([0.10, 0.11, 0.72, 0.72])
    cmap = mcolors.LinearSegmentedColormap.from_list(CMAP_NAME, CMAP_COLORS, N=256)

    continent_region = subset_region(continent, region)
    selected_region = subset_region(selected_fishnet, region)
    if selected_region is None or selected_region.empty:
        selected_region = gpd.GeoDataFrame({"GRID_ID": []}, geometry=[], crs=continent.crs)

    continent_region.plot(ax=ax, color=BASEMAP_FACE, edgecolor=BASEMAP_EDGE, linewidth=0.35)

    if not selected_region.empty:
        has_value = selected_region["delta_contribution"].notna()
        selected_no_data = selected_region[~has_value]
        selected_data = selected_region[has_value]

        if not selected_no_data.empty:
            selected_no_data.plot(ax=ax, color=NO_DATA_COLOR, edgecolor="none", linewidth=0)

        if not selected_data.empty:
            sig_mask = selected_data["delta_sig"].fillna(False).astype(bool)
            selected_sig = selected_data[sig_mask]
            selected_non_sig = selected_data[~sig_mask]

            if not selected_non_sig.empty:
                selected_non_sig.plot(ax=ax, color=NON_SIG_COLOR, edgecolor="none", linewidth=0)
            if not selected_sig.empty:
                selected_sig.plot(
                    ax=ax,
                    column="delta_contribution",
                    cmap=cmap,
                    norm=norm,
                    edgecolor="none",
                    linewidth=0,
                )
        else:
            ax.text(
                0.5,
                0.5,
                "No effect cells",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color=EMPTY_TEXT_COLOR,
            )
    else:
        ax.text(
            0.5,
            0.5,
            "No selected cells",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color=EMPTY_TEXT_COLOR,
        )

    continent_region.boundary.plot(ax=ax, color=BASEMAP_EDGE, linewidth=0.35)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.85, 0.18, 0.04, 0.60])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label("Human Footprint effect", fontsize=12)
    cbar.ax.tick_params(direction="in", labelsize=11)
    cbar.ax.yaxis.label.set_size(12)

    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.set_xlim(float(region["lon_min"]), float(region["lon_max"]))
    ax.set_ylim(float(region["lat_min"]), float(region["lat_max"]))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)

    fig.savefig(output_path, dpi=OUTPUT_DPI, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading world base map: {CONTINENT_SHP_PATH}", flush=True)
    continent = gpd.read_file(CONTINENT_SHP_PATH).to_crs("EPSG:4326")
    print(f"Base map loaded: {len(continent)} feature(s), CRS={continent.crs}", flush=True)

    pooled_ranges = load_group_ranges_from_csv()
    year_popd_df = load_target_year_popd(Path(DATASET["basic_dir"]), TARGET_YEARS)
    effect_all = load_target_year_effects(Path(DATASET["effect_source"]), TARGET_YEARS)

    range_info_by_group: dict[int, dict[str, object]] = {}
    for group_id in TARGET_GROUPS:
        range_row = pooled_ranges.loc[pooled_ranges["parameter_group"] == group_id].iloc[0]
        lower = float(range_row["param_bin_lower"])
        upper = float(range_row["param_bin_upper"])
        range_info_by_group[group_id] = {
            "lower": lower,
            "upper": upper,
            "x_axis_range": str(range_row.get("x_axis_range", f"[{lower:.4g}, {upper:.4g}]")),
        }
        print(
            f"\nPreparing PopD group G{group_id:02d}: "
            f"csv range = [{lower:.6g}, {upper:.6g}]",
            flush=True,
        )

    for year in TARGET_YEARS:
        layer = WUI_LAYER_TEMPLATE.format(wui_year=year)
        grid_ids_by_group: dict[int, list[int]] = {}
        for group_id in TARGET_GROUPS:
            lower = float(range_info_by_group[group_id]["lower"])
            upper = float(range_info_by_group[group_id]["upper"])
            grid_ids = select_grid_ids_for_year(year_popd_df, year, lower, upper)
            grid_ids_by_group[group_id] = grid_ids
            print(
                f"Year {year} G{group_id:02d}: selected GRID_ID = {len(grid_ids):,}",
                flush=True,
            )

        union_ids = sorted({grid_id for ids in grid_ids_by_group.values() for grid_id in ids})
        if union_ids:
            print(
                f"Loading WUI layer {layer}: union selected GRID_ID = {len(union_ids):,}",
                flush=True,
            )
            union_fishnet = read_wui_fishnet_filtered(WUI_GDB_PATH, layer, union_ids)
            if "GRID_ID" not in union_fishnet.columns:
                raise ValueError(f"GRID_ID field not found in WUI fishnet layer: {layer}")
            union_fishnet["GRID_ID"] = pd.to_numeric(union_fishnet["GRID_ID"], errors="coerce")
            union_fishnet = union_fishnet.dropna(subset=["GRID_ID"]).copy()
            union_fishnet["GRID_ID"] = union_fishnet["GRID_ID"].astype(int)
            union_fishnet = union_fishnet.to_crs("EPSG:4326")
        else:
            union_fishnet = gpd.GeoDataFrame({"GRID_ID": []}, geometry=[], crs="EPSG:4326")

        output_path = OUTPUT_DIR / f"Spatial_distribution_PopD_group12_group20_{DATASET['output_slug']}_{year}.png"
        fishnet_by_group = {
            group_id: union_fishnet[union_fishnet["GRID_ID"].isin(grid_ids_by_group[group_id])].copy()
            for group_id in TARGET_GROUPS
        }
        plot_group_year_map(
            continent=continent,
            fishnet_by_group=fishnet_by_group,
            year=year,
            range_info_by_group=range_info_by_group,
            output_path=output_path,
        )
        print(f"Figure saved to: {output_path}", flush=True)

        effect_year = effect_all[effect_all["Year"] == year][["GRID_ID", "delta_contribution", "delta_sig"]]
        selected_fishnet = union_fishnet.merge(effect_year, on="GRID_ID", how="left")
        norm = build_effect_norm(selected_fishnet)

        for region in REGIONS:
            local_output_path = (
                OUTPUT_DIR
                / f"Spatial_distribution_PopD_group12_group20_Human Footprint_effect_{DATASET['output_slug']}_{year}_{region['label']}.png"
            )
            plot_local_effect_region_map(
                continent=continent,
                selected_fishnet=selected_fishnet,
                year=year,
                region=region,
                norm=norm,
                output_path=local_output_path,
            )
            print(f"Figure saved to: {local_output_path}", flush=True)


if __name__ == "__main__":
    main()


