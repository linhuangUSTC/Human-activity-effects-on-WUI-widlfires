#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称: GDP 第 2 组与第 20 组的全球空间分布图

功能简介:
    1) 直接读取「将GDP_人口密度_道路密度分成20组_计算组内的HFI影响_柱状图.py」
       输出的 Grouped_Human Footprint_effect_2003_2022_group_summary.csv
    2) 再读取 2005、2010、2015、2020 四个年份的 GDP 数据
    3) 将当年 GDP 落在第 2 组与第 20 组范围内的 WUI 格网筛出
    4) 直接按年份分别输出单张全球分布图，不再拼成 2×2 合图

输入:
    - B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\归因分析\与GDP等参数的分组关联性分析\20年所有数据，含未通过显著性检验数据\将GDP_人口密度_道路密度分成20组_计算组内的HFI影响\Grouped_Human Footprint_effect_2003_2022_group_summary.csv
    - I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic
    - I:\Processing data\渔网分类建模\Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic
    - I:\Processing data\gdb集合\WUI.gdb
    - I:\Data_huanglin\map\world continent boundary\continent.shp

输出:
    - B:\WUI\Picture\Spatial_distribution_GDP_group02_group20_Fire_frequency_2005.png
    - B:\WUI\Picture\Spatial_distribution_GDP_group02_group20_Fire_frequency_2010.png
    - B:\WUI\Picture\Spatial_distribution_GDP_group02_group20_Fire_frequency_2015.png
    - B:\WUI\Picture\Spatial_distribution_GDP_group02_group20_Fire_frequency_2020.png
    - B:\WUI\Picture\Spatial_distribution_GDP_group02_group20_Normalized_burn_area_2005.png
    - B:\WUI\Picture\Spatial_distribution_GDP_group02_group20_Normalized_burn_area_2010.png
    - B:\WUI\Picture\Spatial_distribution_GDP_group02_group20_Normalized_burn_area_2015.png
    - B:\WUI\Picture\Spatial_distribution_GDP_group02_group20_Normalized_burn_area_2020.png
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False


TARGET_YEARS = [2005, 2010, 2015, 2020]
FIRE_YEAR_START = 2003
FIRE_YEAR_END = 2022
YEAR_DESC = f"{FIRE_YEAR_START}-{FIRE_YEAR_END}"
N_GROUPS = 20
TARGET_GROUPS = [2, 20]
CHUNK_SIZE = 250000
MAX_IDS_FOR_WHERE = 50000
WHERE_CHUNK_SIZE = 3000
GDP_COLUMN = "GDP"

WUI_GDB_PATH = r"I:\Processing data\gdb集合\WUI.gdb"
WUI_LAYER_TEMPLATE = "Grid_Clip_{wui_year}_WUI"
CONTINENT_SHP_PATH = r"I:\Data_huanglin\map\world continent boundary\continent.shp"
OUTPUT_DIR = Path(r"B:\WUI\Picture")
OUTPUT_DPI = 600
GROUP_SUMMARY_CSV = Path(
    r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\归因分析\与GDP等参数的分组关联性分析\20年所有数据，含未通过显著性检验数据\将GDP_人口密度_道路密度分成20组_计算组内的HFI影响\Grouped_Human Footprint_effect_2003_2022_group_summary.csv"
)

BASEMAP_FACE = "#ffffff"
BASEMAP_EDGE = "#000000"
EMPTY_TEXT_COLOR = "#666666"
GROUP_COLORS = {
    2: "#d73027",
    20: "#4575b4",
}

DATASETS = [
    {
        "csv_metric_slug": "Fire_frequency",
        "output_slug": "Fire_frequency",
        "title_label": "Fire frequency",
        "basic_dir": Path(r"I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic"),
    },
    {
        "csv_metric_slug": "Fire_area",
        "output_slug": "Normalized_burn_area",
        "title_label": "Normalized burned area",
        "basic_dir": Path(r"I:\Processing data\渔网分类建模\Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic"),
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


def load_group_ranges_from_csv(dataset: dict[str, object]) -> pd.DataFrame:
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
        (df["metric_slug"] == dataset["csv_metric_slug"])
        & (df["grouping_parameter"] == "GDP")
        & (df["parameter_group"].isin(TARGET_GROUPS))
    ].copy()
    if subset.empty:
        raise ValueError(
            f"No GDP group ranges found in {GROUP_SUMMARY_CSV.name} for metric_slug={dataset['csv_metric_slug']}."
        )
    return subset.sort_values("parameter_group").reset_index(drop=True)


def load_target_year_gdp(basic_dir: Path, target_years: list[int]) -> pd.DataFrame:
    csv_files = sorted(glob.glob(str(basic_dir / "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {basic_dir}")

    target_year_set = set(int(year) for year in target_years)
    parts: list[pd.DataFrame] = []
    print(f"Reading target-year GDP from: {basic_dir}")

    for csv_path in csv_files:
        header = pd.read_csv(csv_path, nrows=0)
        cols = header.columns.tolist()
        grid_col = _pick_column(cols, ["GRID_ID", "grid_id", "Grid_ID"])
        year_col = _pick_column(cols, ["Year", "YEAR", "year"])
        missing = [name for name, value in {"GRID_ID": grid_col, "Year": year_col}.items() if value is None]
        if missing or GDP_COLUMN not in cols:
            required = missing.copy()
            if GDP_COLUMN not in cols:
                required.append(GDP_COLUMN)
            raise ValueError(f"Missing required column(s) {required} in: {csv_path}")

        for chunk in pd.read_csv(
            csv_path,
            usecols=[grid_col, year_col, GDP_COLUMN],
            dtype={grid_col: "string"},
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            chunk = chunk.rename(columns={grid_col: "GRID_ID", year_col: "Year"})
            chunk["Year"] = pd.to_numeric(chunk["Year"], errors="coerce")
            chunk["GRID_ID"] = pd.to_numeric(chunk["GRID_ID"], errors="coerce")
            chunk[GDP_COLUMN] = pd.to_numeric(chunk[GDP_COLUMN], errors="coerce")
            chunk = chunk[chunk["Year"].isin(target_year_set)]
            chunk = chunk.dropna(subset=["Year", "GRID_ID", GDP_COLUMN])
            if chunk.empty:
                continue

            chunk["Year"] = chunk["Year"].astype(int)
            chunk["GRID_ID"] = chunk["GRID_ID"].round().astype(int)
            parts.append(chunk[["Year", "GRID_ID", GDP_COLUMN]].copy())

    if not parts:
        raise ValueError(f"No GDP records found for target years {target_years} in: {basic_dir}")

    combined = pd.concat(parts, ignore_index=True)
    combined = (
        combined.groupby(["Year", "GRID_ID"], as_index=False)[GDP_COLUMN]
        .mean()
        .sort_values(["Year", "GRID_ID"])
        .reset_index(drop=True)
    )
    print(
        f"Loaded target-year GDP rows: {len(combined):,}; "
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


def build_group_year_lookup(year_gdp_df: pd.DataFrame, lower: float, upper: float) -> dict[int, list[int]]:
    selected = year_gdp_df[year_gdp_df[GDP_COLUMN].between(lower, upper, inclusive="both")].copy()
    lookup: dict[int, list[int]] = {}
    for year in TARGET_YEARS:
        ids = (
            selected.loc[selected["Year"] == year, "GRID_ID"]
            .dropna()
            .astype(int)
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        lookup[year] = ids
    return lookup


def select_grid_ids_for_year(year_gdp_df: pd.DataFrame, year: int, lower: float, upper: float) -> list[int]:
    selected = year_gdp_df[
        (year_gdp_df["Year"] == year)
        & year_gdp_df[GDP_COLUMN].between(lower, upper, inclusive="both")
    ].copy()
    return (
        selected["GRID_ID"]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )


def load_fishnet_by_year(group_lookup: dict[int, list[int]]) -> dict[int, gpd.GeoDataFrame]:
    fishnet_by_year: dict[int, gpd.GeoDataFrame] = {}
    for year in TARGET_YEARS:
        grid_ids = group_lookup.get(year, [])
        if not grid_ids:
            fishnet_by_year[year] = gpd.GeoDataFrame({"GRID_ID": []}, geometry=[], crs="EPSG:4326")
            continue

        layer = WUI_LAYER_TEMPLATE.format(wui_year=year)
        print(f"Loading WUI layer {layer}: selected GRID_ID = {len(grid_ids):,}")
        fishnet = read_wui_fishnet_filtered(WUI_GDB_PATH, layer, grid_ids)
        if "GRID_ID" not in fishnet.columns:
            raise ValueError(f"GRID_ID field not found in WUI fishnet layer: {layer}")

        fishnet["GRID_ID"] = pd.to_numeric(fishnet["GRID_ID"], errors="coerce")
        fishnet = fishnet.dropna(subset=["GRID_ID"]).copy()
        fishnet["GRID_ID"] = fishnet["GRID_ID"].astype(int)
        fishnet_by_year[year] = fishnet.to_crs("EPSG:4326")
    return fishnet_by_year


def summarize_selected_years(year_gdp_df: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    selected = year_gdp_df[year_gdp_df[GDP_COLUMN].between(lower, upper, inclusive="both")].copy()
    return (
        selected.groupby("Year", as_index=False)
        .agg(
            n_cells=("GRID_ID", "nunique"),
            gdp_min=(GDP_COLUMN, "min"),
            gdp_max=(GDP_COLUMN, "max"),
        )
        .sort_values("Year")
        .reset_index(drop=True)
    )


def plot_group_year_map(
    *,
    continent: gpd.GeoDataFrame,
    fishnet_by_group: dict[int, gpd.GeoDataFrame],
    dataset: dict[str, object],
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
    ax.set_title(
        f"GDP groups G02 and G20 for {dataset['title_label']} ({year})",
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading world base map: {CONTINENT_SHP_PATH}", flush=True)
    continent = gpd.read_file(CONTINENT_SHP_PATH).to_crs("EPSG:4326")
    print(f"Base map loaded: {len(continent)} feature(s), CRS={continent.crs}", flush=True)

    for dataset in DATASETS:
        print(f"\n{'=' * 80}", flush=True)
        print(f"Processing dataset: {dataset['title_label']}", flush=True)

        pooled_ranges = load_group_ranges_from_csv(dataset)
        year_gdp_df = load_target_year_gdp(Path(dataset["basic_dir"]), TARGET_YEARS)

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
                f"\nPreparing GDP group G{group_id:02d}: "
                f"csv range = [{lower:.6g}, {upper:.6g}]",
                flush=True,
            )

        for year in TARGET_YEARS:
            layer = WUI_LAYER_TEMPLATE.format(wui_year=year)
            grid_ids_by_group: dict[int, list[int]] = {}
            for group_id in TARGET_GROUPS:
                lower = float(range_info_by_group[group_id]["lower"])
                upper = float(range_info_by_group[group_id]["upper"])
                grid_ids = select_grid_ids_for_year(year_gdp_df, year, lower, upper)
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

            output_path = (
                OUTPUT_DIR
                / f"Spatial_distribution_GDP_group02_group20_{dataset['output_slug']}_{year}.png"
            )
            plot_group_year_map(
                continent=continent,
                fishnet_by_group={
                    group_id: union_fishnet[union_fishnet["GRID_ID"].isin(grid_ids_by_group[group_id])].copy()
                    for group_id in TARGET_GROUPS
                },
                dataset=dataset,
                year=year,
                range_info_by_group=range_info_by_group,
                output_path=output_path,
            )
            print(f"Figure saved to: {output_path}", flush=True)


if __name__ == "__main__":
    main()



