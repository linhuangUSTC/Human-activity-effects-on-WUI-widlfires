#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称: HFI影响的空间聚类全球分布图

功能简介:
    1) 读取 Fire num 与 Fire area 的 HFI_effect_*.csv，仅提取 GRID_ID 与 Year
    2) 读取「边际效应曲线层次聚类分类.py」输出的 Human Footprint_effect_curve_cluster_assignments.csv
    3) 根据 effect CSV 文件名恢复 category_id，并将每个 (Year, GRID_ID) 匹配到所属 cluster
    4) 默认仅绘制 2005、2010、2015、2020 四个 WUI 对应年份
    5) 根据 Year 选择对应的 WUI 渔网图层
    6) 每个年份将 Fire frequency 与 Normalized burned area 组合为 1x2 全球分布图
    7) 同时额外输出 Fire frequency 与 Normalized burned area 的单独全球分布图
    8) 3 个 cluster 使用 3 种颜色表示，legend 保持一行放在底部
    9) 输出图片到 B:\WUI\Picture
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False


FIRE_YEAR_START = 2003
FIRE_YEAR_END = 2022

CSV_ROOT = r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv"
CLUSTER_ROOT = r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\层次聚类\边际效应曲线层次聚类分类"

WUI_GDB_PATH = r"I:\Processing data\gdb集合\WUI.gdb"
WUI_YEARS = [2005, 2010, 2015, 2020]
WUI_LAYER_TEMPLATE = "Grid_Clip_{wui_year}_WUI"

CONTINENT_SHP_PATH = r"I:\Data_huanglin\map\world continent boundary\continent.shp"

OUTPUT_ROOT = r"B:\WUI\Picture"
DEFAULT_OUTPUT_DPI = 600
COMBINED_OUTPUT_FILENAME_TEMPLATE = "Human Footprint影响空间聚类_Fire_frequency_Normalized_burned_area_{year}.jpg"

CHUNK_SIZE = 250000
MAX_IDS_FOR_WHERE = 50000
WHERE_CHUNK_SIZE = 3000

CLUSTER_COLORS = {
    1: "#82caee",
    2: "#d32f23",
    3: "#f29f1b",
}
SHARED_CLUSTER_NAMES = {
    1: "Progressive Suppression",
    2: "Intermediate Response",
    3: "High-Human Footprint Enhancement",
}
UNASSIGNED_COLOR = "#bdbdbd"
BASEMAP_FACE = "#ffffff"
BASEMAP_EDGE = "#000000"
TITLE_FONT_SIZE = 30
AXIS_LABEL_FONT_SIZE = 24
AXIS_TICK_FONT_SIZE = 22
YEAR_FONT_SIZE = 28
LEGEND_FONT_SIZE = 30
LATITUDE_TICKS = list(range(-60, 76, 30))
SINGLE_PANEL_FIGSIZE = (36, 12)
MULTI_PANEL_FIGSIZE = (15, 10)

DATASETS = [
    {
        "key": "Fire num",
        "effect_source": os.path.join(CSV_ROOT, "Fire num"),
        "cluster_csv": os.path.join(CLUSTER_ROOT, "Human Footprint_effect_curve_cluster_assignments_Fire_num.csv"),
        "output_dir": OUTPUT_ROOT,
        "title_metric": "Fire frequency",
    },
    {
        "key": "Fire area",
        "effect_source": os.path.join(CSV_ROOT, "Fire area"),
        "cluster_csv": os.path.join(CLUSTER_ROOT, "Human Footprint_effect_curve_cluster_assignments_Fire_area.csv"),
        "output_dir": OUTPUT_ROOT,
        "title_metric": "Normalized burned area",
    },
]


@dataclass(frozen=True)
class EffectFileSpec:
    path: str
    category_id: str
    cluster: int
    cluster_label: str
    grid_col: str
    year_col: str


def _chunks(items: list[int], chunk_size: int) -> Iterable[list[int]]:
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
    for name in candidates:
        if name in columns:
            return name
    return None


def _resolve_effect_files(effect_source: str) -> list[str]:
    if os.path.isfile(effect_source):
        return [effect_source]
    if os.path.isdir(effect_source):
        return sorted(glob.glob(os.path.join(effect_source, "HFI_effect_*.csv")))
    return sorted(glob.glob(effect_source))


def extract_category_id(path: str) -> str:
    name = Path(path).stem
    match = re.search(
        r"HFI_effect_((face|mix)_[A-D]_[A-Z]+)_(?:fire_num_with_WUI_offset|fire_area)$",
        name,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"Unrecognized effect CSV name pattern: {name}")
    return match.group(1)


def load_cluster_assignments(assignments_path: str) -> pd.DataFrame:
    if not os.path.exists(assignments_path):
        raise FileNotFoundError(f"Cluster assignment CSV not found: {assignments_path}")

    df = pd.read_csv(
        assignments_path,
        usecols=["category_id", "cluster", "cluster_label"],
        low_memory=False,
    )
    df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce")
    df = df.dropna(subset=["category_id", "cluster"]).copy()
    df["cluster"] = df["cluster"].astype(int)
    df["cluster_label"] = df["cluster_label"].fillna("").astype(str)
    df["cluster_label"] = df["cluster"].map(SHARED_CLUSTER_NAMES).fillna(df["cluster_label"]).astype(str)

    conflict_counts = df.groupby("category_id")["cluster"].nunique()
    bad_categories = conflict_counts[conflict_counts > 1]
    if not bad_categories.empty:
        raise ValueError(
            "Cluster assignment CSV contains conflicting cluster IDs for category_id(s): "
            + ", ".join(bad_categories.index.tolist()[:10])
        )

    df = df.sort_values(["category_id", "cluster"]).drop_duplicates(subset=["category_id"], keep="first")
    print(f"Loaded {len(df)} cluster assignments from: {assignments_path}")
    return df


def prepare_effect_file_specs(effect_source: str, assignments: pd.DataFrame) -> list[EffectFileSpec]:
    cluster_lookup = assignments.set_index("category_id")[["cluster", "cluster_label"]].to_dict("index")
    csv_files = _resolve_effect_files(effect_source)
    if not csv_files:
        raise FileNotFoundError(f"No effect CSV found by: {effect_source}")

    specs: list[EffectFileSpec] = []
    skipped_categories: list[str] = []

    for csv_path in csv_files:
        category_id = extract_category_id(csv_path)
        cluster_info = cluster_lookup.get(category_id)
        if cluster_info is None:
            skipped_categories.append(category_id)
            continue

        header = pd.read_csv(csv_path, nrows=0)
        cols = header.columns.tolist()
        grid_col = _pick_column(cols, ["GRID_ID", "grid_id", "Grid_ID"])
        year_col = _pick_column(cols, ["Year", "YEAR", "year", "Fire_Year", "fire_year", "年份"])

        missing = [name for name, value in {"GRID_ID": grid_col, "Year": year_col}.items() if value is None]
        if missing:
            raise ValueError(f"Missing required column(s) {missing} in: {csv_path}")

        specs.append(
            EffectFileSpec(
                path=csv_path,
                category_id=category_id,
                cluster=int(cluster_info["cluster"]),
                cluster_label=str(cluster_info["cluster_label"]),
                grid_col=grid_col,
                year_col=year_col,
            )
        )

    if skipped_categories:
        print(
            "Warning: cluster assignments missing for category_id(s): "
            + ", ".join(sorted(set(skipped_categories))[:10])
        )

    print(f"Prepared {len(specs)} effect CSV specifications from: {effect_source}")
    return specs


def load_clustered_grid_data_for_years(specs: list[EffectFileSpec], target_years: list[int]) -> pd.DataFrame:
    target_years_set = set(int(y) for y in target_years)
    parts: list[pd.DataFrame] = []

    for spec in specs:
        for chunk in pd.read_csv(
            spec.path,
            usecols=[spec.grid_col, spec.year_col],
            dtype={spec.grid_col: "string"},
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            chunk = chunk.rename(columns={spec.grid_col: "GRID_ID", spec.year_col: "Year"})
            chunk["Year"] = pd.to_numeric(chunk["Year"], errors="coerce")
            chunk["GRID_ID"] = pd.to_numeric(chunk["GRID_ID"], errors="coerce")
            chunk = chunk[chunk["Year"].isin(target_years_set)]
            chunk = chunk.dropna(subset=["Year", "GRID_ID"])
            if chunk.empty:
                continue

            chunk["Year"] = chunk["Year"].astype(int)
            chunk["GRID_ID"] = chunk["GRID_ID"].astype(int)
            chunk["category_id"] = spec.category_id
            chunk["cluster"] = spec.cluster
            chunk["cluster_label"] = spec.cluster_label
            parts.append(chunk[["Year", "GRID_ID", "category_id", "cluster", "cluster_label"]].copy())

    if not parts:
        return pd.DataFrame(columns=["Year", "GRID_ID", "category_id", "cluster", "cluster_label"])

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=["Year", "GRID_ID", "category_id", "cluster", "cluster_label"])

    cluster_nunique = combined.groupby(["Year", "GRID_ID"])["cluster"].nunique()
    conflict_count = int((cluster_nunique > 1).sum())
    if conflict_count:
        print(
            f"Warning: found {conflict_count} (Year, GRID_ID) record(s) mapped to multiple clusters. "
            "Keeping the first record after sorting by category_id."
        )

    combined = (
        combined.sort_values(["Year", "GRID_ID", "category_id"])
        .groupby(["Year", "GRID_ID"], as_index=False)
        .agg(
            category_id=("category_id", "first"),
            cluster=("cluster", "first"),
            cluster_label=("cluster_label", "first"),
        )
    )

    print(
        "Clustered grid data loaded: "
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
        print(
            f"GRID_ID count = {len(grid_ids)} > {max_ids_for_where}, reading full WUI fishnet layer instead..."
        )
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
            msg = str(exc)
            if "Invalid SQL query" in msg and current_chunk_size > 1:
                new_chunk_size = max(1, current_chunk_size // 2)
                if new_chunk_size == current_chunk_size:
                    raise
                print(
                    f"Invalid SQL query with WHERE_CHUNK_SIZE={current_chunk_size}, retrying with {new_chunk_size}..."
                )
                current_chunk_size = new_chunk_size
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


def parse_year_filter_from_env() -> list[int]:
    env_value = os.environ.get("HFI_CLUSTER_YEARS", "").strip()
    if not env_value:
        return WUI_YEARS.copy()

    years: set[int] = set()
    for part in env_value.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start_year = int(start_text)
            end_year = int(end_text)
            years.update(range(min(start_year, end_year), max(start_year, end_year) + 1))
        else:
            years.add(int(item))

    filtered_years = sorted(y for y in years if FIRE_YEAR_START <= y <= FIRE_YEAR_END)
    if not filtered_years:
        raise ValueError("HFI_CLUSTER_YEARS did not resolve to any valid fire year.")
    return filtered_years


def filter_datasets_from_env(datasets: list[dict[str, str]]) -> list[dict[str, str]]:
    env_value = os.environ.get("HFI_CLUSTER_DATASETS", "").strip()
    if not env_value:
        return datasets

    allowed = {item.strip().lower() for item in env_value.split(",") if item.strip()}
    filtered = [config for config in datasets if config["key"].lower() in allowed]
    if not filtered:
        raise ValueError(
            "HFI_CLUSTER_DATASETS did not match any dataset key. "
            f"Available keys: {', '.join(config['key'] for config in datasets)}"
        )
    return filtered


def output_dpi_from_env() -> int:
    env_value = os.environ.get("HFI_CLUSTER_OUTPUT_DPI", "").strip()
    if not env_value:
        return DEFAULT_OUTPUT_DPI
    dpi = int(env_value)
    if dpi <= 0:
        raise ValueError("HFI_CLUSTER_OUTPUT_DPI must be a positive integer.")
    return dpi


def draw_cluster_map_on_axis(
    *,
    ax: plt.Axes,
    continent: gpd.GeoDataFrame,
    fishnet: gpd.GeoDataFrame,
    fire_year: int,
    title_metric: str,
    cluster_name_map: dict[int, str],
) -> bool:
    continent.plot(ax=ax, color=BASEMAP_FACE, edgecolor=BASEMAP_EDGE, linewidth=0.25)

    if fishnet.empty:
        ax.text(0.02, 0.92, f"{fire_year}", transform=ax.transAxes, fontsize=YEAR_FONT_SIZE, verticalalignment="top")
        ax.set_title(title_metric, fontsize=TITLE_FONT_SIZE, pad=12)
        ax.set_xlabel("Longitude", fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_ylabel("Latitude", fontsize=AXIS_LABEL_FONT_SIZE)
        ax.tick_params(axis="both", which="major", labelsize=AXIS_TICK_FONT_SIZE)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 75)
        ax.set_yticks(LATITUDE_TICKS)
        ax.grid(False)
        return False

    unassigned = fishnet[fishnet["cluster"].isna()]
    if not unassigned.empty:
        unassigned.plot(ax=ax, color=UNASSIGNED_COLOR, edgecolor="none", linewidth=0)

    for cluster_id in sorted(cluster_name_map):
        cluster_cells = fishnet[fishnet["cluster"] == cluster_id]
        if cluster_cells.empty:
            continue
        print(f"Cluster {cluster_id} cells: {len(cluster_cells)}")
        cluster_cells.plot(
            ax=ax,
            color=CLUSTER_COLORS.get(cluster_id, UNASSIGNED_COLOR),
            edgecolor="none",
            linewidth=0,
            alpha=0.9,
        )

    continent.boundary.plot(ax=ax, color=BASEMAP_EDGE, linewidth=0.25)

    ax.text(0.02, 0.92, f"{fire_year}", transform=ax.transAxes, fontsize=YEAR_FONT_SIZE, verticalalignment="top")
    ax.set_title(title_metric, fontsize=TITLE_FONT_SIZE, pad=12)
    ax.set_xlabel("Longitude", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel("Latitude", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=AXIS_TICK_FONT_SIZE)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 75)
    ax.set_yticks(LATITUDE_TICKS)
    ax.grid(False)
    return not unassigned.empty


def build_cluster_legend_handles(
    cluster_name_map: dict[int, str],
    *,
    include_unassigned: bool,
) -> list[Patch]:
    handles = [
        Patch(
            facecolor=CLUSTER_COLORS.get(cluster_id, UNASSIGNED_COLOR),
            edgecolor="none",
            label=f"Cluster {cluster_id}: {cluster_name_map[cluster_id]}",
        )
        for cluster_id in sorted(cluster_name_map)
    ]
    if include_unassigned:
        handles.append(Patch(facecolor=UNASSIGNED_COLOR, edgecolor="none", label="Unassigned"))
    return handles


def build_group_output_path(output_root: str, datasets: list[dict[str, str]], fire_year: int) -> str:
    if len(datasets) == 2:
        filename = COMBINED_OUTPUT_FILENAME_TEMPLATE.format(year=fire_year)
    else:
        metric_text = datasets[0]["title_metric"].replace(" ", "_")
        filename = f"Human Footprint影响空间聚类_{metric_text}_{fire_year}.jpg"
    return os.path.join(output_root, filename)


def plot_cluster_map_group(
    *,
    continent: gpd.GeoDataFrame,
    fishnet_by_key: dict[str, gpd.GeoDataFrame],
    fire_year: int,
    datasets: list[dict[str, str]],
    cluster_name_map: dict[int, str],
    output_dpi: int,
    output_path: str,
) -> None:
    n_panels = len(datasets)
    if n_panels == 1:
        figsize = SINGLE_PANEL_FIGSIZE
    else:
        figsize = (MULTI_PANEL_FIGSIZE[0] * n_panels, MULTI_PANEL_FIGSIZE[1])

    fig, axes = plt.subplots(1, n_panels, figsize=figsize, facecolor="white")
    if n_panels == 1:
        axes = [axes]
    else:
        axes = list(axes)

    include_unassigned = False
    for idx, (ax, config) in enumerate(zip(axes, datasets)):
        fishnet = fishnet_by_key.get(config["key"])
        if fishnet is None:
            fishnet = gpd.GeoDataFrame(columns=["GRID_ID", "cluster", "cluster_label"], geometry=[], crs=continent.crs)
        include_unassigned = draw_cluster_map_on_axis(
            ax=ax,
            continent=continent,
            fishnet=fishnet,
            fire_year=fire_year,
            title_metric=config["title_metric"],
            cluster_name_map=cluster_name_map,
        ) or include_unassigned
        if idx == 0:
            ax.yaxis.set_label_position("left")
            ax.yaxis.tick_left()
            ax.tick_params(axis="y", labelleft=True, labelright=False, left=True, right=False, pad=4)
        else:
            ax.set_ylabel("Latitude", fontsize=AXIS_LABEL_FONT_SIZE)
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.tick_params(axis="y", labelleft=False, labelright=True, left=False, right=True, pad=4)

    legend_handles = build_cluster_legend_handles(cluster_name_map, include_unassigned=include_unassigned)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.07 if n_panels == 1 else 0.12),
        ncol=len(legend_handles),
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.2,
    )

    if n_panels == 1:
        ax = axes[0]
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
            labeltop=False,
            labelleft=False,
            labelright=False,
        )
        fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.15, wspace=0.03)
    else:
        fig.subplots_adjust(left=0.06, right=0.96, top=0.92, bottom=0.17, wspace=0.03)
    fig.savefig(output_path, dpi=output_dpi, bbox_inches="tight", format="jpg")
    plt.close(fig)
    print(f"Figure saved to: {output_path}")


def main() -> None:
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    fire_years = parse_year_filter_from_env()
    output_dpi = output_dpi_from_env()
    datasets = filter_datasets_from_env(DATASETS)
    years_by_wui_year: dict[int, list[int]] = {}
    for fire_year in fire_years:
        wui_year = wui_year_for_fire_year(fire_year)
        years_by_wui_year.setdefault(wui_year, []).append(fire_year)

    print(f"Loading world base map: {CONTINENT_SHP_PATH}")
    continent = gpd.read_file(CONTINENT_SHP_PATH)
    continent = continent.to_crs("EPSG:4326")
    print(f"Base map loaded: {len(continent)} feature(s), CRS={continent.crs}")

    print(f"Selected fire years: {fire_years}")
    print(f"Selected datasets: {[config['key'] for config in datasets]}")
    print(f"Output DPI: {output_dpi}")

    dataset_runtime: list[dict[str, object]] = []
    shared_cluster_name_map: dict[int, str] = {}
    for config in datasets:
        print(f"\n{'=' * 80}")
        print(f"Preparing dataset: {config['key']}")
        assignments = load_cluster_assignments(config["cluster_csv"])
        cluster_name_map = (
            assignments.sort_values(["cluster", "category_id"])
            .drop_duplicates(subset=["cluster"], keep="first")
            .set_index("cluster")["cluster_label"]
            .to_dict()
        )
        shared_cluster_name_map.update(cluster_name_map)
        specs = prepare_effect_file_specs(config["effect_source"], assignments)
        dataset_runtime.append(
            {
                "config": config,
                "cluster_name_map": cluster_name_map,
                "specs": specs,
            }
        )

    for wui_year in sorted(years_by_wui_year):
        year_group = years_by_wui_year[wui_year]
        layer = WUI_LAYER_TEMPLATE.format(wui_year=wui_year)

        print(f"\n{'-' * 60}")
        print(f"Loading clustered grid data for years {year_group[0]}-{year_group[-1]} ...")

        group_df_by_key: dict[str, pd.DataFrame] = {}
        union_grid_ids: set[int] = set()
        for runtime in dataset_runtime:
            config = runtime["config"]
            group_df = load_clustered_grid_data_for_years(runtime["specs"], year_group)
            group_df_by_key[config["key"]] = group_df
            if not group_df.empty:
                union_grid_ids.update(group_df["GRID_ID"].dropna().astype(int).tolist())

        if not union_grid_ids:
            print(f"Skip WUI {wui_year}: no clustered grid data for years {year_group[0]}-{year_group[-1]}.")
            continue

        union_grid_ids_sorted = sorted(union_grid_ids)
        print(f"Loading WUI fishnet layer: {layer} (years {year_group[0]}-{year_group[-1]})")
        fishnet_base = read_wui_fishnet_filtered(WUI_GDB_PATH, layer, union_grid_ids_sorted)
        if "GRID_ID" not in fishnet_base.columns:
            raise ValueError(f"GRID_ID field not found in WUI fishnet layer: {layer}")

        fishnet_base["GRID_ID"] = pd.to_numeric(fishnet_base["GRID_ID"], errors="coerce")
        fishnet_base = fishnet_base.dropna(subset=["GRID_ID"]).copy()
        fishnet_base["GRID_ID"] = fishnet_base["GRID_ID"].astype(int)
        fishnet_base = fishnet_base.to_crs("EPSG:4326")

        found_ids = set(fishnet_base["GRID_ID"].tolist())
        missing_ids = set(union_grid_ids_sorted) - found_ids
        if missing_ids:
            print(f"Warning: {len(missing_ids)} GRID_ID(s) not found in {layer}. They will be skipped.")

        for fire_year in year_group:
            print(f"\n--- Drawing Year {fire_year} (1x{len(datasets)}, WUI={wui_year}) ---")
            fishnet_by_key: dict[str, gpd.GeoDataFrame] = {}
            has_any_data = False

            for runtime in dataset_runtime:
                config = runtime["config"]
                group_df = group_df_by_key[config["key"]]
                year_df = group_df[group_df["Year"] == fire_year][["GRID_ID", "category_id", "cluster", "cluster_label"]]

                if year_df.empty:
                    empty_fishnet = fishnet_base.iloc[0:0].copy()
                    empty_fishnet["category_id"] = pd.Series(dtype="object")
                    empty_fishnet["cluster"] = pd.Series(dtype="float")
                    empty_fishnet["cluster_label"] = pd.Series(dtype="object")
                    fishnet_by_key[config["key"]] = empty_fishnet
                    continue

                has_any_data = True
                year_ids = set(year_df["GRID_ID"].astype(int).tolist())
                fishnet_year = fishnet_base[fishnet_base["GRID_ID"].isin(year_ids)].copy()
                fishnet_year = fishnet_year.merge(year_df, on="GRID_ID", how="left")
                fishnet_by_key[config["key"]] = fishnet_year

            if not has_any_data:
                print(f"Skip Year {fire_year}: no clustered grid records.")
                continue

            output_path = build_group_output_path(OUTPUT_ROOT, datasets, fire_year)
            plot_cluster_map_group(
                continent=continent,
                fishnet_by_key=fishnet_by_key,
                fire_year=fire_year,
                datasets=datasets,
                cluster_name_map=shared_cluster_name_map,
                output_dpi=output_dpi,
                output_path=output_path,
            )

            if len(datasets) > 1:
                for runtime in dataset_runtime:
                    config = runtime["config"]
                    single_output_path = build_group_output_path(OUTPUT_ROOT, [config], fire_year)
                    plot_cluster_map_group(
                        continent=continent,
                        fishnet_by_key={config["key"]: fishnet_by_key.get(config["key"])},
                        fire_year=fire_year,
                        datasets=[config],
                        cluster_name_map=runtime["cluster_name_map"],
                        output_dpi=output_dpi,
                        output_path=single_output_path,
                    )


if __name__ == "__main__":
    main()
