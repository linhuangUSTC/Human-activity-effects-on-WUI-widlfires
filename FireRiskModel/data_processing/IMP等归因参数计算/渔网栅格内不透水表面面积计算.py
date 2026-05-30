#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称:
    渔网栅格内不透水表面面积计算

功能简介:
    1. 从 GAIA 全球城市边界数据中读取 2005/2010/2015/2020 年的 GUB shp。
    2. 将 GUB 数据投影到 World Cylindrical Equal Area (ESRI:54034)，并输出到原始 shp 所在目录。
    3. 读取 WUI.gdb 中对应年份的 Grid_Clip_{year}_WUI 图层。
    4. 计算每个 GRID_ID 下 WUI 与 GUB 的相交面积。
    5. 将结果输出到 B:\WUI\Picture 下的 CSV。
"""

from __future__ import annotations

import gc
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely


YEARS = [2005, 2010, 2015, 2020]
TARGET_CRS = "ESRI:54034"
BATCH_SIZE = 50000

GUB_DIR = Path(
    r"I:\Data_huanglin\map\urban boundary\Mapping global urban boundaries from the global artificial impervious area (GAIA) data"
)
WUI_GDB_PATH = Path(r"I:\Processing data\gdb集合\WUI.gdb")
WUI_LAYER_TEMPLATE = "Grid_Clip_{year}_WUI"
OUTPUT_DIR = Path(r"B:\WUI\Picture")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_gub_source_path(year: int) -> Path:
    return GUB_DIR / f"GUB_Global_{year}.shp"


def get_gub_projected_path(year: int) -> Path:
    return GUB_DIR / f"GUB_Global_{year}_World_Cylindrical_Equal_Area.shp"


def get_output_csv_path(year: int) -> Path:
    return OUTPUT_DIR / f"GUB_area_within_WUI_grid_{year}.csv"


def delete_shapefile_family(shp_path: Path) -> None:
    suffixes = [".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx", ".shp.xml", ".qix"]
    for suffix in suffixes:
        candidate = shp_path.with_suffix(suffix) if suffix != ".shp.xml" else shp_path.with_name(f"{shp_path.stem}.shp.xml")
        if candidate.exists():
            candidate.unlink()


def standardize_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf

    gdf = gdf.loc[gdf.geometry.notna()].copy()
    if gdf.empty:
        return gdf

    invalid_mask = ~gdf.geometry.is_valid
    if invalid_mask.any():
        gdf.loc[invalid_mask, "geometry"] = shapely.make_valid(gdf.loc[invalid_mask, "geometry"].values)

    gdf = gdf.loc[~gdf.geometry.is_empty].copy()
    return gdf


def ensure_projected_gub(year: int) -> Path:
    source_path = get_gub_source_path(year)
    projected_path = get_gub_projected_path(year)

    if not source_path.exists():
        raise FileNotFoundError(f"GUB source shapefile not found: {source_path}")

    if projected_path.exists():
        print(f"[{year}] projected GUB already exists: {projected_path.name}")
        return projected_path

    print(f"[{year}] reading original GUB shapefile...")
    gub_gdf = gpd.read_file(source_path, engine="pyogrio")
    if gub_gdf.crs is None:
        raise ValueError(f"GUB source shapefile has no CRS: {source_path}")

    print(f"[{year}] projecting GUB to {TARGET_CRS}...")
    gub_projected = gub_gdf.to_crs(TARGET_CRS)

    if projected_path.exists():
        delete_shapefile_family(projected_path)

    print(f"[{year}] writing projected shapefile...")
    gub_projected.to_file(projected_path, driver="ESRI Shapefile")

    del gub_gdf, gub_projected
    gc.collect()
    return projected_path


def read_wui_layer(year: int) -> gpd.GeoDataFrame:
    layer = WUI_LAYER_TEMPLATE.format(year=year)
    print(f"[{year}] reading WUI layer: {layer}")
    wui_gdf = gpd.read_file(
        WUI_GDB_PATH,
        layer=layer,
        columns=["GRID_ID"],
        engine="pyogrio",
    )
    if wui_gdf.empty:
        raise ValueError(f"WUI layer is empty: {layer}")

    wui_gdf["GRID_ID"] = pd.to_numeric(wui_gdf["GRID_ID"], errors="coerce").astype("Int64")
    wui_gdf = wui_gdf.dropna(subset=["GRID_ID"]).copy()

    if wui_gdf.crs is None:
        raise ValueError(f"WUI layer has no CRS: {layer}")
    if str(wui_gdf.crs) != TARGET_CRS and wui_gdf.crs.to_string() != TARGET_CRS:
        wui_gdf = wui_gdf.to_crs(TARGET_CRS)

    wui_gdf = standardize_geometries(wui_gdf[["GRID_ID", "geometry"]])
    if wui_gdf.empty:
        raise ValueError(f"WUI layer has no valid geometry after cleaning: {layer}")
    return wui_gdf


def build_zero_result_frame(wui_gdf: gpd.GeoDataFrame, year: int) -> pd.DataFrame:
    result = (
        wui_gdf[["GRID_ID"]]
        .drop_duplicates()
        .sort_values("GRID_ID")
        .reset_index(drop=True)
        .copy()
    )
    result.insert(0, "Year", year)
    result["GUB_area_km2"] = 0.0
    return result


def aggregate_gub_area_by_grid(year: int, projected_gub_path: Path, wui_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    info = pyogrio.read_info(projected_gub_path)
    total_features = int(info["features"])
    if total_features <= 0:
        raise ValueError(f"No features found in projected GUB shapefile: {projected_gub_path}")

    print(f"[{year}] projected GUB features: {total_features:,}")
    print(f"[{year}] WUI polygons: {len(wui_gdf):,}")

    grid_union_geometry: dict[int, object] = {}
    wui_geometry_lookup = wui_gdf.geometry.rename("wui_geometry")
    wui_join = wui_gdf[["GRID_ID", "geometry"]]

    for offset in range(0, total_features, BATCH_SIZE):
        batch_end = min(offset + BATCH_SIZE, total_features)
        print(f"[{year}] processing GUB batch {offset:,} - {batch_end:,} ...")

        gub_batch = pyogrio.read_dataframe(
            projected_gub_path,
            skip_features=offset,
            max_features=BATCH_SIZE,
            columns=[],
        )
        if gub_batch.empty:
            continue

        gub_batch = standardize_geometries(gub_batch[["geometry"]])
        if gub_batch.empty:
            continue

        joined = gpd.sjoin(gub_batch, wui_join, how="inner", predicate="intersects")
        if joined.empty:
            del gub_batch, joined
            gc.collect()
            continue

        joined = joined.join(wui_geometry_lookup, on="index_right")
        intersected = shapely.intersection(joined.geometry.values, joined["wui_geometry"].values)
        area_values = shapely.area(intersected)

        valid_mask = np.isfinite(area_values) & (area_values > 0)
        if not valid_mask.any():
            del gub_batch, joined, intersected, area_values
            gc.collect()
            continue

        clipped_df = pd.DataFrame(
            {
                "GRID_ID": joined.loc[valid_mask, "GRID_ID"].astype(np.int64).to_numpy(),
                "geometry": list(intersected[valid_mask]),
            }
        )

        for grid_id, group in clipped_df.groupby("GRID_ID", sort=False):
            batch_union = shapely.union_all(group["geometry"].to_numpy(dtype=object))
            previous_union = grid_union_geometry.get(int(grid_id))
            if previous_union is None:
                grid_union_geometry[int(grid_id)] = batch_union
            else:
                grid_union_geometry[int(grid_id)] = shapely.union_all(
                    np.array([previous_union, batch_union], dtype=object)
                )

        del gub_batch, joined, intersected, area_values, clipped_df
        gc.collect()

    result = build_zero_result_frame(wui_gdf, year)
    if grid_union_geometry:
        union_areas = shapely.area(np.array(list(grid_union_geometry.values()), dtype=object))
        stats_df = pd.DataFrame(
            {
                "GRID_ID": list(grid_union_geometry.keys()),
                "GUB_area_km2": union_areas.astype(float) / 1_000_000.0,
            }
        )
        result = result.merge(stats_df, on="GRID_ID", how="left", suffixes=("", "_calc"))
        result["GUB_area_km2"] = result["GUB_area_km2_calc"].fillna(result["GUB_area_km2"])
        result = result.drop(columns=["GUB_area_km2_calc"])

    result["GUB_area_km2"] = result["GUB_area_km2"].astype(float)
    return result


def process_year(year: int) -> pd.DataFrame:
    projected_gub_path = ensure_projected_gub(year)
    wui_gdf = read_wui_layer(year)
    result_df = aggregate_gub_area_by_grid(year, projected_gub_path, wui_gdf)

    output_csv = get_output_csv_path(year)
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(
        f"[{year}] saved CSV: {output_csv} | "
        f"rows={len(result_df):,}, nonzero_grids={(result_df['GUB_area_km2'] > 0).sum():,}"
    )

    del wui_gdf
    gc.collect()
    return result_df


def main() -> None:
    print("=== GUB area within WUI grids calculation ===")
    all_results: list[pd.DataFrame] = []

    for year in YEARS:
        print(f"\n=== Processing year {year} ===")
        year_df = process_year(year)
        all_results.append(year_df)

    combined_df = pd.concat(all_results, ignore_index=True)
    combined_output = OUTPUT_DIR / "GUB_area_within_WUI_grid_all_years.csv"
    combined_df.to_csv(combined_output, index=False, encoding="utf-8-sig")
    print(f"\nCombined CSV saved to: {combined_output}")
    print("All years completed.")


if __name__ == "__main__":
    main()
