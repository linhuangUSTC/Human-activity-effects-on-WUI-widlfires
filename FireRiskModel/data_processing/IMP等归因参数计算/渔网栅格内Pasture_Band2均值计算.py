#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称:
    渔网栅格内 Pasture Band2 均值计算

功能简介:
    1. 读取源栅格的 band2。
    2. 将 band2 投影到与项目现有脚本一致的 World Cylindrical Equal Area (ESRI:54034)。
    3. 将投影后的 band2 输出回原始 TIF 所在目录。
    4. 读取 WUI.gdb 中各年份的 Grid_Clip_{year}_WUI 图层。
    5. 将 WUI 鱼网栅格化到投影后 raster 的像元网格上，并按 GRID_ID 统计 band2 的像元均值。
    6. 输出各年份 CSV 到 B:\WUI\Picture。
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import calculate_default_transform, reproject
import shapely


TARGET_CRS = "ESRI:54034"
YEARS = [2005, 2010, 2015, 2020]
BATCH_SIZE = 100_000

SOURCE_TIF = Path(
    r"I:\Data_huanglin\Cropland and Pasture\global-agland-2015\global-agland-2015\outputs\all_correct_to_FAO_scale_itr3_fr_0\agland_map_output_3.tif"
)
PROJECTED_TIF = SOURCE_TIF.with_name("pasture_map_output_3_band2_World_Cylindrical_Equal_Area.tif")
OUTPUT_DIR = Path(r"B:\WUI\Picture")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WUI_GDB_CANDIDATES = [
    Path("I:/Processing data/gdb\u96c6\u5408/WUI.gdb"),
    Path(r"I:\Processing data\WUI.gdb"),
]
WUI_LAYER_TEMPLATE = "Grid_Clip_{year}_WUI"


def log(message: str) -> None:
    print(message, flush=True)


def resolve_wui_gdb_path() -> Path:
    for candidate in WUI_GDB_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No valid WUI.gdb path found. Checked: {0}".format(
            ", ".join(str(path) for path in WUI_GDB_CANDIDATES)
        )
    )


def get_output_csv_path(year: int) -> Path:
    return OUTPUT_DIR / f"Pasture_band2_mean_within_WUI_grid_{year}.csv"


def standardize_geometries(gdf):
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


def ensure_projected_band2() -> Path:
    if not SOURCE_TIF.exists():
        raise FileNotFoundError(f"Source TIF not found: {SOURCE_TIF}")

    if PROJECTED_TIF.exists():
        with rasterio.open(PROJECTED_TIF) as src:
            if src.count == 1 and src.crs and src.crs.to_string() == TARGET_CRS:
                log(f"Projected band2 already exists: {PROJECTED_TIF}")
                return PROJECTED_TIF

    log("Projecting source raster band2 to ESRI:54034 ...")
    with rasterio.open(SOURCE_TIF) as src:
        if src.crs is None:
            raise ValueError(f"Source TIF has no CRS: {SOURCE_TIF}")
        if src.count < 2:
            raise ValueError(f"Source TIF has fewer than 2 bands: {SOURCE_TIF}")

        transform, width, height = calculate_default_transform(
            src.crs,
            TARGET_CRS,
            src.width,
            src.height,
            *src.bounds,
        )

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            crs=TARGET_CRS,
            transform=transform,
            width=width,
            height=height,
            dtype="float64",
            nodata=np.nan,
            compress="LZW",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="IF_SAFER",
        )

        if PROJECTED_TIF.exists():
            PROJECTED_TIF.unlink()

        with rasterio.open(PROJECTED_TIF, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 2),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=TARGET_CRS,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )

    log(f"Projected band2 saved to: {PROJECTED_TIF}")
    return PROJECTED_TIF


def load_projected_raster(projected_tif: Path) -> tuple[np.ndarray, rasterio.Affine]:
    with rasterio.open(projected_tif) as src:
        if src.crs is None or src.crs.to_string() != TARGET_CRS:
            raise ValueError(f"Projected TIF CRS is not {TARGET_CRS}: {projected_tif}")
        band_data = src.read(1)
        transform = src.transform

    band_data = np.asarray(band_data, dtype=np.float64)
    return band_data, transform


def build_zone_raster(gdb_path: Path, layer: str, raster_shape: tuple[int, int], raster_transform) -> tuple[np.ndarray, np.ndarray]:
    info = pyogrio.read_info(gdb_path, layer=layer)
    total_features = int(info["features"])
    if total_features <= 0:
        raise ValueError(f"WUI layer is empty: {layer}")

    zone_raster = np.zeros(raster_shape, dtype=np.int32)
    grid_ids_collected: list[np.ndarray] = []

    log(f"[{layer}] features: {total_features:,}")

    for offset in range(0, total_features, BATCH_SIZE):
        batch_end = min(offset + BATCH_SIZE, total_features)
        log(f"[{layer}] rasterizing batch {offset:,} - {batch_end:,} ...")

        batch = pyogrio.read_dataframe(
            gdb_path,
            layer=layer,
            skip_features=offset,
            max_features=BATCH_SIZE,
            columns=["GRID_ID"],
        )
        if batch.empty:
            continue

        batch["GRID_ID"] = pd.to_numeric(batch["GRID_ID"], errors="coerce")
        batch = batch.dropna(subset=["GRID_ID"]).copy()
        if batch.empty:
            continue

        batch["GRID_ID"] = batch["GRID_ID"].astype(np.int64)
        batch = standardize_geometries(batch[["GRID_ID", "geometry"]])
        if batch.empty:
            continue

        grid_ids = batch["GRID_ID"].to_numpy(dtype=np.int64, copy=False)
        grid_ids_collected.append(grid_ids.copy())

        shapes = (
            (geom, int(grid_id))
            for geom, grid_id in zip(batch.geometry.values, grid_ids)
        )
        rasterize(
            shapes=shapes,
            out=zone_raster,
            transform=raster_transform,
            fill=0,
            all_touched=False,
            dtype="int32",
        )

        del batch, grid_ids, shapes
        gc.collect()

    if not grid_ids_collected:
        raise ValueError(f"No valid GRID_ID geometry found in layer: {layer}")

    unique_grid_ids = np.unique(np.concatenate(grid_ids_collected))
    return zone_raster, unique_grid_ids


def calculate_zonal_mean(year: int, band_data: np.ndarray, zone_raster: np.ndarray, grid_ids: np.ndarray) -> pd.DataFrame:
    valid_mask = np.isfinite(band_data) & (zone_raster > 0)
    covered_pixels = int(np.count_nonzero(valid_mask))
    log(f"[{year}] covered pixels for zonal mean: {covered_pixels:,}")

    zone_values = zone_raster[valid_mask]
    raster_values = band_data[valid_mask]

    sums = np.bincount(zone_values, weights=raster_values)
    counts = np.bincount(zone_values)

    result = pd.DataFrame(
        {
            "Year": year,
            "GRID_ID": grid_ids.astype(np.int64, copy=False),
        }
    )

    result["Pasture_Band2_Mean"] = np.nan
    result["Raster_Pixel_Count"] = 0

    valid_grid_mask = grid_ids < len(counts)
    valid_grid_ids = grid_ids[valid_grid_mask]
    grid_counts = counts[valid_grid_ids].astype(np.int64, copy=False)
    grid_means = np.divide(
        sums[valid_grid_ids],
        counts[valid_grid_ids],
        out=np.full(valid_grid_ids.shape, np.nan, dtype=np.float64),
        where=counts[valid_grid_ids] > 0,
    )

    result.loc[valid_grid_mask, "Raster_Pixel_Count"] = grid_counts
    result.loc[valid_grid_mask, "Pasture_Band2_Mean"] = grid_means
    result = result.sort_values("GRID_ID").reset_index(drop=True)
    return result


def process_year(year: int, gdb_path: Path, band_data: np.ndarray, raster_transform) -> pd.DataFrame:
    layer = WUI_LAYER_TEMPLATE.format(year=year)
    log("")
    log("=" * 70)
    log(f"Start processing year {year}")

    zone_raster, grid_ids = build_zone_raster(
        gdb_path=gdb_path,
        layer=layer,
        raster_shape=band_data.shape,
        raster_transform=raster_transform,
    )
    result_df = calculate_zonal_mean(year, band_data, zone_raster, grid_ids)

    output_csv = get_output_csv_path(year)
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    log(
        f"[{year}] saved CSV: {output_csv} | "
        f"rows={len(result_df):,}, nonnull_mean={(result_df['Pasture_Band2_Mean'].notna()).sum():,}"
    )

    del zone_raster, grid_ids
    gc.collect()
    return result_df


def main() -> None:
    log("=== Pasture band2 mean within WUI grids calculation ===")
    gdb_path = resolve_wui_gdb_path()
    projected_tif = ensure_projected_band2()
    band_data, raster_transform = load_projected_raster(projected_tif)

    for year in YEARS:
        process_year(
            year=year,
            gdb_path=gdb_path,
            band_data=band_data,
            raster_transform=raster_transform,
        )
        gc.collect()

    log("All years completed.")


if __name__ == "__main__":
    main()
