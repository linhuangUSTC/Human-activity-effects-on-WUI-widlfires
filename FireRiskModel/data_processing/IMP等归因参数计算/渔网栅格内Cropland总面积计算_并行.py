#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称:
    渔网栅格内 Cropland 总面积计算

功能简介:
    1. 在 Python 3 环境下使用 fiona + rasterio + shapely 处理 WUI 与 landcover 数据。
    2. 按项目统一时间映射关系，将 landcover 年份映射到参考 WUI 年份：
       2003-2007 -> 2005，2008-2012 -> 2010，2013-2017 -> 2015，2018-2022 -> 2020。
    3. 读取参考年份的 WUI 图层与 landcover match CSV。
    4. 逐个 GRID_ID 裁剪匹配 TIF 的对应年份波段。
    5. 统计 Cropland 类别（像元值 10/11/12/20）的总面积。
    6. 输出每年的 CSV 到 B:\WUI\Picture。
"""

from __future__ import annotations

import csv
import gc
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable

import fiona
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom


YEARS = list(range(2003, 2023))
TARGET_PIXEL_VALUES = {10, 11, 12, 20}
OUTPUT_DIR = r"B:\WUI\Picture"
MATCH_CSV_TEMPLATE = r"I:\Processing data\landcover match\Grid_Clip_{wui_year}_WUI_matches.csv"
OUTPUT_CSV_TEMPLATE = r"B:\WUI\Picture\Grid_Clip_{year}_WUI_Cropland_Area.csv"
WUI_GDB_CANDIDATES = [
    r"I:\Processing data\gdb集合\WUI.gdb",
    r"I:\Processing data\WUI.gdb",
]
WUI_YEAR_GROUPS = {
    2005: range(2003, 2008),
    2010: range(2008, 2013),
    2015: range(2013, 2018),
    2020: range(2018, 2023),
}
SAVE_BATCH_SIZE = 5000
MAX_WORKERS_PER_GROUP = 5


def log(message: str) -> None:
    print(message)


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def resolve_wui_gdb_path() -> str:
    for candidate in WUI_GDB_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("No valid WUI.gdb path found. Checked: {0}".format(", ".join(WUI_GDB_CANDIDATES)))


def get_reference_wui_year(year: int) -> int:
    for wui_year, year_range in WUI_YEAR_GROUPS.items():
        if year in year_range:
            return wui_year
    raise ValueError("No reference WUI year mapping found for year: {0}".format(year))


def get_available_layers(gdb_path: str) -> set[str]:
    return set(fiona.listlayers(gdb_path))


def get_wui_layer_name(reference_wui_year: int) -> str:
    return "Grid_Clip_{0}_WUI".format(reference_wui_year)


def year_to_band_index(year: int) -> int:
    return year - 1999


def get_output_csv_path(year: int) -> str:
    return OUTPUT_CSV_TEMPLATE.format(year=year)


def get_grouped_years(years: Iterable[int]) -> list[tuple[int, list[int]]]:
    grouped: dict[int, list[int]] = {}
    for year in years:
        reference_wui_year = get_reference_wui_year(year)
        grouped.setdefault(reference_wui_year, []).append(year)
    return [(reference_wui_year, sorted(group_years)) for reference_wui_year, group_years in sorted(grouped.items())]


def load_tif_mapping_from_csv(csv_path: str) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            grid_id = str(row["GRID_ID"]).strip()
            tif_field = (row.get("Matching_TIF_Files") or "").strip()
            tif_files = [path for path in tif_field.split("; ") if path]
            mapping[grid_id] = tif_files
    return mapping


def _valid_mask_from_band(band_data: np.ndarray | np.ma.MaskedArray, nodata_value: float | int | None) -> tuple[np.ndarray, np.ndarray]:
    if np.ma.isMaskedArray(band_data):
        data = np.asarray(band_data.data)
        valid_mask = ~np.asarray(band_data.mask)
    else:
        data = np.asarray(band_data)
        valid_mask = np.ones(data.shape, dtype=bool)

    if np.issubdtype(data.dtype, np.floating):
        valid_mask &= ~np.isnan(data)

    if nodata_value is not None:
        try:
            if np.isnan(nodata_value):
                valid_mask &= ~np.isnan(data)
            else:
                valid_mask &= data != nodata_value
        except TypeError:
            valid_mask &= data != nodata_value

    return data, valid_mask


def calculate_target_area_km2(
    geometry: dict,
    geometry_crs,
    tif_files: Iterable[str],
    band_index: int,
) -> float:
    total_area_km2 = 0.0

    for tif_file in tif_files:
        if not tif_file or not os.path.exists(tif_file):
            continue

        try:
            with rasterio.open(tif_file) as src:
                if band_index < 1 or band_index > src.count:
                    log("Warning: invalid band index {0} for file {1}, skipped.".format(band_index, tif_file))
                    continue

                geometry_for_raster = geometry
                if geometry_crs and src.crs and geometry_crs != src.crs:
                    geometry_for_raster = transform_geom(geometry_crs, src.crs, geometry)

                try:
                    clipped, _ = mask(
                        src,
                        [geometry_for_raster],
                        crop=True,
                        indexes=band_index,
                        filled=False,
                    )
                except ValueError:
                    continue

                # rasterio.mask.mask returns a 2D array when indexes is a single int.
                # Using clipped[0] here would only keep the first row and severely
                # underestimate the area.
                band_data = clipped if clipped.ndim == 2 else clipped[0]
                data, valid_mask = _valid_mask_from_band(band_data, src.nodata)
                if data.size == 0:
                    del clipped, band_data, data, valid_mask
                    continue

                target_mask = np.isin(data.astype(np.int32, copy=False), list(TARGET_PIXEL_VALUES))
                target_pixel_count = int(np.count_nonzero(valid_mask & target_mask))
                if target_pixel_count <= 0:
                    del clipped, band_data, data, valid_mask, target_mask
                    continue

                pixel_area_m2 = abs(src.res[0] * src.res[1])
                total_area_km2 += (target_pixel_count * pixel_area_m2) / 1_000_000.0
                del clipped, band_data, data, valid_mask, target_mask
        except Exception as exc:
            log("Warning: failed to process raster {0}".format(tif_file))
            log("Error: {0}".format(exc))
            continue

    return round(total_area_km2, 6)


def write_results(results: list[dict[str, object]], output_csv: str, is_first_save: bool) -> None:
    mode = "w" if is_first_save else "a"
    with open(output_csv, mode, encoding="utf-8-sig", newline="") as csvfile:
        fieldnames = ["Year", "GRID_ID", "Cropland_Area_km2"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if is_first_save:
            writer.writeheader()
        writer.writerows(results)


def process_year(year: int, gdb_path: str, available_layers: set[str]) -> bool:
    reference_wui_year = get_reference_wui_year(year)
    wui_layer = get_wui_layer_name(reference_wui_year)
    csv_path = MATCH_CSV_TEMPLATE.format(wui_year=reference_wui_year)
    output_csv = get_output_csv_path(year)
    band_index = year_to_band_index(year)

    if wui_layer not in available_layers or not os.path.exists(csv_path):
        log("Skipped year {0}: missing WUI layer {1} or match CSV {2}".format(year, wui_layer, csv_path))
        return False

    log("")
    log("=" * 60)
    log("Start processing year {0}".format(year))
    log("Reference WUI year: {0}".format(reference_wui_year))
    log("WUI layer: {0}".format(wui_layer))
    log("Match CSV: {0}".format(csv_path))
    log("Output CSV: {0}".format(output_csv))
    log("Target band: {0}".format(band_index))
    log("Cropland class values: {0}".format(sorted(TARGET_PIXEL_VALUES)))

    grid_tif_mapping = load_tif_mapping_from_csv(csv_path)
    pending_grid_ids = set(grid_tif_mapping.keys())
    log("Loaded {0} match records from CSV".format(len(grid_tif_mapping)))

    results: list[dict[str, object]] = []
    first_save_done = False
    processed_count = 0
    batch_start_time = time.time()

    with fiona.open(gdb_path, layer=wui_layer) as source:
        geometry_crs = source.crs_wkt or source.crs

        for feature in source:
            if not pending_grid_ids:
                break

            grid_id_value = feature["properties"].get("GRID_ID")
            if grid_id_value is None:
                continue

            grid_id = str(grid_id_value).strip()
            if grid_id not in pending_grid_ids:
                continue

            pending_grid_ids.remove(grid_id)
            geometry = feature.get("geometry")
            if not geometry:
                continue

            total_area_km2 = calculate_target_area_km2(
                geometry=geometry,
                geometry_crs=geometry_crs,
                tif_files=grid_tif_mapping.get(grid_id, []),
                band_index=band_index,
            )

            results.append(
                {
                    "Year": year,
                    "GRID_ID": grid_id,
                    "Cropland_Area_km2": total_area_km2,
                }
            )
            processed_count += 1

            if processed_count % SAVE_BATCH_SIZE == 0:
                elapsed_time = time.time() - batch_start_time
                log("Processed {0} GRID_ID values".format(processed_count))
                log("Elapsed time for {0} GRID_ID values: {1:.2f} seconds".format(SAVE_BATCH_SIZE, elapsed_time))
                write_results(results, output_csv, is_first_save=not first_save_done)
                first_save_done = True
                results.clear()
                batch_start_time = time.time()
                gc.collect()

    if results:
        log("Saving remaining {0} GRID_ID results...".format(len(results)))
        write_results(results, output_csv, is_first_save=not first_save_done)
        results.clear()
        gc.collect()

    log("Completed year {0}. Processed GRID_ID values: {1}".format(year, processed_count))
    if pending_grid_ids:
        log("Unmatched GRID_ID values left after scanning WUI layer: {0}".format(len(pending_grid_ids)))
    del grid_tif_mapping, pending_grid_ids, results
    gc.collect()
    return True


def process_year_group_parallel(
    reference_wui_year: int,
    group_years: list[int],
    gdb_path: str,
    available_layers: set[str],
) -> tuple[list[int], list[int]]:
    processed_years: list[int] = []
    skipped_years: list[int] = []
    max_workers = min(MAX_WORKERS_PER_GROUP, len(group_years), max(1, os.cpu_count() or 1))

    log("")
    log("#" * 70)
    log(
        "Start parallel group for reference WUI year {0}: landcover years {1}, workers={2}".format(
            reference_wui_year, group_years, max_workers
        )
    )

    if max_workers <= 1:
        for year in group_years:
            success = process_year(year, gdb_path, available_layers)
            if success:
                processed_years.append(year)
            else:
                skipped_years.append(year)
        return processed_years, skipped_years

    with ProcessPoolExecutor(max_workers=max_workers, max_tasks_per_child=1) as executor:
        future_to_year = {
            executor.submit(process_year, year, gdb_path, available_layers): year
            for year in group_years
        }
        for future in as_completed(future_to_year):
            year = future_to_year[future]
            try:
                success = future.result()
            except Exception as exc:
                log("Year {0} failed: {1}".format(year, exc))
                skipped_years.append(year)
                continue

            if success:
                processed_years.append(year)
            else:
                skipped_years.append(year)
    del future_to_year
    gc.collect()

    log(
        "Completed parallel group for reference WUI year {0}. Processed={1}, Skipped={2}".format(
            reference_wui_year, sorted(processed_years), sorted(skipped_years)
        )
    )
    return sorted(processed_years), sorted(skipped_years)


def process_all_years() -> None:
    ensure_output_dir()
    gdb_path = resolve_wui_gdb_path()
    available_layers = get_available_layers(gdb_path)
    years = YEARS
    processed_years: list[int] = []
    skipped_years: list[int] = []
    existing_output_years = sorted(year for year in years if os.path.exists(get_output_csv_path(year)))
    existing_output_years_set = set(existing_output_years)
    pending_years = [year for year in years if year not in existing_output_years_set]

    log("Selected years: {0}".format(years))
    if existing_output_years:
        log("Skipped years because CSV already exists: {0}".format(existing_output_years))

    if not pending_years:
        log("")
        log("All requested years already have CSV outputs. Nothing to process.")
        return

    for reference_wui_year, group_years in get_grouped_years(pending_years):
        group_processed, group_skipped = process_year_group_parallel(
            reference_wui_year=reference_wui_year,
            group_years=group_years,
            gdb_path=gdb_path,
            available_layers=available_layers,
        )
        processed_years.extend(group_processed)
        skipped_years.extend(group_skipped)
        gc.collect()

    log("")
    log("All processing finished.")
    log("Processed years: {0}".format(sorted(processed_years)))
    log("Skipped years: {0}".format(sorted(existing_output_years + skipped_years)))


def main() -> None:
    process_all_years()


if __name__ == "__main__":
    main()
