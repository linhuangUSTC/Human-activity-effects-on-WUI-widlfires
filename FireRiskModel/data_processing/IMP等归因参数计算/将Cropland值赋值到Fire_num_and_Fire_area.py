#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称:
    将 Cropland 值赋值到 Fire num / Fire area Basic CSV

功能简介:
    1. 读取 I:\Processing data\Cropland 下各年份的 Cropland 统计结果。
    2. 直接按照 GRID_ID 和 Year 一一对应关系构建 lookup 表。
    3. 将 Cropland_Area_km2 回填到 Fire num / Fire area 的 Basic CSV。
    4. 输出到新的目录，不覆盖原始 Basic 文件。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


CROPLAND_DIR = Path(r"I:\Processing data\Cropland")
DATASET_CONFIGS = [
    {
        "name": "Fire_num_Basic",
        "input_dir": Path(r"I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic"),
        "output_dir": Path(
            r"I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic_with_Cropland"
        ),
    },
    {
        "name": "Fire_area_Basic",
        "input_dir": Path(r"I:\Processing data\渔网分类建模\Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic"),
        "output_dir": Path(
            r"I:\Processing data\渔网分类建模\Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic_with_Cropland"
        ),
    },
]
SOURCE_VALUE_COLUMN = "Cropland_Area_km2"
TARGET_VALUE_COLUMN = "Cropland_Area_km2"
EXPECTED_YEARS = list(range(2003, 2023))


def normalize_grid_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def normalize_year(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def resolve_source_grid_id_column(csv_path: Path) -> str:
    header_df = pd.read_csv(csv_path, nrows=0)
    columns = list(header_df.columns)
    if "GRID_ID" in columns:
        return "GRID_ID"
    if "s" in columns:
        return "s"
    raise ValueError(f"Missing GRID_ID column in Cropland CSV: {csv_path}")


def load_cropland_lookup() -> tuple[pd.DataFrame, list[int], list[int]]:
    cropland_parts: list[pd.DataFrame] = []
    loaded_years: set[int] = set()

    source_files = sorted(CROPLAND_DIR.glob("Grid_Clip_*_WUI_Cropland_Area.csv"))
    if not source_files:
        raise FileNotFoundError(f"No Cropland CSV files found in {CROPLAND_DIR}")

    for csv_path in source_files:
        grid_id_column = resolve_source_grid_id_column(csv_path)
        df = pd.read_csv(
            csv_path,
            usecols=["Year", grid_id_column, SOURCE_VALUE_COLUMN],
            low_memory=False,
        )
        if grid_id_column != "GRID_ID":
            df = df.rename(columns={grid_id_column: "GRID_ID"})

        df["GRID_ID_key"] = normalize_grid_id(df["GRID_ID"])
        df["Year_key"] = normalize_year(df["Year"])
        df[TARGET_VALUE_COLUMN] = pd.to_numeric(df[SOURCE_VALUE_COLUMN], errors="coerce")
        df = df.dropna(subset=["GRID_ID_key", "Year_key", TARGET_VALUE_COLUMN]).copy()
        if df.empty:
            continue

        loaded_years.update(df["Year_key"].astype(int).unique().tolist())
        df["Source_File"] = csv_path.name
        cropland_parts.append(df[["GRID_ID_key", "Year_key", TARGET_VALUE_COLUMN, "Source_File"]])

    if not cropland_parts:
        raise ValueError(f"No valid Cropland records found in {CROPLAND_DIR}")

    lookup_df = pd.concat(cropland_parts, ignore_index=True)
    lookup_df = (
        lookup_df.sort_values(["GRID_ID_key", "Year_key", "Source_File"])
        .drop_duplicates(subset=["GRID_ID_key", "Year_key"], keep="last")
        .reset_index(drop=True)
    )

    missing_years = sorted(set(EXPECTED_YEARS) - set(loaded_years))
    return lookup_df, sorted(loaded_years), missing_years


def process_basic_file(csv_path: Path, output_dir: Path, cropland_lookup: pd.DataFrame) -> dict[str, object]:
    df = pd.read_csv(csv_path, low_memory=False)
    if "GRID_ID" not in df.columns or "Year" not in df.columns:
        raise ValueError(f"Missing GRID_ID or Year columns in {csv_path}")

    df["GRID_ID_key"] = normalize_grid_id(df["GRID_ID"])
    df["Year_key"] = normalize_year(df["Year"])

    merged = df.merge(
        cropland_lookup[["GRID_ID_key", "Year_key", TARGET_VALUE_COLUMN]],
        how="left",
        on=["GRID_ID_key", "Year_key"],
        suffixes=("", "_new"),
    )

    merged_column_name = f"{TARGET_VALUE_COLUMN}_new"
    if merged_column_name in merged.columns:
        source_series = pd.to_numeric(merged[merged_column_name], errors="coerce")
        merged = merged.drop(columns=[merged_column_name])
    else:
        source_series = pd.to_numeric(merged[TARGET_VALUE_COLUMN], errors="coerce")

    missing_rows = int(source_series.isna().sum())
    matched_rows = int(source_series.notna().sum())
    matched_nonzero_rows = int((source_series.fillna(0.0) > 0).sum())
    merged[TARGET_VALUE_COLUMN] = source_series

    output_path = output_dir / csv_path.name
    merged = merged.drop(columns=["GRID_ID_key", "Year_key"])
    merged.to_csv(output_path, index=False, encoding="utf-8-sig", na_rep="")

    return {
        "filename": csv_path.name,
        "rows": len(merged),
        "matched_rows": matched_rows,
        "matched_nonzero_rows": matched_nonzero_rows,
        "missing_rows": missing_rows,
        "output_csv": str(output_path),
    }


def process_dataset(dataset_config: dict[str, object], cropland_lookup: pd.DataFrame) -> None:
    input_dir = Path(dataset_config["input_dir"])
    output_dir = Path(dataset_config["output_dir"])
    dataset_name = str(dataset_config["name"])

    if not input_dir.exists():
        raise FileNotFoundError(f"Basic input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    basic_files = sorted(input_dir.glob("*.csv"))
    if not basic_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    summary_rows: list[dict[str, object]] = []
    print(f"\n=== Processing {dataset_name} ===")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Processing {len(basic_files)} Basic CSV files...")
    for idx, csv_path in enumerate(basic_files, start=1):
        summary = process_basic_file(csv_path, output_dir, cropland_lookup)
        summary["dataset_name"] = dataset_name
        summary_rows.append(summary)
        print(
            f"[{idx}/{len(basic_files)}] {csv_path.name}: "
            f"rows={summary['rows']:,}, matched_nonzero_rows={summary['matched_nonzero_rows']:,}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "Cropland_value_assignment_summary.csv"
    try:
        summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        saved_summary_path = summary_csv
    except PermissionError:
        fallback_summary_csv = output_dir / f"Cropland_value_assignment_summary_{dataset_name}.csv"
        summary_df.to_csv(fallback_summary_csv, index=False, encoding="utf-8-sig")
        saved_summary_path = fallback_summary_csv
        print(
            f"Warning: summary file was locked, saved fallback summary instead: {fallback_summary_csv}"
        )

    print(f"Completed dataset: {dataset_name}")
    print(f"Summary saved to: {saved_summary_path}")


def main() -> None:
    print(f"Loading Cropland lookup tables from: {CROPLAND_DIR}")
    cropland_lookup, loaded_years, missing_years = load_cropland_lookup()
    print(
        f"Cropland lookup loaded: rows={len(cropland_lookup):,}, "
        f"unique GRID_ID-Year pairs={cropland_lookup[['GRID_ID_key', 'Year_key']].drop_duplicates().shape[0]:,}"
    )
    print(f"Loaded years: {loaded_years}")
    if missing_years:
        print(f"Warning: missing Cropland source years: {missing_years}")

    for dataset_config in DATASET_CONFIGS:
        process_dataset(dataset_config, cropland_lookup)


if __name__ == "__main__":
    main()
