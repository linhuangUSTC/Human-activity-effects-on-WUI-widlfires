#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称:
    将 GUB 面积赋值到 Fire num / Fire area Basic CSV

功能简介:
    1. 读取 I:\Processing data\Impervious surface 下的 GUB 面积统计结果。
    2. 按项目统一时间映射关系扩展到实际年份：
       2005 -> 2003-2007
       2010 -> 2008-2012
       2015 -> 2013-2017
       2020 -> 2018-2022
    3. 根据 GRID_ID 和 Year，将 GUB_area_km2 回填到 Fire num / Fire area 的 Basic CSV。
    4. 输出到新的目录，不覆盖原始 Basic 文件。
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd


IMPERVIOUS_DIR = Path(r"I:\Processing data\Impervious surface")
DATASET_CONFIGS = [
    {
        "name": "Fire_num_Basic",
        "input_dir": Path(r"I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic"),
        "output_dir": Path(
            r"I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic_with_GUB_area"
        ),
    },
    {
        "name": "Fire_area_Basic",
        "input_dir": Path(r"I:\Processing data\渔网分类建模\Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic"),
        "output_dir": Path(
            r"I:\Processing data\渔网分类建模\Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic_with_GUB_area"
        ),
    },
]

REFERENCE_YEARS = [2005, 2010, 2015, 2020]
YEAR_MAPPING = {
    2005: list(range(2003, 2008)),
    2010: list(range(2008, 2013)),
    2015: list(range(2013, 2018)),
    2020: list(range(2018, 2023)),
}


def normalize_grid_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def normalize_year(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def load_gub_lookup() -> pd.DataFrame:
    gub_parts: list[pd.DataFrame] = []

    for reference_year in REFERENCE_YEARS:
        csv_path = IMPERVIOUS_DIR / f"GUB_area_within_WUI_grid_{reference_year}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing GUB input CSV: {csv_path}")

        df = pd.read_csv(csv_path, usecols=["GRID_ID", "GUB_area_km2"], low_memory=False)
        df["GRID_ID_key"] = normalize_grid_id(df["GRID_ID"])
        df["GUB_area_km2"] = pd.to_numeric(df["GUB_area_km2"], errors="coerce")
        df = df.dropna(subset=["GRID_ID_key", "GUB_area_km2"]).copy()

        expanded_parts: list[pd.DataFrame] = []
        for actual_year in YEAR_MAPPING[reference_year]:
            year_df = df[["GRID_ID_key", "GUB_area_km2"]].copy()
            year_df["Year_key"] = actual_year
            year_df["Reference_WUI_Year"] = reference_year
            expanded_parts.append(year_df)

        gub_parts.append(pd.concat(expanded_parts, ignore_index=True))

    lookup_df = pd.concat(gub_parts, ignore_index=True)
    lookup_df = (
        lookup_df.sort_values(["GRID_ID_key", "Year_key", "Reference_WUI_Year"])
        .drop_duplicates(subset=["GRID_ID_key", "Year_key"], keep="last")
        .reset_index(drop=True)
    )
    return lookup_df


def process_basic_file(csv_path: Path, output_dir: Path, gub_lookup: pd.DataFrame) -> dict[str, object]:
    df = pd.read_csv(csv_path, low_memory=False)
    if "GRID_ID" not in df.columns or "Year" not in df.columns:
        raise ValueError(f"Missing GRID_ID or Year columns in {csv_path}")

    df["GRID_ID_key"] = normalize_grid_id(df["GRID_ID"])
    df["Year_key"] = normalize_year(df["Year"])

    merged = df.merge(
        gub_lookup[["GRID_ID_key", "Year_key", "GUB_area_km2"]],
        how="left",
        on=["GRID_ID_key", "Year_key"],
        suffixes=("", "_new"),
    )

    if "GUB_area_km2_new" in merged.columns:
        source_series = pd.to_numeric(merged["GUB_area_km2_new"], errors="coerce")
        merged = merged.drop(columns=["GUB_area_km2_new"])
    else:
        source_series = pd.to_numeric(merged["GUB_area_km2"], errors="coerce")

    missing_rows = int(source_series.isna().sum())
    matched_rows = int(source_series.notna().sum())
    matched_nonzero_rows = int((source_series.fillna(0.0) > 0).sum())
    merged["GUB_area_km2"] = source_series

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


def process_dataset(dataset_config: dict[str, object], gub_lookup: pd.DataFrame) -> None:
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
        summary = process_basic_file(csv_path, output_dir, gub_lookup)
        summary["dataset_name"] = dataset_name
        summary_rows.append(summary)
        print(
            f"[{idx}/{len(basic_files)}] {csv_path.name}: "
            f"rows={summary['rows']:,}, matched_nonzero_rows={summary['matched_nonzero_rows']:,}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "GUB_area_assignment_summary.csv"
    try:
        summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        saved_summary_path = summary_csv
    except PermissionError:
        fallback_summary_csv = output_dir / f"GUB_area_assignment_summary_{dataset_name}.csv"
        summary_df.to_csv(fallback_summary_csv, index=False, encoding="utf-8-sig")
        saved_summary_path = fallback_summary_csv
        print(
            f"Warning: summary file was locked, saved fallback summary instead: {fallback_summary_csv}"
        )

    print(f"Completed dataset: {dataset_name}")
    print(f"Summary saved to: {saved_summary_path}")
def main() -> None:
    print(f"Loading GUB lookup tables for reference years: {REFERENCE_YEARS}")
    gub_lookup = load_gub_lookup()
    print(
        f"GUB lookup loaded: rows={len(gub_lookup):,}, "
        f"unique GRID_ID-Year pairs={gub_lookup[['GRID_ID_key', 'Year_key']].drop_duplicates().shape[0]:,}"
    )

    for dataset_config in DATASET_CONFIGS:
        process_dataset(dataset_config, gub_lookup)


if __name__ == "__main__":
    main()
