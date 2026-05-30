#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
统计 All / Interface / Intermix WUI 中，人类活动(Human Footprint)对
Fire frequency 和 Normalized burned area 的平均影响，并输出为简单 xlsx。

输入:
    - B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num\HFI_effect_*.csv
    - B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area\HFI_effect_*.csv

输出:
    - B:\WUI\Picture\Human_activity_effect_mean_Interface_Intermix_2003_2022.xlsx
"""

from __future__ import annotations

import glob
import math
import re
from pathlib import Path

import pandas as pd


YEAR_START = 2003
YEAR_END = 2022
YEAR_TAG = f"{YEAR_START}_{YEAR_END}"
CHUNK_SIZE = 500_000

KEY_YEAR_COLUMN = "Year"
EFFECT_COLUMN = "diff_contribution"

OUTPUT_DIR = Path(r"B:\WUI\Picture")
OUTPUT_PATH = OUTPUT_DIR / f"Human_activity_effect_mean_Interface_Intermix_{YEAR_TAG}.xlsx"

DATASETS = [
    {
        "metric": "Fire frequency",
        "metric_slug": "Fire_frequency",
        "effect_dir": Path(r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num"),
        "effect_suffix": "fire_num_with_WUI_offset",
    },
    {
        "metric": "Normalized burned area",
        "metric_slug": "Normalized_burned_area",
        "effect_dir": Path(r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area"),
        "effect_suffix": "fire_area",
    },
]

def parse_wui_mode_from_effect_file(csv_path: str, suffix: str) -> str:
    stem = Path(csv_path).stem
    pattern = rf"HFI_effect_((face|mix)_[A-D]_[A-Z]+)_{suffix}$"
    match = re.search(pattern, stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unrecognized HFI effect file name: {stem}")
    return match.group(2).lower()


def summarize_effect_files(dataset: dict[str, object]) -> list[dict[str, object]]:
    effect_dir = Path(str(dataset["effect_dir"]))
    effect_suffix = str(dataset["effect_suffix"])
    effect_files = sorted(glob.glob(str(effect_dir / "HFI_effect_*.csv")))
    if not effect_files:
        raise FileNotFoundError(f"No HFI effect CSV files found in: {effect_dir}")

    accumulators = {
        "face": {"sum": 0.0, "sum_sq": 0.0, "count": 0, "file_count": 0},
        "mix": {"sum": 0.0, "sum_sq": 0.0, "count": 0, "file_count": 0},
    }

    for csv_path in effect_files:
        try:
            wui_mode = parse_wui_mode_from_effect_file(csv_path, effect_suffix)
        except ValueError as exc:
            print(f"Skip file: {exc}")
            continue
        if wui_mode not in accumulators:
            continue

        has_valid_rows = False
        for chunk in pd.read_csv(
            csv_path,
            usecols=[KEY_YEAR_COLUMN, EFFECT_COLUMN],
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            chunk[KEY_YEAR_COLUMN] = pd.to_numeric(chunk[KEY_YEAR_COLUMN], errors="coerce")
            chunk[EFFECT_COLUMN] = pd.to_numeric(chunk[EFFECT_COLUMN], errors="coerce")
            chunk = chunk[
                chunk[KEY_YEAR_COLUMN].between(YEAR_START, YEAR_END, inclusive="both")
            ].dropna(subset=[EFFECT_COLUMN])
            if chunk.empty:
                continue

            has_valid_rows = True
            values = chunk[EFFECT_COLUMN]
            accumulators[wui_mode]["sum"] += float(values.sum())
            accumulators[wui_mode]["sum_sq"] += float((values * values).sum())
            accumulators[wui_mode]["count"] += int(values.count())

        if has_valid_rows:
            accumulators[wui_mode]["file_count"] += 1

    all_accumulator = {
        "sum": accumulators["face"]["sum"] + accumulators["mix"]["sum"],
        "sum_sq": accumulators["face"]["sum_sq"] + accumulators["mix"]["sum_sq"],
        "count": accumulators["face"]["count"] + accumulators["mix"]["count"],
        "file_count": accumulators["face"]["file_count"] + accumulators["mix"]["file_count"],
    }

    rows: list[dict[str, object]] = []
    for wui_label, accumulator in [
        ("All", all_accumulator),
        ("Interface", accumulators["face"]),
        ("Intermix", accumulators["mix"]),
    ]:
        rows.append(build_summary_row(dataset, wui_label, accumulator))
    return rows


def build_summary_row(
    dataset: dict[str, object],
    wui_label: str,
    accumulator: dict[str, float | int],
) -> dict[str, object]:
    total_count = int(accumulator["count"])
    total_sum = float(accumulator["sum"])
    total_sum_sq = float(accumulator["sum_sq"])
    mean_effect = total_sum / total_count if total_count else pd.NA
    if total_count > 1:
        variance = max((total_sum_sq - (total_sum * total_sum / total_count)) / (total_count - 1), 0.0)
        se_effect = math.sqrt(variance) / math.sqrt(total_count)
    else:
        se_effect = pd.NA

    return {
        "metric": str(dataset["metric"]),
        "metric_slug": str(dataset["metric_slug"]),
        "wui_type": wui_label,
        "year_start": YEAR_START,
        "year_end": YEAR_END,
        "n_samples": total_count,
        "source_file_count": int(accumulator["file_count"]),
        "mean_human_activity_effect": mean_effect,
        "se_human_activity_effect": se_effect,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        rows.extend(summarize_effect_files(dataset))

    output_table = pd.DataFrame(rows)
    output_table.to_excel(OUTPUT_PATH, sheet_name="mean", index=False)
    print(f"Saved mean table to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
