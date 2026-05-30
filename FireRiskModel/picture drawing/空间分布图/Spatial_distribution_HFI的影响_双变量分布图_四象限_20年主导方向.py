#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称: HFI 对 Fire frequency 与 Normalized burned area 影响的四象限主导方向空间分布图（2003-2022）

功能简介:
    1. 从 HFI 净效应固定效应结果 CSV 中分别读取 Fire frequency 和 Normalized burned area 的
       diff_contribution 以及 delta_sig。
    2. 仅保留 GRID_ID、Year、diff_contribution、delta_sig，并在全局范围内合并。
    3. 对两个响应变量都显著（delta_sig == TRUE）的记录，以 0 为界划分四象限。
    4. 对每个 GRID_ID，统计 2003-2022 年间四象限出现频数，频数最多的方向定义为该格网主导方向。
       - 若某格网不足 20 年，仅基于其已有年份统计。
       - 若某格网从未出现“双变量均显著”的年份，则保留为灰色。
       - 若“不显著”年份数多于任一显著方向的频数，则该格网归为不显著（灰色）。
       - 若主导频数并列，则优先选择平均合成效应强度更高的方向；若仍并列，则优先最近年份出现的方向。
    5. 合并 2005/2010/2015/2020 四套 WUI 渔网图层，并输出一张全球主导方向图。

输入数据:
    - Fire frequency HFI effect CSV:
      B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num
    - Normalized burned area HFI effect CSV:
      B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area
    - WUI 渔网:
      I:\Processing data\gdb集合\WUI.gdb
    - 大陆底图:
      I:\Data_huanglin\map\world continent boundary\continent.shp

输出:
    - B:\WUI\Picture\Bivariate_Human Footprint_Effect_FireFrequency_NormalizedFireArea_Dominant_2003_2022.png
"""

from __future__ import annotations

import glob
import os
from typing import Iterable

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

FONT_SCALE = 1.2
LEGEND_FONT_SIZE = 14


def _fs(size: float) -> float:
    return size * FONT_SCALE


FIRE_FREQUENCY_EFFECT_DIR = r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num"
NORMALIZED_FIRE_AREA_EFFECT_DIR = r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area"

WUI_GDB_PATH = r"I:\Processing data\gdb集合\WUI.gdb"
WUI_YEARS = [2005, 2010, 2015, 2020]
WUI_LAYER_TEMPLATE = "Grid_Clip_{wui_year}_WUI"

CONTINENT_SHP_PATH = r"I:\Data_huanglin\map\world continent boundary\continent.shp"

OUTPUT_DIR = r"B:\WUI\Picture"
OUTPUT_FILENAME = "Bivariate_Human Footprint_Effect_FireFrequency_NormalizedFireArea_Dominant_2003_2022.png"
OUTPUT_DPI = 1200

FIRE_YEAR_START = 2003
FIRE_YEAR_END = 2022

MAX_IDS_FOR_WHERE = 50000
WHERE_CHUNK_SIZE = 3000

BASEMAP_FACE = "#ffffff"
BASEMAP_EDGE = "#000000"
NON_SIG_COLOR = "#dcdcdc"

BIVARIATE_COLORS = {
    (1, 1): "#63BE7B",
    (2, 1): "#F3BB6E",
    (1, 2): "#6BA6E1",
    (2, 2): "#F68183",
}


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


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    values = series.astype(str).str.strip().str.upper()
    return values.isin({"TRUE", "T", "1", "YES", "Y"})


def _bool_mask(series: pd.Series) -> pd.Series:
    return series.astype("boolean").fillna(False).astype(bool)


def read_effect_from_csv_dir(source_dir: str, effect_col: str, sig_col: str) -> pd.DataFrame:
    csv_files = _resolve_effect_files(source_dir)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {source_dir}")

    parts: list[pd.DataFrame] = []
    print(f"Reading effect '{effect_col}' from {len(csv_files)} CSV file(s): {source_dir}")

    for csv_path in csv_files:
        header = pd.read_csv(csv_path, nrows=0)
        cols = header.columns.tolist()

        grid_col = _pick_column(cols, ["GRID_ID", "grid_id", "Grid_ID"])
        year_col = _pick_column(cols, ["Year", "YEAR", "year", "Fire_Year", "fire_year", "年份"])
        value_col = _pick_column(cols, ["diff_contribution", "delta_contribution", "delta"])
        sig_value_col = _pick_column(cols, ["delta_sig", "DELTA_SIG", "Delta_Sig"])

        missing = [
            k
            for k, v in {"GRID_ID": grid_col, "Year": year_col, effect_col: value_col, sig_col: sig_value_col}.items()
            if v is None
        ]
        if missing:
            raise ValueError(f"Missing required column(s) {missing} in: {csv_path}")

        df = pd.read_csv(
            csv_path,
            usecols=[grid_col, year_col, value_col, sig_value_col],
            dtype={grid_col: "string"},
            low_memory=False,
        )
        df = df.rename(
            columns={
                grid_col: "GRID_ID",
                year_col: "Year",
                value_col: effect_col,
                sig_value_col: sig_col,
            }
        )

        df["GRID_ID"] = pd.to_numeric(df["GRID_ID"], errors="coerce").astype("Int64")
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        df[effect_col] = pd.to_numeric(df[effect_col], errors="coerce")
        df[sig_col] = _coerce_bool_series(df[sig_col])
        df = df.dropna(subset=["GRID_ID", "Year"])
        df = df[df["Year"].between(FIRE_YEAR_START, FIRE_YEAR_END)]
        if df.empty:
            continue

        parts.append(df[["GRID_ID", "Year", effect_col, sig_col]])

    if not parts:
        raise ValueError(f"No valid records loaded from: {source_dir}")

    combined = pd.concat(parts, ignore_index=True)
    combined = (
        combined.groupby(["Year", "GRID_ID"], as_index=False)
        .agg(
            **{
                effect_col: (effect_col, "mean"),
                sig_col: (sig_col, "max"),
            }
        )
        .dropna(subset=["GRID_ID", "Year"])
    )

    print(
        f"Loaded '{effect_col}': "
        f"years={combined['Year'].nunique()}, rows={len(combined)}, unique GRID_ID={combined['GRID_ID'].nunique()}"
    )
    return combined


def load_bivariate_metrics() -> pd.DataFrame:
    fire_frequency = read_effect_from_csv_dir(
        source_dir=FIRE_FREQUENCY_EFFECT_DIR,
        effect_col="Fire_frequency_effect",
        sig_col="Fire_frequency_sig",
    )
    normalized_fire_area = read_effect_from_csv_dir(
        source_dir=NORMALIZED_FIRE_AREA_EFFECT_DIR,
        effect_col="Normalized_fire_area_effect",
        sig_col="Normalized_fire_area_sig",
    )

    merged = pd.merge(
        fire_frequency,
        normalized_fire_area,
        on=["Year", "GRID_ID"],
        how="inner",
    )
    merged = merged.sort_values(["Year", "GRID_ID"]).reset_index(drop=True)

    print(
        "Merged metrics: "
        f"years={merged['Year'].nunique()}, rows={len(merged)}, "
        f"rows_with_both={len(merged)}, "
        f"rows_joint_sig={(_bool_mask(merged['Fire_frequency_sig']) & _bool_mask(merged['Normalized_fire_area_sig'])).sum()}"
    )
    return merged


def read_wui_fishnet_filtered(
    gdb_path: str,
    layer: str,
    grid_ids: list[int],
    max_ids_for_where: int = MAX_IDS_FOR_WHERE,
    chunk_size: int = WHERE_CHUNK_SIZE,
) -> gpd.GeoDataFrame:
    if not grid_ids:
        print(f"No GRID_ID provided, reading full WUI fishnet layer: {layer}")
        return gpd.read_file(gdb_path, layer=layer, columns=["GRID_ID"], engine="pyogrio")

    if len(grid_ids) > max_ids_for_where:
        print(
            f"GRID_ID count = {len(grid_ids)} > {max_ids_for_where}, "
            f"reading full WUI fishnet layer instead: {layer}"
        )
        return gpd.read_file(gdb_path, layer=layer, columns=["GRID_ID"], engine="pyogrio")

    print(f"Reading WUI fishnet by GRID_ID filter (layer={layer}, n={len(grid_ids)}) ...")
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
            msg = str(exc)
            if "Invalid SQL query" in msg and current_chunk_size > 1:
                new_chunk_size = max(1, current_chunk_size // 2)
                if new_chunk_size == current_chunk_size:
                    raise
                print(
                    f"Invalid SQL query with WHERE_CHUNK_SIZE={current_chunk_size}, "
                    f"retrying with {new_chunk_size}..."
                )
                current_chunk_size = new_chunk_size
                continue
            raise


def wui_year_for_fire_year(fire_year: int) -> int:
    if fire_year < FIRE_YEAR_START or fire_year > FIRE_YEAR_END:
        raise ValueError(f"Fire year {fire_year} out of range [{FIRE_YEAR_START}, {FIRE_YEAR_END}]")

    base_wui_year = min(WUI_YEARS)
    wui_year = base_wui_year + 5 * ((fire_year - FIRE_YEAR_START) // 5)
    if wui_year not in WUI_YEARS:
        raise ValueError(f"No WUI layer configured for fire year {fire_year} (computed WUI year {wui_year}).")
    return wui_year


def classify_sign_classes(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    classes = pd.Series(pd.NA, index=values.index, dtype="Int64")
    valid = values.dropna()
    if valid.empty:
        return classes

    classes.loc[valid.index] = pd.Series(np.where(valid < 0, 1, 2), index=valid.index, dtype="Int64")
    return classes


def summarize_dominant_quadrants(metric_all: pd.DataFrame) -> pd.DataFrame:
    df = metric_all.copy()
    df["joint_sig"] = _bool_mask(df["Fire_frequency_sig"]) & _bool_mask(df["Normalized_fire_area_sig"])
    df["Fire_frequency_class"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df["Normalized_fire_area_class"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    joint_sig_idx = df["joint_sig"]
    df.loc[joint_sig_idx, "Fire_frequency_class"] = classify_sign_classes(df.loc[joint_sig_idx, "Fire_frequency_effect"])
    df.loc[joint_sig_idx, "Normalized_fire_area_class"] = classify_sign_classes(
        df.loc[joint_sig_idx, "Normalized_fire_area_effect"]
    )

    valid_class = df["Fire_frequency_class"].notna() & df["Normalized_fire_area_class"].notna()
    df["bivariate_key"] = pd.Series([pd.NA] * len(df), dtype="object")
    df.loc[valid_class, "bivariate_key"] = pd.Series(
        list(
            zip(
                df.loc[valid_class, "Fire_frequency_class"].astype(int),
                df.loc[valid_class, "Normalized_fire_area_class"].astype(int),
            )
        ),
        index=df.index[valid_class],
        dtype="object",
    )

    available_summary = (
        df.groupby("GRID_ID", as_index=False)
        .agg(
            available_years=("Year", "nunique"),
            first_year=("Year", "min"),
            last_year=("Year", "max"),
        )
    )

    sig_df = df.loc[df["bivariate_key"].notna(), [
        "GRID_ID",
        "Year",
        "bivariate_key",
        "Fire_frequency_effect",
        "Normalized_fire_area_effect",
    ]].copy()
    if sig_df.empty:
        raise ValueError("No jointly significant quadrant records found across 2003-2022.")

    sig_df["effect_strength"] = np.sqrt(
        sig_df["Fire_frequency_effect"].astype(float) ** 2 +
        sig_df["Normalized_fire_area_effect"].astype(float) ** 2
    )

    dominant_candidates = (
        sig_df.groupby(["GRID_ID", "bivariate_key"], as_index=False)
        .agg(
            dominant_count=("Year", "size"),
            sig_years=("Year", "nunique"),
            mean_effect_strength=("effect_strength", "mean"),
            latest_year=("Year", "max"),
        )
        .sort_values(
            ["GRID_ID", "dominant_count", "mean_effect_strength", "latest_year", "bivariate_key"],
            ascending=[True, False, False, False, True],
        )
    )

    dominant = dominant_candidates.drop_duplicates(subset=["GRID_ID"], keep="first").copy()
    dominant = dominant.rename(columns={"bivariate_key": "dominant_bivariate_key"})

    summary = available_summary.merge(
        dominant[["GRID_ID", "dominant_bivariate_key", "dominant_count", "sig_years", "mean_effect_strength"]],
        on="GRID_ID",
        how="left",
    )
    summary["non_sig_years"] = summary["available_years"] - pd.to_numeric(summary["sig_years"], errors="coerce").fillna(0)
    summary["sig_years"] = pd.to_numeric(summary["sig_years"], errors="coerce").fillna(0).astype(int)
    summary["dominant_count"] = pd.to_numeric(summary["dominant_count"], errors="coerce").fillna(0).astype(int)
    summary["non_sig_years"] = pd.to_numeric(summary["non_sig_years"], errors="coerce").fillna(0).astype(int)
    summary["has_sig_direction"] = summary["sig_years"] > 0
    summary["non_sig_dominant"] = summary["non_sig_years"] > summary["dominant_count"]
    summary.loc[summary["non_sig_dominant"], "dominant_bivariate_key"] = pd.NA
    summary.loc[summary["non_sig_dominant"], "has_sig_direction"] = False

    print(
        "Dominant-direction summary: "
        f"GRID_ID with any data={len(summary)}, "
        f"GRID_ID with dominant significant direction={(summary['has_sig_direction']).sum()}, "
        f"GRID_ID classified as non-significant={(~summary['has_sig_direction']).sum()}, "
        f"GRID_ID forced to non-significant by higher non-significant frequency={summary['non_sig_dominant'].sum()}"
    )
    return summary


def load_combined_wui_fishnet(metric_all: pd.DataFrame) -> gpd.GeoDataFrame:
    available_years = sorted(metric_all["Year"].dropna().astype(int).unique().tolist())
    fishnet_parts: list[gpd.GeoDataFrame] = []

    for wui_year in WUI_YEARS:
        year_group = [year for year in available_years if wui_year_for_fire_year(year) == wui_year]
        if not year_group:
            continue

        needed_ids = sorted(
            set(
                metric_all.loc[metric_all["Year"].isin(year_group), "GRID_ID"]
                .dropna()
                .astype(int)
                .tolist()
            )
        )
        if not needed_ids:
            continue

        layer = WUI_LAYER_TEMPLATE.format(wui_year=wui_year)
        fishnet_part = read_wui_fishnet_filtered(WUI_GDB_PATH, layer, needed_ids)
        if "GRID_ID" not in fishnet_part.columns:
            raise ValueError(f"GRID_ID field not found in WUI fishnet layer: {layer}")

        fishnet_part["GRID_ID"] = pd.to_numeric(fishnet_part["GRID_ID"], errors="coerce")
        fishnet_part = fishnet_part.dropna(subset=["GRID_ID"]).copy()
        fishnet_part["GRID_ID"] = fishnet_part["GRID_ID"].astype(int)
        fishnet_part = fishnet_part.to_crs(epsg=4326)
        fishnet_part["source_wui_year"] = wui_year
        fishnet_parts.append(fishnet_part)
        print(f"Loaded WUI layer {layer}: {len(fishnet_part)} filtered features")

    if not fishnet_parts:
        raise ValueError("No WUI fishnet features were loaded for the available years.")

    combined = pd.concat(fishnet_parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=["GRID_ID"], keep="first").copy()
    if not isinstance(combined, gpd.GeoDataFrame):
        combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=fishnet_parts[0].crs)

    print(f"Combined WUI fishnet features after GRID_ID deduplication: {len(combined)}")
    return combined


def add_bivariate_legend(ax: plt.Axes) -> None:
    legend_fontsize = _fs(LEGEND_FONT_SIZE)
    cell_width = 23.0
    cell_height = 22.0

    quadrant_rectangles = [
        ((1, 2), -162.0, -12.0),
        ((2, 2), -139.0, -12.0),
        ((1, 1), -162.0, -34.0),
        ((2, 1), -139.0, -34.0),
    ]
    for key, x0, y0 in quadrant_rectangles:
        ax.add_patch(
            mpatches.Rectangle(
                (x0, y0),
                cell_width,
                cell_height,
                facecolor=BIVARIATE_COLORS[key],
                edgecolor="white",
                linewidth=0.6,
                alpha=0.8,
                zorder=21,
            )
        )

    ax.text(-152.0, 13.0, "Inhibition", ha="center", va="bottom", fontsize=legend_fontsize, zorder=22)
    ax.text(-121.0, 13.0, "Amplification", ha="center", va="bottom", fontsize=legend_fontsize, zorder=22)
    ax.text(-113.0, -1.0, "Amplification", ha="left", va="center", fontsize=legend_fontsize, zorder=22)
    ax.text(-113.0, -23.0, "Inhibition", ha="left", va="center", fontsize=legend_fontsize, zorder=22)
    ax.text(-137.0, -39.0, "Fire frequency", ha="center", va="top", fontsize=legend_fontsize, zorder=22)
    ax.text(
        -172.0,
        -12.0,
        "Normalized burned area",
        ha="center",
        va="center",
        rotation=90,
        fontsize=legend_fontsize,
        zorder=22,
    )

    ax.add_patch(
        mpatches.Rectangle(
            (-162.0, -52.2),
            16.0,
            4.5,
            facecolor=NON_SIG_COLOR,
            edgecolor="none",
            alpha=0.8,
            zorder=21,
        )
    )
    ax.text(-142.0, -50.0, "No significant", ha="left", va="center", fontsize=legend_fontsize, zorder=22)


def plot_dominant_map(*, continent: gpd.GeoDataFrame, fishnet: gpd.GeoDataFrame, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(15, 10))

    if continent is not None:
        continent.plot(ax=ax, color=BASEMAP_FACE, edgecolor=BASEMAP_EDGE, linewidth=0.5)

    ax.set_facecolor("white")
    ax.set_xlim([-180, 180])
    ax.set_ylim([-60, 75])

    if not fishnet.empty:
        non_sig_mask = ~fishnet["has_sig_direction"].fillna(False)
        sig_mask = fishnet["has_sig_direction"].fillna(False) & fishnet["dominant_bivariate_key"].notna()

        fishnet_non_sig = fishnet[non_sig_mask]
        fishnet_sig = fishnet[sig_mask]

        if not fishnet_non_sig.empty:
            fishnet_non_sig.plot(ax=ax, color=NON_SIG_COLOR, edgecolor="none", linewidth=0, alpha=1.0)

        for key, color in BIVARIATE_COLORS.items():
            subset = fishnet_sig[fishnet_sig["dominant_bivariate_key"] == key]
            if not subset.empty:
                subset.plot(ax=ax, color=color, edgecolor="none", linewidth=0, alpha=0.9)

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
        labelleft=False,
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    add_bivariate_legend(ax)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.055, top=0.995)
    plt.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {output_path}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading global map data...")
    continent = gpd.read_file(CONTINENT_SHP_PATH)
    continent = continent.to_crs("EPSG:4326")
    print(f"Loaded continent layer: {len(continent)} features")

    metric_all = load_bivariate_metrics()
    metric_all["GRID_ID"] = metric_all["GRID_ID"].astype(int)
    metric_all["Year"] = metric_all["Year"].astype(int)

    dominant_summary = summarize_dominant_quadrants(metric_all)
    fishnet = load_combined_wui_fishnet(metric_all)
    fishnet = fishnet.merge(dominant_summary, on="GRID_ID", how="inner")

    missing_available = fishnet["available_years"].lt(FIRE_YEAR_END - FIRE_YEAR_START + 1).sum()
    print(
        "Coverage summary: "
        f"grids on map={len(fishnet)}, "
        f"grids with fewer than 20 available years={int(missing_available)}"
    )

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    plot_dominant_map(continent=continent, fishnet=fishnet, output_path=output_path)

    print("\n=== Dominant-direction map completed ===")


if __name__ == "__main__":
    main()
