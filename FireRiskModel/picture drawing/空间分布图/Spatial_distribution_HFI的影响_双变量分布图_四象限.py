#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称: HFI 对 Fire frequency 与 Normalized burned area 影响的双变量空间分布图

功能简介:
    1. 从 HFI 净效应固定效应结果 CSV 中分别读取 Fire frequency 和 Normalized burned area 的
       diff_contribution 以及 delta_sig。
    2. 仅保留 GRID_ID、Year、diff_contribution、delta_sig，并在全局范围内合并。
    3. 根据 Year 选择对应的 WUI 渔网图层:
       - 2003-2007 -> Grid_Clip_2005_WUI
       - 2008-2012 -> Grid_Clip_2010_WUI
       - 2013-2017 -> Grid_Clip_2015_WUI
       - 2018-2022 -> Grid_Clip_2020_WUI
    4. 仅保留两个响应变量都存在结果的格网，并对其中两个响应变量都显著
       （delta_sig == TRUE）的格网做双变量颜色分类。
    5. 有数据但不显著的格网用灰色表示；不再单独绘制 No data。
    6. 底图、渔网和版式尽量保持与「Spatial_distribution_主导气候带类型.py」一致。

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
    - B:\WUI\Picture\Bivariate_Human Footprint_Effect_FireFrequency_NormalizedFireArea_YYYY.png
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

# 设置字体
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

FONT_SCALE = 1.2
LEGEND_FONT_SIZE = 14.7


def _fs(size: float) -> float:
    return size * FONT_SCALE

# =========================
# 路径与年份配置
# =========================
FIRE_FREQUENCY_EFFECT_DIR = r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num"
NORMALIZED_FIRE_AREA_EFFECT_DIR = r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area"

WUI_GDi_PATH = r"I:\Processing data\gdb集合\WUI.gdb"
WUI_YEARS = [2005, 2010, 2015, 2020]
WUI_LAYER_TEMPLATE = "Grid_Clip_{wui_year}_WUI"

CONTINENT_SHP_PATH = r"I:\Data_huanglin\map\world continent boundary\continent.shp"

OUTPUT_DIR = r"B:\WUI\Picture"
OUTPUT_FILENAME_TEMPLATE = "Bivariate_Human Footprint_Effect_FireFrequency_NormalizedFireArea_{year}.png"
OUTPUT_DPI = 1200

FIRE_YEAR_START = int(os.getenv("BIVAR_FIRE_YEAR_START", "2003"))
FIRE_YEAR_END = int(os.getenv("BIVAR_FIRE_YEAR_END", "2022"))

# 读取渔网时按 GRID_ID 过滤，避免一次性读整层
MAX_IDS_FOR_WHERE = 50000
WHERE_CHUNK_SIZE = 3000

# =========================
# 绘图样式
# =========================
iASEMAP_FACE = "#ffffff"
iASEMAP_EDGE = "#000000"
NON_SIG_COLOR = "#dcdcdc"

# 2x2 四象限双变量配色:
# 横轴 = Fire frequency, 纵轴 = Normalized burned area
# 以 0 为分界线:
#   - 两个都为正: 红色
#   - 两个都为负: 蓝色
#   - Fire frequency 为正 / Normalized burned area 为负: 橙色
#   - Fire frequency 为负 / Normalized burned area 为正: 绿色
BIVARIATE_COLORS = {
    (1, 1): "#5AE085",
    (2, 1): "#FFA72C",
    (1, 2): "#51A2F3",
    (2, 2): "#F36568",
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


def read_effect_from_csv_dir(
    source_dir: str,
    effect_col: str,
    sig_col: str,
) -> pd.DataFrame:
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
        except ValueError as e:
            msg = str(e)
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

    # 以 0 为分界线:
    # 1 = 负值, 2 = 正值或 0
    classes.loc[valid.index] = pd.Series(np.where(valid < 0, 1, 2), index=valid.index, dtype="Int64")
    return classes


def add_bivariate_legend(
    ax: plt.Axes,
) -> None:
    legend_fontsize = _fs(LEGEND_FONT_SIZE)
    cell_width = 23.0
    cell_height = 22.0

    quadrant_rectangles = [
        ((1, 2), -155.0, -12.0),
        ((2, 2), -132.0, -12.0),
        ((1, 1), -155.0, -34.0),
        ((2, 1), -132.0, -34.0),
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
                zorder=21,
            )
        )

    ax.text(
        -144.5,
        13.5,
        "Negative",
        ha="center",
        va="bottom",
        fontsize=legend_fontsize,
        zorder=22,
    )
    ax.text(
        -121,
        13.5,
        "Positive",
        ha="center",
        va="bottom",
        fontsize=legend_fontsize,
        zorder=22,
    )
    ax.text(
        -106.0,
        -1.0,
        "Positive",
        ha="left",
        va="center",
        fontsize=legend_fontsize,
        zorder=22,
    )
    ax.text(
        -106.0,
        -23.0,
        "Negative",
        ha="left",
        va="center",
        fontsize=legend_fontsize,
        zorder=22,
    )
    ax.text(
        -130.0,
        -39.0,
        "Fire frequency",
        ha="center",
        va="top",
        fontsize=legend_fontsize,
        zorder=22,
    )
    ax.text(
        -165.0,
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
            (-155.0, -52.2),
            16.0,
            4.5,
            facecolor=NON_SIG_COLOR,
            edgecolor="none",
            zorder=21,
        )
    )
    ax.text(
        -135.0,
        -50,
        "No significant",
        ha="left",
        va="center",
        fontsize=legend_fontsize,
        zorder=22,
    )


def plot_bivariate_map(
    *,
    continent: gpd.GeoDataFrame,
    fishnet: gpd.GeoDataFrame,
    fire_year: int,
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(15, 10))

    if continent is not None:
        continent.plot(ax=ax, color=iASEMAP_FACE, edgecolor=iASEMAP_EDGE, linewidth=0.5)

    ax.set_facecolor("white")
    ax.set_xlim([-180, 180])
    ax.set_ylim([-60, 75])

    if not fishnet.empty:
        joint_sig_mask = (
            _bool_mask(fishnet["Fire_frequency_sig"]) &
            _bool_mask(fishnet["Normalized_fire_area_sig"])
        )
        non_sig_mask = ~joint_sig_mask
        sig_mask = joint_sig_mask & fishnet["bivariate_key"].notna()

        fishnet_non_sig = fishnet[non_sig_mask]
        fishnet_has_data = fishnet[sig_mask]

        if not fishnet_non_sig.empty:
            fishnet_non_sig.plot(ax=ax, color=NON_SIG_COLOR, edgecolor="none", linewidth=0, alpha=1.0)

        for key, color in BIVARIATE_COLORS.items():
            subset = fishnet_has_data[fishnet_has_data["bivariate_key"] == key]
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
    ax.text(0.02, 0.92, f"{fire_year}", transform=ax.transAxes, fontsize=20, verticalalignment="top")

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

    available_years = sorted(metric_all["Year"].unique().tolist())
    print(f"Available years: {available_years}")

    fishnet_cache: dict[int, gpd.GeoDataFrame] = {}
    for wui_year in WUI_YEARS:
        year_group = [y for y in available_years if wui_year_for_fire_year(y) == wui_year]
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

        layer = WUI_LAYER_TEMPLATE.format(wui_year=wui_year)
        fishnet_base = read_wui_fishnet_filtered(WUI_GDi_PATH, layer, needed_ids)
        if "GRID_ID" not in fishnet_base.columns:
            raise ValueError(f"GRID_ID field not found in WUI fishnet layer: {layer}")

        fishnet_base["GRID_ID"] = pd.to_numeric(fishnet_base["GRID_ID"], errors="coerce")
        fishnet_base = fishnet_base.dropna(subset=["GRID_ID"])
        fishnet_base["GRID_ID"] = fishnet_base["GRID_ID"].astype(int)
        fishnet_base = fishnet_base.to_crs(epsg=4326)
        fishnet_cache[wui_year] = fishnet_base
        print(f"Cached WUI layer {layer}: {len(fishnet_base)} features")

    for fire_year in available_years:
        print(f"\n=== Processing year {fire_year} ===")
        year_metric = metric_all.loc[
            metric_all["Year"] == fire_year,
            [
                "GRID_ID",
                "Fire_frequency_effect",
                "Fire_frequency_sig",
                "Normalized_fire_area_effect",
                "Normalized_fire_area_sig",
            ],
        ].copy()

        year_metric["joint_sig"] = (
            _bool_mask(year_metric["Fire_frequency_sig"]) &
            _bool_mask(year_metric["Normalized_fire_area_sig"])
        )

        year_metric["Fire_frequency_class"] = pd.Series(pd.NA, index=year_metric.index, dtype="Int64")
        year_metric["Normalized_fire_area_class"] = pd.Series(pd.NA, index=year_metric.index, dtype="Int64")
        joint_sig_idx = year_metric["joint_sig"]
        year_metric.loc[joint_sig_idx, "Fire_frequency_class"] = classify_sign_classes(
            year_metric.loc[joint_sig_idx, "Fire_frequency_effect"]
        )
        year_metric.loc[joint_sig_idx, "Normalized_fire_area_class"] = classify_sign_classes(
            year_metric.loc[joint_sig_idx, "Normalized_fire_area_effect"]
        )

        valid_class = (
            year_metric["Fire_frequency_class"].notna() &
            year_metric["Normalized_fire_area_class"].notna()
        )
        year_metric["bivariate_key"] = pd.Series([pd.NA] * len(year_metric), dtype="object")
        year_metric.loc[valid_class, "bivariate_key"] = pd.Series(
            list(
                zip(
                    year_metric.loc[valid_class, "Fire_frequency_class"].astype(int),
                    year_metric.loc[valid_class, "Normalized_fire_area_class"].astype(int),
                )
            ),
            index=year_metric.index[valid_class],
            dtype="object",
        )

        wui_year = wui_year_for_fire_year(fire_year)
        fishnet_base = fishnet_cache[wui_year]
        year_ids = set(year_metric["GRID_ID"].astype(int).tolist())
        fishnet_year = fishnet_base[fishnet_base["GRID_ID"].isin(year_ids)].copy()
        fishnet_year = fishnet_year.merge(year_metric, on="GRID_ID", how="left")

        output_path = os.path.join(
            OUTPUT_DIR,
            OUTPUT_FILENAME_TEMPLATE.format(year=fire_year),
        )
        plot_bivariate_map(
            continent=continent,
            fishnet=fishnet_year,
            fire_year=fire_year,
            output_path=output_path,
        )

    print("\n=== All years completed ===")


if __name__ == "__main__":
    main()


