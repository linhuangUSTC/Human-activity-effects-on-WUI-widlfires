#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称: Human Footprint对Fire frequency及Normalized burned area的影响除以HFI后的全球分布图
功能简介:
    1) 分别读取 Fire frequency 与 Normalized burned area 的 HFI_effect_*.csv
    2) 仅保留 GRID_ID、Year、HFI、(delta/diff)_contribution、delta_sig（Year=2003-2022），每年绘制一张图
    3) 将 Human Footprint 的影响值除以 Human Footprint 本身的值，即绘制 delta_contribution / HFI
    4) 根据 Year 选择 WUI 渔网图层：
       - 2003-2007 -> Grid_Clip_2005_WUI
       - 2008-2012 -> Grid_Clip_2010_WUI
       - 2013-2017 -> Grid_Clip_2015_WUI
       - 2018-2022 -> Grid_Clip_2020_WUI
    5) 按 GRID_ID 匹配并用 colormap 绘制显著格网的 ratio_contribution
    6) 不显著格网（delta_sig == FALSE）用灰色表示
    7) 以 I:/Data_huanglin/map/world continent boundary/continent.shp 为底图（WGS-84）
    8) 输出图片到 B:/WUI/Picture
"""

from __future__ import annotations

import glob
import math
import os
from typing import Iterable

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置英文显示，使用 Arial 字体
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 用户可配置参数
# =========================
FIRE_YEAR_START = 2003
FIRE_YEAR_END = 2022

# WUI 渔网（FileGDB）
WUI_GDB_PATH = r"I:\Processing data\gdb集合\WUI.gdb"
WUI_YEARS = [2005, 2010, 2015, 2020]
WUI_LAYER_TEMPLATE = "Grid_Clip_{wui_year}_WUI"

# 世界大陆边界（底图）
CONTINENT_SHP_PATH = r"I:\Data_huanglin\map\world continent boundary\continent.shp"

OUTPUT_DIR = r"B:\WUI\Picture"
OUTPUT_DPI = 2400

DATASETS = [
    {
        "metric_title": "Fire frequency",
        "effect_source": r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num",
        "output_filename_template": "Human Footprint对Fire_frequency的影响除以HFI全球分布图_{year}.jpg",
        "colorbar_label": "Human Footprint effect on fire frequency / Human Footprint",
    },
    {
        "metric_title": "Normalized burned area",
        "effect_source": r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area",
        "output_filename_template": "Human Footprint对Normalized_burn_area的影响除以HFI全球分布图_{year}.jpg",
        "colorbar_label": "Human Footprint effect on normalized burned area / Human Footprint",
    },
]

# 绘图样式
# 以0为分界：负值用5级蓝色，正值用5级红色，使用离散色阶而非连续色带
CMAP_NAME = "BlueRedDiscrete"
NEGATIVE_CMAP_COLORS = ["#08306B", "#08519C", "#2171B5", "#6BAED6", "#C6DBEF"]
POSITIVE_CMAP_COLORS = ["#FEE0D2", "#FC9272", "#FB6A4A", "#DE2D26", "#A50F15"]
CMAP_COLORS = NEGATIVE_CMAP_COLORS + POSITIVE_CMAP_COLORS
NO_DATA_COLOR = "#d9d9d9"
NON_SIG_COLOR = "#bdbdbd"
BASEMAP_FACE = "#ffffff"
BASEMAP_EDGE = "#000000"

# 读取渔网时按 GRID_ID 过滤的阈值（避免 where 子句过长）
MAX_IDS_FOR_WHERE = 50000
WHERE_CHUNK_SIZE = 3000


def _chunks(items: list[int], chunk_size: int) -> Iterable[list[int]]:
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def _resolve_effect_files(effect_source: str) -> list[str]:
    if os.path.isfile(effect_source):
        return [effect_source]
    if os.path.isdir(effect_source):
        return sorted(glob.glob(os.path.join(effect_source, "HFI_effect_*.csv")))
    return sorted(glob.glob(effect_source))


def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
    for name in candidates:
        if name in columns:
            return name
    return None


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    values = series.astype(str).str.strip().str.upper()
    return values.isin({"TRUE", "T", "1", "YES", "Y"})


def load_effect_data(effect_source: str, year_start: int, year_end: int) -> pd.DataFrame:
    files = _resolve_effect_files(effect_source)
    if not files:
        raise FileNotFoundError(f"No effect CSV found by: {effect_source}")

    print(f"Found {len(files)} effect CSV file(s).")
    parts: list[pd.DataFrame] = []

    for csv_path in files:
        header = pd.read_csv(csv_path, nrows=0)
        cols = header.columns.tolist()

        grid_col = _pick_column(cols, ["GRID_ID", "grid_id", "Grid_ID"])
        year_col = _pick_column(cols, ["Year", "YEAR", "year", "Fire_Year", "fire_year", "年份"])
        hfi_col = _pick_column(cols, ["HFI", "hfi", "Human_Footprint", "human_footprint"])
        sig_col = _pick_column(cols, ["delta_sig", "DELTA_SIG", "Delta_Sig"])
        contrib_col = _pick_column(
            cols,
            [
                "delta_contribution",
                "diff_contribution",
                "delta",
            ],
        )

        missing = [
            k
            for k, v in {"GRID_ID": grid_col, "Year": year_col, "HFI": hfi_col, "delta_sig": sig_col}.items()
            if v is None
        ]
        if missing:
            raise ValueError(f"Missing required column(s) {missing} in: {csv_path}")
        if contrib_col is None:
            raise ValueError(f"Missing contribution column in: {csv_path}")

        df = pd.read_csv(
            csv_path,
            usecols=[grid_col, year_col, hfi_col, contrib_col, sig_col],
            low_memory=False,
            dtype={grid_col: "string"},
        )
        df = df.rename(
            columns={
                grid_col: "GRID_ID",
                year_col: "Year",
                hfi_col: "HFI",
                contrib_col: "delta_contribution",
                sig_col: "delta_sig",
            }
        )
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        df = df[df["Year"].between(year_start, year_end)]
        if df.empty:
            continue

        df["GRID_ID"] = pd.to_numeric(df["GRID_ID"], errors="coerce").astype("Int64")
        df["HFI"] = pd.to_numeric(df["HFI"], errors="coerce")
        df["delta_contribution"] = pd.to_numeric(df["delta_contribution"], errors="coerce")
        df["delta_sig"] = _coerce_bool_series(df["delta_sig"])
        df = df.dropna(subset=["GRID_ID"])

        parts.append(df[["GRID_ID", "Year", "HFI", "delta_contribution", "delta_sig"]])

    if not parts:
        raise ValueError(f"No records found for Year in [{year_start}, {year_end}].")

    combined = pd.concat(parts, ignore_index=True)
    combined = (
        combined.groupby(["Year", "GRID_ID"], as_index=False)
        .agg(
            HFI=("HFI", "mean"),
            delta_contribution=("delta_contribution", "mean"),
            delta_sig=("delta_sig", "max"),
        )
        .dropna(subset=["Year", "GRID_ID"])
    )

    hfi_nonzero = combined["HFI"].notna() & (combined["HFI"] != 0)
    combined["ratio_contribution"] = np.nan
    combined.loc[hfi_nonzero, "ratio_contribution"] = (
        combined.loc[hfi_nonzero, "delta_contribution"] / combined.loc[hfi_nonzero, "HFI"]
    )

    print(
        "Effect data loaded: "
        f"years={combined['Year'].nunique()}, "
        f"rows={len(combined)}, "
        f"unique GRID_ID={combined['GRID_ID'].nunique()}, "
        f"nonzero_hfi_rows={int(hfi_nonzero.sum())}, "
        f"zero_or_missing_hfi_rows={int((~hfi_nonzero).sum())}"
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
        except ValueError as e:
            msg = str(e)
            if "Invalid SQL query" in msg and current_chunk_size > 1:
                new_chunk_size = max(1, current_chunk_size // 2)
                if new_chunk_size == current_chunk_size:
                    raise
                print(f"Invalid SQL query with WHERE_CHUNK_SIZE={current_chunk_size}, retrying with {new_chunk_size}...")
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


def plot_fire_year_map(
    *,
    continent: gpd.GeoDataFrame,
    fishnet: gpd.GeoDataFrame,
    fire_year: int,
    colorbar_label: str,
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(15, 10))

    continent.plot(ax=ax, color=BASEMAP_FACE, edgecolor=BASEMAP_EDGE, linewidth=0.25)

    if fishnet.empty:
        ax.text(0.02, 0.92, f"{fire_year}", transform=ax.transAxes, fontsize=20, verticalalignment="top")
        ax.set_xlabel("Longitude", fontsize=16)
        ax.set_ylabel("Latitude", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 75)
        ax.grid(False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight", format="jpg")
        plt.close(fig)
        print(f"Figure saved to: {output_path} (no matched GRID_ID)")
        return

    if "ratio_contribution" not in fishnet.columns or "delta_sig" not in fishnet.columns:
        raise ValueError("Fishnet is missing required columns: ratio_contribution / delta_sig.")

    has_value = fishnet["ratio_contribution"].notna()
    fishnet_no_data = fishnet[~has_value]
    fishnet_data = fishnet[has_value]

    if len(fishnet_no_data) > 0:
        fishnet_no_data.plot(ax=ax, color=NO_DATA_COLOR, edgecolor="none", linewidth=0)

    if len(fishnet_data) > 0:
        sig_mask = fishnet_data["delta_sig"].fillna(False).astype(bool)
        fishnet_sig = fishnet_data[sig_mask]
        fishnet_non_sig = fishnet_data[~sig_mask]

        if len(fishnet_non_sig) > 0:
            fishnet_non_sig.plot(ax=ax, color=NON_SIG_COLOR, edgecolor="none", linewidth=0)

        values = fishnet_data["ratio_contribution"].astype(float)
        abs_max = max(abs(values.min()), abs(values.max()))
        if pd.isna(abs_max) or abs_max == 0:
            abs_max = 1
        step = abs_max / 5
        boundaries = [(-abs_max + i * step) for i in range(6)] + [(i * step) for i in range(1, 6)]
        cmap = mcolors.ListedColormap(CMAP_COLORS, name=CMAP_NAME)
        norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)

        if len(fishnet_sig) > 0:
            fishnet_sig.plot(
                ax=ax,
                column="ratio_contribution",
                cmap=cmap,
                norm=norm,
                edgecolor="none",
                linewidth=0,
            )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(
            sm,
            ax=ax,
            shrink=0.72,
            aspect=35,
            fraction=0.05,
            pad=0.08,
            orientation="horizontal",
            label=colorbar_label,
            drawedges=True,
        )
        exponent = int(math.floor(math.log10(abs_max))) if abs_max > 0 else 0
        scale = 10 ** exponent
        cbar.set_ticks(boundaries)
        cbar.ax.set_xticklabels([f"{value / scale:.2f}" for value in boundaries])
        cbar.ax.text(
            1.01,
            0.5,
            f"1e{exponent}",
            transform=cbar.ax.transAxes,
            ha="left",
            va="center",
            fontsize=18,
        )
        cbar.ax.tick_params(direction="in")
        cbar.ax.tick_params(labelsize=18)
        cbar.ax.xaxis.label.set_size(20)

        print(f"Significant cells (delta_sig=TRUE): {len(fishnet_sig)}")
    else:
        print("No ratio_contribution values found for this year.")

    continent.boundary.plot(ax=ax, color=BASEMAP_EDGE, linewidth=0.25)
    ax.text(0.02, 0.92, f"{fire_year}", transform=ax.transAxes, fontsize=20, verticalalignment="top")
    ax.set_xlabel("Longitude", fontsize=16)
    ax.set_ylabel("Latitude", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 75)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight", format="jpg")
    plt.close(fig)
    print(f"Figure saved to: {output_path}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fire_years = list(range(FIRE_YEAR_START, FIRE_YEAR_END + 1))

    print(f"Loading world base map: {CONTINENT_SHP_PATH}")
    continent = gpd.read_file(CONTINENT_SHP_PATH)
    continent = continent.to_crs("EPSG:4326")
    print(f"Base map loaded: {len(continent)} feature(s), CRS={continent.crs}")

    for dataset in DATASETS:
        print(f"\n{'=' * 80}")
        print(f"Processing dataset: {dataset['metric_title']}")

        print(f"Loading effect CSV(s) from: {dataset['effect_source']}")
        effect_all = load_effect_data(dataset["effect_source"], FIRE_YEAR_START, FIRE_YEAR_END)
        effect_all["Year"] = effect_all["Year"].astype(int)
        effect_all["GRID_ID"] = effect_all["GRID_ID"].astype(int)

        available_years = sorted(effect_all["Year"].unique().tolist())
        missing_years = [y for y in fire_years if y not in set(available_years)]
        if missing_years:
            print(f"Warning: missing Year(s) in effect CSV: {missing_years}")

        years_by_wui_year: dict[int, list[int]] = {}
        for y in fire_years:
            wui_year = wui_year_for_fire_year(y)
            years_by_wui_year.setdefault(wui_year, []).append(y)

        for wui_year in sorted(years_by_wui_year):
            year_group = years_by_wui_year[wui_year]
            layer = WUI_LAYER_TEMPLATE.format(wui_year=wui_year)

            group_effect = effect_all[effect_all["Year"].isin(year_group)]
            if group_effect.empty:
                print(f"Skip WUI {wui_year}: no effect data in years {year_group[0]}-{year_group[-1]}.")
                continue

            union_grid_ids = (
                group_effect["GRID_ID"]
                .dropna()
                .astype(int)
                .drop_duplicates()
                .sort_values()
                .tolist()
            )

            print(f"\n{'=' * 60}")
            print(f"Loading WUI fishnet layer: {layer}  (years {year_group[0]}-{year_group[-1]})")
            fishnet_base = read_wui_fishnet_filtered(WUI_GDB_PATH, layer, union_grid_ids)
            if "GRID_ID" not in fishnet_base.columns:
                raise ValueError(f"GRID_ID field not found in WUI fishnet layer: {layer}")

            fishnet_base["GRID_ID"] = pd.to_numeric(fishnet_base["GRID_ID"], errors="coerce")
            fishnet_base = fishnet_base.dropna(subset=["GRID_ID"])
            fishnet_base["GRID_ID"] = fishnet_base["GRID_ID"].astype(int)
            fishnet_base = fishnet_base.to_crs("EPSG:4326")

            found_ids = set(fishnet_base["GRID_ID"].tolist())
            missing_ids = set(union_grid_ids) - found_ids
            if missing_ids:
                print(f"Warning: {len(missing_ids)} GRID_ID(s) not found in {layer}. They will be skipped.")

            for fire_year in year_group:
                print(f"\n--- Drawing Year {fire_year} (WUI={wui_year}) ---")
                effect_year = effect_all[
                    effect_all["Year"] == fire_year
                ][["GRID_ID", "ratio_contribution", "delta_sig"]]
                if effect_year.empty:
                    print(f"Skip Year {fire_year}: no effect records.")
                    continue

                year_ids = set(effect_year["GRID_ID"].astype(int).tolist())
                fishnet_year = fishnet_base[fishnet_base["GRID_ID"].isin(year_ids)]
                fishnet_year = fishnet_year.merge(effect_year, on="GRID_ID", how="left")

                output_path = os.path.join(OUTPUT_DIR, dataset["output_filename_template"].format(year=fire_year))
                plot_fire_year_map(
                    continent=continent,
                    fishnet=fishnet_year,
                    fire_year=fire_year,
                    colorbar_label=dataset["colorbar_label"],
                    output_path=output_path,
                )


if __name__ == "__main__":
    main()
