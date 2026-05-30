#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称: 覆盖2003-2022年HFI影响四象限结果的大洲面积加权比例统计图

功能简介:
    1. 复用「按照landcover以及气候带分类统计HFI的影响方向_四象限_覆盖所有年份_堆叠图_.py」
       的四象限定义:
       - 横轴: Fire frequency effect
       - 纵轴: Normalized burned area effect
       - 仅统计两个响应变量都显著(delta_sig == TRUE)且能落入四象限的 GRID_ID。
    2. 处理 2003-2022 全部年份数据。
    3. 根据年份映射对应的 WUI 时段标签表:
       - 2003-2007 -> 2005
       - 2008-2012 -> 2010
       - 2013-2017 -> 2015
       - 2018-2022 -> 2020
    4. 通过 I:\Processing data\Grid_Clip_{wui_year}_WUI.csv 读取大洲、WUI类型和WUI面积。
    5. 按大洲统计 Interface / Intermix 中四象限的面积加权比例，并输出一张100%堆叠柱状图。

输出:
    - B:\WUI\Picture\Bivariate_HFI_quadrant_proportion_by_continent_all_years_Interface_Intermix.png
    - B:\WUI\Picture\Bivariate_HFI_quadrant_proportion_by_continent_all_years.xlsx
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

FIRE_YEAR_START = 2003
FIRE_YEAR_END = 2022
CHUNK_SIZE = 300_000

FIRE_FREQUENCY_EFFECT_DIR = r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num"
NORMALIZED_FIRE_AREA_EFFECT_DIR = r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area"
WUI_STATS_TEMPLATE = r"I:\Processing data\Grid_Clip_{wui_year}_WUI.csv"
WUI_YEARS = [2005, 2010, 2015, 2020]

OUTPUT_DIR = Path(r"B:\WUI\Picture")
OUTPUT_FIGURE = OUTPUT_DIR / "Bivariate_HFI_quadrant_proportion_by_continent_all_years_Interface_Intermix.png"
OUTPUT_TABLE = OUTPUT_DIR / "Bivariate_HFI_quadrant_proportion_by_continent_all_years.xlsx"

BAR_ALPHA = 0.50
INTERFACE_BAR_ALPHA = 0.70
INTERMIX_BAR_ALPHA = 0.50
BAR_WIDTH = 0.46
REFERENCE_CATEGORY_COUNT = 5
XTICK_LABEL_FONT_SIZE = 20
GROUPED_BAR_WIDTH_SCALE = 0.312
GROUPED_BAR_OFFSET_SCALE = 0.58
INTERMIX_HATCH = "//"

CONTINENT_CODE_LABELS = {
    1: "Africa",
    2: "Asia",
    3: "Oceania",
    4: "North America",
    5: "Oceania",
    6: "South America",
    8: "Europe",
}
CONTINENT_ORDER = [
    "Africa",
    "Asia",
    "Europe",
    "North America",
    "South America",
    "Oceania",
]
CONTINENT_LABELS = {
    "Africa": "Africa",
    "Asia": "Asia",
    "Europe": "Europe",
    "North America": "North\nAmerica",
    "South America": "South\nAmerica",
    "Oceania": "Oceania",
}

QUADRANT_ORDER = [(1, 1), (1, 2), (2, 1), (2, 2)]
QUADRANT_LABELS = {
    (1, 1): "FF- / NBA-",
    (1, 2): "FF- / NBA+",
    (2, 1): "FF+ / NBA-",
    (2, 2): "FF+ / NBA+",
}
QUADRANT_COLORS = {
    (1, 1): "#63BE7B",
    (2, 1): "#F3BB6E",
    (1, 2): "#6BA6E1",
    (2, 2): "#F68183",
}


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


def _bool_mask(series: pd.Series) -> pd.Series:
    return series.astype("boolean").fillna(False).astype(bool)


def classify_sign_classes(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    classes = pd.Series(pd.NA, index=values.index, dtype="Int64")
    valid = values.dropna()
    if valid.empty:
        return classes
    classes.loc[valid.index] = pd.Series(np.where(valid < 0, 1, 2), index=valid.index, dtype="Int64")
    return classes


def wui_year_for_fire_year(fire_year: int) -> int:
    if fire_year < FIRE_YEAR_START or fire_year > FIRE_YEAR_END:
        raise ValueError(f"Fire year {fire_year} out of range [{FIRE_YEAR_START}, {FIRE_YEAR_END}].")
    wui_year = min(WUI_YEARS) + 5 * ((fire_year - FIRE_YEAR_START) // 5)
    if wui_year not in WUI_YEARS:
        raise ValueError(f"No WUI year configured for fire year {fire_year}.")
    return wui_year


def read_effect_from_csv_dir(
    source_dir: str,
    effect_col: str,
    sig_col: str,
    year_start: int = FIRE_YEAR_START,
    year_end: int = FIRE_YEAR_END,
) -> pd.DataFrame:
    csv_files = _resolve_effect_files(source_dir)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {source_dir}")

    parts: list[pd.DataFrame] = []
    print(f"Reading {len(csv_files)} effect CSV(s) from: {source_dir}")

    for csv_path in csv_files:
        header = pd.read_csv(csv_path, nrows=0)
        cols = header.columns.tolist()

        grid_col = _pick_column(cols, ["GRID_ID", "grid_id", "Grid_ID"])
        year_col = _pick_column(cols, ["Year", "YEAR", "year", "Fire_Year", "fire_year", "年份"])
        value_col = _pick_column(cols, ["diff_contribution", "delta_contribution", "delta"])
        sig_value_col = _pick_column(cols, ["delta_sig", "DELTA_SIG", "Delta_Sig"])

        missing = [
            k
            for k, v in {
                "GRID_ID": grid_col,
                "Year": year_col,
                effect_col: value_col,
                sig_col: sig_value_col,
            }.items()
            if v is None
        ]
        if missing:
            raise ValueError(f"Missing required column(s) {missing} in: {csv_path}")

        chunk_iter = pd.read_csv(
            csv_path,
            usecols=[grid_col, year_col, value_col, sig_value_col],
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )
        for chunk in chunk_iter:
            chunk = chunk.rename(
                columns={
                    grid_col: "GRID_ID",
                    year_col: "Year",
                    value_col: effect_col,
                    sig_value_col: sig_col,
                }
            )
            chunk["GRID_ID"] = pd.to_numeric(chunk["GRID_ID"], errors="coerce").astype("Int64")
            chunk["Year"] = pd.to_numeric(chunk["Year"], errors="coerce").astype("Int64")
            chunk = chunk[chunk["Year"].between(year_start, year_end)]
            if chunk.empty:
                continue

            chunk[effect_col] = pd.to_numeric(chunk[effect_col], errors="coerce")
            chunk[sig_col] = _coerce_bool_series(chunk[sig_col])
            chunk = chunk.dropna(subset=["GRID_ID", "Year"])
            if chunk.empty:
                continue
            parts.append(chunk[["GRID_ID", "Year", effect_col, sig_col]])

    if not parts:
        raise ValueError(f"No valid records loaded from {source_dir} for years [{year_start}, {year_end}]")

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
        f"Loaded {effect_col}: rows={len(combined)}, unique GRID_ID={combined['GRID_ID'].nunique()}"
    )
    return combined


def load_bivariate_metrics() -> pd.DataFrame:
    fire_frequency = read_effect_from_csv_dir(
        source_dir=FIRE_FREQUENCY_EFFECT_DIR,
        effect_col="Fire_frequency_effect",
        sig_col="Fire_frequency_sig",
        year_start=FIRE_YEAR_START,
        year_end=FIRE_YEAR_END,
    )
    normalized_fire_area = read_effect_from_csv_dir(
        source_dir=NORMALIZED_FIRE_AREA_EFFECT_DIR,
        effect_col="Normalized_fire_area_effect",
        sig_col="Normalized_fire_area_sig",
        year_start=FIRE_YEAR_START,
        year_end=FIRE_YEAR_END,
    )

    merged = pd.merge(
        fire_frequency,
        normalized_fire_area,
        on=["Year", "GRID_ID"],
        how="inner",
    ).sort_values(["Year", "GRID_ID"]).reset_index(drop=True)

    if merged.empty:
        raise ValueError(
            f"No merged Fire frequency / Normalized burned area records found for years [{FIRE_YEAR_START}, {FIRE_YEAR_END}]."
        )

    merged["joint_sig"] = (
        _bool_mask(merged["Fire_frequency_sig"]) &
        _bool_mask(merged["Normalized_fire_area_sig"])
    )
    merged["Fire_frequency_class"] = pd.Series(pd.NA, index=merged.index, dtype="Int64")
    merged["Normalized_fire_area_class"] = pd.Series(pd.NA, index=merged.index, dtype="Int64")

    joint_sig_idx = merged["joint_sig"]
    merged.loc[joint_sig_idx, "Fire_frequency_class"] = classify_sign_classes(
        merged.loc[joint_sig_idx, "Fire_frequency_effect"]
    )
    merged.loc[joint_sig_idx, "Normalized_fire_area_class"] = classify_sign_classes(
        merged.loc[joint_sig_idx, "Normalized_fire_area_effect"]
    )

    valid_class = (
        merged["Fire_frequency_class"].notna() &
        merged["Normalized_fire_area_class"].notna()
    )
    merged["bivariate_key"] = pd.Series([pd.NA] * len(merged), dtype="object")
    merged.loc[valid_class, "bivariate_key"] = pd.Series(
        list(
            zip(
                merged.loc[valid_class, "Fire_frequency_class"].astype(int),
                merged.loc[valid_class, "Normalized_fire_area_class"].astype(int),
            )
        ),
        index=merged.index[valid_class],
        dtype="object",
    )

    quadrant_df = merged.loc[merged["bivariate_key"].notna()].copy()
    if quadrant_df.empty:
        raise ValueError(
            f"All GRID_ID records for years [{FIRE_YEAR_START}, {FIRE_YEAR_END}] are non-significant or unclassified; no quadrant data available."
        )

    quadrant_df["GRID_ID"] = quadrant_df["GRID_ID"].astype(int)
    quadrant_df["Year"] = quadrant_df["Year"].astype(int)
    quadrant_df["WUI_Year"] = quadrant_df["Year"].map(wui_year_for_fire_year).astype(int)
    print(
        f"Quadrant-classified year-grid records for [{FIRE_YEAR_START}, {FIRE_YEAR_END}]: rows={len(quadrant_df)}, "
        f"unique GRID_ID={quadrant_df['GRID_ID'].nunique()}, "
        f"unique years={quadrant_df['Year'].nunique()}"
    )
    return quadrant_df


def load_wui_attributes() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for wui_year in WUI_YEARS:
        csv_path = WUI_STATS_TEMPLATE.format(wui_year=wui_year)
        df = pd.read_csv(
            csv_path,
            usecols=["GRID_ID", "main_WUI", "All_WUI_area", "Continent"],
            low_memory=False,
        )
        df["GRID_ID"] = pd.to_numeric(df["GRID_ID"], errors="coerce")
        df = df.dropna(subset=["GRID_ID"]).copy()
        df["GRID_ID"] = df["GRID_ID"].astype(int)
        df["main_WUI"] = df["main_WUI"].astype(str).str.strip().str.lower()
        df["wui_type"] = df["main_WUI"].map({"face": "Interface", "mix": "Intermix"}).fillna("Unknown")
        df["wui_area"] = pd.to_numeric(df["All_WUI_area"], errors="coerce").fillna(0.0)
        df["Continent"] = pd.to_numeric(df["Continent"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["Continent"]).copy()
        df = df[df["Continent"] != 0].copy()
        df.loc[df["Continent"] == 5, "Continent"] = 3
        df["continent_name"] = df["Continent"].astype(int).map(CONTINENT_CODE_LABELS)
        df = df[df["continent_name"].isin(CONTINENT_ORDER)].copy()
        df["WUI_Year"] = wui_year
        parts.append(df[["WUI_Year", "GRID_ID", "wui_type", "wui_area", "continent_name"]])

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=["WUI_Year", "GRID_ID"], keep="first")
    return combined


def merge_quadrant_labels(quadrant_df: pd.DataFrame) -> pd.DataFrame:
    wui_attribute_df = load_wui_attributes()
    merged = quadrant_df.merge(wui_attribute_df, on=["WUI_Year", "GRID_ID"], how="left")
    merged["wui_type"] = merged["wui_type"].fillna("Unknown")
    merged["wui_area"] = pd.to_numeric(merged["wui_area"], errors="coerce").fillna(0.0)
    return merged


def summarize_quadrant_proportions(
    df: pd.DataFrame,
    *,
    group_col: str,
    group_order: list[str],
) -> pd.DataFrame:
    area_sums = (
        df.loc[df[group_col].isin(group_order)]
        .groupby([group_col, "bivariate_key"], observed=False)["wui_area"]
        .sum()
        .unstack(fill_value=0)
        .reindex(index=group_order, columns=QUADRANT_ORDER, fill_value=0)
    )
    area_sums.index.name = group_col

    totals = area_sums.sum(axis=1)
    proportions = area_sums.div(totals.replace(0, np.nan), axis=0) * 100.0

    summary = area_sums.stack().rename("weighted_area").reset_index()
    summary["group_total_area"] = summary[group_col].map(totals.to_dict())
    percentage_values: list[float] = []
    for group, quadrant in zip(summary[group_col], summary["bivariate_key"]):
        if group in proportions.index and quadrant in proportions.columns:
            percentage_values.append(float(proportions.at[group, quadrant]))
        else:
            percentage_values.append(np.nan)
    summary["percentage"] = percentage_values
    summary["quadrant_label"] = summary["bivariate_key"].map(QUADRANT_LABELS)
    return summary


def build_plot_table(
    summary: pd.DataFrame,
    *,
    group_col: str,
    group_order: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    pivot = (
        summary.pivot(index=group_col, columns="bivariate_key", values="percentage")
        .reindex(index=group_order, columns=QUADRANT_ORDER)
        .fillna(0.0)
    )
    totals = (
        summary[[group_col, "group_total_area"]]
        .drop_duplicates()
        .set_index(group_col)["group_total_area"]
        .reindex(group_order)
        .fillna(0.0)
        .astype(float)
        .to_dict()
    )
    return pivot, totals


def prepare_continent_plot_data(scope_df: pd.DataFrame, scope_name: str) -> tuple[pd.DataFrame, dict[str, float]] | None:
    if scope_df["GRID_ID"].nunique() == 0:
        print(f"All GRID_ID data are missing for {scope_name} in years [{FIRE_YEAR_START}, {FIRE_YEAR_END}].")
        return None

    continent_main = scope_df.loc[scope_df["continent_name"].isin(CONTINENT_ORDER)].copy()
    if continent_main.empty:
        print(
            f"All GRID_ID data are missing for continent statistics in {scope_name} "
            f"({FIRE_YEAR_START}-{FIRE_YEAR_END})."
        )
        return None

    omitted_continent = len(scope_df) - len(continent_main)
    print(f"{scope_name}: omitted from continent bars (missing or unsupported continent): {omitted_continent}")

    continent_summary = summarize_quadrant_proportions(
        continent_main,
        group_col="continent_name",
        group_order=CONTINENT_ORDER,
    )
    continent_plot, continent_totals = build_plot_table(
        continent_summary,
        group_col="continent_name",
        group_order=CONTINENT_ORDER,
    )
    return continent_plot, continent_totals


def build_output_table(
    plot_table: pd.DataFrame,
    totals: dict[str, float],
    *,
    scope_name: str,
) -> pd.DataFrame:
    table = plot_table.copy()
    table.columns = [QUADRANT_LABELS[col] for col in table.columns]
    table = table.reset_index().rename(columns={"continent_name": "group_code"})
    table["group_label"] = table["group_code"]
    table["scope"] = scope_name
    table["grouping_type"] = "Continent"
    table["group_total_area"] = table["group_code"].map(totals).astype(float)
    ordered_columns = [
        "scope",
        "grouping_type",
        "group_code",
        "group_label",
        "group_total_area",
        QUADRANT_LABELS[(1, 1)],
        QUADRANT_LABELS[(1, 2)],
        QUADRANT_LABELS[(2, 1)],
        QUADRANT_LABELS[(2, 2)],
    ]
    return table.loc[:, ordered_columns]


def write_output_tables(merged: pd.DataFrame, output_table: Path) -> None:
    scope_frames = {
        "All": merged,
        "Interface": merged.loc[merged["wui_type"] == "Interface"].copy(),
        "Intermix": merged.loc[merged["wui_type"] == "Intermix"].copy(),
    }
    with pd.ExcelWriter(output_table, engine="openpyxl") as writer:
        for scope_name, scope_df in scope_frames.items():
            prepared = prepare_continent_plot_data(scope_df, scope_name)
            if prepared is None:
                continue
            plot_table, totals = prepared
            output_df = build_output_table(
                plot_table,
                totals,
                scope_name=scope_name,
            )
            output_df.to_excel(writer, sheet_name=f"{scope_name}_continent", index=False)
    print(f"Saved table to: {output_table}")


def plot_grouped_stacked_percentage_bars(
    ax: plt.Axes,
    interface_plot_table: pd.DataFrame,
    interface_totals: dict[str, float],
    intermix_plot_table: pd.DataFrame,
    intermix_totals: dict[str, float],
    *,
    x_labels: list[str],
    y_label: str,
    xtick_fontsize: float,
) -> None:
    x = np.arange(len(interface_plot_table.index))
    single_width = BAR_WIDTH * len(interface_plot_table.index) / REFERENCE_CATEGORY_COUNT
    bar_width = single_width * GROUPED_BAR_WIDTH_SCALE
    bar_offset = bar_width * GROUPED_BAR_OFFSET_SCALE

    interface_bottom = np.zeros(len(interface_plot_table.index), dtype=float)
    intermix_bottom = np.zeros(len(intermix_plot_table.index), dtype=float)

    for quadrant in QUADRANT_ORDER:
        color = QUADRANT_COLORS[quadrant]
        interface_values = interface_plot_table[quadrant].to_numpy(dtype=float)
        intermix_values = intermix_plot_table[quadrant].to_numpy(dtype=float)

        ax.bar(
            x - bar_offset,
            interface_values,
            bottom=interface_bottom,
            width=bar_width,
            color=color,
            edgecolor="none",
            linewidth=0.0,
            alpha=INTERFACE_BAR_ALPHA,
        )
        ax.bar(
            x + bar_offset,
            intermix_values,
            bottom=intermix_bottom,
            width=bar_width,
            color=color,
            edgecolor=color,
            linewidth=0.8,
            hatch=INTERMIX_HATCH,
            alpha=INTERMIX_BAR_ALPHA,
        )
        interface_bottom += interface_values
        intermix_bottom += intermix_values

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=xtick_fontsize)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylabel(y_label, fontsize=17)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)

    for xpos, group in enumerate(interface_plot_table.index):
        if interface_totals.get(group, 0) == 0:
            ax.text(xpos - bar_offset, 50, "NA", ha="center", va="center", fontsize=12, color="black")
        if intermix_totals.get(group, 0) == 0:
            ax.text(xpos + bar_offset, 50, "NA", ha="center", va="center", fontsize=12, color="black")


def render_continent_interface_intermix_combined(
    interface_df: pd.DataFrame,
    intermix_df: pd.DataFrame,
    output_figure: Path,
) -> bool:
    interface_prepared = prepare_continent_plot_data(interface_df, "Interface")
    intermix_prepared = prepare_continent_plot_data(intermix_df, "Intermix")
    if interface_prepared is None or intermix_prepared is None:
        return False

    interface_plot, interface_totals = interface_prepared
    intermix_plot, intermix_totals = intermix_prepared

    fig, ax = plt.subplots(1, 1, figsize=(14.8, 5.44))

    plot_grouped_stacked_percentage_bars(
        ax,
        interface_plot,
        interface_totals,
        intermix_plot,
        intermix_totals,
        x_labels=[CONTINENT_LABELS[item] for item in CONTINENT_ORDER],
        y_label="WUI area proportion",
        xtick_fontsize=XTICK_LABEL_FONT_SIZE,
    )

    legend_color = QUADRANT_COLORS[(2, 2)]
    legend_handles = [
        Patch(
            facecolor=legend_color,
            edgecolor="none",
            linewidth=0.0,
            alpha=INTERFACE_BAR_ALPHA,
        ),
        Patch(
            facecolor=legend_color,
            edgecolor=legend_color,
            linewidth=0.8,
            hatch=INTERMIX_HATCH,
            alpha=INTERMIX_BAR_ALPHA,
        ),
    ]
    fig.legend(
        handles=legend_handles,
        labels=["Interface", "Intermix"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.055),
        ncol=2,
        frameon=False,
        fontsize=20,
        handlelength=2.2,
        columnspacing=1.8,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.18)
    fig.savefig(str(output_figure), dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {output_figure}")
    return True


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    quadrant_df = load_bivariate_metrics()
    merged = merge_quadrant_labels(quadrant_df)

    if merged["GRID_ID"].nunique() == 0:
        raise ValueError(
            f"All GRID_ID data are missing after merging quadrant results for years [{FIRE_YEAR_START}, {FIRE_YEAR_END}]."
        )

    ok = render_continent_interface_intermix_combined(
        merged.loc[merged["wui_type"] == "Interface"].copy(),
        merged.loc[merged["wui_type"] == "Intermix"].copy(),
        OUTPUT_FIGURE,
    )
    if not ok:
        raise ValueError("Failed to render continent Interface/Intermix stacked-bar figure.")
    write_output_tables(merged, OUTPUT_TABLE)


if __name__ == "__main__":
    main()
