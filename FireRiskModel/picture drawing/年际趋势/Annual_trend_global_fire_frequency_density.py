#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称:
    全球野火频率与归一化面积年际趋势图绘制程序（模板时间趋势图风格）

功能简介:
    1. 读取 2005、2010、2015、2020 四期 WUI 网格数据。
    2. 计算全球范围内 Interface、Intermix、All 三类 WUI 的年际平均值。
    3. 以与 Word 模板“时间趋势图”相近的风格绘制趋势图：
       细原始折线 + 点、粗平滑趋势线、95% 置信区间半透明带、极简坐标轴。
    4. 输出两张高分辨率 JPG 图。

输入文件:
    - I:/Processing data/Grid_Clip_YYYY_WUI.csv

输出文件:
    - B:/WUI/Picture/Global_fire_frequency_and_normalized_burn_area_annual_trend_template_style.jpg
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from scipy.stats import kendalltau, theilslopes
from statsmodels.nonparametric.smoothers_lowess import lowess


plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 15
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 17
plt.rcParams["ytick.labelsize"] = 17
plt.rcParams["legend.fontsize"] = 18

OUTPUT_DIR = Path(r"B:\WUI\Picture")
COMBINED_OUTPUT_FILENAME = "Global_fire_frequency_and_normalized_burn_area_annual_trend_template_style.jpg"
WUI_YEARS = [2005, 2010, 2015, 2020]
LOWESS_FRAC = 0.4
SMOOTH_GRID_SIZE = 240
CONFIDENCE_Z = 1.96
RAW_LINE_ALPHA = 0.65
RIBBON_ALPHA = 0.18
FIGURE_WIDTH = 16.8
FIGURE_HEIGHT = 7.2

SERIES_CONFIGS = [
    {
        "key": "Interface",
        "label": "Interface",
        "color": "#1F77B4",
        "marker": "o",
        "area_field": "Interface_WUI_area",
    },
    {
        "key": "Intermix",
        "label": "Intermix",
        "color": "#FF7F00",
        "marker": "o",
        "area_field": "Intermix_WUI_area",
    },
    {
        "key": "All",
        "label": "All",
        "color": "#C85C5C",
        "marker": "o",
        "area_field": "All_WUI_area",
    },
]

STATS_CONFIGS = [
    {
        "name": "Fire Frequency per WUI",
        "panel_letter": "a",
        "interface_field": lambda year: f"Interface_Fire_num_{year}",
        "intermix_field": lambda year: f"Intermix_Fire_num_{year}",
        "all_field": lambda year: f"All_Fire_num_{year}",
        "ylabel": "Fire frequency",
        "output_filename": "Global_fire_frequency_per_wui_annual_trend_template_style.jpg",
    },
    {
        "name": "Normalized burned area",
        "panel_letter": "b",
        "interface_field": lambda year: f"Interface_Fire_area_{year}",
        "intermix_field": lambda year: f"Intermix_Fire_area_{year}",
        "all_field": lambda year: f"All_Fire_area_{year}",
        "ylabel": "Normalized burned area",
        "output_filename": "Global_normalized_burn_area_annual_trend_template_style.jpg",
    },
]


def build_fire_year_to_wui_year() -> dict[int, int]:
    mapping: dict[int, int] = {}
    for wui_year in WUI_YEARS:
        for offset in range(-2, 3):
            mapping[wui_year + offset] = wui_year
    return mapping


FIRE_YEAR_TO_WUI_YEAR = build_fire_year_to_wui_year()


def load_wui_grid(year: int) -> pd.DataFrame:
    csv_path = Path(f"I:/Processing data/Grid_Clip_{year}_WUI.csv")
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded {len(df):,} rows for WUI year {year}.")
    return df


def calculate_global_annual_stats(df: pd.DataFrame, base_year: int, stat_config: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for fire_year in range(base_year - 2, base_year + 3):
        required_value_fields = {
            "Interface": str(stat_config["interface_field"](fire_year)),
            "Intermix": str(stat_config["intermix_field"](fire_year)),
            "All": str(stat_config["all_field"](fire_year)),
        }
        required_area_fields = {series["key"]: str(series["area_field"]) for series in SERIES_CONFIGS}
        missing = [
            field
            for field in list(required_value_fields.values()) + list(required_area_fields.values())
            if field not in df.columns
        ]
        if missing:
            print(f"Skip fire year {fire_year}: missing field(s) {missing}")
            continue

        row: dict[str, float | int] = {
            "Year": fire_year,
            "WUI_Year": FIRE_YEAR_TO_WUI_YEAR[fire_year],
        }
        for series in SERIES_CONFIGS:
            series_key = str(series["key"])
            value_field = required_value_fields[series_key]
            area_field = required_area_fields[series_key]

            values = pd.to_numeric(df[value_field], errors="coerce")
            areas = pd.to_numeric(df[area_field], errors="coerce")
            valid = values.notna() & areas.notna() & np.isfinite(values) & np.isfinite(areas) & (areas > 0)
            density = values.loc[valid].to_numpy(dtype=float) / areas.loc[valid].to_numpy(dtype=float)
            n = int(density.shape[0])
            mean_value = float(np.mean(density)) if n > 0 else np.nan
            if n <= 1:
                se_value = 0.0
            else:
                se_value = float(np.std(density, ddof=1) / np.sqrt(n))
            row[f"{series_key}_mean"] = mean_value
            row[f"{series_key}_se"] = se_value
            row[f"{series_key}_n"] = n
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Year").reset_index(drop=True)


def combine_annual_stats(year_data_dict: dict[int, pd.DataFrame], stat_config: dict[str, object]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for base_year, df in sorted(year_data_dict.items()):
        annual_df = calculate_global_annual_stats(df, base_year, stat_config)
        if not annual_df.empty:
            parts.append(annual_df)

    if not parts:
        raise ValueError(f"No annual statistics available for {stat_config['name']}.")

    combined = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates(subset=["Year"], keep="first")
        .sort_values("Year")
        .reset_index(drop=True)
    )
    return combined


def compute_smooth_band(x: np.ndarray, y: np.ndarray, se: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    se = np.asarray(se, dtype=float)

    if x.size == 1:
        grid = np.array([x[0], x[0] + 1e-6], dtype=float)
        fit = np.array([y[0], y[0]], dtype=float)
        band = np.array([max(se[0], 0.0), max(se[0], 0.0)], dtype=float)
        return grid, fit - band, fit + band

    grid = np.linspace(float(x.min()), float(x.max()), SMOOTH_GRID_SIZE)
    if x.size >= 4:
        smooth = lowess(y, x, frac=LOWESS_FRAC, return_sorted=True)
        smooth_se = lowess(np.maximum(se, 0.0), x, frac=min(LOWESS_FRAC, 0.55), return_sorted=True)
        fit = np.interp(grid, smooth[:, 0], smooth[:, 1])
        se_fit = np.interp(grid, smooth_se[:, 0], smooth_se[:, 1])
    else:
        fit = np.interp(grid, x, y)
        se_fit = np.interp(grid, x, np.maximum(se, 0.0))

    lower = fit - CONFIDENCE_Z * se_fit
    upper = fit + CONFIDENCE_Z * se_fit
    return grid, lower, upper


def print_trend_summary(annual_df: pd.DataFrame, stat_config: dict[str, object]) -> None:
    print(f"\n=== {stat_config['name']} ===")
    for series in SERIES_CONFIGS:
        mean_col = f"{series['key']}_mean"
        subset = annual_df[["Year", mean_col]].dropna()
        years = subset["Year"].to_numpy(dtype=float)
        values = subset[mean_col].to_numpy(dtype=float)
        tau, p_value = kendalltau(years, values)
        slope, intercept, _, _ = theilslopes(values, years, alpha=0.95)
        print(
            f"{series['label']}: tau={tau:.3f}, p={p_value:.3g}, "
            f"Sen slope={slope:.3e}, intercept={intercept:.3e}"
        )


def draw_trend_panel(
    ax: plt.Axes,
    annual_df: pd.DataFrame,
    stat_config: dict[str, object],
    *,
    show_legend: bool,
) -> None:
    legend_handles: list[Line2D] = []
    available_years = annual_df["Year"].dropna().astype(int).tolist()
    xticks = [year for year in [2005, 2010, 2015, 2020] if year in available_years]
    if not xticks:
        xticks = available_years

    for series in SERIES_CONFIGS:
        mean_col = f"{series['key']}_mean"
        se_col = f"{series['key']}_se"
        subset = annual_df[["Year", mean_col, se_col]].dropna().copy()
        if subset.empty:
            continue

        x = subset["Year"].to_numpy(dtype=float)
        y = subset[mean_col].to_numpy(dtype=float)
        se = subset[se_col].to_numpy(dtype=float)

        grid, lower, upper = compute_smooth_band(x, y, se)
        fit = (lower + upper) / 2.0

        ax.fill_between(grid, lower, upper, color=str(series["color"]), alpha=RIBBON_ALPHA, linewidth=0, zorder=1)
        ax.plot(grid, fit, color=str(series["color"]), linewidth=3.0, alpha=0.98, zorder=3)
        ax.plot(x, y, color=str(series["color"]), linewidth=0.9, alpha=RAW_LINE_ALPHA, zorder=2)
        ax.scatter(
            x,
            y,
            s=18,
            color=str(series["color"]),
            edgecolors="white",
            linewidths=0.45,
            alpha=0.9,
            zorder=4,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=str(series["color"]),
                linewidth=3.0,
                marker=str(series["marker"]),
                markersize=9,
                markerfacecolor=str(series["color"]),
                markeredgecolor="white",
                markeredgewidth=0.45,
                label=str(series["label"]),
            )
        )

    ax.set_xlabel("Year")
    ax.set_ylabel(str(stat_config["ylabel"]))
    ax.set_xticks(xticks)
    ax.set_xlim(min(available_years) - 0.8, max(available_years) + 0.8)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    formatter = ScalarFormatter(useMathText=False)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(formatter)
    ax.tick_params(axis="both", direction="in", length=7, width=0.9, color="#5c5c5c", pad=6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.spines["left"].set_color("#6a6a6a")
    ax.spines["bottom"].set_color("#6a6a6a")
    if show_legend:
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            frameon=False,
            handlelength=2.2,
            handletextpad=0.7,
            borderaxespad=0.4,
        )
    ax.text(
        -0.12,
        1.02,
        str(stat_config["panel_letter"]),
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=29,
        fontweight="bold",
        color="black",
    )


def plot_combined_trends(stat_payloads: list[dict[str, object]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), facecolor="white")
    for ax, payload in zip(axes, stat_payloads):
        draw_trend_panel(
            ax,
            annual_df=payload["annual_df"],
            stat_config=payload["stat_config"],
            show_legend=bool(payload["show_legend"]),
        )

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.13, top=0.96, wspace=0.22)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / COMBINED_OUTPUT_FILENAME
    fig.savefig(output_path, dpi=1200, bbox_inches="tight", facecolor=fig.get_facecolor(), format="jpg")
    plt.close(fig)
    print(f"Saved figure to: {output_path}")
    return output_path


def main() -> None:
    print("=== Global Fire Trend Program (Template Style) ===")
    year_data_dict: dict[int, pd.DataFrame] = {}
    for year in WUI_YEARS:
        year_data_dict[year] = load_wui_grid(year)

    stat_payloads: list[dict[str, object]] = []
    for stat_config in STATS_CONFIGS:
        annual_df = combine_annual_stats(year_data_dict, stat_config)
        print_trend_summary(annual_df, stat_config)
        stat_payloads.append(
            {
                "annual_df": annual_df,
                "stat_config": stat_config,
                "show_legend": str(stat_config["name"]) == "Normalized burned area",
            }
        )

    plot_combined_trends(stat_payloads)

    print("\nAll figures completed.")


if __name__ == "__main__":
    main()
