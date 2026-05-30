#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称:
    HFI 对 Fire frequency 与 Normalized burned area 的年际趋势图（模板时间趋势图风格）

功能简介:
    1. 分别读取 Fire num 与 Fire area 文件夹下所有 HFI_effect_*.csv。
    2. 按文件名前缀将分类聚合为 Interface(face_*)、Intermix(mix_*) 和 All。
    3. 对每个指标分别绘制三类 WUI 的年际均值趋势图：
       细原始折线 + 点、粗 LOWESS 平滑线、95% 置信区间半透明带。
    4. 输出两张高分辨率 PNG 图。

输出:
    - B:\WUI\Picture\Annual_trend_Human Footprint_effect_on_Fire_frequency_and_Normalized_burn_area_mean_se.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
COMBINED_OUTPUT_FILENAME = "Annual_trend_Human Footprint_effect_on_Fire_frequency_and_Normalized_burn_area_mean_se.png"
YEAR_START = 2003
YEAR_END = 2022
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
    },
    {
        "key": "Intermix",
        "label": "Intermix",
        "color": "#FF7F00",
    },
    {
        "key": "All",
        "label": "All",
        "color": "#C85C5C",
    },
]

METRICS = [
    {
        "name": "Fire frequency",
        "input_dir": Path(r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num"),
        "output_filename": "Annual_trend_Human Footprint_effect_on_Fire_frequency_mean_se.png",
        "ylabel": "Human activity effects on fire frequency",
        "panel_letter": "a",
    },
    {
        "name": "Normalized burned area",
        "input_dir": Path(r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area"),
        "output_filename": "Annual_trend_Human Footprint_effect_on_Normalized_burn_area_mean_se.png",
        "ylabel": "Human activity effects on normalized burned area",
        "panel_letter": "b",
    },
]


def infer_wui_type(csv_path: Path) -> str:
    name = csv_path.name.lower()
    if "hfi_effect_face_" in name:
        return "Interface"
    if "hfi_effect_mix_" in name:
        return "Intermix"
    raise ValueError(f"Cannot infer WUI type from filename: {csv_path.name}")


def load_effect_data_by_series(input_dir: Path) -> dict[str, pd.DataFrame]:
    csv_files = sorted(input_dir.glob("HFI_effect_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No HFI effect CSV files found in: {input_dir}")

    print(f"Found {len(csv_files)} CSV file(s) in {input_dir}.")
    parts: list[pd.DataFrame] = []
    type_file_counts = {"Interface": 0, "Intermix": 0}

    for csv_path in csv_files:
        wui_type = infer_wui_type(csv_path)
        type_file_counts[wui_type] += 1

        header = pd.read_csv(csv_path, nrows=0)
        required = {"Year", "GRID_ID", "diff_contribution"}
        missing = required.difference(header.columns.tolist())
        if missing:
            raise ValueError(f"Missing columns {sorted(missing)} in: {csv_path}")

        df = pd.read_csv(
            csv_path,
            usecols=["Year", "GRID_ID", "diff_contribution"],
            dtype={"GRID_ID": "string"},
            low_memory=False,
        )
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        df["GRID_ID"] = pd.to_numeric(df["GRID_ID"], errors="coerce").astype("Int64")
        df["diff_contribution"] = pd.to_numeric(df["diff_contribution"], errors="coerce")
        df = df.dropna(subset=["Year", "GRID_ID", "diff_contribution"])
        df = df[df["Year"].between(YEAR_START, YEAR_END)].copy()
        if df.empty:
            continue
        df["WUI_Type"] = wui_type
        parts.append(df)

    if not parts:
        raise ValueError(
            f"No valid rows found in {input_dir} for year range [{YEAR_START}, {YEAR_END}]."
        )

    combined = pd.concat(parts, ignore_index=True)
    by_type = (
        combined.groupby(["WUI_Type", "Year", "GRID_ID"], as_index=False)
        .agg(diff_contribution=("diff_contribution", "mean"))
        .sort_values(["WUI_Type", "Year", "GRID_ID"])
        .reset_index(drop=True)
    )
    all_df = (
        combined.groupby(["Year", "GRID_ID"], as_index=False)
        .agg(diff_contribution=("diff_contribution", "mean"))
        .sort_values(["Year", "GRID_ID"])
        .reset_index(drop=True)
    )

    series_frames = {
        "Interface": by_type.loc[by_type["WUI_Type"] == "Interface", ["Year", "GRID_ID", "diff_contribution"]]
        .copy(),
        "Intermix": by_type.loc[by_type["WUI_Type"] == "Intermix", ["Year", "GRID_ID", "diff_contribution"]]
        .copy(),
        "All": all_df.copy(),
    }

    print(
        "CSV counts by WUI type: "
        f"Interface={type_file_counts['Interface']}, "
        f"Intermix={type_file_counts['Intermix']}, "
        f"All={len(csv_files)}"
    )
    for series in SERIES_CONFIGS:
        series_df = series_frames[series["key"]]
        print(
            f"{series['label']}: years={series_df['Year'].nunique()}, "
            f"rows={len(series_df):,}, unique GRID_ID={series_df['GRID_ID'].nunique():,}"
        )
    return series_frames


def build_annual_stats(series_df: pd.DataFrame) -> pd.DataFrame:
    annual = (
        series_df.groupby("Year", as_index=False)
        .agg(
            mean=("diff_contribution", "mean"),
            std=("diff_contribution", "std"),
            n=("diff_contribution", "size"),
        )
        .sort_values("Year")
        .reset_index(drop=True)
    )
    annual["std"] = annual["std"].fillna(0.0)
    annual["se"] = annual["std"] / np.sqrt(annual["n"])
    return annual[["Year", "mean", "se", "n"]]


def compute_smooth_band(x: np.ndarray, y: np.ndarray, se: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    se = np.asarray(se, dtype=float)

    if x.size == 1:
        grid = np.array([x[0], x[0] + 1e-6], dtype=float)
        fit = np.array([y[0], y[0]], dtype=float)
        band = np.array([CONFIDENCE_Z * max(se[0], 0.0), CONFIDENCE_Z * max(se[0], 0.0)], dtype=float)
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


def print_trend_summary(annual_stats_map: dict[str, pd.DataFrame], metric_name: str) -> None:
    print(f"\n=== {metric_name} ===")
    for series in SERIES_CONFIGS:
        annual_df = annual_stats_map[series["key"]]
        years = annual_df["Year"].to_numpy(dtype=float)
        values = annual_df["mean"].to_numpy(dtype=float)
        tau, p_value = kendalltau(years, values)
        slope, intercept, _, _ = theilslopes(values, years, alpha=0.95)
        print(
            f"{series['label']}: tau={tau:.3f}, p={p_value:.3g}, "
            f"Sen slope={slope:.3e}, intercept={intercept:.3e}"
        )


def draw_trend_panel(
    ax: plt.Axes,
    annual_stats_map: dict[str, pd.DataFrame],
    *,
    ylabel: str,
    panel_letter: str,
    show_legend: bool,
) -> None:
    legend_handles: list[Line2D] = []
    available_years: list[int] = sorted(
        {
            int(year)
            for annual_df in annual_stats_map.values()
            for year in annual_df["Year"].dropna().astype(int).tolist()
        }
    )
    xticks = [year for year in [2005, 2010, 2015, 2020] if year in available_years]
    if not xticks:
        xticks = available_years

    for series in SERIES_CONFIGS:
        annual_df = annual_stats_map[series["key"]]
        if annual_df.empty:
            continue

        x = annual_df["Year"].to_numpy(dtype=float)
        y = annual_df["mean"].to_numpy(dtype=float)
        se = annual_df["se"].to_numpy(dtype=float)
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
                marker="o",
                markersize=9,
                markerfacecolor=str(series["color"]),
                markeredgecolor="white",
                markeredgewidth=0.45,
                label=str(series["label"]),
            )
        )

    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
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
        panel_letter,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=29,
        fontweight="bold",
        color="black",
    )


def plot_combined_trends(metric_payloads: list[dict[str, object]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), facecolor="white")
    for ax, payload in zip(axes, metric_payloads):
        draw_trend_panel(
            ax,
            annual_stats_map=payload["annual_stats_map"],
            ylabel=str(payload["ylabel"]),
            panel_letter=str(payload["panel_letter"]),
            show_legend=bool(payload["show_legend"]),
        )

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.13, top=0.96, wspace=0.22)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / COMBINED_OUTPUT_FILENAME
    fig.savefig(output_path, dpi=1200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Figure saved to: {output_path}")
    return output_path


def main() -> None:
    metric_payloads: list[dict[str, object]] = []
    for metric in METRICS:
        print(f"\n=== Processing {metric['name']} ===")
        series_frames = load_effect_data_by_series(metric["input_dir"])
        annual_stats_map = {
            series["key"]: build_annual_stats(series_frames[series["key"]])
            for series in SERIES_CONFIGS
        }
        print_trend_summary(annual_stats_map, metric["name"])
        metric_payloads.append(
            {
                "annual_stats_map": annual_stats_map,
                "ylabel": metric["ylabel"],
                "panel_letter": metric["panel_letter"],
                "show_legend": metric["name"] == "Normalized burned area",
            }
        )

    plot_combined_trends(metric_payloads)


if __name__ == "__main__":
    main()
