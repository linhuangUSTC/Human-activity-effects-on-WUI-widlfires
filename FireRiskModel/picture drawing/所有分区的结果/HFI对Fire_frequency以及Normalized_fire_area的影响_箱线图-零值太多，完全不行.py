#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称:
    HFI 对 Fire frequency 与 Normalized burned area 影响的箱线图

功能简介:
    1. 读取以下目录中的 HFI 净效应固定效应结果 CSV:
       - B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num
       - B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area
    2. 每个 CSV 按 Year 计算 diff_contribution 的年均值。
    3. 按 climate zone 与 landcover 组织 Interface / Intermix 两类 WUI 的年均值分布。
    4. 直接输出两张单独箱线图，并显示均值:
       - Human Footprint effect on Fire frequency
       - Human Footprint effect on Normalized burned area

输出:
    - B:\WUI\Picture\Ranking_Human Footprint_effect_boxplot_Fire_frequency.png
    - B:\WUI\Picture\Ranking_Human Footprint_effect_boxplot_Normalized_burn_area.png
"""

from __future__ import annotations

import glob
import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator


FIRE_NUM_DIR = r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num"
FIRE_AREA_DIR = r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area"

OUTPUT_DIR = r"B:\WUI\Picture"
OUTPUT_PNG_FIRE_NUM = "Ranking_Human Footprint_effect_boxplot_Fire_frequency.png"
OUTPUT_PNG_FIRE_AREA = "Ranking_Human Footprint_effect_boxplot_Normalized_burn_area.png"

YEAR_START = 2003
YEAR_END = 2022

CLIMATE_ORDER = ["A", "B", "C", "D"]
MODE_ORDER = ["face", "mix"]
LANDCOVER_ORDER = ["CRP", "FST", "GRS", "SHR", "WET"]

POSITIVE_COLOR = "#F28E2B"
NEGATIVE_COLOR = "#4E79A7"
AX_FACE = "#ffffff"
ZERO_LINE_COLOR = "#7f7f7f"
EDGE_COLOR = "#4d4d4d"

plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11


def choose_exponent(values: np.ndarray) -> int:
    finite_vals = values[np.isfinite(values)]
    if finite_vals.size == 0:
        return 0
    max_abs = float(np.max(np.abs(finite_vals)))
    if max_abs < 1e-12:
        return 0
    return int(np.floor(np.log10(max_abs)))


def extract_category_id(path: str) -> str:
    name = Path(path).stem
    match = re.search(r"HFI_effect_((?:face|mix)_[^_]+_[^_]+)", name, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    trimmed = re.sub(r"^HFI_effect_", "", name, flags=re.IGNORECASE)
    parts = trimmed.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else trimmed


def split_category_parts(category_id: str) -> tuple[str, str, str]:
    parts = category_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected category_id: {category_id}")
    return parts[0].lower(), parts[1].upper(), parts[2].upper()


def save_with_fallback_figure(fig: plt.Figure, path: str) -> str:
    try:
        fig.savefig(path, dpi=1200, bbox_inches="tight", facecolor=fig.get_facecolor())
        return path
    except PermissionError:
        stem, ext = os.path.splitext(path)
        alt_path = f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        fig.savefig(alt_path, dpi=1200, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Target PNG is locked, saved fallback file: {alt_path}")
        return alt_path


def load_metric_annual(input_dir: str, response_name: str) -> pd.DataFrame:
    csv_files = sorted(glob.glob(os.path.join(input_dir, "HFI_effect_*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    rows: list[pd.DataFrame] = []
    print(f"Reading {len(csv_files)} file(s) for {response_name}: {input_dir}")

    for csv_path in csv_files:
        df = pd.read_csv(csv_path, usecols=["Year", "diff_contribution"], low_memory=False)
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        df["diff_contribution"] = pd.to_numeric(df["diff_contribution"], errors="coerce")
        df = df.dropna(subset=["Year", "diff_contribution"])
        df = df[df["Year"].between(YEAR_START, YEAR_END)]
        if df.empty:
            continue

        annual = df.groupby("Year", as_index=False)["diff_contribution"].mean()
        category_id = extract_category_id(csv_path)
        mode, climate, landcover = split_category_parts(category_id)
        annual["category_id"] = category_id
        annual["mode"] = mode
        annual["climate"] = climate
        annual["landcover"] = landcover
        annual["response"] = response_name
        annual = annual.rename(columns={"diff_contribution": "annual_mean_effect"})
        rows.append(annual)

    if not rows:
        raise ValueError(f"No valid annual rows from: {input_dir}")

    out = pd.concat(rows, ignore_index=True)
    out["mode"] = pd.Categorical(out["mode"], MODE_ORDER, ordered=True)
    out["climate"] = pd.Categorical(out["climate"], CLIMATE_ORDER, ordered=True)
    out["landcover"] = pd.Categorical(out["landcover"], LANDCOVER_ORDER, ordered=True)
    return out.sort_values(["climate", "landcover", "mode", "Year"]).reset_index(drop=True)


def coerce_array(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value.astype(float, copy=False)
    elif isinstance(value, (list, tuple, pd.Series)):
        arr = np.asarray(value, dtype=float)
    else:
        return np.array([], dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr.astype(float, copy=False)


def build_plot_frame(metric_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        metric_df.groupby(["climate", "landcover", "mode"], observed=True)["annual_mean_effect"]
        .apply(lambda s: s.dropna().to_numpy(dtype=float))
        .reset_index(name="values")
    )

    full_index = pd.MultiIndex.from_product(
        [CLIMATE_ORDER, LANDCOVER_ORDER, MODE_ORDER],
        names=["climate", "landcover", "mode"],
    )
    completed = (
        grouped.set_index(["climate", "landcover", "mode"])[["values"]]
        .reindex(full_index)
        .reset_index()
    )
    completed["values"] = completed["values"].apply(coerce_array)
    completed["mean_effect"] = completed["values"].apply(
        lambda arr: float(arr.mean()) if arr.size else np.nan
    )
    return completed


def format_metric_ylabel(metric_label: str) -> str:
    if metric_label == "Human Footprint effect on Fire frequency":
        return "Human Footprint effect on\nFire frequency"
    if metric_label == "Human Footprint effect on Normalized burned area":
        return "Human Footprint effect on\nNormalized burned area"
    return metric_label


def build_figure(metric_df: pd.DataFrame, *, metric_label: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(24, 8.2), facecolor="white")
    plot_df = build_plot_frame(metric_df)

    category_positions = np.arange(len(CLIMATE_ORDER) * len(LANDCOVER_ORDER), dtype=float)
    base_df = pd.DataFrame(
        [
            (climate, landcover, pos)
            for pos, (climate, landcover) in enumerate(
                [(climate, landcover) for climate in CLIMATE_ORDER for landcover in LANDCOVER_ORDER]
            )
        ],
        columns=["climate", "landcover", "position"],
    )
    plot_df = plot_df.merge(base_df, on=["climate", "landcover"], how="left", validate="many_to_one")

    box_offset = 0.16
    box_width = 0.22
    plot_df["x"] = np.where(
        plot_df["mode"].astype(str) == "face",
        plot_df["position"] - box_offset,
        plot_df["position"] + box_offset,
    )
    plot_df["fill_color"] = np.where(plot_df["mean_effect"] >= 0, POSITIVE_COLOR, NEGATIVE_COLOR)
    plot_df["hatch"] = np.where(plot_df["mode"].astype(str) == "face", "", "///")

    ax.set_facecolor(AX_FACE)
    ax.axhline(0.0, color=ZERO_LINE_COLOR, linewidth=1.0, linestyle="--", zorder=1)
    ax.set_axisbelow(True)

    for _, row in plot_df.iterrows():
        values = row["values"]
        if values.size == 0:
            continue

        box = ax.boxplot(
            [values],
            positions=[float(row["x"])],
            widths=box_width,
            patch_artist=True,
            showmeans=True,
            showfliers=False,
            manage_ticks=False,
            meanprops={
                "marker": "D",
                "markerfacecolor": "#ffffff",
                "markeredgecolor": EDGE_COLOR,
                "markersize": 6.5,
                "markeredgewidth": 1.0,
            },
            boxprops={
                "facecolor": str(row["fill_color"]),
                "edgecolor": EDGE_COLOR,
                "linewidth": 1.2,
                "alpha": 0.82,
            },
            whiskerprops={
                "color": EDGE_COLOR,
                "linewidth": 1.1,
            },
            capprops={
                "color": EDGE_COLOR,
                "linewidth": 1.1,
            },
            medianprops={
                "color": "#111111",
                "linewidth": 1.35,
            },
        )
        box["boxes"][0].set_hatch(str(row["hatch"]))

    for separator_x in [4.5, 9.5, 14.5]:
        ax.axvline(separator_x, color="#a8a8a8", linewidth=1.8, linestyle="-", zorder=2)

    ax.set_xticks(category_positions)
    ax.set_xticklabels(
        [landcover for _climate in CLIMATE_ORDER for landcover in LANDCOVER_ORDER],
        rotation=0,
        ha="center",
        va="top",
        fontsize=17,
    )
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_xlim(-0.7, len(category_positions) - 0.3)
    ax.set_ylabel(format_metric_ylabel(metric_label), fontsize=23)

    all_values = np.concatenate([arr for arr in plot_df["values"] if arr.size > 0]) if any(
        arr.size > 0 for arr in plot_df["values"]
    ) else np.array([0.0], dtype=float)
    y_min = float(np.nanmin(all_values))
    y_max = float(np.nanmax(all_values))
    if y_min == y_max:
        y_min -= 0.1
        y_max += 0.1
    pad = (y_max - y_min) * 0.08
    ax.set_ylim(y_min - pad, y_max + pad)

    exponent = choose_exponent(all_values)
    scale_factor = 10.0 ** exponent
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val / scale_factor:.1f}"))
    ax.tick_params(axis="y", labelsize=20)
    ax.text(
        0.0,
        1.01,
        f"1e{exponent:+d}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=19,
        color=EDGE_COLOR,
    )

    climate_centers = {
        "A": 2.0,
        "B": 7.0,
        "C": 12.0,
        "D": 17.0,
    }
    for climate, x_center in climate_centers.items():
        ax.text(
            x_center,
            0.045,
            climate,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=21,
            fontweight="bold",
            color="#4f4f4f",
        )

    legend_handles = [
        Patch(facecolor="#ffffff", edgecolor=EDGE_COLOR, linewidth=1.2, label="Interface"),
        Patch(facecolor="#ffffff", edgecolor=EDGE_COLOR, linewidth=1.2, hatch="///", label="Intermix"),
        Patch(facecolor=POSITIVE_COLOR, edgecolor="none", linewidth=0.0, label="Positive effect"),
        Patch(facecolor=NEGATIVE_COLOR, edgecolor="none", linewidth=0.0, label="Negative effect"),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            markerfacecolor="#ffffff",
            markeredgecolor=EDGE_COLOR,
            markeredgewidth=1.0,
            markersize=7,
            label="Mean",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        ncol=3,
        frameon=False,
        fontsize=18,
        handlelength=1.8,
        columnspacing=1.0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.spines["left"].set_color(EDGE_COLOR)
    ax.spines["bottom"].set_color(EDGE_COLOR)

    fig.subplots_adjust(left=0.11, right=0.99, top=0.92, bottom=0.20)
    return fig


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fire_num = load_metric_annual(FIRE_NUM_DIR, "Fire frequency")
    fire_area = load_metric_annual(FIRE_AREA_DIR, "Normalized burned area")

    fig = build_figure(
        fire_num,
        metric_label="Human Footprint effect on Fire frequency",
    )
    output_png = save_with_fallback_figure(fig, os.path.join(OUTPUT_DIR, OUTPUT_PNG_FIRE_NUM))
    plt.close(fig)
    print(f"Figure saved to: {output_png}")

    fig = build_figure(
        fire_area,
        metric_label="Human Footprint effect on Normalized burned area",
    )
    output_png = save_with_fallback_figure(fig, os.path.join(OUTPUT_DIR, OUTPUT_PNG_FIRE_AREA))
    plt.close(fig)
    print(f"Figure saved to: {output_png}")


if __name__ == "__main__":
    main()
