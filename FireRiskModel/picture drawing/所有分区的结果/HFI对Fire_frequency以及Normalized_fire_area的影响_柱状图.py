#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称:
    HFI 对 Fire frequency 与 Normalized burned area 影响的柱状图

功能简介:
    1. 读取以下目录中的 HFI 净效应固定效应结果 CSV:
       - B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num
       - B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area
    2. 每个 CSV 按 Year 计算 diff_contribution 的年均值。
    3. 按 category_id 汇总 2003-2022 年年均值的跨年平均和标准误。
    4. 直接输出两张单独柱状图:
       - Human Footprint effect on Fire frequency
       - Human Footprint effect on Normalized burned area

输出:
    - B:\WUI\Picture\Ranking_Human Footprint_effect_barplot_Fire_frequency.png
    - B:\WUI\Picture\Ranking_Human Footprint_effect_barplot_Normalized_burn_area.png
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
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

FIRE_NUM_DIR = r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num"
FIRE_AREA_DIR = r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area"

OUTPUT_DIR = r"B:\WUI\Picture"
OUTPUT_PNG_FIRE_NUM = "Ranking_Human Footprint_effect_barplot_Fire_frequency.png"
OUTPUT_PNG_FIRE_AREA = "Ranking_Human Footprint_effect_barplot_Normalized_burn_area.png"

YEAR_START = 2003
YEAR_END = 2022

TOP_ROW = ["face_A", "face_B", "face_C", "face_D"]
BOTTOM_ROW = ["mix_A", "mix_B", "mix_C", "mix_D"]
CLIMATE_ORDER = ["A", "B", "C", "D"]
MODE_ORDER = ["face", "mix"]
LANDCOVER_ORDER = ["CRP", "FST", "GRS", "SHR", "WET"]
CATEGORY_ORDER = [f"{mode}_{climate}_{landcover}" for climate in CLIMATE_ORDER for landcover in LANDCOVER_ORDER for mode in MODE_ORDER]

POSITIVE_COLOR = "#F28E2B"
NEGATIVE_COLOR = "#4E79A7"
AX_FACE = "#ffffff"
GRID_COLOR = "#e2e2e2"
ZERO_LINE_COLOR = "#7f7f7f"
EDGE_COLOR = "#4d4d4d"
CATEGORY_STEP = 0.84

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


def split_category_id(category_id: str) -> tuple[str, str]:
    parts = category_id.split("_")
    if len(parts) < 3:
        return category_id, category_id
    return f"{parts[0]}_{parts[1]}", parts[2]


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


def summarize_categories(metric_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metric_df.groupby(["mode", "climate", "landcover"], as_index=False, observed=True)
        .agg(
            mean_effect=("annual_mean_effect", "mean"),
            sd_effect=("annual_mean_effect", "std"),
            n_years=("annual_mean_effect", "size"),
        )
    )
    summary["sd_effect"] = summary["sd_effect"].fillna(0.0)
    summary["se_effect"] = summary["sd_effect"] / np.sqrt(summary["n_years"].clip(lower=1))
    summary["category_id"] = (
        summary["mode"].astype(str) + "_" + summary["climate"].astype(str) + "_" + summary["landcover"].astype(str)
    )
    summary["category_id"] = pd.Categorical(summary["category_id"], CATEGORY_ORDER, ordered=True)
    summary["mode"] = pd.Categorical(summary["mode"], MODE_ORDER, ordered=True)
    summary["climate"] = pd.Categorical(summary["climate"], CLIMATE_ORDER, ordered=True)
    summary["landcover"] = pd.Categorical(summary["landcover"], LANDCOVER_ORDER, ordered=True)
    return summary.sort_values(["climate", "landcover", "mode"]).reset_index(drop=True)


def build_plot_frame(metric_summary: pd.DataFrame) -> pd.DataFrame:
    full_index = pd.MultiIndex.from_product(
        [CLIMATE_ORDER, LANDCOVER_ORDER, MODE_ORDER],
        names=["climate", "landcover", "mode"],
    )
    completed = (
        metric_summary.set_index(["climate", "landcover", "mode"])[["mean_effect", "se_effect"]]
        .reindex(full_index)
        .reset_index()
    )
    return completed


def format_metric_ylabel(metric_label: str) -> str:
    if metric_label == "Human Footprint effect on Fire frequency":
        return "Human Footprint effect on\nFire frequency"
    if metric_label == "Human Footprint effect on Normalized burned area":
        return "Human Footprint effect on\nNormalized burned area"
    return metric_label


def build_figure(metric_summary: pd.DataFrame, *, metric_label: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(24, 8.2), facecolor="white")
    plot_df = build_plot_frame(metric_summary)

    category_positions = np.arange(len(CLIMATE_ORDER) * len(LANDCOVER_ORDER), dtype=float) * CATEGORY_STEP
    category_pairs = [(climate, landcover) for climate in CLIMATE_ORDER for landcover in LANDCOVER_ORDER]
    base_df = pd.DataFrame(
        [
            (climate, landcover, pos)
            for pos, (climate, landcover) in zip(category_positions, category_pairs)
        ],
        columns=["climate", "landcover", "position"],
    )
    plot_df = plot_df.merge(base_df, on=["climate", "landcover"], how="left", validate="many_to_one")

    bar_offset = 0.15
    bar_width = 0.28
    plot_df["x"] = np.where(plot_df["mode"].astype(str) == "face", plot_df["position"] - bar_offset, plot_df["position"] + bar_offset)
    plot_df["fill_color"] = np.where(plot_df["mean_effect"] >= 0, POSITIVE_COLOR, NEGATIVE_COLOR)
    plot_df["hatch"] = np.where(plot_df["mode"].astype(str) == "face", "", "///")

    ax.set_facecolor(AX_FACE)
    ax.axhline(0.0, color=ZERO_LINE_COLOR, linewidth=1.0, linestyle="--", zorder=1)
    ax.set_axisbelow(True)

    for _, row in plot_df.iterrows():
        if not np.isfinite(row["mean_effect"]):
            continue
        edgecolor = "#000000" if row["mode"] == "mix" else "none"
        ax.bar(
            row["x"],
            row["mean_effect"],
            width=bar_width,
            color=row["fill_color"],
            edgecolor=edgecolor,
            linewidth=0.0,
            alpha=0.82,
            hatch=str(row["hatch"]),
            zorder=3,
        )

    climate_boundary_positions = [
        (category_positions[idx] + category_positions[idx + 1]) / 2.0
        for idx in [4, 9, 14]
    ]
    for separator_x in climate_boundary_positions:
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
    ax.set_xlim(category_positions[0] - 0.55, category_positions[-1] + 0.55)
    ax.set_ylabel(format_metric_ylabel(metric_label), fontsize=23)

    values = plot_df["mean_effect"].to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    y_min = float(np.nanmin(finite_values)) if finite_values.size else -1.0
    y_max = float(np.nanmax(finite_values)) if finite_values.size else 1.0
    if y_min == y_max:
        y_min -= 0.1
        y_max += 0.1
    pad = (y_max - y_min) * 0.08
    ax.set_ylim(y_min - pad, y_max + pad)
    y_lower, y_upper = ax.get_ylim()
    y_span = y_upper - y_lower
    na_label_y = 0.0 + y_span * 0.015
    na_label_y = min(max(na_label_y, y_lower + y_span * 0.03), y_upper - y_span * 0.03)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    exponent = choose_exponent(np.array([*values], dtype=float))
    scale_factor = 10.0 ** exponent
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

    category_na = (
        plot_df.groupby(["climate", "landcover"], as_index=False, observed=True)["mean_effect"]
        .apply(lambda s: s.isna().sum())
        .rename(columns={"mean_effect": "n_missing"})
        .merge(base_df, on=["climate", "landcover"], how="left", validate="one_to_one")
    )
    for _, row in category_na.iterrows():
        if int(row["n_missing"]) == 2:
            ax.text(
                float(row["position"]),
                na_label_y,
                "NA",
                ha="center",
                va="bottom",
                fontsize=16,
                color="#000000",
                fontweight="bold",
            )
    partial_na = plot_df.loc[plot_df["mean_effect"].isna()].copy()
    for _, row in partial_na.iterrows():
        same_group = category_na.loc[
            (category_na["climate"] == row["climate"]) & (category_na["landcover"] == row["landcover"]),
            "n_missing",
        ]
        if not same_group.empty and int(same_group.iloc[0]) == 1:
            ax.text(
                float(row["x"]),
                na_label_y,
                "NA",
                ha="center",
                va="bottom",
                fontsize=16,
                color="#000000",
                fontweight="bold",
            )

    climate_centers = {
        climate: float(np.mean(category_positions[idx * len(LANDCOVER_ORDER):(idx + 1) * len(LANDCOVER_ORDER)]))
        for idx, climate in enumerate(CLIMATE_ORDER)
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
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        ncol=2,
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

    fire_num_summary = summarize_categories(fire_num)
    fire_area_summary = summarize_categories(fire_area)

    fig = build_figure(
        fire_num_summary,
        metric_label="Human Footprint effect on Fire frequency",
    )
    output_png = save_with_fallback_figure(fig, os.path.join(OUTPUT_DIR, OUTPUT_PNG_FIRE_NUM))
    plt.close(fig)
    print(f"Figure saved to: {output_png}")

    fig = build_figure(
        fire_area_summary,
        metric_label="Human Footprint effect on Normalized burned area",
    )
    output_png = save_with_fallback_figure(fig, os.path.join(OUTPUT_DIR, OUTPUT_PNG_FIRE_AREA))
    plt.close(fig)
    print(f"Figure saved to: {output_png}")


if __name__ == "__main__":
    main()
