#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称: HFI 对 Fire frequency 与 Normalized burned area 影响的分类排序图

功能简介:
    1. 从以下目录分别读取所有 HFI 净效应固定效应结果 CSV:
       - B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num
       - B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area
    2. 以每个 CSV 代表一个分类，统计其中 HFI 对响应变量的净影响:
       - 先按 Year 计算 diff_contribution 的年均值
       - 再对 2003-2022 年的年均值求跨年平均和标准误
       - 同时统计 delta_sig == TRUE 的格网比例
    3. 绘制一张双面排序图:
       - 左侧: HFI 对 Fire frequency 的影响排序
       - 右侧: HFI 对 Normalized burned area 的影响排序
       - 点大小表示显著格网比例
       - 颜色表示影响方向，负值为蓝色，正值为暖色

输出:
    - B:\WUI\Picture\Ranking_Human Footprint_effect_on_Fire_frequency_and_Normalized_burn_area.png
    - B:\WUI\Picture\Ranking_Human Footprint_effect_summary.csv
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import math


# =========================
# 参数配置
# =========================
FIRE_NUM_DIR = r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num"
FIRE_AREA_DIR = r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area"

OUTPUT_DIR = r"B:\WUI\Picture"
OUTPUT_PNG = "Ranking_Human Footprint_effect_on_Fire_frequency_and_Normalized_burn_area.png"
OUTPUT_CSV = "Ranking_Human Footprint_effect_summary.csv"

YEAR_START = 2003
YEAR_END = 2022


# =========================
# 绘图样式
# =========================
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

FIGURE_FACE = "#ffffff"
PANEL_FACE = "#fffdf9"
STRIPE_FACE = "#f7f2ea"
GRID_COLOR = "#d9cfbf"
ZERO_LINE_COLOR = "#7f7f7f"
POS_COLOR = "#d94801"
NEG_COLOR = "#2b8cbe"
NEAR_ZERO_COLOR = "#c7b299"


def extract_category_id(path: str) -> str:
    name = Path(path).stem
    match = re.search(r"HFI_effect_((?:face|mix)_[^_]+_[^_]+)", name, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    trimmed = re.sub(r"^HFI_effect_", "", name, flags=re.IGNORECASE)
    parts = trimmed.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else trimmed


def coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    values = series.astype(str).str.strip().str.upper()
    return values.isin({"TRUE", "T", "1", "YES", "Y"})


def format_effect(x: float, _: int | None = None) -> str:
    if abs(x) < 1e-12:
        return "0"
    if abs(x) < 0.1:
        return f"{x:.1e}"
    return f"{x:.2f}"


def color_for_value(value: float) -> str:
    if value < 0:
        return NEG_COLOR
    if value > 0:
        return POS_COLOR
    return NEAR_ZERO_COLOR


def save_with_fallback_csv(df: pd.DataFrame, path: str) -> str:
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        stem, ext = os.path.splitext(path)
        alt_path = f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        df.to_csv(alt_path, index=False, encoding="utf-8-sig")
        print(f"Target CSV is locked, saved fallback file: {alt_path}")
        return alt_path


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


def summarize_metric(
    input_dir: str,
    *,
    response_name: str,
) -> pd.DataFrame:
    csv_files = sorted(glob.glob(os.path.join(input_dir, "HFI_effect_*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    rows: list[dict[str, object]] = []
    print(f"Reading {len(csv_files)} file(s) for {response_name}: {input_dir}")

    for csv_path in csv_files:
        header = pd.read_csv(csv_path, nrows=0)
        columns = set(header.columns.tolist())
        required = {"Year", "diff_contribution", "delta_sig"}
        missing = sorted(required.difference(columns))
        if missing:
            raise ValueError(f"Missing columns {missing} in {csv_path}")

        df = pd.read_csv(
            csv_path,
            usecols=["Year", "diff_contribution", "delta_sig"],
            low_memory=False,
        )
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        df["diff_contribution"] = pd.to_numeric(df["diff_contribution"], errors="coerce")
        df["delta_sig"] = coerce_bool_series(df["delta_sig"])
        df = df.dropna(subset=["Year", "diff_contribution"])
        df = df[df["Year"].between(YEAR_START, YEAR_END)]
        if df.empty:
            continue

        annual_mean = df.groupby("Year", as_index=False)["diff_contribution"].mean()
        annual_mean = annual_mean.sort_values("Year").reset_index(drop=True)

        n_years = int(len(annual_mean))
        mean_effect = float(annual_mean["diff_contribution"].mean())
        if n_years <= 1:
            se_effect = 0.0
        else:
            se_effect = float(annual_mean["diff_contribution"].std(ddof=1) / math.sqrt(n_years))

        sig_share = float(df["delta_sig"].mean())
        rows.append(
            {
                "category_id": extract_category_id(csv_path),
                "response": response_name,
                "mean_effect": mean_effect,
                "se_effect": se_effect,
                "significant_share": sig_share,
                "n_years": n_years,
                "n_rows": int(len(df)),
                "source_csv": os.path.basename(csv_path),
            }
        )

    if not rows:
        raise ValueError(f"No valid summary rows built from: {input_dir}")

    summary = pd.DataFrame(rows).sort_values("mean_effect", ascending=False).reset_index(drop=True)
    print(
        f"Built summary for {response_name}: "
        f"categories={len(summary)}, mean range=[{summary['mean_effect'].min():.3e}, {summary['mean_effect'].max():.3e}]"
    )
    return summary


def plot_rank_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    x_label: str,
    max_abs_effect: float,
) -> None:
    data = summary.sort_values("mean_effect", ascending=False).reset_index(drop=True)
    y = np.arange(len(data))
    values = data["mean_effect"].to_numpy(dtype=float)
    errors = data["se_effect"].to_numpy(dtype=float)
    if max_abs_effect <= 0:
        sizes = np.full(len(data), 85.0)
    else:
        sizes = 55 + 260 * (np.abs(values) / max_abs_effect)
    colors = [color_for_value(v) for v in values]

    ax.set_facecolor(PANEL_FACE)

    for yi in y:
        if yi % 2 == 0:
            ax.axhspan(yi - 0.5, yi + 0.5, color=STRIPE_FACE, zorder=0)

    ax.axvline(0, color=ZERO_LINE_COLOR, linewidth=1.0, linestyle="--", zorder=1)

    for yi, value, color in zip(y, values, colors):
        ax.hlines(
            yi,
            xmin=min(0.0, value),
            xmax=max(0.0, value),
            color=color,
            linewidth=2.3,
            alpha=0.95,
            zorder=2,
        )

    ax.errorbar(
        values,
        y,
        xerr=errors,
        fmt="none",
        ecolor="#4d4d4d",
        elinewidth=1.05,
        capsize=2.5,
        alpha=0.95,
        zorder=3,
    )
    ax.scatter(
        values,
        y,
        s=sizes,
        c=colors,
        edgecolors="white",
        linewidths=1.1,
        alpha=0.98,
        zorder=4,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(data["category_id"])
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.75)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(FuncFormatter(format_effect))

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#b9ae9d")

    ax.set_xlabel(x_label)


def build_figure(fire_num_summary: pd.DataFrame, fire_area_summary: pd.DataFrame) -> plt.Figure:
    max_abs_effect = float(
        max(
            fire_num_summary["mean_effect"].abs().max(),
            fire_area_summary["mean_effect"].abs().max(),
        )
    )
    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(19, 14),
        facecolor=FIGURE_FACE,
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.28},
    )

    plot_rank_panel(
        ax_left,
        fire_num_summary,
        x_label="Mean annual diff_contribution",
        max_abs_effect=max_abs_effect,
    )
    plot_rank_panel(
        ax_right,
        fire_area_summary,
        x_label="Mean annual diff_contribution",
        max_abs_effect=max_abs_effect,
    )

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=NEG_COLOR, markeredgecolor="white", markersize=9, label="Negative mean effect"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=POS_COLOR, markeredgecolor="white", markersize=9, label="Positive mean effect"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
        fontsize=10.5,
    )

    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.05, top=0.94)
    return fig


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fire_num_summary = summarize_metric(FIRE_NUM_DIR, response_name="Fire frequency")
    fire_area_summary = summarize_metric(FIRE_AREA_DIR, response_name="Normalized burned area")

    summary_all = pd.concat([fire_num_summary, fire_area_summary], ignore_index=True)
    summary_csv_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV)
    summary_csv_path = save_with_fallback_csv(summary_all, summary_csv_path)
    print(f"Summary CSV saved to: {summary_csv_path}")

    fig = build_figure(fire_num_summary, fire_area_summary)
    output_png_path = os.path.join(OUTPUT_DIR, OUTPUT_PNG)
    output_png_path = save_with_fallback_figure(fig, output_png_path)
    plt.close(fig)
    print(f"Figure saved to: {output_png_path}")


if __name__ == "__main__":
    main()


