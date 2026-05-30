# -*- coding: utf-8 -*-
r"""
程序名称：按照 landcover 及气候带分类统计 Fire frequency 与 Normalized burned area

功能：
    1) 读取指定目录下所有 CSV
    2) 直接读取 Fire_frequency 与 Normalized_fire_area 列，不再自行计算
    3) 按文件名模式（face/mix + A/B/C/D）汇总统计
    4) 计算每组均值与标准误（SE）
    5) 使用分组柱状图 + 误差棒绘制
    6) 分别按气候带（A/B/C/D）和 landcover 汇总

输出：
    - B:\WUI\Picture\Landcover_Fire_Frequency_Grouped_Panel_1x2.png
    - B:\WUI\Picture\Landcover_Normalized_Burned_Area_Grouped_Panel_1x2.png
"""

from __future__ import annotations

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


# 设置字体
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False


# 输入与输出路径
FIRE_NUM_DIR = r"I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic"
FIRE_AREA_DIR = r"I:\Processing data\渔网分类建模\Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic"
OUTPUT_DIR = r"B:\WUI\Picture"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 分组配置
GROUPS = ["A", "B", "C", "D"]
GROUP_LABELS = ["A", "B", "C", "D"]
LANDCOVERS = ["CRP", "FST", "GRS", "SHR", "WET"]
LANDCOVER_LABELS = ["CRP", "FST", "GRS", "SHR", "WET"]
MODES = ["face", "mix"]

FACE_COLOR = "#72B7E2"
MIX_COLOR = "#FDB366"
PLOT_FONT_SIZE = 36 * 1.4
SUBPLOT_TITLE_FONT_SIZE = PLOT_FONT_SIZE
FIGURE_WIDTH = 20.5
FIGURE_HEIGHT = 9


METRICS = [
    {
        "name": "Fire frequency",
        "input_dir": FIRE_NUM_DIR,
        "metric_col": "Fire_frequency",
        "filename_pattern": re.compile(r"^(face|mix)_([ABCD])_([A-Z]+)_fire_num_with_WUI_offset$"),
        "output_panel": "Landcover_Fire_Frequency_Grouped_Panel_1x2.png",
        "ylabel": "Fire frequency",
    },
    {
        "name": "Normalized burned area",
        "input_dir": FIRE_AREA_DIR,
        "metric_col": "Normalized_fire_area",
        "filename_pattern": re.compile(r"^(face|mix)_([ABCD])_([A-Z]+)_fire_area$"),
        "output_panel": "Landcover_Normalized_Burned_Area_Grouped_Panel_1x2.png",
        "ylabel": "Normalized burned area",
    },
]


def compute_mean_se(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return np.nan, np.nan
    mean = float(values.mean())
    if values.size == 1:
        return mean, 0.0
    se = float(values.std(ddof=1) / np.sqrt(values.size))
    return mean, se


def summarize_metric(
    *,
    input_dir: str,
    metric_col: str,
    filename_pattern: re.Pattern[str],
) -> tuple[dict[tuple[str, str], tuple[float, float]], dict[tuple[str, str], tuple[float, float]]]:
    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    values_by_group: dict[tuple[str, str], list[np.ndarray]] = {
        (mode, group): [] for mode in MODES for group in GROUPS
    }
    values_by_lc: dict[tuple[str, str], list[np.ndarray]] = {
        (mode, lc): [] for mode in MODES for lc in LANDCOVERS
    }

    matched_files = 0
    for csv_path in csv_files:
        name = os.path.splitext(os.path.basename(csv_path))[0]
        match = filename_pattern.match(name)
        if not match:
            continue

        mode, group, lc = match.groups()
        df = pd.read_csv(csv_path, usecols=[metric_col])
        values = pd.to_numeric(df[metric_col], errors="coerce").to_numpy()
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue

        matched_files += 1
        values_by_group[(mode, group)].append(values)
        if lc in LANDCOVERS:
            values_by_lc[(mode, lc)].append(values)

    if matched_files == 0:
        raise ValueError(f"No matched CSV files found in {input_dir}. Check file name pattern.")

    summary_group: dict[tuple[str, str], tuple[float, float]] = {}
    for mode in MODES:
        for group in GROUPS:
            arrays = values_by_group[(mode, group)]
            merged = np.concatenate(arrays) if arrays else np.array([], dtype=float)
            summary_group[(mode, group)] = compute_mean_se(merged)

    summary_lc: dict[tuple[str, str], tuple[float, float]] = {}
    for mode in MODES:
        for lc in LANDCOVERS:
            arrays = values_by_lc[(mode, lc)]
            merged = np.concatenate(arrays) if arrays else np.array([], dtype=float)
            summary_lc[(mode, lc)] = compute_mean_se(merged)

    return summary_group, summary_lc


def extract_mode_series(
    summary: dict[tuple[str, str], tuple[float, float]],
    categories: list[str],
    *,
    mode: str,
) -> tuple[list[float], list[float]]:
    means: list[float] = []
    ses: list[float] = []
    for category in categories:
        mean, se = summary[(mode, category)]
        means.append(mean)
        ses.append(se)
    return means, ses


def _get_summary_limits(
    summaries: list[dict[tuple[str, str], tuple[float, float]]],
    categories_list: list[list[str]],
) -> tuple[float, float]:
    lower_candidates: list[float] = []
    upper_candidates: list[float] = []
    for summary, categories in zip(summaries, categories_list):
        for mode in MODES:
            means, ses = extract_mode_series(summary, categories, mode=mode)
            for mean, se in zip(means, ses):
                if np.isfinite(mean):
                    err = se if np.isfinite(se) else 0.0
                    lower_candidates.append(float(mean - err))
                    upper_candidates.append(float(mean + err))

    if not lower_candidates or not upper_candidates:
        return -1.0, 1.0

    y_min = min(lower_candidates)
    y_max = max(upper_candidates)
    if y_min == y_max:
        margin = max(abs(y_min) * 0.12, 1.0)
    else:
        margin = (y_max - y_min) * 0.12
    return y_min - margin, y_max + margin


def _draw_grouped_bars(
    ax: plt.Axes,
    *,
    categories: list[str],
    category_labels: list[str],
    summary: dict[tuple[str, str], tuple[float, float]],
    metric_name: str,
    show_ylabel: bool,
    ylabel: str,
    show_legend: bool,
    panel_title: str,
) -> None:
    means_face, ses_face = extract_mode_series(summary, categories, mode="face")
    means_mix, ses_mix = extract_mode_series(summary, categories, mode="mix")

    bar_positions = np.arange(len(categories))
    bar_width = 0.34
    side_margin = 0.55

    ax.bar(
        bar_positions - bar_width / 2,
        means_face,
        yerr=ses_face,
        capsize=4,
        width=bar_width,
        color=FACE_COLOR,
        edgecolor="#333333",
        linewidth=1.0,
        alpha=0.85,
        label="Interface",
    )
    ax.bar(
        bar_positions + bar_width / 2,
        means_mix,
        yerr=ses_mix,
        capsize=4,
        width=bar_width,
        color=MIX_COLOR,
        edgecolor="#333333",
        linewidth=1.0,
        alpha=0.85,
        label="Intermix",
    )

    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=PLOT_FONT_SIZE)
    else:
        ax.set_ylabel("")
    ax.set_title(panel_title, fontsize=SUBPLOT_TITLE_FONT_SIZE, pad=14)
    ax.set_xlabel("")
    ax.set_xlim(-side_margin, len(categories) - 1 + side_margin)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(category_labels, fontsize=PLOT_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=PLOT_FONT_SIZE)
    if metric_name == "Fire frequency":
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_legend:
        ax.legend(
            loc="upper right",
            ncol=1,
            fontsize=PLOT_FONT_SIZE,
            frameon=False,
            handlelength=1.4,
            labelspacing=0.35,
            borderaxespad=0.3,
        )


def plot_grouped_panel(
    *,
    summary_group: dict[tuple[str, str], tuple[float, float]],
    summary_lc: dict[tuple[str, str], tuple[float, float]],
    metric_name: str,
    ylabel: str,
    output_path: str,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
        sharey=True,
        gridspec_kw={"width_ratios": [len(GROUPS), len(LANDCOVERS)]},
    )

    y_min, y_max = _get_summary_limits(
        [summary_group, summary_lc],
        [GROUPS, LANDCOVERS],
    )

    _draw_grouped_bars(
        axes[0],
        categories=GROUPS,
        category_labels=GROUP_LABELS,
        summary=summary_group,
        metric_name=metric_name,
        show_ylabel=True,
        ylabel=ylabel,
        show_legend=False,
        panel_title="Climate zone",
    )
    _draw_grouped_bars(
        axes[1],
        categories=LANDCOVERS,
        category_labels=LANDCOVER_LABELS,
        summary=summary_lc,
        metric_name=metric_name,
        show_ylabel=False,
        ylabel=ylabel,
        show_legend=True,
        panel_title="Land cover",
    )

    for ax in axes:
        ax.set_ylim(y_min, y_max)

    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.20, top=0.88, wspace=0.10)
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {output_path}")


def main() -> None:
    for metric in METRICS:
        print(f"Processing: {metric['name']}")
        summary_group, summary_lc = summarize_metric(
            input_dir=metric["input_dir"],
            metric_col=metric["metric_col"],
            filename_pattern=metric["filename_pattern"],
        )

        plot_grouped_panel(
            summary_group=summary_group,
            summary_lc=summary_lc,
            metric_name=metric["name"],
            ylabel=metric["ylabel"],
            output_path=os.path.join(OUTPUT_DIR, metric["output_panel"]),
        )


if __name__ == "__main__":
    main()
