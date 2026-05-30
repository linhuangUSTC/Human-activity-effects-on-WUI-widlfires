# -*- coding: utf-8 -*-
r"""
程序名称：按照 landcover 及气候带分类统计 Fire frequency 与 Normalized burned area（柱状图+显著性）

功能：
    1) 读取指定目录下所有 CSV
    2) 直接读取 Fire_frequency 与 Normalized_fire_area 列，不再自行计算
    3) 按文件名模式（face/mix + A/B/C/D）汇总统计
    4) 保留每个类别下的原始样本分布
    5) 使用均值 + 标准误（SE）分组柱状图绘制 Interface 与 Intermix 的对比
    6) 计算组间差异，并用括号与显著性星号标注
    7) 每个指标输出一张 1×2 组图：左侧为 Climate zone，右侧为 Land cover

输出：
    - B:\WUI\Picture\Landcover_Fire_Frequency_Grouped_Bar_Significance_Panel_1x2.png
    - B:\WUI\Picture\Landcover_Normalized_Burned_Area_Grouped_Bar_Significance_Panel_1x2.png
    - B:\WUI\Picture\Landcover_Fire_Data_Grouped_Bar_Mean_SD_Table.xlsx
"""

from __future__ import annotations

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.stats import ttest_ind


plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False


FIRE_NUM_DIR = r"I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic"
FIRE_AREA_DIR = r"I:\Processing data\渔网分类建模\Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic"
OUTPUT_DIR = r"B:\WUI\Picture"
os.makedirs(OUTPUT_DIR, exist_ok=True)


GROUPS = ["A", "B", "C", "D"]
GROUP_LABELS = ["A", "B", "C", "D"]
LANDCOVERS = ["CRP", "FST", "GRS", "SHR", "WET"]
LANDCOVER_LABELS = ["CRP", "FST", "GRS", "SHR", "WET"]
MODES = ["face", "mix"]

FACE_COLOR = "#72B7E2"
MIX_COLOR = "#FDB366"
EDGE_COLOR = "#4A4A4A"
PLOT_FONT_SIZE = 30
SUBPLOT_TITLE_FONT_SIZE = PLOT_FONT_SIZE
FIGURE_WIDTH = 20.5
FIGURE_HEIGHT = 9
OUTPUT_MEAN_SD_TABLE = "Landcover_Fire_Data_Grouped_Bar_Mean_SD_Table.xlsx"


METRICS = [
    {
        "name": "Fire frequency",
        "input_dir": FIRE_NUM_DIR,
        "metric_col": "Fire_frequency",
        "filename_pattern": re.compile(r"^(face|mix)_([ABCD])_([A-Z]+)_fire_num_with_WUI_offset$"),
        "output_panel": "Landcover_Fire_Frequency_Grouped_Bar_Significance_Panel_1x2.png",
        "ylabel": "Fire frequency",
    },
    {
        "name": "Normalized burned area",
        "input_dir": FIRE_AREA_DIR,
        "metric_col": "Normalized_fire_area",
        "filename_pattern": re.compile(r"^(face|mix)_([ABCD])_([A-Z]+)_fire_area$"),
        "output_panel": "Landcover_Normalized_Burned_Area_Grouped_Bar_Significance_Panel_1x2.png",
        "ylabel": "Normalized burned area",
    },
]


def compute_mean_se(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    mean = float(values.mean())
    if values.size == 1:
        return mean, 0.0
    se = float(values.std(ddof=1) / np.sqrt(values.size))
    return mean, se


def compute_mean_std(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    mean = float(values.mean())
    if values.size == 1:
        return mean, 0.0
    std = float(values.std(ddof=1))
    return mean, std


def collect_metric_values(
    *,
    input_dir: str,
    metric_col: str,
    filename_pattern: re.Pattern[str],
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], np.ndarray]]:
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
        values = pd.to_numeric(df[metric_col], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue

        matched_files += 1
        values_by_group[(mode, group)].append(values)
        if lc in LANDCOVERS:
            values_by_lc[(mode, lc)].append(values)

    if matched_files == 0:
        raise ValueError(f"No matched CSV files found in {input_dir}. Check file name pattern.")

    merged_group: dict[tuple[str, str], np.ndarray] = {}
    for mode in MODES:
        for group in GROUPS:
            arrays = values_by_group[(mode, group)]
            merged_group[(mode, group)] = np.concatenate(arrays) if arrays else np.array([np.nan], dtype=float)

    merged_lc: dict[tuple[str, str], np.ndarray] = {}
    for mode in MODES:
        for lc in LANDCOVERS:
            arrays = values_by_lc[(mode, lc)]
            merged_lc[(mode, lc)] = np.concatenate(arrays) if arrays else np.array([np.nan], dtype=float)

    return merged_group, merged_lc


def _get_value_limits(
    distributions: list[dict[tuple[str, str], np.ndarray]],
    categories_list: list[list[str]],
) -> tuple[float, float]:
    lower_candidates: list[float] = []
    upper_candidates: list[float] = []
    for distribution, categories in zip(distributions, categories_list):
        for mode in MODES:
            for category in categories:
                values = distribution[(mode, category)]
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                mean, se = compute_mean_se(values)
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
        margin = (y_max - y_min) * 0.22
    return y_min - margin, y_max + margin


def _significance_marker(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _compute_p_value(values_a: np.ndarray, values_b: np.ndarray) -> float:
    valid_a = values_a[np.isfinite(values_a)]
    valid_b = values_b[np.isfinite(values_b)]
    if valid_a.size < 2 or valid_b.size < 2:
        return np.nan
    if np.allclose(valid_a, valid_a[0]) and np.allclose(valid_b, valid_b[0]) and np.isclose(valid_a[0], valid_b[0]):
        return np.nan
    result = ttest_ind(valid_a, valid_b, equal_var=False, nan_policy="omit")
    return float(result.pvalue) if np.isfinite(result.pvalue) else np.nan


def _draw_significance_bracket(
    ax: plt.Axes,
    *,
    x_left: float,
    x_right: float,
    y_left_anchor: float,
    y_right_anchor: float,
    y_line: float,
    text: str,
    fontsize: float,
    direction: int,
    text_gap: float,
) -> None:
    ax.plot(
        [x_left, x_left, x_right, x_right],
        [y_left_anchor, y_line, y_line, y_right_anchor],
        color="#9A9A9A",
        linewidth=1.3,
        solid_capstyle="round",
        zorder=6,
    )
    ax.text(
        (x_left + x_right) / 2.0,
        y_line + direction * text_gap,
        text,
        ha="center",
        va="bottom" if direction > 0 else "top",
        fontsize=fontsize,
        color="#333333",
        zorder=7,
    )


def _bar_endpoint_for_bracket(mean: float, se: float) -> float:
    err = se if np.isfinite(se) else 0.0
    return mean + err if mean >= 0 else mean - err


def _bracket_anchor_for_bar(
    mean: float,
    se: float,
    *,
    direction: int,
    anchor_gap: float,
    mixed_sign: bool,
) -> float:
    if mixed_sign and mean < 0:
        return direction * anchor_gap
    return _bar_endpoint_for_bracket(mean, se) + direction * anchor_gap


def _draw_grouped_bars(
    ax: plt.Axes,
    *,
    categories: list[str],
    category_labels: list[str],
    distribution: dict[tuple[str, str], np.ndarray],
    metric_name: str,
    show_ylabel: bool,
    ylabel: str,
    show_legend: bool,
    panel_title: str,
    y_limits: tuple[float, float],
) -> None:
    positions = np.arange(len(categories), dtype=float)
    offset = 0.18
    bar_width = 0.30
    side_margin = 0.60
    face_means: list[float] = []
    face_ses: list[float] = []
    mix_means: list[float] = []
    mix_ses: list[float] = []
    significance_markers: list[str] = []

    for category in categories:
        face_values = distribution[("face", category)]
        mix_values = distribution[("mix", category)]
        face_mean, face_se = compute_mean_se(face_values)
        mix_mean, mix_se = compute_mean_se(mix_values)
        face_means.append(face_mean)
        face_ses.append(face_se if np.isfinite(face_se) else 0.0)
        mix_means.append(mix_mean)
        mix_ses.append(mix_se if np.isfinite(mix_se) else 0.0)
        significance_markers.append(_significance_marker(_compute_p_value(face_values, mix_values)))

    ax.bar(
        positions - offset,
        face_means,
        yerr=face_ses,
        capsize=4,
        width=bar_width,
        color=FACE_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=1.2,
        alpha=0.85,
        label="Interface",
        error_kw={"ecolor": EDGE_COLOR, "elinewidth": 1.1, "capsize": 4},
        zorder=4,
    )
    ax.bar(
        positions + offset,
        mix_means,
        yerr=mix_ses,
        capsize=4,
        width=bar_width,
        color=MIX_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=1.2,
        alpha=0.85,
        label="Intermix",
        error_kw={"ecolor": EDGE_COLOR, "elinewidth": 1.1, "capsize": 4},
        zorder=4,
    )

    ax.set_ylim(*y_limits)
    y_span = y_limits[1] - y_limits[0]
    if not np.isfinite(y_span) or y_span <= 0:
        y_span = 1.0
    bracket_gap = y_span * 0.028
    anchor_gap = y_span * 0.010
    text_gap = y_span * 0.010
    text_fontsize = max(PLOT_FONT_SIZE * 0.72, 16)

    for idx, marker in enumerate(significance_markers):
        if not marker:
            continue
        face_mean = face_means[idx]
        mix_mean = mix_means[idx]
        if not np.isfinite(face_mean) or not np.isfinite(mix_mean):
            continue
        mixed_sign = (face_mean < 0 < mix_mean) or (mix_mean < 0 < face_mean)
        if face_mean <= 0 and mix_mean <= 0:
            direction = -1
        else:
            direction = 1

        face_anchor = _bracket_anchor_for_bar(
            face_mean,
            face_ses[idx],
            direction=direction,
            anchor_gap=anchor_gap,
            mixed_sign=mixed_sign,
        )
        mix_anchor = _bracket_anchor_for_bar(
            mix_mean,
            mix_ses[idx],
            direction=direction,
            anchor_gap=anchor_gap,
            mixed_sign=mixed_sign,
        )
        if direction < 0:
            y_line = min(face_anchor, mix_anchor) - bracket_gap
        else:
            y_line = max(face_anchor, mix_anchor) + bracket_gap

        _draw_significance_bracket(
            ax,
            x_left=positions[idx] - offset,
            x_right=positions[idx] + offset,
            y_left_anchor=face_anchor,
            y_right_anchor=mix_anchor,
            y_line=y_line,
            text=marker,
            fontsize=text_fontsize,
            direction=direction,
            text_gap=text_gap,
        )

    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=PLOT_FONT_SIZE)
        ax.yaxis.set_label_coords(-0.14, 0.50)
    else:
        ax.set_ylabel("")
    ax.set_title("")
    ax.set_xlabel(panel_title, fontsize=SUBPLOT_TITLE_FONT_SIZE, labelpad=16)
    ax.set_xlim(-side_margin, len(categories) - 1 + side_margin)
    ax.set_xticks(positions)
    ax.set_xticklabels(category_labels, fontsize=PLOT_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=PLOT_FONT_SIZE)
    if metric_name == "Fire frequency":
        ax.set_yticks([0.00, 0.01, 0.02, 0.03])
        ax.set_yticklabels(["0.00", "0.01", "0.02", "0.03"], fontsize=PLOT_FONT_SIZE)
    ax.grid(True, axis="y", linestyle="--", alpha=0.32, linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_legend:
        ax.legend(
            loc="upper right",
            ncol=1,
            fontsize=max(PLOT_FONT_SIZE * 0.74, 16),
            frameon=False,
            handlelength=1.4,
            labelspacing=0.35,
            borderaxespad=0.35,
        )


def plot_grouped_panel(
    *,
    group_distribution: dict[tuple[str, str], np.ndarray],
    lc_distribution: dict[tuple[str, str], np.ndarray],
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

    y_min, y_max = _get_value_limits(
        [group_distribution, lc_distribution],
        [GROUPS, LANDCOVERS],
    )

    _draw_grouped_bars(
        axes[0],
        categories=GROUPS,
        category_labels=GROUP_LABELS,
        distribution=group_distribution,
        metric_name=metric_name,
        show_ylabel=True,
        ylabel=ylabel,
        show_legend=False,
        panel_title="Climate zone",
        y_limits=(y_min, y_max),
    )
    _draw_grouped_bars(
        axes[1],
        categories=LANDCOVERS,
        category_labels=LANDCOVER_LABELS,
        distribution=lc_distribution,
        metric_name=metric_name,
        show_ylabel=False,
        ylabel=ylabel,
        show_legend=True,
        panel_title="Land cover",
        y_limits=(y_min, y_max),
    )

    for ax in axes:
        ax.set_ylim(y_min, y_max)

    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.20, top=0.88, wspace=0.10)
    first_ax_position = axes[0].get_position()
    axes[0].set_position([
        first_ax_position.x0 + 0.025,
        first_ax_position.y0,
        first_ax_position.width,
        first_ax_position.height,
    ])
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {output_path}")


def build_mean_sd_rows(
    *,
    metric_name: str,
    grouping_type: str,
    categories: list[str],
    category_labels: list[str],
    distribution: dict[tuple[str, str], np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    mode_labels = {"face": "Interface", "mix": "Intermix"}
    for category, category_label in zip(categories, category_labels):
        combined_values = np.concatenate([distribution[(mode, category)] for mode in MODES])
        combined_mean, combined_std = compute_mean_std(combined_values)
        rows.append(
            {
                "metric": metric_name,
                "grouping_type": grouping_type,
                "category_code": category,
                "category_label": category_label,
                "wui_type": "All",
                "mean": combined_mean,
                "std": combined_std,
            }
        )
        for mode in MODES:
            mean, std = compute_mean_std(distribution[(mode, category)])
            rows.append(
                {
                    "metric": metric_name,
                    "grouping_type": grouping_type,
                    "category_code": category,
                    "category_label": category_label,
                    "wui_type": mode_labels[mode],
                    "mean": mean,
                    "std": std,
                }
            )
    return rows


def write_mean_sd_table(rows: list[dict[str, object]], output_path: str) -> None:
    pd.DataFrame(rows).to_excel(output_path, sheet_name="mean_sd", index=False)
    print(f"Mean/SD table saved to: {output_path}")


def main() -> None:
    mean_sd_rows: list[dict[str, object]] = []
    for metric in METRICS:
        print(f"Processing: {metric['name']}")
        group_distribution, lc_distribution = collect_metric_values(
            input_dir=metric["input_dir"],
            metric_col=metric["metric_col"],
            filename_pattern=metric["filename_pattern"],
        )
        mean_sd_rows.extend(
            build_mean_sd_rows(
                metric_name=str(metric["name"]),
                grouping_type="Climate zone",
                categories=GROUPS,
                category_labels=GROUP_LABELS,
                distribution=group_distribution,
            )
        )
        mean_sd_rows.extend(
            build_mean_sd_rows(
                metric_name=str(metric["name"]),
                grouping_type="Land cover",
                categories=LANDCOVERS,
                category_labels=LANDCOVER_LABELS,
                distribution=lc_distribution,
            )
        )

        plot_grouped_panel(
            group_distribution=group_distribution,
            lc_distribution=lc_distribution,
            metric_name=metric["name"],
            ylabel=metric["ylabel"],
            output_path=os.path.join(OUTPUT_DIR, metric["output_panel"]),
        )
    write_mean_sd_table(
        mean_sd_rows,
        os.path.join(OUTPUT_DIR, OUTPUT_MEAN_SD_TABLE),
    )


if __name__ == "__main__":
    main()
