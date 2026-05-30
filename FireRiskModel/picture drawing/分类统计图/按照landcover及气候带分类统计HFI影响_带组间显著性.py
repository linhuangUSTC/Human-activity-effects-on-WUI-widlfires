# -*- coding: utf-8 -*-
r"""
程序名称：按照 landcover 及气候带分类统计 HFI 影响（柱状图+显著性）

功能：
    1) 读取 HFI 净效应固定效应结果 CSV
    2) 直接读取 diff_contribution，不再重新计算
    3) 按文件名模式（face/mix + A/B/C/D）汇总 HFI 对
       Fire frequency 和 Normalized burned area 的影响
    4) 保留每个类别下的原始样本分布
    5) 使用均值 + 标准误（SE）分组柱状图绘制 Interface 与 Intermix 的对比
    6) 计算组间差异，并用括号与显著性星号标注
    7) 输出一张 2×2 组合图：
       - 上排：Fire frequency
       - 下排：Normalized burned area
       - 左列：Climate zone
       - 右列：Land cover

输出：
    - B:\WUI\Picture\Landcover_Human_Activity_Effect_Grouped_Bar_Significance_Panel_2x2.png
    - B:\WUI\Picture\Landcover_Human_Activity_Effect_Grouped_Bar_Mean_SD_Table.xlsx
"""

from __future__ import annotations

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from scipy.stats import ttest_ind


plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False


FIRE_NUM_DIR = r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num"
FIRE_AREA_DIR = r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area"
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
COMBINED_OUTPUT_PANEL = "Landcover_Human_Activity_Effect_Grouped_Bar_Significance_Panel_2x2.png"
OUTPUT_MEAN_SD_TABLE = "Landcover_Human_Activity_Effect_Grouped_Bar_Mean_SD_Table.xlsx"

# Slightly smaller than the previous version to reduce crowding.
PLOT_FONT_SIZE = 34
SUBPLOT_TITLE_FONT_SIZE = 32
FIGURE_WIDTH = 20.5
FIGURE_HEIGHT = 15.5


METRICS = [
    {
        "name": "Human activity effect on Fire frequency",
        "input_dir": FIRE_NUM_DIR,
        "metric_col": "diff_contribution",
        "filename_pattern": re.compile(r"^HFI_effect_(face|mix)_([ABCD])_([A-Z]+)_fire_num_with_WUI_offset$"),
        "ylabel": "Human activity effects on\nfire frequency",
        "legend_loc": "upper right",
    },
    {
        "name": "Human activity effect on Normalized burned area",
        "input_dir": FIRE_AREA_DIR,
        "metric_col": "diff_contribution",
        "filename_pattern": re.compile(r"^HFI_effect_(face|mix)_([ABCD])_([A-Z]+)_fire_area$"),
        "ylabel": "Human activity effects on\nnormalized burned area",
        "legend_loc": "lower right",
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
        margin = (y_max - y_min) * 0.34
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
    if (
        np.allclose(valid_a, valid_a[0])
        and np.allclose(valid_b, valid_b[0])
        and np.isclose(valid_a[0], valid_b[0])
    ):
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
    """
    Anchor significance whiskers to the visible outer end of each bar.
    Positive bars connect from their top; negative bars connect from their bottom.
    """
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
    legend_loc: str,
    panel_title: str,
    panel_label: str,
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
        ax.yaxis.set_label_coords(-0.17, 0.46)
    else:
        ax.set_ylabel("")
    ax.set_title("")
    ax.set_xlabel(panel_title, fontsize=SUBPLOT_TITLE_FONT_SIZE, labelpad=16)
    ax.set_xlim(-side_margin, len(categories) - 1 + side_margin)
    ax.set_xticks(positions)
    ax.set_xticklabels(category_labels, fontsize=PLOT_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=PLOT_FONT_SIZE)
    if metric_name == "Human activity effect on Fire frequency":
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    sci_formatter = ScalarFormatter(useMathText=False)
    sci_formatter.set_scientific(True)
    sci_formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(sci_formatter)
    ax.yaxis.get_offset_text().set_size(PLOT_FONT_SIZE - 2)
    ax.grid(True, axis="y", linestyle="--", alpha=0.32, linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        -0.10,
        1.02,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=PLOT_FONT_SIZE,
        fontweight="bold",
        color="#222222",
    )

    if show_legend:
        ax.legend(
            loc=legend_loc,
            ncol=1,
            fontsize=max(PLOT_FONT_SIZE * 0.74, 16),
            frameon=False,
            handlelength=1.4,
            labelspacing=0.35,
            borderaxespad=0.35,
        )


def plot_grouped_panel(
    *,
    metric_payloads: list[tuple[dict[str, object], dict[tuple[str, str], np.ndarray], dict[tuple[str, str], np.ndarray]]],
    output_path: str,
) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
        sharey="row",
        gridspec_kw={"width_ratios": [len(GROUPS), len(LANDCOVERS)]},
    )

    panel_labels = ["a", "b", "c", "d"]
    panel_index = 0

    for row_idx, (metric, group_distribution, lc_distribution) in enumerate(metric_payloads):
        y_min, y_max = _get_value_limits(
            [group_distribution, lc_distribution],
            [GROUPS, LANDCOVERS],
        )

        _draw_grouped_bars(
            axes[row_idx, 0],
            categories=GROUPS,
            category_labels=GROUP_LABELS,
            distribution=group_distribution,
            metric_name=str(metric["name"]),
            show_ylabel=True,
            ylabel=str(metric["ylabel"]),
            show_legend=(row_idx == 0),
            legend_loc="upper right",
            panel_title="Climate zone",
            panel_label=panel_labels[panel_index],
            y_limits=(y_min, y_max),
        )
        panel_index += 1

        _draw_grouped_bars(
            axes[row_idx, 1],
            categories=LANDCOVERS,
            category_labels=LANDCOVER_LABELS,
            distribution=lc_distribution,
            metric_name=str(metric["name"]),
            show_ylabel=False,
            ylabel=str(metric["ylabel"]),
            show_legend=False,
            legend_loc=str(metric["legend_loc"]),
            panel_title="Land cover",
            panel_label=panel_labels[panel_index],
            y_limits=(y_min, y_max),
        )
        panel_index += 1

        axes[row_idx, 0].set_ylim(y_min, y_max)
        axes[row_idx, 1].set_ylim(y_min, y_max)

    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.12, top=0.96, wspace=0.10, hspace=0.28)
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


def write_mean_sd_table(
    metric_payloads: list[tuple[dict[str, object], dict[tuple[str, str], np.ndarray], dict[tuple[str, str], np.ndarray]]],
    output_path: str,
) -> None:
    rows: list[dict[str, object]] = []
    for metric, group_distribution, lc_distribution in metric_payloads:
        rows.extend(
            build_mean_sd_rows(
                metric_name=str(metric["name"]),
                grouping_type="Climate zone",
                categories=GROUPS,
                category_labels=GROUP_LABELS,
                distribution=group_distribution,
            )
        )
        rows.extend(
            build_mean_sd_rows(
                metric_name=str(metric["name"]),
                grouping_type="Land cover",
                categories=LANDCOVERS,
                category_labels=LANDCOVER_LABELS,
                distribution=lc_distribution,
            )
        )

    pd.DataFrame(rows).to_excel(output_path, sheet_name="mean_sd", index=False)
    print(f"Mean/SD table saved to: {output_path}")


def main() -> None:
    metric_payloads: list[
        tuple[
            dict[str, object],
            dict[tuple[str, str], np.ndarray],
            dict[tuple[str, str], np.ndarray],
        ]
    ] = []

    for metric in METRICS:
        print(f"Processing: {metric['name']}")
        group_distribution, lc_distribution = collect_metric_values(
            input_dir=metric["input_dir"],
            metric_col=metric["metric_col"],
            filename_pattern=metric["filename_pattern"],
        )
        metric_payloads.append((metric, group_distribution, lc_distribution))

    plot_grouped_panel(
        metric_payloads=metric_payloads,
        output_path=os.path.join(OUTPUT_DIR, COMBINED_OUTPUT_PANEL),
    )
    write_mean_sd_table(
        metric_payloads,
        os.path.join(OUTPUT_DIR, OUTPUT_MEAN_SD_TABLE),
    )


if __name__ == "__main__":
    main()
