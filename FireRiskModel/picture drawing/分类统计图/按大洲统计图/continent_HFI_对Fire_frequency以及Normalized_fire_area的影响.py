#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称: 按大洲绘制 HFI 对 Fire frequency 和 Normalized burned area 影响的分组柱状图

功能简介:
    1. 读取
       - B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num
       - B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area
       中全部 HFI_effect_*.csv。
    2. 提取 Year、GRID_ID 和 diff_contribution。
    3. 按火灾年份与 WUI 年份对应关系进行匹配:
       - 2003-2007 -> 2005
       - 2008-2012 -> 2010
       - 2013-2017 -> 2015
       - 2018-2022 -> 2020
    4. 通过 I:\Processing data\Grid_Clip_{year}_WUI.csv 获取 GRID_ID 对应的大洲信息。
    5. 结合 I:\Data_huanglin\map\world continent boundary\continent.shp
       校验和标准化大洲名称。
    6. 分别绘制 HFI 对 Fire frequency 和 Normalized burned area 影响的洲别分组柱状图：
       - Interface 与 Intermix 的均值 + 标准误（SE）
       - 组间差异用括号与显著性星号标注
       - 图形样式与“按照landcover及气候带分类统计起火数据_带组间显著性.py”保持一致

输出:
    - B:\WUI\Picture\continent_grouped_bar_Human Footprint_effect_on_Fire_frequency_and_Normalized_burn_area.png
    - B:\WUI\Picture\continent_HFI_effect_Grouped_Bar_Mean_SD_Table.xlsx
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from scipy.stats import ttest_ind


# =========================
# Configuration
# =========================
OUTPUT_DIR = Path(r"B:\WUI\Picture")

CONTINENT_SHP = Path(r"I:\Data_huanglin\map\world continent boundary\continent.shp")
WUI_CSV_TEMPLATE = r"I:\Processing data\Grid_Clip_{year}_WUI.csv"
WUI_YEARS = [2005, 2010, 2015, 2020]

YEAR_START = 2003
YEAR_END = 2022
CHUNKSIZE = 200_000

METRICS = [
    {
        "name": "Fire frequency",
        "key": "fire_frequency",
        "input_dir": Path(
            r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num"
        ),
        "ylabel": "Human activity effects on\nfire frequency",
        "output_filename": "continent_grouped_bar_Human Footprint_effect_on_Fire_frequency.png",
    },
    {
        "name": "Normalized burned area",
        "key": "normalized_fire_area",
        "input_dir": Path(
            r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area"
        ),
        "ylabel": "Human activity effects on\nnormalized burned area",
        "output_filename": "continent_grouped_bar_Human Footprint_effect_on_Normalized_burn_area.png",
    },
]

CONTINENT_CODE_MAPPING = {
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
CONTINENT_DISPLAY_LABELS = {
    "Africa": "Africa",
    "Asia": "Asia",
    "Europe": "Europe",
    "North America": "North\nAmerica",
    "South America": "South\nAmerica",
    "Oceania": "Oceania",
}

ALIASES = {
    "Australia": "Oceania",
}

WUI_MODE_ORDER = ["face", "mix"]
WUI_MODE_LABELS = {
    "face": "Interface",
    "mix": "Intermix",
}
FACE_COLOR = "#72B7E2"
MIX_COLOR = "#FDB366"
EDGE_COLOR = "#4A4A4A"


# =========================
# Plot style
# =========================
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 26
plt.rcParams["axes.titlesize"] = 38
plt.rcParams["axes.labelsize"] = 30
plt.rcParams["xtick.labelsize"] = 26
plt.rcParams["ytick.labelsize"] = 28
PLOT_FONT_SIZE = 28
SUBPLOT_TITLE_FONT_SIZE = 30
FIGURE_WIDTH = 14.5
FIGURE_HEIGHT = 8.8
COMBINED_OUTPUT_FILENAME = (
    "continent_grouped_bar_Human Footprint_effect_on_"
    "Fire_frequency_and_Normalized_burn_area.png"
)
OUTPUT_MEAN_SD_TABLE = "continent_HFI_effect_Grouped_Bar_Mean_SD_Table.xlsx"


def build_fire_year_to_wui_year() -> dict[int, int]:
    mapping: dict[int, int] = {}
    for wui_year in WUI_YEARS:
        for offset in range(-2, 3):
            mapping[wui_year + offset] = wui_year
    return mapping


FIRE_YEAR_TO_WUI_YEAR = build_fire_year_to_wui_year()


def normalize_continent_name(name: object) -> str | None:
    if pd.isna(name):
        return None
    text = str(name).strip()
    if not text:
        return None
    text = ALIASES.get(text, text)
    if text == "Antarctica":
        return None
    return text


def load_valid_continent_names() -> set[str]:
    gdf = gpd.read_file(CONTINENT_SHP)[["CONTINENT"]]
    names = {
        normalized
        for value in gdf["CONTINENT"].dropna().tolist()
        if (normalized := normalize_continent_name(value)) is not None
    }
    if not names:
        raise ValueError(f"No valid continent names found in: {CONTINENT_SHP}")
    return names


def load_continent_lookup(valid_continent_names: set[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    for year in WUI_YEARS:
        csv_path = WUI_CSV_TEMPLATE.format(year=year)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"WUI CSV not found: {csv_path}")

        df = pd.read_csv(
            csv_path,
            usecols=["GRID_ID", "Continent"],
            dtype={"GRID_ID": "string"},
            low_memory=False,
        )

        df["GRID_ID"] = df["GRID_ID"].astype("string").str.strip()
        df["Continent"] = pd.to_numeric(df["Continent"], errors="coerce").astype("Int64")
        df["continent_name"] = df["Continent"].map(CONTINENT_CODE_MAPPING).map(
            normalize_continent_name
        )
        df = df.dropna(subset=["GRID_ID", "continent_name"])
        df = df[df["continent_name"].isin(valid_continent_names)]
        df = df[["GRID_ID", "continent_name"]].drop_duplicates()
        df["WUI_Year"] = year
        parts.append(df)

        print(
            f"Loaded continent lookup from Grid_Clip_{year}_WUI.csv: "
            f"{len(df):,} GRID_ID matched to continents."
        )

    if not parts:
        raise ValueError("No continent lookup tables were loaded.")

    return pd.concat(parts, ignore_index=True).drop_duplicates(["WUI_Year", "GRID_ID"])


def parse_wui_mode_from_filename(csv_path: str) -> str | None:
    filename = Path(csv_path).name
    if filename.startswith("HFI_effect_face_"):
        return "face"
    if filename.startswith("HFI_effect_mix_"):
        return "mix"
    return None


def collect_metric_values(
    metric: dict[str, object], continent_lookup: pd.DataFrame
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], int, int]:
    input_dir = Path(str(metric["input_dir"]))
    csv_files = sorted(glob.glob(str(input_dir / "HFI_effect_*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No HFI effect CSV files found in: {input_dir}")

    chunk_lists: dict[str, list[np.ndarray]] = {continent: [] for continent in CONTINENT_ORDER}
    mode_chunk_lists: dict[str, dict[str, list[np.ndarray]]] = {
        continent: {mode: [] for mode in WUI_MODE_ORDER}
        for continent in CONTINENT_ORDER
    }
    matched_rows = 0
    unmatched_rows = 0

    print(f"\nProcessing {metric['name']} ({len(csv_files)} files)")

    for file_index, csv_path in enumerate(csv_files, start=1):
        wui_mode = parse_wui_mode_from_filename(csv_path)
        if wui_mode not in WUI_MODE_ORDER:
            print(f"Skip file with unknown WUI mode: {Path(csv_path).name}")
            continue

        for chunk in pd.read_csv(
            csv_path,
            usecols=["Year", "GRID_ID", "diff_contribution"],
            dtype={"GRID_ID": "string"},
            chunksize=CHUNKSIZE,
            low_memory=False,
        ):
            chunk["Year"] = pd.to_numeric(chunk["Year"], errors="coerce").astype("Int64")
            chunk["diff_contribution"] = pd.to_numeric(
                chunk["diff_contribution"], errors="coerce"
            )
            chunk["GRID_ID"] = chunk["GRID_ID"].astype("string").str.strip()
            chunk = chunk.dropna(subset=["Year", "GRID_ID", "diff_contribution"])
            chunk = chunk[chunk["Year"].between(YEAR_START, YEAR_END)]
            if chunk.empty:
                continue

            chunk["WUI_Year"] = chunk["Year"].map(FIRE_YEAR_TO_WUI_YEAR).astype("Int64")
            chunk = chunk.dropna(subset=["WUI_Year"])
            if chunk.empty:
                continue

            merged = chunk.merge(
                continent_lookup,
                how="left",
                on=["WUI_Year", "GRID_ID"],
            )

            matched_rows += int(merged["continent_name"].notna().sum())
            unmatched_rows += int(merged["continent_name"].isna().sum())

            merged = merged.dropna(subset=["continent_name"])
            if merged.empty:
                continue

            for continent, group in merged.groupby("continent_name"):
                if continent not in chunk_lists:
                    continue
                values = group["diff_contribution"].to_numpy(dtype=np.float32, copy=True)
                if values.size:
                    chunk_lists[continent].append(values)
                    mode_chunk_lists[continent][wui_mode].append(values)

    continent_values: dict[str, np.ndarray] = {}
    continent_mode_values: dict[str, dict[str, np.ndarray]] = {}
    for continent in CONTINENT_ORDER:
        pieces = chunk_lists.get(continent, [])
        if pieces:
            continent_values[continent] = np.concatenate(pieces)
        else:
            continent_values[continent] = np.array([], dtype=np.float32)

        continent_mode_values[continent] = {}
        for mode in WUI_MODE_ORDER:
            mode_pieces = mode_chunk_lists[continent].get(mode, [])
            if mode_pieces:
                continent_mode_values[continent][mode] = np.concatenate(mode_pieces)
            else:
                continent_mode_values[continent][mode] = np.array([], dtype=np.float32)

    print(
        f"Completed {metric['name']}: matched_rows={matched_rows:,}, "
        f"unmatched_rows={unmatched_rows:,}"
    )
    return continent_values, continent_mode_values, matched_rows, unmatched_rows


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


def _get_value_limits(
    continent_mode_values: dict[str, dict[str, np.ndarray]],
    continents: list[str],
    *,
    metric_key: str,
) -> tuple[float, float]:
    lower_candidates: list[float] = []
    upper_candidates: list[float] = []
    for continent in continents:
        for mode in WUI_MODE_ORDER:
            values = continent_mode_values[continent][mode]
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
    if metric_key == "fire_frequency":
        margin_ratio = 0.04
        abs_floor_ratio = 0.02
    else:
        margin_ratio = 0.05
        abs_floor_ratio = 0.025
    scale_ref = max(abs(y_min), abs(y_max), 1e-8)
    margin = max((y_max - y_min) * margin_ratio, abs_floor_ratio * scale_ref)
    return y_min - margin, y_max + margin


def draw_metric_grouped_bars(
    ax: plt.Axes,
    metric: dict[str, object],
    continent_mode_values: dict[str, dict[str, np.ndarray]],
    *,
    show_legend: bool,
) -> None:
    continents = [
        continent
        for continent in CONTINENT_ORDER
        if any(continent_mode_values[continent][mode].size > 0 for mode in WUI_MODE_ORDER)
    ]
    if not continents:
        raise ValueError(f"No matched continent data available for {metric['name']}")

    positions = np.arange(len(continents), dtype=float)
    offset = 0.18
    bar_width = 0.30
    side_margin = 0.60

    face_means: list[float] = []
    face_ses: list[float] = []
    mix_means: list[float] = []
    mix_ses: list[float] = []
    significance_markers: list[str] = []

    for continent in continents:
        face_values = continent_mode_values[continent]["face"]
        mix_values = continent_mode_values[continent]["mix"]
        face_mean, face_se = compute_mean_se(face_values)
        mix_mean, mix_se = compute_mean_se(mix_values)
        face_means.append(face_mean)
        face_ses.append(face_se if np.isfinite(face_se) else 0.0)
        mix_means.append(mix_mean)
        mix_ses.append(mix_se if np.isfinite(mix_se) else 0.0)
        significance_markers.append(_significance_marker(_compute_p_value(face_values, mix_values)))

    y_limits = _get_value_limits(continent_mode_values, continents, metric_key=str(metric["key"]))
    y_span = y_limits[1] - y_limits[0]
    bracket_gap = y_span * 0.028
    anchor_gap = y_span * 0.010
    text_gap = y_span * 0.010
    text_fontsize = max(PLOT_FONT_SIZE * 0.72, 16)
    final_y_min = y_limits[0]
    final_y_max = y_limits[1]

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

    for idx, marker in enumerate(significance_markers):
        if not marker:
            continue
        face_mean = face_means[idx]
        mix_mean = mix_means[idx]
        if not np.isfinite(face_mean) or not np.isfinite(mix_mean):
            continue
        mixed_sign = (face_mean < 0 < mix_mean) or (mix_mean < 0 < face_mean)
        direction = -1 if (face_mean <= 0 and mix_mean <= 0) else 1

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
        y_line = min(face_anchor, mix_anchor) - bracket_gap if direction < 0 else max(face_anchor, mix_anchor) + bracket_gap

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
        if direction < 0:
            final_y_min = min(final_y_min, y_line - 2.2 * text_gap)
        else:
            final_y_max = max(final_y_max, y_line + 2.2 * text_gap)

    ax.set_ylim(final_y_min, final_y_max)
    ax.set_ylabel(str(metric["ylabel"]), fontsize=PLOT_FONT_SIZE)
    ax.yaxis.set_label_coords(-0.065, 0.50)
    ax.set_xlabel("Continent", fontsize=SUBPLOT_TITLE_FONT_SIZE, labelpad=16)
    ax.set_xlim(-side_margin, len(continents) - 1 + side_margin)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [CONTINENT_DISPLAY_LABELS.get(continent, continent) for continent in continents],
        fontsize=PLOT_FONT_SIZE - 1,
        rotation=0,
        ha="center",
    )
    ax.tick_params(axis="y", labelsize=PLOT_FONT_SIZE)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    sci_formatter = ScalarFormatter(useMathText=False)
    sci_formatter.set_scientific(True)
    sci_formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(sci_formatter)
    ax.yaxis.get_offset_text().set_size(PLOT_FONT_SIZE - 2)
    ax.grid(True, axis="y", linestyle="--", alpha=0.32, linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_legend:
        ax.legend(
            loc="upper right",
            ncol=1,
            fontsize=max(PLOT_FONT_SIZE * 0.95, 22),
            frameon=False,
            handlelength=1.4,
            labelspacing=0.35,
            borderaxespad=0.35,
        )


def build_mean_sd_rows(
    *,
    metric_name: str,
    continent_values: dict[str, np.ndarray],
    continent_mode_values: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for continent in CONTINENT_ORDER:
        if (
            continent_values[continent].size == 0
            and all(continent_mode_values[continent][mode].size == 0 for mode in WUI_MODE_ORDER)
        ):
            continue

        mean, std = compute_mean_std(continent_values[continent])
        rows.append(
            {
                "metric": metric_name,
                "grouping_type": "Continent",
                "category_code": continent,
                "category_label": continent,
                "wui_type": "All",
                "mean": mean,
                "std": std,
            }
        )

        for mode in WUI_MODE_ORDER:
            mode_mean, mode_std = compute_mean_std(continent_mode_values[continent][mode])
            rows.append(
                {
                    "metric": metric_name,
                    "grouping_type": "Continent",
                    "category_code": continent,
                    "category_label": continent,
                    "wui_type": WUI_MODE_LABELS[mode],
                    "mean": mode_mean,
                    "std": mode_std,
                }
            )
    return rows


def write_mean_sd_table(rows: list[dict[str, object]], output_path: Path) -> None:
    pd.DataFrame(rows).to_excel(output_path, sheet_name="mean_sd", index=False)
    print(f"Mean/SD table saved to: {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    valid_continent_names = load_valid_continent_names()
    continent_lookup = load_continent_lookup(valid_continent_names)

    metric_payloads: list[tuple[dict[str, object], dict[str, dict[str, np.ndarray]], int, int]] = []
    mean_sd_rows: list[dict[str, object]] = []
    for metric in METRICS:
        continent_values, continent_mode_values, matched_rows, unmatched_rows = collect_metric_values(
            metric, continent_lookup
        )
        metric_payloads.append((metric, continent_mode_values, matched_rows, unmatched_rows))
        mean_sd_rows.extend(
            build_mean_sd_rows(
                metric_name=str(metric["name"]),
                continent_values=continent_values,
                continent_mode_values=continent_mode_values,
            )
        )
        print(
            f"Finished {metric['name']}: matched_rows={matched_rows:,}, "
            f"unmatched_rows={unmatched_rows:,}"
        )

    fig, axes = plt.subplots(2, 1, figsize=(FIGURE_WIDTH * 1.25, FIGURE_HEIGHT * 2.0))
    panel_labels = ["a", "b"]
    for idx, (ax, (metric, continent_mode_values, _, _), show_legend) in enumerate(zip(
        axes,
        metric_payloads,
        [True, False],
        strict=False,
    )):
        draw_metric_grouped_bars(
            ax,
            metric,
            continent_mode_values,
            show_legend=show_legend,
        )
        ax.text(
            -0.09,
            1.02,
            panel_labels[idx],
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=PLOT_FONT_SIZE + 2,
            fontweight="bold",
            color="#222222",
        )

    output_path = OUTPUT_DIR / COMBINED_OUTPUT_FILENAME
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.11, top=0.965, hspace=0.34)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {output_path}")

    write_mean_sd_table(
        mean_sd_rows,
        OUTPUT_DIR / OUTPUT_MEAN_SD_TABLE,
    )


if __name__ == "__main__":
    main()
