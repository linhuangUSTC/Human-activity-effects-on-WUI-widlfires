#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称: 边际效应曲线 HFI 峰谷值分析

功能:
    1) 读取 Fire num 与 Fire area 的 HFI 净效应固定效应结果 CSV
    2) 基于 HFI 与 diff_contribution 重建每个分类 ID 的平滑边际效应曲线
    3) 提取每个分类的 HFI 峰值位置（peak_hfi）、谷值位置（trough_hfi）及对应效应
    4) 关联边际效应曲线分类结果，检查不同 cluster / 气候带 / landcover / WUI 类型下的规律
    5) 对 Fire frequency 与 Normalized burn area 分别输出峰值和谷值的排序图、cluster 分布图
    6) 将 climate-landcover 热图分别按峰值和谷值合并为共享 colormap 的单行总图

输出:
    - B:\WUI\Picture\Human Footprint_peak_pattern_sorted_peak_*.png
    - B:\WUI\Picture\Human Footprint_peak_pattern_cluster_distribution_*.png
    - B:\WUI\Picture\Human Footprint_peak_pattern_climate_landcover_combined.png
    - B:\WUI\Picture\Human Footprint_trough_pattern_sorted_trough_*.png
    - B:\WUI\Picture\Human Footprint_trough_pattern_cluster_distribution_*.png
    - B:\WUI\Picture\Human Footprint_trough_pattern_climate_landcover_combined.png
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from statsmodels.nonparametric.smoothers_lowess import lowess


CSV_ROOT = r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv"
OUTPUT_ROOT = r"B:\WUI\Picture"
CLASSIFICATION_ROOT = (
    r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\层次聚类"
    r"\边际效应曲线层次聚类分类"
)

LOWESS_FRAC = 0.22
N_BINS = 80
N_GRID = 240


DATASETS = [
    {
        "key": "Fire num",
        "csv_dir": os.path.join(CSV_ROOT, "Fire num"),
        "classification_csv": os.path.join(
            CLASSIFICATION_ROOT,
            "Human Footprint_effect_curve_cluster_assignments_Fire_num.csv",
        ),
        "effect_ylabel": "Effect on Fire frequency",
        "title_metric": "Fire frequency",
        "heatmap_title": "",
        "output_slug": "Fire_frequency",
    },
    {
        "key": "Fire area",
        "csv_dir": os.path.join(CSV_ROOT, "Fire area"),
        "classification_csv": os.path.join(
            CLASSIFICATION_ROOT,
            "Human Footprint_effect_curve_cluster_assignments_Fire_area.csv",
        ),
        "effect_ylabel": "Effect on Normalized burn area",
        "title_metric": "Normalized burn area",
        "heatmap_title": "",
        "output_slug": "Normalized_burn_area",
    },
]


plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 9

PEAK_COLOR = "#4c78a8"
TROUGH_COLOR = "#e45756"
PANEL_FACE = "#ffffff"
GRID_COLOR = "#d9d9d9"
STRIPE_FACE = "#f7f7f7"
HEATMAP_TEXT_FONTSIZE = 45
HEATMAP_ANNOTATION_FONTSIZE = HEATMAP_TEXT_FONTSIZE
HEATMAP_CBAR_LABEL_FONTSIZE = HEATMAP_TEXT_FONTSIZE
HEATMAP_CBAR_TICK_FONTSIZE = HEATMAP_TEXT_FONTSIZE
HEATMAP_AXIS_LABEL_FONTSIZE = HEATMAP_TEXT_FONTSIZE
HEATMAP_SUPTITLE_FONTSIZE = HEATMAP_TEXT_FONTSIZE
HEATMAP_PANEL_TITLE_FONTSIZE = HEATMAP_TEXT_FONTSIZE
HEATMAP_PANEL_TITLE_PAD = 14
HEATMAP_COMBINED_WIDTH = 47.5
HEATMAP_COMBINED_HEIGHT = 11
HEATMAP_OUTPUT_DPI = 800
HEATMAP_INNER_WSPACE = 0.06
HEATMAP_DATASET_WSPACE = 0.1
HEATMAP_COLORBAR_WSPACE = 0.03
HEATMAP_COLORBAR_WIDTH_RATIO = 0.045
EXTREME_CONFIGS = [
    {
        "key": "peak",
        "value_col": "peak_hfi",
        "effect_col": "peak_effect",
        "label": "Peak",
        "color": PEAK_COLOR,
        "sorted_output_pattern": "Human Footprint_peak_pattern_sorted_peak_{output_slug}.png",
        "cluster_output_pattern": "Human Footprint_peak_pattern_cluster_distribution_{output_slug}.png",
        "heatmap_output": "Human Footprint_peak_pattern_climate_landcover_combined.png",
        "heatmap_value_label": "Median peak Human Footprint",
    },
    {
        "key": "trough",
        "value_col": "trough_hfi",
        "effect_col": "trough_effect",
        "label": "Trough",
        "color": TROUGH_COLOR,
        "sorted_output_pattern": "Human Footprint_trough_pattern_sorted_trough_{output_slug}.png",
        "cluster_output_pattern": "Human Footprint_trough_pattern_cluster_distribution_{output_slug}.png",
        "heatmap_output": "Human Footprint_trough_pattern_climate_landcover_combined.png",
        "heatmap_value_label": "Median trough Human Footprint",
    },
]
HEATMAP_CLIMATE_ORDER = ["A", "B", "C", "D"]
HEATMAP_LANDCOVER_ORDER = ["CRP", "FST", "GRS", "SHR", "WET"]
HEATMAP_MODE_ORDER = ["face", "mix"]


@dataclass
class HeatmapPayload:
    heatmap_title: str
    mode_tables: dict[str, pd.DataFrame]
    finite_values: np.ndarray


def extract_metadata(path: str) -> tuple[str, str, str, str]:
    name = Path(path).stem
    match = re.search(
        r"HFI_effect_((face|mix)_([A-D])_([A-Z]+))_(?:fire_num_with_WUI_offset|fire_area)$",
        name,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"Unrecognized file name pattern: {name}")
    return match.group(1), match.group(2), match.group(3), match.group(4)


def format_hfi(value: float, _: int | None = None) -> str:
    return f"{value:.0f}"


def build_smoothed_curve(hfi: np.ndarray, diff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hfi = np.asarray(hfi, dtype=float)
    diff = np.asarray(diff, dtype=float)
    valid = np.isfinite(hfi) & np.isfinite(diff)
    hfi = hfi[valid]
    diff = diff[valid]
    order = np.argsort(hfi)
    hfi = hfi[order]
    diff = diff[order]

    hfi_min = float(hfi.min())
    hfi_max = float(hfi.max())
    if hfi_max - hfi_min < 1e-12:
        grid = np.linspace(hfi_min, hfi_max + 1e-6, N_GRID)
        return grid, np.full_like(grid, diff.mean())

    bin_edges = np.linspace(hfi_min, hfi_max, N_BINS + 1)
    bin_idx = np.digitize(hfi, bin_edges) - 1
    valid_bin = (bin_idx >= 0) & (bin_idx < len(bin_edges) - 1)
    bin_idx = bin_idx[valid_bin]
    hfi = hfi[valid_bin]
    diff = diff[valid_bin]

    counts = np.bincount(bin_idx, minlength=len(bin_edges) - 1)
    sum_hfi = np.bincount(bin_idx, weights=hfi, minlength=len(bin_edges) - 1)
    sum_diff = np.bincount(bin_idx, weights=diff, minlength=len(bin_edges) - 1)
    mask = counts > 0

    x = sum_hfi[mask] / counts[mask]
    y = sum_diff[mask] / counts[mask]
    if len(x) >= 4:
        y = lowess(y, x, frac=LOWESS_FRAC, return_sorted=False)

    grid = np.linspace(hfi_min, hfi_max, N_GRID)
    curve = np.interp(grid, x, y, left=y[0], right=y[-1])
    return grid, curve


def load_classification_table(classification_csv: str) -> pd.DataFrame:
    normalized = os.path.normpath(classification_csv)
    if os.path.exists(normalized):
        class_df = pd.read_csv(normalized, low_memory=False)
        required_cols = {"category_id", "cluster", "cluster_label"}
        missing = required_cols.difference(class_df.columns)
        if missing:
            raise ValueError(
                f"Classification CSV missing required column(s) {sorted(missing)}: {normalized}"
            )
        print(f"Using classification results: {normalized}")
        return class_df

    print(f"Warning: classification CSV not found: {normalized}. Cluster-based summaries will be unassigned.")
    return pd.DataFrame(columns=["category_id", "cluster", "cluster_label"])


def build_mode_tables(climate_lc_summary: pd.DataFrame, median_value_col: str) -> tuple[dict[str, pd.DataFrame], np.ndarray]:
    mode_tables: dict[str, pd.DataFrame] = {}
    finite_values_list: list[np.ndarray] = []
    for mode_name in HEATMAP_MODE_ORDER:
        mode_table = (
            climate_lc_summary.loc[climate_lc_summary["mode"] == mode_name]
            .pivot(index="climate_zone", columns="landcover", values=median_value_col)
            .reindex(index=HEATMAP_CLIMATE_ORDER, columns=HEATMAP_LANDCOVER_ORDER)
        )
        mode_tables[mode_name] = mode_table
        finite_values = mode_table.to_numpy(dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size > 0:
            finite_values_list.append(finite_values)

    if finite_values_list:
        return mode_tables, np.concatenate(finite_values_list)
    return mode_tables, np.array([], dtype=float)


def save_combined_heatmap(
    heatmap_payloads: list[HeatmapPayload],
    *,
    output_filename: str,
    colorbar_label: str,
) -> None:
    if not heatmap_payloads:
        return

    finite_values_list = [payload.finite_values for payload in heatmap_payloads if payload.finite_values.size > 0]
    if finite_values_list:
        combined_values = np.concatenate(finite_values_list)
        heatmap_vmin = float(np.nanmin(combined_values))
        heatmap_vmax = float(np.nanmax(combined_values))
    else:
        heatmap_vmin, heatmap_vmax = 0.0, 1.0
    if abs(heatmap_vmax - heatmap_vmin) < 1e-12:
        heatmap_vmax = heatmap_vmin + 1.0

    heatmap_cmap = sns.color_palette("YlOrRd", as_cmap=True)
    mode_title_map = {"face": "Interface", "mix": "Intermix"}
    colorbar_ticks = np.arange(
        np.ceil(heatmap_vmin / 10.0) * 10.0,
        np.floor(heatmap_vmax / 10.0) * 10.0 + 0.1,
        10.0,
    )

    fig_heat = plt.figure(figsize=(HEATMAP_COMBINED_WIDTH, HEATMAP_COMBINED_HEIGHT), facecolor="white")
    heat_grid = fig_heat.add_gridspec(
        1,
        2,
        width_ratios=[len(heatmap_payloads), HEATMAP_COLORBAR_WIDTH_RATIO],
        wspace=HEATMAP_COLORBAR_WSPACE,
    )
    dataset_area_grid = heat_grid[0, 0].subgridspec(1, len(heatmap_payloads), wspace=HEATMAP_DATASET_WSPACE)
    cbar_ax = fig_heat.add_subplot(heat_grid[0, 1])
    colorbar = None
    dataset_title_specs = []
    panel_axes = []

    for dataset_idx, payload in enumerate(heatmap_payloads):
        dataset_axes = []
        dataset_grid = dataset_area_grid[0, dataset_idx].subgridspec(
            1,
            len(HEATMAP_MODE_ORDER),
            wspace=HEATMAP_INNER_WSPACE,
        )
        for mode_idx, mode_name in enumerate(HEATMAP_MODE_ORDER):
            panel_idx = dataset_idx * len(HEATMAP_MODE_ORDER) + mode_idx
            ax = fig_heat.add_subplot(dataset_grid[0, mode_idx])
            mode_table = payload.mode_tables[mode_name]
            show_cbar = dataset_idx == len(heatmap_payloads) - 1 and mode_idx == len(HEATMAP_MODE_ORDER) - 1
            sns.heatmap(
                mode_table,
                ax=ax,
                cmap=heatmap_cmap,
                annot=True,
                fmt=".1f",
                annot_kws={"fontsize": HEATMAP_ANNOTATION_FONTSIZE, "fontname": "Arial"},
                linewidths=0.5,
                linecolor="white",
                vmin=heatmap_vmin,
                vmax=heatmap_vmax,
                cbar=show_cbar,
                cbar_ax=cbar_ax if show_cbar else None,
                cbar_kws={"label": colorbar_label} if show_cbar else None,
                square=True,
            )
            na_positions = np.argwhere(mode_table.isna().to_numpy())
            for row_idx, col_idx in na_positions:
                ax.text(
                    col_idx + 0.5,
                    row_idx + 0.5,
                    "NA",
                    ha="center",
                    va="center",
                    fontsize=HEATMAP_ANNOTATION_FONTSIZE,
                    fontname="Arial",
                    color="#4f4f4f",
                )
            ax.set_title(
                mode_title_map[mode_name],
                fontsize=HEATMAP_PANEL_TITLE_FONTSIZE,
                fontweight="bold",
                fontname="Arial",
                pad=HEATMAP_PANEL_TITLE_PAD,
            )
            ax.set_xlabel("Landcover", fontsize=HEATMAP_AXIS_LABEL_FONTSIZE, fontname="Arial")
            ax.set_ylabel(
                "Climate zone" if mode_idx == 0 else "",
                fontsize=HEATMAP_AXIS_LABEL_FONTSIZE,
                fontname="Arial",
            )
            ax.tick_params(axis="both", labelsize=HEATMAP_TEXT_FONTSIZE)
            for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
                tick_label.set_fontname("Arial")
            if mode_idx != 0:
                ax.tick_params(axis="y", left=False, labelleft=False)
            if show_cbar and ax.collections:
                colorbar = ax.collections[0].colorbar
            dataset_axes.append(ax)
            panel_axes.append(ax)

        if payload.heatmap_title and dataset_axes:
            dataset_title_specs.append((payload.heatmap_title, dataset_axes[0], dataset_axes[-1]))

    if colorbar is not None:
        colorbar.set_label(colorbar_label, fontsize=HEATMAP_CBAR_LABEL_FONTSIZE, fontname="Arial")
        colorbar.ax.tick_params(labelsize=HEATMAP_CBAR_TICK_FONTSIZE)
        if colorbar_ticks.size > 0:
            colorbar.set_ticks(colorbar_ticks.tolist())
            colorbar.set_ticklabels([f"{int(tick)}" for tick in colorbar_ticks])
        for tick_label in colorbar.ax.get_yticklabels():
            tick_label.set_fontname("Arial")

    fig_heat.subplots_adjust(top=0.89, left=0.05, right=0.965, bottom=0.11)
    if colorbar is not None and panel_axes:
        ref_ax = panel_axes[-1]
        cbar_pos = cbar_ax.get_position()
        ref_pos = ref_ax.get_position()
        cbar_ax.set_position([cbar_pos.x0, ref_pos.y0, cbar_pos.width, ref_pos.height])
    for heatmap_title, left_ax, right_ax in dataset_title_specs:
        left = left_ax.get_position().x0
        right = right_ax.get_position().x1
        fig_heat.text(
            (left + right) / 2,
            0.955,
            heatmap_title,
            ha="center",
            va="bottom",
            fontsize=HEATMAP_SUPTITLE_FONTSIZE,
            fontweight="bold",
            fontname="Arial",
        )
    heatmap_output_path = os.path.join(OUTPUT_ROOT, output_filename)
    fig_heat.savefig(heatmap_output_path, dpi=HEATMAP_OUTPUT_DPI, facecolor=fig_heat.get_facecolor())
    plt.close(fig_heat)
    print(f"Saved combined climate-landcover heatmap figure to: {heatmap_output_path}")


def plot_sorted_extreme(
    plot_df: pd.DataFrame,
    config: dict[str, str],
    extreme_config: dict[str, str],
) -> None:
    fig_sorted, ax = plt.subplots(figsize=(11.2, 11.2), facecolor="white")
    y = np.arange(len(plot_df))
    for i in y:
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=STRIPE_FACE, zorder=0)
    ax.hlines(
        y,
        xmin=plot_df["hfi_min"],
        xmax=plot_df[str(extreme_config["value_col"])],
        color=str(extreme_config["color"]),
        linewidth=1.7,
        alpha=0.9,
    )
    ax.scatter(
        plot_df[str(extreme_config["value_col"])],
        y,
        s=46,
        c=str(extreme_config["color"]),
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["category_id"])
    ax.set_xlabel(f"{extreme_config['label']} Human Footprint")
    ax.set_ylabel("Category ID")
    ax.set_title(
        f"Sorted {str(extreme_config['label']).lower()} Human Footprint by category: {config['title_metric']}",
        fontweight="bold",
    )
    ax.grid(True, axis="x", color=GRID_COLOR, alpha=0.6)
    ax.xaxis.set_major_formatter(FuncFormatter(format_hfi))
    ax.set_facecolor(PANEL_FACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig_sorted.subplots_adjust(top=0.93, left=0.2, right=0.98, bottom=0.08)
    output_path = os.path.join(
        OUTPUT_ROOT,
        str(extreme_config["sorted_output_pattern"]).format(output_slug=config["output_slug"]),
    )
    fig_sorted.savefig(output_path, dpi=1200, bbox_inches="tight", facecolor=fig_sorted.get_facecolor())
    plt.close(fig_sorted)
    print(f"Saved sorted {str(extreme_config['key'])} figure to: {output_path}")


def plot_cluster_distribution(
    extreme_df: pd.DataFrame,
    extreme_config: dict[str, str],
    config: dict[str, str],
) -> None:
    fig_cluster, ax = plt.subplots(figsize=(10.2, 6.8), facecolor="white")
    sns.boxplot(
        data=extreme_df,
        x=str(extreme_config["value_col"]),
        y="cluster_label",
        ax=ax,
        color="#e6e6e6",
        fliersize=0,
        linewidth=1.0,
    )
    sns.stripplot(
        data=extreme_df,
        x=str(extreme_config["value_col"]),
        y="cluster_label",
        dodge=False,
        size=6,
        alpha=0.9,
        color=str(extreme_config["color"]),
        ax=ax,
    )
    ax.set_title(f"{extreme_config['label']} Human Footprint distribution by cluster", fontweight="bold")
    ax.set_xlabel(f"{extreme_config['label']} Human Footprint")
    ax.set_ylabel("")
    ax.grid(True, axis="x", color=GRID_COLOR, alpha=0.6)
    ax.xaxis.set_major_formatter(FuncFormatter(format_hfi))
    ax.set_facecolor(PANEL_FACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig_cluster.subplots_adjust(top=0.9, left=0.2, right=0.98, bottom=0.12)
    output_path = os.path.join(
        OUTPUT_ROOT,
        str(extreme_config["cluster_output_pattern"]).format(output_slug=config["output_slug"]),
    )
    fig_cluster.savefig(output_path, dpi=1200, bbox_inches="tight", facecolor=fig_cluster.get_facecolor())
    plt.close(fig_cluster)
    print(f"Saved {str(extreme_config['key'])} cluster distribution figure to: {output_path}")


def analyze_dataset(config: dict[str, str]) -> dict[str, HeatmapPayload]:
    csv_files = sorted(glob.glob(os.path.join(config["csv_dir"], "HFI_effect_*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {config['csv_dir']}")

    class_df = load_classification_table(config["classification_csv"])

    rows: list[dict[str, object]] = []
    for csv_path in csv_files:
        category_id, mode, climate_zone, landcover = extract_metadata(csv_path)
        df = pd.read_csv(csv_path, usecols=["HFI", "diff_contribution"], low_memory=False)
        hfi = pd.to_numeric(df["HFI"], errors="coerce").to_numpy(dtype=float)
        diff = pd.to_numeric(df["diff_contribution"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(hfi) & np.isfinite(diff)
        hfi = hfi[valid]
        diff = diff[valid]
        if hfi.size < 10:
            continue

        grid, curve = build_smoothed_curve(hfi, diff)
        peak_idx = int(np.argmax(curve))
        peak_hfi = float(grid[peak_idx])
        peak_effect = float(curve[peak_idx])
        trough_idx = int(np.argmin(curve))
        trough_hfi = float(grid[trough_idx])
        trough_effect = float(curve[trough_idx])
        hfi_min = float(grid[0])
        hfi_max = float(grid[-1])

        row = {
            "category_id": category_id,
            "mode": mode,
            "climate_zone": climate_zone,
            "landcover": landcover,
            "hfi_min": hfi_min,
            "hfi_max": hfi_max,
            "peak_hfi": peak_hfi,
            "peak_effect": peak_effect,
            "trough_hfi": trough_hfi,
            "trough_effect": trough_effect,
            "cluster": np.nan,
            "cluster_label": "",
        }

        if not class_df.empty:
            match = class_df.loc[class_df["category_id"] == category_id]
            if not match.empty:
                row["cluster"] = int(match.iloc[0]["cluster"])
                row["cluster_label"] = match.iloc[0].get("cluster_label", "")
        rows.append(row)

    extreme_df = pd.DataFrame(rows)
    if extreme_df.empty:
        return {}

    heatmap_payloads: dict[str, HeatmapPayload] = {}
    for extreme_config in EXTREME_CONFIGS:
        value_col = str(extreme_config["value_col"])
        plot_df = extreme_df.sort_values(value_col, ascending=True).reset_index(drop=True)
        plot_sorted_extreme(plot_df, config, extreme_config)
        plot_cluster_distribution(extreme_df, extreme_config, config)

        climate_lc_summary = (
            extreme_df.groupby(["mode", "climate_zone", "landcover"], dropna=False)
            .agg(
                n=("category_id", "count"),
                **{
                    f"mean_{value_col}": (value_col, "mean"),
                    f"median_{value_col}": (value_col, "median"),
                },
            )
            .reset_index()
            .sort_values(["mode", "climate_zone", "landcover"])
        )
        mode_tables, finite_values = build_mode_tables(climate_lc_summary, f"median_{value_col}")
        heatmap_payloads[str(extreme_config["key"])] = HeatmapPayload(
            heatmap_title=config["heatmap_title"],
            mode_tables=mode_tables,
            finite_values=finite_values,
        )
    return heatmap_payloads


def main() -> None:
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    heatmap_payloads_by_extreme = {str(item["key"]): [] for item in EXTREME_CONFIGS}
    for config in DATASETS:
        print(f"\n=== Processing {config['key']} ===")
        dataset_payloads = analyze_dataset(config)
        for extreme_config in EXTREME_CONFIGS:
            extreme_key = str(extreme_config["key"])
            payload = dataset_payloads.get(extreme_key)
            if payload is not None:
                heatmap_payloads_by_extreme[extreme_key].append(payload)

    for extreme_config in EXTREME_CONFIGS:
        save_combined_heatmap(
            heatmap_payloads_by_extreme[str(extreme_config["key"])],
            output_filename=str(extreme_config["heatmap_output"]),
            colorbar_label=str(extreme_config["heatmap_value_label"]),
        )


if __name__ == "__main__":
    main()
