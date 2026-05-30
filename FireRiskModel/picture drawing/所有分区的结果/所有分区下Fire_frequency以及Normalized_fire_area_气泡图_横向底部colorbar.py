#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称:
    不同分区下 Fire frequency 与 Normalized burned area 的气泡矩阵图比较（底部横向 colorbar）

功能简介:
    1. 读取以下目录中的分类建模基础数据:
       - I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic
       - I:\Processing data\渔网分类建模\Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic
    2. 依据文件名识别 partition (face/mix + climate zone + landcover)。
    3. 对每个 partition，直接对 2003-2022 年全部样本计算 Fire frequency / Normalized burned area 的总体均值。
    4. 分别输出两张双联气泡矩阵图:
       - 左: Interface WUI (face)
       - 右: Intermix WUI (mix)
    5. 圆点大小保持一致，仅用颜色表示数值大小，并共享位于底部的横向 colorbar。

输出:
    - B:\WUI\Picture\Partition_fire_frequency_bubble_comparison_horizontal_colorbar.png
    - B:\WUI\Picture\Partition_normalized_burn_area_bubble_comparison_horizontal_colorbar.png
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator


FIRE_NUM_DIR = Path(
    r"I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic"
)
FIRE_AREA_DIR = Path(
    r"I:\Processing data\渔网分类建模\Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic"
)
OUTPUT_DIR = Path(r"B:\WUI\Picture")

YEAR_START = 2003
YEAR_END = 2022
CLIMATE_ORDER = ["A", "B", "C", "D"]
LANDCOVER_ORDER = ["CRP", "FST", "GRS", "SHR", "WET"]
MODE_ORDER = ["face", "mix"]

INTERFACE_CMAP = LinearSegmentedColormap.from_list(
    "interface_blue_bubbles",
    ["#acd0ff", "#99c3fd", "#76adff", "#4a90ff", "#2769f8"],
    N=256,
)

INTERMIX_CMAP = LinearSegmentedColormap.from_list(
    "intermix_orange_bubbles",
    ["#ffddbf", "#f9d48a", "#ffa968", "#f68828", "#f6662d"],
    N=256,
)

DATASETS = [
    {
        "input_dir": FIRE_NUM_DIR,
        "metric_column": "Fire_frequency",
        "metric_label": "Fire frequency",
        "colorbar_label": "Fire frequency",
        "output_filename": "Partition_fire_frequency_bubble_comparison_horizontal_colorbar.png",
    },
    {
        "input_dir": FIRE_AREA_DIR,
        "metric_column": "Normalized_fire_area",
        "metric_label": "Normalized burned area",
        "colorbar_label": "Normalized burned area",
        "output_filename": "Partition_normalized_burn_area_bubble_comparison_horizontal_colorbar.png",
    },
]


plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 22

UNIFORM_BUBBLE_SIZE = 1300.0


def extract_category_id(path: str) -> str:
    stem = Path(path).stem
    match = re.search(r"((?:face|mix)_[A-D]_[A-Z]+)", stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse category ID from file name: {stem}")
    return match.group(1)


def split_category_id(category_id: str) -> tuple[str, str, str]:
    parts = category_id.split("_")
    if len(parts) != 3:
        raise ValueError(f"Unexpected category ID: {category_id}")
    return parts[0].lower(), parts[1].upper(), parts[2].upper()


def choose_exponent(values: np.ndarray) -> int:
    finite_vals = values[np.isfinite(values)]
    if finite_vals.size == 0:
        return 0
    max_abs = float(np.nanmax(np.abs(finite_vals)))
    if max_abs < 1e-12:
        return 0
    return int(np.floor(np.log10(max_abs)))


def safe_scale(values: np.ndarray, exponent: int) -> np.ndarray:
    factor = 10.0 ** exponent
    if factor == 0:
        return values.copy()
    return values / factor


def load_partition_summary(dataset: dict[str, object]) -> pd.DataFrame:
    input_dir = Path(str(dataset["input_dir"]))
    metric_column = str(dataset["metric_column"])
    csv_files = sorted(glob.glob(str(input_dir / "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    rows: list[dict[str, object]] = []
    print(f"Reading {len(csv_files)} partition file(s) from: {input_dir}")

    for csv_path in csv_files:
        category_id = extract_category_id(csv_path)
        mode, climate, landcover = split_category_id(category_id)

        df = pd.read_csv(csv_path, usecols=["Year", metric_column], low_memory=False)
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df[metric_column] = pd.to_numeric(df[metric_column], errors="coerce")
        df = df.dropna(subset=["Year", metric_column])
        df = df[df["Year"].between(YEAR_START, YEAR_END)]
        if df.empty:
            continue

        rows.append(
            {
                "category_id": category_id,
                "mode": mode,
                "climate": climate,
                "landcover": landcover,
                "overall_mean": float(df[metric_column].mean()),
            }
        )

    if not rows:
        raise ValueError(f"No valid partition summaries derived from: {input_dir}")

    summary = pd.DataFrame(rows)
    summary["mode"] = pd.Categorical(summary["mode"], MODE_ORDER, ordered=True)
    summary["climate"] = pd.Categorical(summary["climate"], CLIMATE_ORDER, ordered=True)
    summary["landcover"] = pd.Categorical(summary["landcover"], LANDCOVER_ORDER, ordered=True)
    return summary.sort_values(["mode", "climate", "landcover"]).reset_index(drop=True)


def build_matrix(summary: pd.DataFrame, mode: str, value_col: str) -> np.ndarray:
    subset = summary[summary["mode"] == mode].copy()
    matrix = np.full((len(CLIMATE_ORDER), len(LANDCOVER_ORDER)), np.nan, dtype=float)
    for i, climate in enumerate(CLIMATE_ORDER):
        for j, landcover in enumerate(LANDCOVER_ORDER):
            matched = subset[
                (subset["climate"] == climate)
                & (subset["landcover"] == landcover)
            ]
            if not matched.empty:
                matrix[i, j] = float(matched.iloc[0][value_col])
    return matrix


def draw_bubble_matrix(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    cmap,
    norm,
    panel_label: str,
    exponent: int,
    vmax_scaled: float,
    show_ylabel: bool,
) -> None:
    scaled = safe_scale(matrix, exponent)

    ax.set_xlim(-0.5, len(LANDCOVER_ORDER) - 0.5)
    ax.set_ylim(len(CLIMATE_ORDER) - 0.5, -0.5)
    ax.set_xticks(np.arange(len(LANDCOVER_ORDER)))
    ax.set_xticklabels(LANDCOVER_ORDER, fontsize=24, fontweight="normal")
    ax.set_yticks(np.arange(len(CLIMATE_ORDER)))
    ax.set_yticklabels(CLIMATE_ORDER, fontsize=24, fontweight="normal")
    ax.set_ylabel("Climate zone" if show_ylabel else "", fontsize=24)
    ax.set_title(panel_label, fontsize=28, pad=12)
    ax.set_box_aspect(len(CLIMATE_ORDER) / len(LANDCOVER_ORDER))

    ax.set_xticks(np.arange(-0.5, len(LANDCOVER_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CLIMATE_ORDER), 1), minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                ax.text(j, i, "NA", ha="center", va="center", fontsize=19, color="#000000")
                continue

            display = scaled[i, j]
            face_color = cmap(norm(display))
            edge_color = face_color

            ax.scatter(
                j,
                i,
                s=UNIFORM_BUBBLE_SIZE,
                c=[face_color],
                edgecolors=edge_color,
                linewidths=0.9,
                zorder=3,
            )

            ax.text(
                j,
                i,
                f"{display:.2f}",
                ha="center",
                va="center",
                fontsize=20,
                color="#111111",
                fontweight="normal",
                zorder=4,
            )

    for spine in ax.spines.values():
        spine.set_visible(False)


def build_figure(summary: pd.DataFrame, dataset: dict[str, object]) -> plt.Figure:
    face_raw = build_matrix(summary, "face", "overall_mean")
    mix_raw = build_matrix(summary, "mix", "overall_mean")

    all_finite_raw = np.concatenate(
        [
            face_raw[np.isfinite(face_raw)],
            mix_raw[np.isfinite(mix_raw)],
        ]
    )
    exponent = choose_exponent(all_finite_raw)

    face_matrix = safe_scale(face_raw, exponent)
    mix_matrix = safe_scale(mix_raw, exponent)

    all_finite = np.concatenate(
        [
            face_matrix[np.isfinite(face_matrix)],
            mix_matrix[np.isfinite(mix_matrix)],
        ]
    )
    vmin = 0.0
    vmax = float(np.nanmax(all_finite)) if all_finite.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    shared_norm = Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(11.8, 7.8), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.16)
    ax_face = fig.add_subplot(gs[0, 0])
    ax_mix = fig.add_subplot(gs[0, 1])

    draw_bubble_matrix(
        ax_face,
        face_raw,
        cmap=INTERFACE_CMAP,
        norm=shared_norm,
        panel_label="Interface",
        exponent=exponent,
        vmax_scaled=vmax,
        show_ylabel=True,
    )
    draw_bubble_matrix(
        ax_mix,
        mix_raw,
        cmap=INTERMIX_CMAP,
        norm=shared_norm,
        panel_label="Intermix",
        exponent=exponent,
        vmax_scaled=vmax,
        show_ylabel=False,
    )

    fig.subplots_adjust(top=0.88, bottom=0.31)
    fig.text(0.5, 0.28, "Land cover", ha="center", va="center", fontsize=24)

    cax_face = fig.add_axes([0.10, 0.22, 0.31, 0.035])
    cax_mix = fig.add_axes([0.53, 0.22, 0.31, 0.035])

    sm_face = ScalarMappable(norm=shared_norm, cmap=INTERFACE_CMAP)
    sm_face.set_array([])
    cbar_face = fig.colorbar(sm_face, cax=cax_face, orientation="horizontal")
    cbar_face.locator = MaxNLocator(nbins=4)
    cbar_face.update_ticks()
    cbar_face.ax.tick_params(labelsize=20)
    cbar_face.outline.set_visible(False)

    sm_mix = ScalarMappable(norm=shared_norm, cmap=INTERMIX_CMAP)
    sm_mix.set_array([])
    cbar_mix = fig.colorbar(sm_mix, cax=cax_mix, orientation="horizontal")
    cbar_mix.locator = MaxNLocator(nbins=4)
    cbar_mix.update_ticks()
    cbar_mix.ax.tick_params(labelsize=20)
    cbar_mix.outline.set_visible(False)

    fig.text(0.5, 0.145, str(dataset["colorbar_label"]), ha="center", va="center", fontsize=28)
    fig.text(0.85, 0.2175, f"×1e{exponent:+d}", ha="left", va="center", fontsize=22)
    return fig


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        print(f"\n{'=' * 80}")
        print(f"Processing: {dataset['metric_label']}")
        summary = load_partition_summary(dataset)
        fig = build_figure(summary, dataset)
        output_path = OUTPUT_DIR / str(dataset["output_filename"])
        fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
