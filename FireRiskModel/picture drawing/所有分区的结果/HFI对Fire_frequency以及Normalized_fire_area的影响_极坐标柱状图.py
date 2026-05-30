#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称:
    HFI 对 Fire frequency 与 Normalized burned area 影响的分组环形图

功能简介:
    1. 读取以下目录中的 HFI 净效应固定效应结果 CSV:
       - B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num
       - B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area
    2. 以每个 CSV 代表一个分类，先按 Year 计算 diff_contribution 的年均值，
       再对 2003-2022 年的年均值求跨年平均和标准误。
    3. 将分类按前两个字段分成 8 组:
       - face_A, face_B, face_C, face_D, mix_A, mix_B, mix_C, mix_D
       每组内部按最后一个字段固定顺序绘制:
       - CRP, FST, GRS, SHR, WET
    4. 分别绘制两个单轨环形分组图:
       - Human Footprint effect on Fire frequency
       - Human Footprint effect on Normalized burned area
       蓝色表示负效应, 暖色表示正效应。

输出:
    - B:\WUI\Picture\Ranking_Human Footprint_effect_circular_grouped_Fire_frequency.png
    - B:\WUI\Picture\Ranking_Human Footprint_effect_circular_grouped_Normalized_burn_area.png
"""

from __future__ import annotations

import glob
import math
import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIRE_NUM_DIR = r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num"
FIRE_AREA_DIR = r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area"

OUTPUT_DIR = r"B:\WUI\Picture"
OUTPUT_PNG_FIRE_NUM = "Ranking_Human Footprint_effect_circular_grouped_Fire_frequency.png"
OUTPUT_PNG_FIRE_AREA = "Ranking_Human Footprint_effect_circular_grouped_Normalized_burn_area.png"

YEAR_START = 2003
YEAR_END = 2022

GROUP_ORDER = [
    "face_A",
    "face_B",
    "face_C",
    "face_D",
    "mix_A",
    "mix_B",
    "mix_C",
    "mix_D",
]
LANDCOVER_ORDER = ["CRP", "FST", "GRS", "SHR", "WET"]

plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 24

FIGURE_FACE = "#ffffff"
AXIS_FACE = "#ffffff"
GROUP_BAND_FACE = "#ffffff"
GROUP_LINE_COLOR = "#1f1f1f"
BASELINE_COLOR = "#bfc8ce"
CONNECTOR_COLOR = "#5f848e"
MID_GROUP_FRAME_COLOR = "#8f9aa3"
NEG_COLOR = "#2b83ba"
POS_COLOR = "#ef8a62"
INNER_TRACK_FILL = "#f5f7f8"
OUTER_TRACK_FILL = "#fcf5ef"


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


def coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    values = series.astype(str).str.strip().str.upper()
    return values.isin({"TRUE", "T", "1", "YES", "Y"})


def color_for_value(value: float) -> str:
    return POS_COLOR if value >= 0 else NEG_COLOR


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


def summarize_metric(input_dir: str, response_name: str) -> pd.DataFrame:
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

        df = pd.read_csv(csv_path, usecols=["Year", "diff_contribution", "delta_sig"], low_memory=False)
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

        category_id = extract_category_id(csv_path)
        group_id, landcover = split_category_id(category_id)
        rows.append(
            {
                "category_id": category_id,
                "group_id": group_id,
                "landcover": landcover,
                "response": response_name,
                "mean_effect": mean_effect,
                "se_effect": se_effect,
                "significant_share": float(df["delta_sig"].mean()),
                "n_years": n_years,
                "n_rows": int(len(df)),
                "source_csv": os.path.basename(csv_path),
            }
        )

    if not rows:
        raise ValueError(f"No valid summary rows built from: {input_dir}")
    return pd.DataFrame(rows)


def prepare_metric_summary(input_dir: str, response_name: str) -> pd.DataFrame:
    metric = summarize_metric(input_dir, response_name)
    template = pd.MultiIndex.from_product(
        [GROUP_ORDER, LANDCOVER_ORDER],
        names=["group_id", "landcover"],
    ).to_frame(index=False)
    combined = template.merge(
        metric[
            [
                "group_id",
                "landcover",
                "category_id",
                "mean_effect",
                "se_effect",
                "significant_share",
                "n_years",
                "n_rows",
                "source_csv",
            ]
        ],
        on=["group_id", "landcover"],
        how="left",
    )
    combined["slot_label"] = combined["landcover"]
    combined["slot_index"] = np.arange(len(combined))
    return combined


def radial_text_rotation(theta: float) -> float:
    rotation = 90.0 - np.degrees(theta)
    if rotation > 90:
        rotation -= 180
    if rotation < -90:
        rotation += 180
    return rotation


def tangential_text_rotation(theta: float) -> float:
    rotation = -np.degrees(theta)
    while rotation > 90:
        rotation -= 180
    while rotation < -90:
        rotation += 180
    return rotation


def format_scale_value(value: float) -> str:
    if abs(value) < 1e-12:
        return "0"
    if abs(value) < 0.1:
        return f"{value:.1e}"
    return f"{value:.2f}"


def add_track_background(
    ax: plt.Axes,
    thetas: np.ndarray,
    widths: np.ndarray,
    bottom: float,
    height: float,
    color: str,
) -> None:
    ax.bar(
        thetas,
        height=np.full_like(thetas, height),
        width=widths,
        bottom=np.full_like(thetas, bottom),
        color=color,
        edgecolor="none",
        align="center",
        zorder=0,
    )


def build_circular_figure(combined: pd.DataFrame, *, metric_label: str) -> plt.Figure:
    n_groups = len(GROUP_ORDER)
    n_landcovers = len(LANDCOVER_ORDER)
    gap_slots = 1.55
    total_units = n_groups * n_landcovers + n_groups * gap_slots
    unit_angle = 2 * np.pi / total_units
    slot_width = unit_angle * 0.80

    theta_map: dict[tuple[str, str], float] = {}
    group_bounds: dict[str, tuple[float, float]] = {}

    cursor = 0.0
    for group in GROUP_ORDER:
        group_start = cursor * unit_angle
        for landcover in LANDCOVER_ORDER:
            theta_map[(group, landcover)] = (cursor + 0.5) * unit_angle
            cursor += 1.0
        group_end = cursor * unit_angle
        group_bounds[group] = (group_start, group_end)
        cursor += gap_slots

    combined = combined.copy()
    combined["theta"] = [theta_map[(g, lc)] for g, lc in zip(combined["group_id"], combined["landcover"])]
    combined["width"] = slot_width

    values = combined["mean_effect"].dropna().to_numpy(dtype=float)
    if values.size == 0:
        min_val, max_val = -1.0, 1.0
    else:
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        if min_val >= 0:
            min_val = 0.0
        if max_val <= 0:
            max_val = 0.0
        if np.isclose(min_val, max_val):
            if np.isclose(max_val, 0.0):
                min_val, max_val = -1.0, 1.0
            else:
                span_pad = abs(max_val) * 0.15
                min_val -= span_pad
                max_val += span_pad

    track_base = 1.42
    track_half_height = 0.79
    connector_inner = 0.58
    connector_outer = 0.84
    group_band_bottom = 2.18
    group_band_height = 0.31
    outer_label_radius = 2.74

    fig = plt.figure(figsize=(14.8, 14.8), facecolor=FIGURE_FACE)
    ax = plt.subplot(111, projection="polar")
    ax.set_facecolor(AXIS_FACE)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 3.30)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)

    track_bottom = track_base - track_half_height
    track_top = track_base + track_half_height
    track_span = track_top - track_bottom

    def value_to_radius(value: float) -> float:
        if np.isclose(max_val, min_val):
            return track_base
        return track_bottom + ((value - min_val) / (max_val - min_val)) * track_span

    zero_radius = value_to_radius(0.0)

    valid_background = combined["mean_effect"].notna()
    add_track_background(
        ax,
        combined.loc[valid_background, "theta"].to_numpy(),
        combined.loc[valid_background, "width"].to_numpy(),
        track_bottom,
        track_span,
        INNER_TRACK_FILL,
    )

    theta_line = np.linspace(0, 2 * np.pi, 720)
    ax.plot(theta_line, np.full_like(theta_line, zero_radius), color=BASELINE_COLOR, linewidth=0.95, zorder=1)

    mid_group_bottom = track_bottom - 0.03
    mid_group_height = track_span + 0.06
    for group in GROUP_ORDER:
        start_theta, end_theta = group_bounds[group]
        center_theta = (start_theta + end_theta) / 2
        width = end_theta - start_theta - unit_angle * 0.24
        ax.bar(
            center_theta,
            height=mid_group_height,
            width=width,
            bottom=mid_group_bottom,
            color="none",
            edgecolor=MID_GROUP_FRAME_COLOR,
            linewidth=1.15,
            align="center",
            zorder=1.6,
        )
        ax.bar(
            center_theta,
            height=group_band_height,
            width=width,
            bottom=group_band_bottom,
            color=GROUP_BAND_FACE,
            edgecolor=GROUP_LINE_COLOR,
            linewidth=0.85,
            align="center",
            zorder=2,
        )
        rot = tangential_text_rotation(center_theta)
        ax.text(
            center_theta,
            group_band_bottom + group_band_height / 2,
            group,
            rotation=rot,
            rotation_mode="anchor",
            ha="center",
            va="center",
            fontsize=32,
            color="#000000",
            zorder=3,
        )

    for _, row in combined.iterrows():
        theta = float(row["theta"])

        if pd.notna(row["mean_effect"]):
            value = float(row["mean_effect"])
            r_end = value_to_radius(value)
            ax.bar(
                theta,
                height=max(abs(r_end - zero_radius), 0.0001),
                width=slot_width * 0.48,
                bottom=min(zero_radius, r_end),
                color=color_for_value(value),
                edgecolor="none",
                alpha=0.95,
                zorder=4,
            )
            ax.scatter(
                theta,
                r_end,
                s=24,
                color=color_for_value(value),
                edgecolors="white",
                linewidths=0.55,
                zorder=5,
            )

            rot = radial_text_rotation(theta)
            ax.text(
                theta,
                outer_label_radius,
                row["slot_label"],
                rotation=rot,
                rotation_mode="anchor",
                ha="center",
                va="center",
                fontsize=24,
                color="#000000",
                zorder=6,
            )

    axis_theta = group_bounds[GROUP_ORDER[0]][0] - unit_angle * 1.05
    tick_half_width = unit_angle * 0.16
    axis_label_theta = axis_theta - unit_angle * 0.38
    axis_values = [min_val, 0.0, max_val]
    deduped_axis_values: list[float] = []
    for val in axis_values:
        if not deduped_axis_values or not np.isclose(val, deduped_axis_values[-1]):
            deduped_axis_values.append(val)

    axis_label_rot = radial_text_rotation(axis_theta)
    for val in deduped_axis_values:
        radius = value_to_radius(val)
        ax.plot(
            [axis_theta - tick_half_width, axis_theta + tick_half_width],
            [radius, radius],
            color="#555555",
            linewidth=0.9,
            zorder=11,
        )
        ax.text(
            axis_label_theta,
            radius,
            format_scale_value(val),
            rotation=axis_label_rot,
            rotation_mode="anchor",
            ha="center",
            va="center",
            fontsize=24,
            color="#555555",
            zorder=12,
        )

    ax.bar(
        0,
        height=connector_inner - 0.05,
        width=2 * np.pi,
        bottom=0,
        color="white",
        edgecolor="white",
        zorder=10,
    )
    fig.text(
        0.5,
        0.92,
        metric_label,
        ha="center",
        va="top",
        fontsize=36,
        fontweight="bold",
        color="#3c3c3c",
    )

    fig.subplots_adjust(top=0.94, bottom=0.04, left=0.03, right=0.97)
    return fig


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fire_num = prepare_metric_summary(FIRE_NUM_DIR, "Fire frequency")
    fire_area = prepare_metric_summary(FIRE_AREA_DIR, "Normalized burned area")

    fig = build_circular_figure(fire_num, metric_label="Fire frequency")
    output_png_path = save_with_fallback_figure(fig, os.path.join(OUTPUT_DIR, OUTPUT_PNG_FIRE_NUM))
    plt.close(fig)
    print(f"Figure saved to: {output_png_path}")

    fig = build_circular_figure(fire_area, metric_label="Normalized burned area")
    output_png_path = save_with_fallback_figure(fig, os.path.join(OUTPUT_DIR, OUTPUT_PNG_FIRE_AREA))
    plt.close(fig)
    print(f"Figure saved to: {output_png_path}")


if __name__ == "__main__":
    main()


