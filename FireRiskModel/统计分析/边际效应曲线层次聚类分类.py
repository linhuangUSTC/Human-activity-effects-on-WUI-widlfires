#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称: HFI 边际效应曲线层次聚类分类（Fire frequency / Normalized burn area）

功能:
    1) 分别读取 Fire num 与 Fire area 的 HFI 净效应固定效应结果 CSV
    2) 基于 HFI 与 diff_contribution 重建边际效应曲线
    3) 使用层次聚类 + 相关距离，并固定切分为 3 类
    4) 对两个数据集分别输出分类结果表、聚类树、各自独立的聚类热图、以及 2x2 类型总览图

输入:
    - CSV:
      B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num
      B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area
    - 参考图片目录（不直接用于数值分类，仅用于对应检查）:
      B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\边际效应曲线以及95%误差带\Fire num
      B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\边际效应曲线以及95%误差带\Fire area

输出:
    - B:\WUI\Picture\Human Footprint_effect_curve_*.csv/png
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
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from statsmodels.nonparametric.smoothers_lowess import lowess


# =========================
# 参数配置
# =========================
CSV_ROOT = r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv"
CURVE_PNG_ROOT = r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\边际效应曲线以及95%误差带"
OUTPUT_ROOT = r"B:\WUI\Picture"

OUTPUT_ASSIGNMENTS_TEMPLATE = "Human Footprint_effect_curve_cluster_assignments_{dataset_slug}.csv"
OUTPUT_CENTROIDS_TEMPLATE = "Human Footprint_effect_curve_cluster_centroids_{dataset_slug}.csv"
OUTPUT_DENDROGRAM_TEMPLATE = "Human Footprint_effect_curve_dendrogram_{dataset_slug}.png"
OUTPUT_HEATMAP_TEMPLATE = "Human Footprint_effect_curve_cluster_heatmap_{dataset_slug}.png"
OUTPUT_TYPES_TEMPLATE = "Human Footprint_effect_curve_cluster_types_2x2_{dataset_slug}.png"

SHARED_CLUSTER_NAMES = {
    1: "Progressive inhibition",
    2: "Intermediate response",
    3: "Continuous amplification",
}

DATASETS = [
    {
        "key": "Fire num",
        "csv_dir": os.path.join(CSV_ROOT, "Fire num"),
        "curve_png_dir": os.path.join(CURVE_PNG_ROOT, "Fire num"),
        "output_dir": OUTPUT_ROOT,
        "heatmap_slug": "fire_frequency",
        "file_suffix_regex": r"fire_num_with_WUI_offset",
        "png_filename_template": "GLMM_NaturalSpline_NB_HFI_only_{category_id}_fire_num_with_WUI_offset_marginal_effect.png",
        "effect_ylabel": "Effect on Fire frequency",
        "title_metric": "Fire frequency",
        "cluster_names": SHARED_CLUSTER_NAMES,
    },
    {
        "key": "Fire area",
        "csv_dir": os.path.join(CSV_ROOT, "Fire area"),
        "curve_png_dir": os.path.join(CURVE_PNG_ROOT, "Fire area"),
        "output_dir": OUTPUT_ROOT,
        "heatmap_slug": "normalized_burned_area",
        "file_suffix_regex": r"fire_area",
        "png_filename_template": "GLMM_NaturalSpline_Ordbeta_HFI_only_{category_id}_fire_area_marginal_effect.png",
        "effect_ylabel": "Effect on Normalized burn area",
        "title_metric": "Normalized burn area",
        "cluster_names": SHARED_CLUSTER_NAMES,
    },
]

FIXED_K = 3
N_BINS = 80
N_GRID = 120
LOWESS_FRAC = 0.22


# =========================
# 绘图样式
# =========================
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 20
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20

FIGURE_FACE = "#ffffff"
GRID_COLOR = "#d7d1c7"
ZERO_LINE = "#7d7d7d"
HEATMAP_CMAP = sns.color_palette("vlag", as_cmap=True)
CLUSTER_COLORS = ["#2b8cbe", "#1b9e77", "#e67e22", "#c0392b", "#8e44ad"]


@dataclass
class CurveRecord:
    category_id: str
    mode: str
    climate_zone: str
    landcover: str
    hfi: np.ndarray
    diff: np.ndarray


@dataclass
class HeatmapPayload:
    dataset_slug: str
    title_metric: str
    heat_data_display: np.ndarray
    display_labels: list[str]
    cluster_spans: list[tuple[int, int, int]]
    cluster_breaks: list[int]
    cluster_name_map: dict[int, str]
    abs_clip: float


def extract_metadata(path: str) -> tuple[str, str, str, str]:
    name = Path(path).stem
    match = re.search(
        r"HFI_effect_((face|mix)_([A-D])_([A-Z]+))_(?:fire_num_with_WUI_offset|fire_area)$",
        name,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"Unrecognized file name pattern: {name}")
    category_id = match.group(1)
    mode = match.group(2)
    climate_zone = match.group(3)
    landcover = match.group(4)
    return category_id, mode, climate_zone, landcover


def format_effect(value: float, _: int | None = None) -> str:
    if abs(value) < 1e-12:
        return "0"
    if abs(value) < 0.1:
        return f"{value:.1e}"
    return f"{value:.2f}"


def load_curve_records(csv_dir: str) -> list[CurveRecord]:
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "HFI_effect_*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {csv_dir}")

    records: list[CurveRecord] = []
    for csv_path in csv_files:
        category_id, mode, climate_zone, landcover = extract_metadata(csv_path)
        df = pd.read_csv(csv_path, usecols=["HFI", "diff_contribution"], low_memory=False)
        hfi = pd.to_numeric(df["HFI"], errors="coerce").to_numpy(dtype=np.float32)
        diff = pd.to_numeric(df["diff_contribution"], errors="coerce").to_numpy(dtype=np.float32)
        valid = np.isfinite(hfi) & np.isfinite(diff)
        hfi = hfi[valid]
        diff = diff[valid]
        if hfi.size < 10:
            continue
        order = np.argsort(hfi)
        records.append(
            CurveRecord(
                category_id=category_id,
                mode=mode,
                climate_zone=climate_zone,
                landcover=landcover,
                hfi=hfi[order],
                diff=diff[order],
            )
        )

    if not records:
        raise ValueError("No valid curve records loaded from CSV files.")

    print(f"Loaded {len(records)} curve records from {csv_dir}")
    return records


def build_smoothed_curve(
    hfi: np.ndarray,
    diff: np.ndarray,
    *,
    bin_edges: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    bin_idx = np.digitize(hfi, bin_edges) - 1
    valid_bin = (bin_idx >= 0) & (bin_idx < len(bin_edges) - 1)
    bin_idx = bin_idx[valid_bin]
    hfi = hfi[valid_bin]
    diff = diff[valid_bin]

    counts = np.bincount(bin_idx, minlength=len(bin_edges) - 1)
    sum_hfi = np.bincount(bin_idx, weights=hfi, minlength=len(bin_edges) - 1)
    sum_diff = np.bincount(bin_idx, weights=diff, minlength=len(bin_edges) - 1)
    mask = counts > 0

    if mask.sum() == 0:
        return np.zeros_like(grid, dtype=float)

    x = sum_hfi[mask] / counts[mask]
    y = sum_diff[mask] / counts[mask]

    if len(x) >= 4:
        y_smooth = lowess(y, x, frac=LOWESS_FRAC, return_sorted=False)
    else:
        y_smooth = y

    curve = np.interp(grid, x, y_smooth, left=y_smooth[0], right=y_smooth[-1])
    return curve.astype(float)


def standardize_curve(curve: np.ndarray) -> np.ndarray:
    std = float(curve.std(ddof=0))
    if std < 1e-12:
        return np.zeros_like(curve)
    return (curve - curve.mean()) / std


def robust_correlation_distance(x: np.ndarray, y: np.ndarray) -> float:
    x_std = float(x.std(ddof=0))
    y_std = float(y.std(ddof=0))

    if x_std < 1e-12 and y_std < 1e-12:
        return 0.0 if np.allclose(x, y, atol=1e-12) else 1.0
    if x_std < 1e-12 or y_std < 1e-12:
        return 1.0

    corr = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(corr):
        corr = 0.0
    return float(np.clip(1.0 - corr, 0.0, 2.0))


def compute_correlation_distance_matrix(curve_matrix_std: np.ndarray) -> np.ndarray:
    n = curve_matrix_std.shape[0]
    dist_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            dist = robust_correlation_distance(curve_matrix_std[i], curve_matrix_std[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix


def cluster_curves(curve_matrix_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dist_matrix = compute_correlation_distance_matrix(curve_matrix_std)
    condensed_dist = squareform(dist_matrix, checks=False)
    tree = linkage(condensed_dist, method="average", optimal_ordering=True)
    labels = fcluster(tree, t=FIXED_K, criterion="maxclust")
    print(f"Using fixed hierarchical clusters: k={FIXED_K}")
    return labels, tree

def reorder_clusters(labels: np.ndarray, curve_matrix_raw: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    old_clusters = sorted(np.unique(labels).tolist())
    old_centroids = {old: curve_matrix_raw[labels == old].mean(axis=0) for old in old_clusters}
    ordered_old = sorted(old_clusters, key=lambda old: (old_centroids[old].mean(), old_centroids[old].max()))
    remap = {old: new for new, old in enumerate(ordered_old, start=1)}
    new_labels = np.array([remap[int(label)] for label in labels], dtype=int)
    new_centroids = {remap[old]: old_centroids[old] for old in ordered_old}
    return new_labels, new_centroids


def slugify_filename_part(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_")


def prepare_heatmap_payload(
    records: list[CurveRecord],
    *,
    dataset_slug: str,
    title_metric: str,
    curve_matrix_raw: np.ndarray,
    curve_matrix_std: np.ndarray,
    labels: np.ndarray,
    cluster_name_map: dict[int, str],
) -> HeatmapPayload:
    order = sorted(
        range(len(records)),
        key=lambda i: (labels[i], curve_matrix_raw[i].mean(), curve_matrix_raw[i].max()),
    )
    heat_data = curve_matrix_std[order, :]
    abs_clip = float(np.percentile(np.abs(heat_data), 97))
    if not np.isfinite(abs_clip) or abs_clip <= 0:
        abs_clip = float(np.nanmax(np.abs(heat_data)))
    if not np.isfinite(abs_clip) or abs_clip <= 0:
        abs_clip = 1.0

    display_rows: list[np.ndarray] = []
    display_labels: list[str] = []
    cluster_spans: list[tuple[int, int, int]] = []
    cluster_breaks = []
    prev_cluster = None
    current_cluster_start = None
    for _ in range(1):
        display_rows.append(np.full(heat_data.shape[1], np.nan, dtype=float))
        display_labels.append("")
    for idx in order:
        current_cluster = int(labels[idx])
        if prev_cluster is None:
            current_cluster_start = len(display_rows)
        elif current_cluster != prev_cluster:
            cluster_spans.append((prev_cluster, current_cluster_start, len(display_rows) - 1))
            cluster_breaks.append(len(display_rows))
            for _ in range(2):
                display_rows.append(np.full(heat_data.shape[1], np.nan, dtype=float))
                display_labels.append("")
            current_cluster_start = len(display_rows)
        display_rows.append(curve_matrix_std[idx].astype(float))
        display_labels.append(records[idx].category_id)
        prev_cluster = current_cluster
    if prev_cluster is not None and current_cluster_start is not None:
        cluster_spans.append((prev_cluster, current_cluster_start, len(display_rows) - 1))

    return HeatmapPayload(
        dataset_slug=dataset_slug,
        title_metric=title_metric,
        heat_data_display=np.vstack(display_rows),
        display_labels=display_labels,
        cluster_spans=cluster_spans,
        cluster_breaks=cluster_breaks,
        cluster_name_map=cluster_name_map,
        abs_clip=abs_clip,
    )


def save_individual_heatmap(payload: HeatmapPayload, output_dir: str) -> None:
    heatmap_cmap = HEATMAP_CMAP.copy()
    heatmap_cmap.set_bad("#ffffff")

    abs_clip = payload.abs_clip
    if not np.isfinite(abs_clip) or abs_clip <= 0:
        abs_clip = 1.0

    fig = plt.figure(figsize=(29.184, 42), facecolor=FIGURE_FACE)
    outer_grid = fig.add_gridspec(
        1,
        4,
        width_ratios=[0.08, 1.48, 0.025, 0.055],
        wspace=0.03,
    )
    ax = fig.add_subplot(outer_grid[0, 1])
    cbar_ax = fig.add_subplot(outer_grid[0, 3])

    sns.heatmap(
        payload.heat_data_display,
        ax=ax,
        cmap=heatmap_cmap,
        center=0,
        vmin=-abs_clip,
        vmax=abs_clip,
        cbar=True,
        cbar_ax=cbar_ax,
        cbar_kws={
            "label": "Standardized relative effect",
            "shrink": 0.92,
            "pad": 0.02,
        },
        xticklabels=False,
        yticklabels=payload.display_labels,
        linewidths=0,
    )
    ax.set_title(payload.title_metric, fontsize=68, pad=16)
    ax.set_xlabel("Common Human Footprint grid", fontsize=58, labelpad=14)
    ax.set_ylabel("Partition ID", fontsize=66, labelpad=12)
    ax.tick_params(axis="y", labelsize=52, length=0)
    for br in payload.cluster_breaks:
        ax.hlines(br, *ax.get_xlim(), colors="white", linewidth=2.2)
    label_x = payload.heat_data_display.shape[1] / 2.0
    for cluster_id, start_row, _end_row in payload.cluster_spans:
        if start_row <= 1:
            label_y = start_row - 0.35
        else:
            label_y = start_row - 1.1
        ax.text(
            label_x,
            label_y,
            payload.cluster_name_map[cluster_id],
            ha="center",
            va="center",
            fontsize=56,
            color="black",
            clip_on=False,
        )

    colorbar = ax.collections[0].colorbar if ax.collections else None
    if colorbar is not None:
        colorbar.ax.tick_params(labelsize=60)
        colorbar.set_label("Standardized relative effect", size=62)

    fig.subplots_adjust(left=0.095, right=0.955, top=0.965, bottom=0.07)
    heatmap_path = os.path.join(output_dir, OUTPUT_HEATMAP_TEMPLATE.format(dataset_slug=payload.dataset_slug))
    fig.savefig(heatmap_path, dpi=400, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Heatmap figure saved to: {heatmap_path}")


def build_outputs(
    records: list[CurveRecord],
    *,
    dataset_label: str,
    heatmap_slug: str,
    output_dir: str,
    curve_png_dir: str,
    png_filename_template: str,
    effect_ylabel: str,
    title_metric: str,
    cluster_names: dict[int, str],
    grid: np.ndarray,
    curve_matrix_raw: np.ndarray,
    curve_matrix_std: np.ndarray,
    labels: np.ndarray,
    centroids: dict[int, np.ndarray],
    dendro_tree: np.ndarray,
) -> HeatmapPayload:
    os.makedirs(output_dir, exist_ok=True)
    dataset_slug = slugify_filename_part(dataset_label)
    cluster_ids = sorted(centroids.keys())
    cluster_name_map = {cluster_id: cluster_names.get(cluster_id, f"Type {cluster_id}") for cluster_id in cluster_ids}
    heatmap_payload = prepare_heatmap_payload(
        records,
        dataset_slug=slugify_filename_part(str(heatmap_slug)),
        title_metric=title_metric,
        curve_matrix_raw=curve_matrix_raw,
        curve_matrix_std=curve_matrix_std,
        labels=labels,
        cluster_name_map=cluster_name_map,
    )

    assignments_rows: list[dict[str, object]] = []
    for idx, rec in enumerate(records):
        curve = curve_matrix_raw[idx]
        peak_idx = int(np.argmax(curve))
        trough_idx = int(np.argmin(curve))
        assignments_rows.append(
            {
                "category_id": rec.category_id,
                "mode": rec.mode,
                "climate_zone": rec.climate_zone,
                "landcover": rec.landcover,
                "cluster": int(labels[idx]),
                "cluster_label": cluster_name_map[int(labels[idx])],
                "mean_curve_effect": float(curve.mean()),
                "peak_effect": float(curve[peak_idx]),
                "peak_hfi": float(grid[peak_idx]),
                "trough_effect": float(curve[trough_idx]),
                "trough_hfi": float(grid[trough_idx]),
                "effect_at_min_hfi": float(curve[0]),
                "effect_at_max_hfi": float(curve[-1]),
                "source_png_exists": int(
                    Path(curve_png_dir, rec.mode, png_filename_template.format(category_id=rec.category_id)).exists()
                ),
            }
        )

    assignments = pd.DataFrame(assignments_rows).sort_values(["cluster", "mean_curve_effect", "category_id"])
    assignments.rename(
        columns={
            "peak_hfi": "peak_Human Footprint",
            "trough_hfi": "trough_Human Footprint",
            "effect_at_min_hfi": "effect_at_min_Human Footprint",
            "effect_at_max_hfi": "effect_at_max_Human Footprint",
        }
    ).to_csv(
        os.path.join(output_dir, OUTPUT_ASSIGNMENTS_TEMPLATE.format(dataset_slug=dataset_slug)),
        index=False,
        encoding="utf-8-sig",
    )

    centroid_rows: list[dict[str, float]] = []
    for cluster_id, centroid in centroids.items():
        for hfi_value, effect_value in zip(grid, centroid):
            centroid_rows.append(
                {
                    "cluster": cluster_id,
                    "HFI": float(hfi_value),
                    "centroid_effect": float(effect_value),
                }
            )
    pd.DataFrame(centroid_rows).rename(columns={"HFI": "Human Footprint"}).to_csv(
        os.path.join(output_dir, OUTPUT_CENTROIDS_TEMPLATE.format(dataset_slug=dataset_slug)),
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------
    # Dendrogram figure
    # -------------------------
    fig_d, ax_d = plt.subplots(figsize=(18, 10), facecolor=FIGURE_FACE)
    dendrogram(
        dendro_tree,
        labels=[records[i].category_id for i in range(len(records))],
        leaf_rotation=90,
        leaf_font_size=9,
        color_threshold=None,
        ax=ax_d,
    )
    ax_d.set_xlabel("Category ID", fontsize=18)
    ax_d.set_ylabel("Correlation Distance", fontsize=18)
    ax_d.tick_params(axis="x", labelsize=9)
    ax_d.tick_params(axis="y", labelsize=14)
    dendrogram_path = os.path.join(output_dir, OUTPUT_DENDROGRAM_TEMPLATE.format(dataset_slug=dataset_slug))
    fig_d.savefig(dendrogram_path, dpi=1200, bbox_inches="tight", facecolor=fig_d.get_facecolor())
    plt.close(fig_d)
    print(f"Dendrogram figure saved to: {dendrogram_path}")

    # -------------------------
    # Type figure
    # -------------------------
    n_clusters = len(cluster_ids)
    n_cols = 2
    n_rows = max(1, (n_clusters + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5.25 * n_rows), facecolor=FIGURE_FACE)
    axes = np.atleast_1d(axes).flatten()

    for plot_idx, cluster_id in enumerate(cluster_ids):
        ax = axes[plot_idx]
        ax.set_facecolor("white")
        cluster_color = CLUSTER_COLORS[(cluster_id - 1) % len(CLUSTER_COLORS)]
        member_idx = np.where(labels == cluster_id)[0]
        cluster_curves = curve_matrix_raw[member_idx, :]
        centroid = centroids[cluster_id]
        spread = cluster_curves.std(axis=0, ddof=0) if len(member_idx) > 1 else np.zeros_like(centroid)

        for idx in member_idx:
            ax.plot(grid, curve_matrix_raw[idx], color=cluster_color, alpha=0.16, linewidth=1.0)
        ax.fill_between(
            grid,
            centroid - spread,
            centroid + spread,
            color=cluster_color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(grid, centroid, color=cluster_color, linewidth=3.0)
        ax.axhline(0, color=ZERO_LINE, linestyle="--", linewidth=0.9)
        ax.grid(True, axis="both", color=GRID_COLOR, alpha=0.45, linewidth=0.7)
        ax.set_title(cluster_name_map[cluster_id], fontsize=20, fontweight="bold", pad=8)
        ax.text(
            0.02,
            0.94,
            f"Cluster {cluster_id} | n = {len(member_idx)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=20,
            color="#4b4b4b",
        )
        ax.set_xlabel("Human Footprint", fontsize=18)
        ax.set_ylabel(effect_ylabel, fontsize=18)
        ax.tick_params(axis="both", labelsize=20)
        ax.yaxis.set_major_formatter(FuncFormatter(format_effect))

    for extra_ax in axes[len(cluster_ids):]:
        extra_ax.axis("off")

    fig.suptitle(f"Human Footprint Curve Types: {title_metric} (k = {FIXED_K})", fontsize=20, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.08, top=0.92, hspace=0.28, wspace=0.18)

    type_fig_path = os.path.join(
        output_dir,
        OUTPUT_TYPES_TEMPLATE.format(dataset_slug=dataset_slug),
    )
    fig.savefig(type_fig_path, dpi=1200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Type figure saved to: {type_fig_path}")
    return heatmap_payload


def process_dataset(config: dict[str, str]) -> HeatmapPayload:
    print(f"\n=== Processing {config['key']} ===")
    records = load_curve_records(config["csv_dir"])
    global_hfi_min = min(float(rec.hfi.min()) for rec in records)
    global_hfi_max = max(float(rec.hfi.max()) for rec in records)
    bin_edges = np.linspace(global_hfi_min, global_hfi_max, N_BINS + 1)
    grid = np.linspace(global_hfi_min, global_hfi_max, N_GRID)

    curve_matrix_raw = np.vstack(
        [
            build_smoothed_curve(rec.hfi, rec.diff, bin_edges=bin_edges, grid=grid)
            for rec in records
        ]
    )
    curve_matrix_std = np.vstack([standardize_curve(curve) for curve in curve_matrix_raw])

    labels_raw, dendro_tree = cluster_curves(curve_matrix_std)
    labels, centroids = reorder_clusters(labels_raw, curve_matrix_raw)

    return build_outputs(
        records,
        dataset_label=config["key"],
        heatmap_slug=config["heatmap_slug"],
        output_dir=config["output_dir"],
        curve_png_dir=config["curve_png_dir"],
        png_filename_template=config["png_filename_template"],
        effect_ylabel=config["effect_ylabel"],
        title_metric=config["title_metric"],
        cluster_names=config["cluster_names"],
        grid=grid,
        curve_matrix_raw=curve_matrix_raw,
        curve_matrix_std=curve_matrix_std,
        labels=labels,
        centroids=centroids,
        dendro_tree=dendro_tree,
    )


def main() -> None:
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    heatmap_payloads: list[HeatmapPayload] = []
    for config in DATASETS:
        heatmap_payloads.append(process_dataset(config))
    for payload in heatmap_payloads:
        save_individual_heatmap(payload, OUTPUT_ROOT)


if __name__ == "__main__":
    main()
