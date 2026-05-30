#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称:
    使用 XGBoost 分别建模 Human Footprint 对 Fire frequency 与 Normalized burn area 的影响

功能简介:
    1. 分别从 HFI 固定效应结果 CSV 中读取 Human Footprint 对 Fire frequency / Normalized burn area 的影响(diff_contribution)。
    2. 从对应的 Basic 数据中读取 GDP、Population density、Road density、Cropland、Built environment、Pasture。
    3. 按文件名前缀分别汇总 Interface(face_*) 与 Intermix(mix_*) 样本。
    4. 对每个指标分别建立 Interface 与 Intermix 两套 XGBoost 回归模型。
    5. 输出模型表现、特征重要性、SHAP 结果、模型文件和诊断图。
"""

from __future__ import annotations

import gc
from io import BytesIO
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from shap.plots import colors as shap_colors
from xgboost import XGBRegressor


plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12

OUTPUT_DIR = Path(r"B:\WUI\Picture")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TUNED_PARAMETER_DIR = Path(
    r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\归因分析-XGBoost\XGBoost机制揭示\XGboost参数调优"
)

METRIC_CONFIGS = [
    {
        "name": "Fire frequency",
        "slug": "fire_frequency",
        "effect_dir": Path(r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num"),
        "basic_dir": Path(r"I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic"),
    },
    {
        "name": "Normalized burn area",
        "slug": "normalized_burn_area",
        "effect_dir": Path(r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area"),
        "basic_dir": Path(r"I:\Processing data\渔网分类建模\Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic"),
    },
]

TARGET_COLUMN = "diff_contribution"
BASIC_FEATURE_COLUMNS = ["GDP", "PopD", "Roadsd", "Cropland_Area_km2", "GUB_area_km2", "Pasture_Band2_Mean"]
FEATURE_COLUMNS = BASIC_FEATURE_COLUMNS.copy()
FEATURE_LABELS = {
    "GDP": "GDP",
    "PopD": "Population density",
    "Roadsd": "Road density",
    "Cropland_Area_km2": "Cropland",
    "GUB_area_km2": "Built environment",
    "Pasture_Band2_Mean": "Pasture",
}
TEST_SIZE = 0.20
RANDOM_STATE = 42
BASE_MODEL_CONFIG = {
    "objective": "reg:squarederror",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "tree_method": "hist",
    "eval_metric": "rmse",
}
INTEGER_PARAMETER_NAMES = {"n_estimators", "max_depth", "min_child_weight"}
MODEL_CONFIG_CACHE: dict[tuple[str, str], dict[str, object]] = {}
PREDICTION_PLOT_SAMPLE_SIZE = 50000
SHAP_SAMPLE_SIZE = 20000


def normalize_grid_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def extract_effect_category_id(csv_path: Path) -> str:
    stem = csv_path.stem
    prefix = "HFI_effect_"
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected effect filename: {csv_path.name}")
    return stem[len(prefix):]


def infer_wui_type(category_id: str) -> str:
    if category_id.startswith("face_"):
        return "Interface"
    if category_id.startswith("mix_"):
        return "Intermix"
    raise ValueError(f"Cannot infer WUI type from category_id: {category_id}")


def discover_file_pairs(metric_config: dict[str, object]) -> tuple[dict[str, list[tuple[str, Path, Path]]], pd.DataFrame]:
    effect_dir = Path(metric_config["effect_dir"])
    basic_dir = Path(metric_config["basic_dir"])
    effect_files = {extract_effect_category_id(path): path for path in sorted(effect_dir.glob("HFI_effect_*.csv"))}
    basic_files = {path.stem: path for path in sorted(basic_dir.glob("*.csv"))}

    matched_ids = sorted(set(effect_files).intersection(basic_files))
    missing_effect = sorted(set(basic_files).difference(effect_files))
    missing_basic = sorted(set(effect_files).difference(basic_files))

    if missing_effect:
        print(f"{metric_config['name']} basic-only categories excluded: {', '.join(missing_effect)}")
    if missing_basic:
        print(f"{metric_config['name']} effect-only categories excluded: {', '.join(missing_basic)}")

    pairs_by_type: dict[str, list[tuple[str, Path, Path]]] = {"Interface": [], "Intermix": []}
    inventory_rows: list[dict[str, object]] = []
    for category_id in matched_ids:
        wui_type = infer_wui_type(category_id)
        effect_path = effect_files[category_id]
        basic_path = basic_files[category_id]
        pairs_by_type[wui_type].append((category_id, effect_path, basic_path))
        inventory_rows.append(
            {
                "metric_name": metric_config["name"],
                "category_id": category_id,
                "WUI_Type": wui_type,
                "effect_csv": str(effect_path),
                "basic_csv": str(basic_path),
            }
        )

    inventory_df = pd.DataFrame(inventory_rows)
    print(
        f"{metric_config['name']} matched categories: "
        f"Interface={len(pairs_by_type['Interface'])}, "
        f"Intermix={len(pairs_by_type['Intermix'])}, "
        f"Total={len(matched_ids)}"
    )
    return pairs_by_type, inventory_df


def read_effect_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        usecols=["GRID_ID", "Year", "HFI", "diff_contribution"],
        low_memory=False,
    )
    df["GRID_ID"] = normalize_grid_id(df["GRID_ID"])
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["HFI_effect"] = pd.to_numeric(df["HFI"], errors="coerce").astype(np.float32)
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").astype(np.float32)
    df = df.dropna(subset=["GRID_ID", "Year", "HFI_effect", TARGET_COLUMN]).copy()
    return (
        df.groupby(["GRID_ID", "Year"], as_index=False)
        .agg(HFI_effect=("HFI_effect", "mean"), diff_contribution=(TARGET_COLUMN, "mean"))
    )


def read_basic_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        usecols=["GRID_ID", "Year", "HFI"] + BASIC_FEATURE_COLUMNS,
        low_memory=False,
    )
    df["GRID_ID"] = normalize_grid_id(df["GRID_ID"])
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    for column in BASIC_FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(np.float32)
    df["HFI_basic"] = pd.to_numeric(df["HFI"], errors="coerce").astype(np.float32)
    df = df.dropna(subset=["GRID_ID", "Year", "HFI_basic"] + BASIC_FEATURE_COLUMNS).copy()
    return (
        df.groupby(["GRID_ID", "Year"], as_index=False)
        .agg(**{column: (column, "mean") for column in BASIC_FEATURE_COLUMNS}, HFI_basic=("HFI_basic", "mean"))
    )


def build_dataset_for_wui_type(
    wui_type: str,
    pairs: list[tuple[str, Path, Path]],
    metric_config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_parts: list[pd.DataFrame] = []
    alignment_rows: list[dict[str, object]] = []

    for category_id, effect_path, basic_path in pairs:
        effect_df = read_effect_csv(effect_path)
        basic_df = read_basic_csv(basic_path)
        merged = effect_df.merge(basic_df, how="inner", on=["GRID_ID", "Year"], validate="one_to_one")
        if merged.empty:
            alignment_rows.append(
                {
                    "metric_name": metric_config["name"],
                    "category_id": category_id,
                    "WUI_Type": wui_type,
                    "effect_rows": len(effect_df),
                    "basic_rows": len(basic_df),
                    "matched_rows": 0,
                    "hfi_mean_abs_diff": np.nan,
                    "hfi_max_abs_diff": np.nan,
                }
            )
            continue

        hfi_abs_diff = (merged["HFI_effect"] - merged["HFI_basic"]).abs()
        alignment_rows.append(
            {
                "metric_name": metric_config["name"],
                "category_id": category_id,
                "WUI_Type": wui_type,
                "effect_rows": len(effect_df),
                "basic_rows": len(basic_df),
                "matched_rows": len(merged),
                "hfi_mean_abs_diff": float(hfi_abs_diff.mean()),
                "hfi_max_abs_diff": float(hfi_abs_diff.max()),
            }
        )

        data_parts.append(merged[FEATURE_COLUMNS + [TARGET_COLUMN]].astype(np.float32))
        del effect_df, basic_df, merged
        gc.collect()

    if not data_parts:
        raise ValueError(f"No matched samples available for {metric_config['name']} / {wui_type}.")

    dataset = pd.concat(data_parts, ignore_index=True)
    alignment_df = pd.DataFrame(alignment_rows)
    print(
        f"{metric_config['name']} / {wui_type}: categories={len(pairs)}, "
        f"samples={len(dataset):,}, target_mean={dataset[TARGET_COLUMN].mean():.6e}"
    )
    return dataset, alignment_df


def coerce_parameter_value(parameter: str, value: object) -> object:
    numeric_value = float(value)
    if parameter in INTEGER_PARAMETER_NAMES:
        return int(round(numeric_value))
    return numeric_value


def load_tuned_parameters(metric_config: dict[str, object], wui_type: str) -> dict[str, object]:
    cache_key = (str(metric_config["slug"]), wui_type)
    if cache_key in MODEL_CONFIG_CACHE:
        return MODEL_CONFIG_CACHE[cache_key]

    workbook_path = TUNED_PARAMETER_DIR / f"xgboost_{metric_config['slug']}_{wui_type.lower()}_hfi_effect_tuned_summary.xlsx"
    if not workbook_path.exists():
        raise FileNotFoundError(
            f"Tuned summary workbook not found for {metric_config['name']} / {wui_type}: {workbook_path}"
        )

    parameter_df = pd.read_excel(workbook_path, sheet_name="best_parameters")
    required_columns = {"parameter", "value"}
    if not required_columns.issubset(parameter_df.columns):
        raise ValueError(
            f"Workbook {workbook_path.name} best_parameters sheet is missing required columns: "
            f"{sorted(required_columns)}"
        )

    tuned_params: dict[str, object] = {}
    for _, row in parameter_df.iterrows():
        parameter = str(row["parameter"])
        tuned_params[parameter] = coerce_parameter_value(parameter, row["value"])

    model_config = BASE_MODEL_CONFIG.copy()
    model_config.update(tuned_params)
    MODEL_CONFIG_CACHE[cache_key] = model_config

    print(
        f"{metric_config['name']} / {wui_type}: tuned parameters loaded from {workbook_path.name}"
    )
    return model_config


def create_model(metric_config: dict[str, object], wui_type: str) -> XGBRegressor:
    return XGBRegressor(**load_tuned_parameters(metric_config, wui_type))


def get_booster_score(scores: dict[str, float], feature: str, idx: int) -> float:
    return float(scores.get(feature, scores.get(f"f{idx}", 0.0)))


def rename_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={column: FEATURE_LABELS.get(column, column) for column in df.columns})


def get_metric_display_name(metric_config: dict[str, object]) -> str:
    if str(metric_config["slug"]) == "normalized_burn_area":
        return "Normalized burned area"
    return str(metric_config["name"])


def add_wui_corner_label(ax: plt.Axes, wui_type: str, fontsize: float = 16) -> None:
    ax.text(
        0.985,
        0.04,
        wui_type,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=fontsize,
        color="#333333",
    )


def add_feature_value_colorbar(
    fig: plt.Figure,
    left: float = 0.3,
    right: float = 0.7,
    bottom: float = 0.026,
    height: float = 0.028,
    label_fontsize: float = 12,
    tick_fontsize: float = 11,
) -> None:
    cax = fig.add_axes([left, bottom, right - left, height])
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=shap_colors.red_blue)
    colorbar = fig.colorbar(mapper, cax=cax, orientation="horizontal")
    colorbar.set_ticks([0.0, 1.0])
    colorbar.set_ticklabels(["Low", "High"])
    colorbar.set_label("Feature value", fontsize=label_fontsize, labelpad=4)
    colorbar.ax.tick_params(labelsize=tick_fontsize, length=0, pad=2)


def build_feature_importance_table(model: XGBRegressor, metric_config: dict[str, object], wui_type: str) -> pd.DataFrame:
    booster = model.get_booster()
    gain_scores = booster.get_score(importance_type="gain")
    weight_scores = booster.get_score(importance_type="weight")
    cover_scores = booster.get_score(importance_type="cover")
    rows: list[dict[str, object]] = []
    for idx, feature in enumerate(FEATURE_COLUMNS):
        rows.append(
            {
                "metric_name": metric_config["name"],
                "WUI_Type": wui_type,
                "feature_column": feature,
                "feature_name": FEATURE_LABELS[feature],
                "importance_gain": get_booster_score(gain_scores, feature, idx),
                "importance_weight": get_booster_score(weight_scores, feature, idx),
                "importance_cover": get_booster_score(cover_scores, feature, idx),
                "feature_importances_": float(model.feature_importances_[idx]),
            }
        )
    importance_df = pd.DataFrame(rows).sort_values("importance_gain", ascending=False).reset_index(drop=True)
    total_gain = float(importance_df["importance_gain"].sum())
    if total_gain > 0:
        importance_df["relative_importance_pct"] = importance_df["importance_gain"] / total_gain * 100.0
    else:
        importance_df["relative_importance_pct"] = 0.0
    return importance_df


def figure_to_image(fig: plt.Figure) -> Image.Image:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    image.load()
    buffer.close()
    return image


def save_image(image: Image.Image, output_path: Path) -> Path:
    image.save(output_path)
    image.close()
    return output_path


def build_feature_importance_image(
    importance_df: pd.DataFrame,
    metric_config: dict[str, object],
    wui_type: str,
) -> Image.Image:
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    plot_df = importance_df.sort_values("importance_gain", ascending=True)
    bars = ax.barh(plot_df["feature_name"], plot_df["importance_gain"], color="#4C83B6", alpha=0.88)
    ax.set_xlabel("XGBoost importance (gain)")
    ax.set_ylabel("")
    ax.set_title(f"{metric_config['name']} - {wui_type} feature importance")
    max_gain = float(plot_df["importance_gain"].max()) if not plot_df.empty else 0.0
    for bar, pct in zip(bars, plot_df["relative_importance_pct"]):
        x = float(bar.get_width())
        y = float(bar.get_y() + bar.get_height() / 2.0)
        ax.text(
            x + max(max_gain * 0.02, 1e-8),
            y,
            f"{pct:.1f}%",
            va="center",
            ha="left",
            fontsize=11,
            color="#333333",
        )
    ax.set_xlim(0, max_gain * 1.22 if max_gain > 0 else 1.0)
    fig.tight_layout()
    return figure_to_image(fig)


def build_prediction_scatter_image(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_config: dict[str, object],
    wui_type: str,
    split_label: str,
    r2_value: float,
) -> Image.Image:
    if y_true.size > PREDICTION_PLOT_SAMPLE_SIZE:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(y_true.size, size=PREDICTION_PLOT_SAMPLE_SIZE, replace=False)
        y_true_plot = y_true[idx]
        y_pred_plot = y_pred[idx]
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred

    fig, ax = plt.subplots(figsize=(6.8, 6.3))
    ax.scatter(y_true_plot, y_pred_plot, s=8, alpha=0.28, color="#C85C5C", edgecolors="none")
    min_val = float(min(np.min(y_true_plot), np.min(y_pred_plot)))
    max_val = float(max(np.max(y_true_plot), np.max(y_pred_plot)))
    ax.plot([min_val, max_val], [min_val, max_val], color="#4A4A4A", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Observed HFI effect")
    ax.set_ylabel("Predicted HFI effect")
    ax.set_title(f"{metric_config['name']} - {wui_type} {split_label} prediction")
    ax.text(
        0.04,
        0.96,
        f"R² = {r2_value:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#B0B0B0", "alpha": 0.9},
    )
    fig.tight_layout()
    return figure_to_image(fig)


def compose_horizontal_figure(
    images: list[Image.Image],
    output_path: Path,
) -> Path:
    if len(images) != 2:
        raise ValueError(f"Expected 2 images for a 1x2 panel, got {len(images)}.")

    try:
        cell_width = max(image.width for image in images)
        cell_height = max(image.height for image in images)
        gap = 28
        padding = 24
        canvas_width = cell_width * 2 + gap * 3
        canvas_height = cell_height + gap * 2
        canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")

        positions = [
            (gap, gap),
            (cell_width + gap * 2, gap),
        ]
        for image, (x0, y0) in zip(images, positions):
            x = x0 + (cell_width - image.width) // 2
            y = y0 + (cell_height - image.height) // 2
            canvas.paste(image, (x, y))

        canvas = canvas.crop((0, 0, canvas_width - padding, canvas_height - padding))
        canvas.save(output_path)
    finally:
        for image in images:
            image.close()
    return output_path


def load_image_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_name in ["arial.ttf", "Arial.ttf"]:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def compose_prediction_grid_figure(
    prediction_images_by_wui: dict[str, dict[str, Image.Image]],
    output_path: Path,
) -> Path:
    wui_order = ["Interface", "Intermix"]
    split_order = ["Train", "Test"]
    images = [
        prediction_images_by_wui[wui_type][split_label]
        for wui_type in wui_order
        for split_label in split_order
    ]
    if len(images) != 4:
        raise ValueError(f"Expected 4 images for a 2x2 panel, got {len(images)}.")

    try:
        cell_width = max(image.width for image in images)
        cell_height = max(image.height for image in images)
        left_label_width = 170
        top_padding = 28
        side_padding = 28
        col_gap = 28
        row_gap = 32
        bottom_padding = 28
        canvas_width = left_label_width + cell_width * 2 + col_gap + side_padding * 2
        canvas_height = top_padding + cell_height * 2 + row_gap + bottom_padding
        canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")
        draw = ImageDraw.Draw(canvas)
        label_font = load_image_font(34)

        positions = {
            ("Interface", "Train"): (left_label_width + side_padding, top_padding),
            ("Interface", "Test"): (left_label_width + side_padding + cell_width + col_gap, top_padding),
            ("Intermix", "Train"): (left_label_width + side_padding, top_padding + cell_height + row_gap),
            ("Intermix", "Test"): (
                left_label_width + side_padding + cell_width + col_gap,
                top_padding + cell_height + row_gap,
            ),
        }

        for wui_type in wui_order:
            row_top = positions[(wui_type, "Train")][1]
            row_center_y = row_top + cell_height // 2
            text_bbox = draw.textbbox((0, 0), wui_type, font=label_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.text(
                ((left_label_width - text_width) / 2, row_center_y - text_height / 2),
                wui_type,
                fill="#333333",
                font=label_font,
            )

        for wui_type in wui_order:
            for split_label in split_order:
                image = prediction_images_by_wui[wui_type][split_label]
                x0, y0 = positions[(wui_type, split_label)]
                x = x0 + (cell_width - image.width) // 2
                y = y0 + (cell_height - image.height) // 2
                canvas.paste(image, (x, y))

        canvas.save(output_path)
    finally:
        for image in images:
            image.close()
    return output_path


def compose_importance_grid_figure(
    importance_images_by_metric: dict[str, dict[str, Image.Image]],
    output_path: Path,
) -> Path:
    metric_order = [str(metric_config["name"]) for metric_config in METRIC_CONFIGS]
    wui_order = ["Interface", "Intermix"]
    images = [
        importance_images_by_metric[metric_name][wui_type]
        for metric_name in metric_order
        for wui_type in wui_order
    ]
    if len(images) != 4:
        raise ValueError(f"Expected 4 images for a 2x2 panel, got {len(images)}.")

    try:
        cell_width = max(image.width for image in images)
        cell_height = max(image.height for image in images)
        left_label_width = 260
        top_label_height = 82
        side_padding = 28
        top_padding = 18
        col_gap = 28
        row_gap = 32
        bottom_padding = 28
        canvas_width = left_label_width + cell_width * 2 + col_gap + side_padding * 2
        canvas_height = top_label_height + top_padding + cell_height * 2 + row_gap + bottom_padding
        canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")
        draw = ImageDraw.Draw(canvas)
        row_font = load_image_font(34)
        col_font = load_image_font(34)

        positions = {
            (metric_order[0], "Interface"): (left_label_width + side_padding, top_label_height + top_padding),
            (metric_order[0], "Intermix"): (
                left_label_width + side_padding + cell_width + col_gap,
                top_label_height + top_padding,
            ),
            (metric_order[1], "Interface"): (
                left_label_width + side_padding,
                top_label_height + top_padding + cell_height + row_gap,
            ),
            (metric_order[1], "Intermix"): (
                left_label_width + side_padding + cell_width + col_gap,
                top_label_height + top_padding + cell_height + row_gap,
            ),
        }

        for idx, wui_type in enumerate(wui_order):
            x0 = left_label_width + side_padding + idx * (cell_width + col_gap)
            text_bbox = draw.textbbox((0, 0), wui_type, font=col_font)
            text_width = text_bbox[2] - text_bbox[0]
            draw.text(
                (x0 + (cell_width - text_width) / 2, 18),
                wui_type,
                fill="#333333",
                font=col_font,
            )

        for metric_name in metric_order:
            y0 = positions[(metric_name, "Interface")][1]
            row_center_y = y0 + cell_height // 2
            metric_label = "Normalized burned area" if metric_name == "Normalized burn area" else metric_name
            text_bbox = draw.textbbox((0, 0), metric_label, font=row_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.text(
                ((left_label_width - text_width) / 2, row_center_y - text_height / 2),
                metric_label,
                fill="#333333",
                font=row_font,
            )

        for metric_name in metric_order:
            for wui_type in wui_order:
                image = importance_images_by_metric[metric_name][wui_type]
                x0, y0 = positions[(metric_name, wui_type)]
                x = x0 + (cell_width - image.width) // 2
                y = y0 + (cell_height - image.height) // 2
                canvas.paste(image, (x, y))

        canvas.save(output_path)
    finally:
        for image in images:
            image.close()
    return output_path


def build_shap_importance_image(
    shap_importance_df: pd.DataFrame,
    metric_config: dict[str, object],
    wui_type: str,
) -> Image.Image:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    plot_df = shap_importance_df.sort_values("mean_abs_shap", ascending=True)
    bars = ax.barh(plot_df["feature_name"], plot_df["mean_abs_shap"], color="#4C83B6", alpha=0.88)
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_ylabel("")
    ax.set_title(f"{metric_config['name']} - {wui_type} SHAP importance")
    max_value = float(plot_df["mean_abs_shap"].max()) if not plot_df.empty else 0.0
    for bar, value in zip(bars, plot_df["mean_abs_shap"]):
        x = float(bar.get_width())
        y = float(bar.get_y() + bar.get_height() / 2.0)
        ax.text(
            x + max(max_value * 0.02, 1e-8),
            y,
            f"{value:.3e}",
            va="center",
            ha="left",
            fontsize=10,
            color="#333333",
        )
    ax.set_xlim(0, max_value * 1.22 if max_value > 0 else 1.0)
    fig.tight_layout()
    return figure_to_image(fig)


def build_shap_outputs(
    model: XGBRegressor,
    X_reference: pd.DataFrame,
    metric_config: dict[str, object],
    wui_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    if len(X_reference) > SHAP_SAMPLE_SIZE:
        X_sample = X_reference.sample(n=SHAP_SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    else:
        X_sample = X_reference.reset_index(drop=True)

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_sample, check_additivity=False)
    shap_values = np.asarray(explanation.values, dtype=float)

    shap_importance_df = pd.DataFrame(
        {
            "metric_name": metric_config["name"],
            "WUI_Type": wui_type,
            "feature_column": FEATURE_COLUMNS,
            "feature_name": [FEATURE_LABELS[col] for col in FEATURE_COLUMNS],
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
            "shap_sample_size": len(X_sample),
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return shap_importance_df, X_sample, shap_values


def build_shared_x_shap_beeswarm_figure(
    metric_config: dict[str, object],
    shap_plot_payloads: dict[str, dict[str, object]],
) -> Path:
    wui_order = ["Interface", "Intermix"]
    x_limit = 0.0
    for wui_type in wui_order:
        shap_values = np.asarray(shap_plot_payloads[wui_type]["shap_values"], dtype=float)
        if shap_values.size:
            x_limit = max(x_limit, float(np.nanmax(np.abs(shap_values))))
    x_limit = x_limit * 1.05 if x_limit > 0 else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(16.8, 6.8), sharex=True)
    for idx, wui_type in enumerate(wui_order):
        ax = axes[idx]
        plt.sca(ax)
        shap.summary_plot(
            shap_plot_payloads[wui_type]["shap_values"],
            rename_feature_frame(shap_plot_payloads[wui_type]["X_sample"]),
            show=False,
            max_display=len(FEATURE_COLUMNS),
            plot_size=None,
            color_bar=False,
            sort=False,
        )
        ax.set_title(f"{metric_config['name']} - {wui_type} SHAP beeswarm", fontsize=14)
        ax.set_xlim(-x_limit, x_limit)
        ax.set_xlabel("SHAP value")

    fig.suptitle(f"{metric_config['name']} - SHAP beeswarm comparison", fontsize=16, y=0.995)
    fig.subplots_adjust(left=0.12, right=0.90, top=0.90, bottom=0.12, wspace=0.45)
    output_path = OUTPUT_DIR / f"xgboost_{metric_config['slug']}_shap_beeswarm_compare_shared_x.png"
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def build_shared_x_shap_importance_figure(
    metric_config: dict[str, object],
    shap_tables: dict[str, pd.DataFrame],
) -> Path:
    wui_order = ["Interface", "Intermix"]
    metric_title = get_metric_display_name(metric_config)
    global_max = 0.0
    for wui_type in wui_order:
        table = shap_tables[wui_type]
        if not table.empty:
            global_max = max(global_max, float(table["mean_abs_shap"].max()))
    global_max = global_max * 1.15 if global_max > 0 else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.8), sharex=True)
    colors = {"Interface": "#C85C5C", "Intermix": "#4C83B6"}

    for idx, wui_type in enumerate(wui_order):
        ax = axes[idx]
        plot_df = (
            shap_tables[wui_type]
            .set_index("feature_column")
            .loc[FEATURE_COLUMNS]
            .reset_index()
        )
        bars = ax.barh(plot_df["feature_name"], plot_df["mean_abs_shap"], color=colors[wui_type], alpha=0.9)
        ax.set_ylabel("")
        ax.set_xlim(0, global_max)
        for bar, value in zip(bars, plot_df["mean_abs_shap"]):
            x = float(bar.get_width())
            y = float(bar.get_y() + bar.get_height() / 2.0)
            ax.text(
                x + max(global_max * 0.015, 1e-8),
                y,
                f"{value:.3e}",
                va="center",
                ha="left",
                fontsize=10,
                color="#333333",
            )
        ax.set_xlabel("mean(|SHAP value|)")
        add_wui_corner_label(ax, wui_type)

    fig.suptitle(metric_title, fontsize=18, y=0.975)
    fig.subplots_adjust(left=0.12, right=0.90, top=0.88, bottom=0.16, wspace=0.40)
    add_feature_value_colorbar(fig, left=0.3, right=0.7, bottom=0.02)
    output_path = OUTPUT_DIR / f"xgboost_{metric_config['slug']}_shap_importance_compare_shared_x.png"
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def build_overlay_shap_compare_figure(
    metric_config: dict[str, object],
    shap_plot_payloads: dict[str, dict[str, object]],
    shap_tables: dict[str, pd.DataFrame],
) -> Path:
    wui_order = ["Interface", "Intermix"]
    metric_title = get_metric_display_name(metric_config)
    plot_left = 0.10
    plot_right = 0.98
    title_fontsize = 36
    axis_label_fontsize = 24
    tick_fontsize = 24
    wui_label_fontsize = 32
    center_label_x = -0.29
    order_top_down = shap_tables["Intermix"].sort_values("mean_abs_shap", ascending=False)["feature_column"].tolist()
    feature_indices = [FEATURE_COLUMNS.index(feature) for feature in order_top_down]

    shap_limit = 0.0
    mean_abs_limit = 0.0
    for wui_type in wui_order:
        shap_values = np.asarray(shap_plot_payloads[wui_type]["shap_values"], dtype=float)
        if shap_values.size:
            shap_limit = max(shap_limit, float(np.nanmax(np.abs(shap_values[:, feature_indices]))))
        table = shap_tables[wui_type]
        if not table.empty:
            mean_abs_limit = max(mean_abs_limit, float(table["mean_abs_shap"].max()))
    shap_limit = shap_limit * 1.05 if shap_limit > 0 else 1.0
    mean_abs_limit = mean_abs_limit * 1.15 if mean_abs_limit > 0 else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(20.4, 7.8), sharex=True)
    bar_colors = {
        "Interface": "#72B7E2",
        "Intermix": "#FDB366",
    }

    for idx, wui_type in enumerate(wui_order):
        ax = axes[idx]
        X_plot = rename_feature_frame(shap_plot_payloads[wui_type]["X_sample"][order_top_down])
        shap_values_plot = np.asarray(shap_plot_payloads[wui_type]["shap_values"], dtype=float)[:, feature_indices]

        plt.sca(ax)
        shap.summary_plot(
            shap_values_plot,
            X_plot,
            show=False,
            max_display=len(FEATURE_COLUMNS),
            plot_size=None,
            color_bar=False,
            sort=False,
        )

        ax.set_xlim(-shap_limit, shap_limit)
        ax.axvline(0.0, color="#666666", linewidth=1.0, linestyle="--", zorder=1)
        ax.set_xlabel("SHAP value", fontsize=axis_label_fontsize)
        ax.set_ylabel("")
        ax.tick_params(axis="x", bottom=True, labelbottom=True, pad=2, labelsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)

        ax_top = ax.twiny()
        ax_top.set_zorder(0)
        ax.set_zorder(1)
        ax.patch.set_alpha(0.0)
        ax_top.patch.set_alpha(0.0)
        ax_top.set_xlim(0.0, mean_abs_limit)
        ax_top.set_ylim(ax.get_ylim())

        bar_df = (
            shap_tables[wui_type]
            .set_index("feature_column")
            .loc[list(reversed(order_top_down))]
            .reset_index()
        )
        ax_top.barh(
            y=np.arange(len(bar_df)),
            width=bar_df["mean_abs_shap"].to_numpy(dtype=float),
            left=0.0,
            height=0.76,
            color=bar_colors[wui_type],
            alpha=0.4,
            edgecolor="none",
            zorder=-1,
        )
        ax_top.set_xlabel("Mean(|SHAP value|)", fontsize=axis_label_fontsize, labelpad=10)
        ax_top.tick_params(
            axis="x",
            top=True,
            labeltop=True,
            bottom=False,
            labelbottom=False,
            pad=2,
            labelsize=tick_fontsize,
        )
        ax_top.grid(False)
        ax_top.spines["bottom"].set_visible(False)
        ax_top.spines["left"].set_visible(False)
        ax_top.spines["right"].set_visible(False)

        if idx == 0:
            ax.invert_xaxis()
            ax.yaxis.tick_right()
            ax.tick_params(
                axis="y",
                labelright=False,
                labelleft=False,
                right=False,
                left=False,
                labelsize=tick_fontsize,
            )
            ax.set_yticklabels([])
            ax_top.invert_xaxis()
            ax.text(
                0.015,
                0.04,
                wui_type,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=wui_label_fontsize,
                color="#333333",
            )
        else:
            center_tick_positions = ax.get_yticks()
            center_tick_labels = [label.get_text() for label in ax.get_yticklabels()]
            ax.tick_params(
                axis="y",
                labelleft=False,
                labelright=False,
                left=False,
                right=False,
                labelsize=tick_fontsize,
            )
            for y_pos, label_text in zip(center_tick_positions, center_tick_labels):
                if not label_text:
                    continue
                ax.text(
                    center_label_x,
                    y_pos,
                    label_text,
                    transform=ax.get_yaxis_transform(),
                    ha="center",
                    va="center",
                    fontsize=tick_fontsize,
                    color="#333333",
                    clip_on=False,
                )
            add_wui_corner_label(ax, wui_type, fontsize=wui_label_fontsize)

    fig.suptitle(metric_title, fontsize=title_fontsize, x=(plot_left + 0.90) / 2.0, y=0.99)
    fig.subplots_adjust(left=plot_left, right=0.90, top=0.82, bottom=0.18, wspace=0.58)
    add_feature_value_colorbar(
        fig,
        left=0.3,
        right=0.7,
        bottom=0.03,
        height=0.03,
        label_fontsize=24,
        tick_fontsize=22,
    )
    output_path = OUTPUT_DIR / f"xgboost_{metric_config['slug']}_shap_overlay_compare_shared_x.png"
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def cleanup_legacy_plot_outputs(metric_slug: str, wui_type: str) -> None:
    legacy_paths = [
        OUTPUT_DIR / f"xgboost_{metric_slug}_{wui_type.lower()}_feature_importance.png",
        OUTPUT_DIR / f"xgboost_{metric_slug}_{wui_type.lower()}_train_prediction_scatter.png",
        OUTPUT_DIR / f"xgboost_{metric_slug}_{wui_type.lower()}_test_prediction_scatter.png",
        OUTPUT_DIR / f"xgboost_{metric_slug}_{wui_type.lower()}_shap_beeswarm.png",
        OUTPUT_DIR / f"xgboost_{metric_slug}_{wui_type.lower()}_shap_bar.png",
        OUTPUT_DIR / f"xgboost_{metric_slug}_{wui_type.lower()}_panel_2x2.png",
        OUTPUT_DIR / f"xgboost_{metric_slug}_{wui_type.lower()}_panel_3x2.png",
        OUTPUT_DIR / f"xgboost_{metric_slug}_{wui_type.lower()}_train_test_prediction_panel_1x2.png",
        OUTPUT_DIR / f"xgboost_{metric_slug}_{wui_type.lower()}_shap_panel_1x2.png",
    ]
    for path in legacy_paths:
        if path.exists():
            path.unlink()


def cleanup_metric_level_shap_compare_outputs(metric_slug: str) -> None:
    compare_paths = [
        OUTPUT_DIR / f"xgboost_{metric_slug}_shap_beeswarm_compare_shared_x.png",
        OUTPUT_DIR / f"xgboost_{metric_slug}_shap_importance_compare_shared_x.png",
        OUTPUT_DIR / f"xgboost_{metric_slug}_shap_overlay_compare_shared_x.png",
    ]
    for path in compare_paths:
        if path.exists():
            path.unlink()


def train_model_for_wui_type(
    metric_config: dict[str, object],
    wui_type: str,
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, Image.Image], Image.Image]:
    X = dataset[FEATURE_COLUMNS].astype(np.float32).copy()
    y = dataset[TARGET_COLUMN].to_numpy(dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    print(
        f"{metric_config['name']} / {wui_type}: train_samples={len(y_train):,}, "
        f"test_samples={len(y_test):,}, features={', '.join(FEATURE_LABELS[feature] for feature in FEATURE_COLUMNS)}"
    )

    model = create_model(metric_config, wui_type)
    model.fit(X_train, y_train, verbose=False)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    metrics_df = pd.DataFrame(
        [
            {
                "metric_name": metric_config["name"],
                "WUI_Type": wui_type,
                "n_total": len(dataset),
                "n_train": len(y_train),
                "n_test": len(y_test),
                "train_r2": float(r2_score(y_train, train_pred)),
                "test_r2": float(r2_score(y_test, test_pred)),
                "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_pred))),
                "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
                "train_mae": float(mean_absolute_error(y_train, train_pred)),
                "test_mae": float(mean_absolute_error(y_test, test_pred)),
            }
        ]
    )

    importance_df = build_feature_importance_table(model, metric_config, wui_type)
    shap_importance_df, shap_X_sample, shap_values = build_shap_outputs(
        model,
        X_test,
        metric_config,
        wui_type,
    )

    train_r2_value = float(metrics_df.at[0, "train_r2"])
    test_r2_value = float(metrics_df.at[0, "test_r2"])
    train_prediction_image = build_prediction_scatter_image(
        y_train,
        train_pred,
        metric_config,
        wui_type,
        split_label="Train",
        r2_value=train_r2_value,
    )
    test_prediction_image = build_prediction_scatter_image(
        y_test,
        test_pred,
        metric_config,
        wui_type,
        split_label="Test",
        r2_value=test_r2_value,
    )
    importance_image = build_feature_importance_image(importance_df, metric_config, wui_type)
    cleanup_legacy_plot_outputs(str(metric_config["slug"]), wui_type)

    print(
        f"{metric_config['name']} / {wui_type}: test_r2={metrics_df.at[0, 'test_r2']:.4f}, "
        f"test_rmse={metrics_df.at[0, 'test_rmse']:.6e}"
    )
    print(
        f"{metric_config['name']} / {wui_type}: gain importance image prepared"
    )

    shap_plot_payload = {
        "WUI_Type": wui_type,
        "X_sample": shap_X_sample,
        "shap_values": shap_values,
    }
    prediction_images = {
        "Train": train_prediction_image,
        "Test": test_prediction_image,
    }

    del X, y, X_train, X_test, y_train, y_test, train_pred, test_pred, model, dataset
    gc.collect()
    return metrics_df, importance_df, shap_importance_df, shap_plot_payload, prediction_images, importance_image


def write_outputs(
    metric_config: dict[str, object],
    inventory_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    importance_tables: dict[str, pd.DataFrame],
    shap_tables: dict[str, pd.DataFrame],
) -> Path:
    output_xlsx = OUTPUT_DIR / f"xgboost_{metric_config['slug']}_hfi_effect_summary.xlsx"
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        inventory_df.to_excel(writer, sheet_name="file_inventory", index=False)
        alignment_df.to_excel(writer, sheet_name="alignment_summary", index=False)
        metrics_df.to_excel(writer, sheet_name="model_metrics", index=False)
        for wui_type, table in importance_tables.items():
            table.to_excel(writer, sheet_name=f"{wui_type}_importance", index=False)
        for wui_type, table in shap_tables.items():
            table.to_excel(writer, sheet_name=f"{wui_type}_shap", index=False)
    return output_xlsx


def main() -> None:
    print("=== XGBoost Modeling by WUI Type ===")
    importance_images_by_metric: dict[str, dict[str, Image.Image]] = {}
    for metric_config in METRIC_CONFIGS:
        print(f"\n=== Processing {metric_config['name']} ===")
        pairs_by_type, inventory_df = discover_file_pairs(metric_config)

        all_alignment: list[pd.DataFrame] = []
        all_metrics: list[pd.DataFrame] = []
        importance_tables: dict[str, pd.DataFrame] = {}
        shap_tables: dict[str, pd.DataFrame] = {}
        shap_plot_payloads: dict[str, dict[str, object]] = {}
        prediction_images_by_wui: dict[str, dict[str, Image.Image]] = {}

        cleanup_metric_level_shap_compare_outputs(str(metric_config["slug"]))

        for wui_type in ["Interface", "Intermix"]:
            print(f"\n--- Building dataset for {metric_config['name']} / {wui_type} ---")
            dataset, alignment_df = build_dataset_for_wui_type(wui_type, pairs_by_type[wui_type], metric_config)
            metrics_df, importance_df, shap_importance_df, shap_plot_payload, prediction_images, importance_image = train_model_for_wui_type(
                metric_config,
                wui_type,
                dataset,
            )
            all_alignment.append(alignment_df)
            all_metrics.append(metrics_df)
            importance_tables[wui_type] = importance_df
            shap_tables[wui_type] = shap_importance_df
            shap_plot_payloads[wui_type] = shap_plot_payload
            prediction_images_by_wui[wui_type] = prediction_images
            importance_images_by_metric.setdefault(str(metric_config["name"]), {})[wui_type] = importance_image

        prediction_compare_path = compose_prediction_grid_figure(
            prediction_images_by_wui,
            OUTPUT_DIR / f"xgboost_{metric_config['slug']}_train_test_prediction_panel_2x2.png",
        )
        print(
            f"{metric_config['name']} combined train/test panel saved: "
            f"{prediction_compare_path.name}"
        )

        overlay_compare_path = build_overlay_shap_compare_figure(metric_config, shap_plot_payloads, shap_tables)
        print(
            f"{metric_config['name']} shared-x SHAP overlay saved: "
            f"{overlay_compare_path.name}"
        )

        alignment_all = pd.concat(all_alignment, ignore_index=True)
        metrics_all = pd.concat(all_metrics, ignore_index=True)
        output_xlsx = write_outputs(
            metric_config,
            inventory_df,
            alignment_all,
            metrics_all,
            importance_tables,
            shap_tables,
        )
        print(f"{metric_config['name']} summary workbook saved to: {output_xlsx}")

    importance_compare_path = compose_importance_grid_figure(
        importance_images_by_metric,
        OUTPUT_DIR / "xgboost_gain_importance_panel_2x2.png",
    )
    print(f"Combined gain importance panel saved: {importance_compare_path}")

    print("\nAll XGBoost modeling completed.")


if __name__ == "__main__":
    main()
