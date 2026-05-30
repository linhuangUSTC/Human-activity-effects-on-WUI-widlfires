#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称:
    使用 XGBoost 按 Human Footprint 影响正负分别建模 Fire frequency 与 Normalized burn area

功能简介:
    1. 分别从 HFI 固定效应结果 CSV 中读取 Human Footprint 对 Fire frequency / Normalized burn area 的影响(diff_contribution)。
    2. 从对应的 Basic 数据中读取 GDP、Population density、Road density、Cropland、Built environment、Pasture。
    3. 汇总全部 Interface(face_*) 与 Intermix(mix_*) 样本，但不再按 WUI 类型拆分。
    4. 对每个指标按 diff_contribution 的正负拆分为 Positive 与 Negative 两套 XGBoost 回归模型。
    5. 输出模型表现、特征重要性、SHAP 结果、模型文件和诊断图，输出结构与原程序保持一致。

说明:
    由于当前调参结果 workbook 只有 Interface / Intermix 版本，
    本程序对同一指标下 Interface 与 Intermix 的 best_parameters 取中位数，
    作为 Positive / Negative 两组的共用 XGBoost 参数。
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
    r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\归因分析\XGBoost机制揭示\XGboost参数调优"
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
SIGN_GROUPS = ["Positive", "Negative"]
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
MODEL_CONFIG_CACHE: dict[str, dict[str, object]] = {}
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


def sign_group_slug(sign_group: str) -> str:
    return sign_group.lower()


def discover_file_pairs(metric_config: dict[str, object]) -> tuple[list[tuple[str, str, Path, Path]], pd.DataFrame]:
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

    pairs: list[tuple[str, str, Path, Path]] = []
    inventory_rows: list[dict[str, object]] = []
    interface_count = 0
    intermix_count = 0
    for category_id in matched_ids:
        wui_type = infer_wui_type(category_id)
        if wui_type == "Interface":
            interface_count += 1
        else:
            intermix_count += 1
        effect_path = effect_files[category_id]
        basic_path = basic_files[category_id]
        pairs.append((category_id, wui_type, effect_path, basic_path))
        inventory_rows.append(
            {
                "metric_name": metric_config["name"],
                "category_id": category_id,
                "source_WUI_Type": wui_type,
                "effect_csv": str(effect_path),
                "basic_csv": str(basic_path),
            }
        )

    inventory_df = pd.DataFrame(inventory_rows)
    print(
        f"{metric_config['name']} matched categories: "
        f"Interface={interface_count}, Intermix={intermix_count}, Total={len(matched_ids)}"
    )
    return pairs, inventory_df


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


def build_dataset_for_sign(
    sign_group: str,
    pairs: list[tuple[str, str, Path, Path]],
    metric_config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_parts: list[pd.DataFrame] = []
    alignment_rows: list[dict[str, object]] = []

    for category_id, source_wui_type, effect_path, basic_path in pairs:
        effect_df = read_effect_csv(effect_path)
        basic_df = read_basic_csv(basic_path)
        merged = effect_df.merge(basic_df, how="inner", on=["GRID_ID", "Year"], validate="one_to_one")
        if merged.empty:
            alignment_rows.append(
                {
                    "metric_name": metric_config["name"],
                    "Effect_Sign": sign_group,
                    "category_id": category_id,
                    "source_WUI_Type": source_wui_type,
                    "effect_rows": len(effect_df),
                    "basic_rows": len(basic_df),
                    "matched_rows_before_sign": 0,
                    "matched_rows_after_sign": 0,
                    "zero_effect_rows": 0,
                    "hfi_mean_abs_diff": np.nan,
                    "hfi_max_abs_diff": np.nan,
                }
            )
            continue

        if sign_group == "Positive":
            sign_mask = merged[TARGET_COLUMN] > 0
        elif sign_group == "Negative":
            sign_mask = merged[TARGET_COLUMN] < 0
        else:
            raise ValueError(f"Unexpected sign group: {sign_group}")

        signed = merged.loc[sign_mask].copy()
        hfi_abs_diff = (merged["HFI_effect"] - merged["HFI_basic"]).abs()
        zero_effect_rows = int((merged[TARGET_COLUMN] == 0).sum())
        alignment_rows.append(
            {
                "metric_name": metric_config["name"],
                "Effect_Sign": sign_group,
                "category_id": category_id,
                "source_WUI_Type": source_wui_type,
                "effect_rows": len(effect_df),
                "basic_rows": len(basic_df),
                "matched_rows_before_sign": len(merged),
                "matched_rows_after_sign": len(signed),
                "zero_effect_rows": zero_effect_rows,
                "hfi_mean_abs_diff": float(hfi_abs_diff.mean()),
                "hfi_max_abs_diff": float(hfi_abs_diff.max()),
            }
        )

        if not signed.empty:
            data_parts.append(signed[FEATURE_COLUMNS + [TARGET_COLUMN]].astype(np.float32))

        del effect_df, basic_df, merged, signed
        gc.collect()

    if not data_parts:
        raise ValueError(f"No matched samples available for {metric_config['name']} / {sign_group}.")

    dataset = pd.concat(data_parts, ignore_index=True)
    alignment_df = pd.DataFrame(alignment_rows)
    print(
        f"{metric_config['name']} / {sign_group}: samples={len(dataset):,}, "
        f"target_mean={dataset[TARGET_COLUMN].mean():.6e}"
    )
    return dataset, alignment_df


def coerce_parameter_value(parameter: str, value: object) -> object:
    numeric_value = float(value)
    if parameter in INTEGER_PARAMETER_NAMES:
        return int(round(numeric_value))
    return numeric_value


def load_tuned_parameters(metric_config: dict[str, object]) -> dict[str, object]:
    cache_key = str(metric_config["slug"])
    if cache_key in MODEL_CONFIG_CACHE:
        return MODEL_CONFIG_CACHE[cache_key]

    parameter_tables: list[pd.DataFrame] = []
    source_names: list[str] = []
    for source_wui in ["interface", "intermix"]:
        workbook_path = TUNED_PARAMETER_DIR / f"xgboost_{metric_config['slug']}_{source_wui}_hfi_effect_tuned_summary.xlsx"
        if not workbook_path.exists():
            raise FileNotFoundError(
                f"Tuned summary workbook not found for {metric_config['name']} / {source_wui}: {workbook_path}"
            )
        parameter_df = pd.read_excel(workbook_path, sheet_name="best_parameters")
        required_columns = {"parameter", "value"}
        if not required_columns.issubset(parameter_df.columns):
            raise ValueError(
                f"Workbook {workbook_path.name} best_parameters sheet is missing required columns: "
                f"{sorted(required_columns)}"
            )
        parameter_tables.append(parameter_df.loc[:, ["parameter", "value"]].copy())
        source_names.append(workbook_path.name)

    merged_parameter_df = pd.concat(parameter_tables, ignore_index=True)
    tuned_params: dict[str, object] = {}
    for parameter, group_df in merged_parameter_df.groupby("parameter", sort=False):
        median_value = float(pd.to_numeric(group_df["value"], errors="coerce").median())
        tuned_params[str(parameter)] = coerce_parameter_value(str(parameter), median_value)

    model_config = BASE_MODEL_CONFIG.copy()
    model_config.update(tuned_params)
    MODEL_CONFIG_CACHE[cache_key] = model_config

    print(
        f"{metric_config['name']}: tuned parameters loaded from "
        f"{', '.join(source_names)} and aggregated by median"
    )
    return model_config


def create_model(metric_config: dict[str, object], sign_group: str) -> XGBRegressor:
    _ = sign_group
    return XGBRegressor(**load_tuned_parameters(metric_config))


def get_booster_score(scores: dict[str, float], feature: str, idx: int) -> float:
    return float(scores.get(feature, scores.get(f"f{idx}", 0.0)))


def rename_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={column: FEATURE_LABELS.get(column, column) for column in df.columns})


def get_metric_display_name(metric_config: dict[str, object]) -> str:
    if str(metric_config["slug"]) == "normalized_burn_area":
        return "Normalized burned area"
    return str(metric_config["name"])


def add_group_corner_label(ax: plt.Axes, sign_group: str, fontsize: float = 16) -> None:
    ax.text(
        0.985,
        0.04,
        sign_group,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=fontsize,
        color="#333333",
    )


def add_feature_value_colorbar(
    fig: plt.Figure,
    left: float = 0.30,
    right: float = 0.70,
    bottom: float = 0.04,
    height: float = 0.022,
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


def build_feature_importance_table(model: XGBRegressor, metric_config: dict[str, object], sign_group: str) -> pd.DataFrame:
    booster = model.get_booster()
    gain_scores = booster.get_score(importance_type="gain")
    weight_scores = booster.get_score(importance_type="weight")
    cover_scores = booster.get_score(importance_type="cover")
    rows: list[dict[str, object]] = []
    for idx, feature in enumerate(FEATURE_COLUMNS):
        rows.append(
            {
                "metric_name": metric_config["name"],
                "Effect_Sign": sign_group,
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


def build_feature_importance_image(
    importance_df: pd.DataFrame,
    metric_config: dict[str, object],
    sign_group: str,
) -> Image.Image:
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    plot_df = importance_df.sort_values("importance_gain", ascending=True)
    bars = ax.barh(plot_df["feature_name"], plot_df["importance_gain"], color="#4C83B6", alpha=0.88)
    ax.set_xlabel("XGBoost importance (gain)")
    ax.set_ylabel("")
    ax.set_title(f"{metric_config['name']} - {sign_group} feature importance")
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
    sign_group: str,
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
    ax.set_title(f"{metric_config['name']} - {sign_group} {split_label} prediction")
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


def load_image_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_name in ["arial.ttf", "Arial.ttf"]:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def compose_prediction_grid_figure(
    prediction_images_by_group: dict[str, dict[str, Image.Image]],
    output_path: Path,
) -> Path:
    split_order = ["Train", "Test"]
    images = [
        prediction_images_by_group[sign_group][split_label]
        for sign_group in SIGN_GROUPS
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
            ("Positive", "Train"): (left_label_width + side_padding, top_padding),
            ("Positive", "Test"): (left_label_width + side_padding + cell_width + col_gap, top_padding),
            ("Negative", "Train"): (left_label_width + side_padding, top_padding + cell_height + row_gap),
            ("Negative", "Test"): (
                left_label_width + side_padding + cell_width + col_gap,
                top_padding + cell_height + row_gap,
            ),
        }

        for sign_group in SIGN_GROUPS:
            row_top = positions[(sign_group, "Train")][1]
            row_center_y = row_top + cell_height // 2
            text_bbox = draw.textbbox((0, 0), sign_group, font=label_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.text(
                ((left_label_width - text_width) / 2, row_center_y - text_height / 2),
                sign_group,
                fill="#333333",
                font=label_font,
            )

        for sign_group in SIGN_GROUPS:
            for split_label in split_order:
                image = prediction_images_by_group[sign_group][split_label]
                x0, y0 = positions[(sign_group, split_label)]
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
    images = [
        importance_images_by_metric[metric_name][sign_group]
        for metric_name in metric_order
        for sign_group in SIGN_GROUPS
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
            (metric_order[0], "Positive"): (left_label_width + side_padding, top_label_height + top_padding),
            (metric_order[0], "Negative"): (
                left_label_width + side_padding + cell_width + col_gap,
                top_label_height + top_padding,
            ),
            (metric_order[1], "Positive"): (
                left_label_width + side_padding,
                top_label_height + top_padding + cell_height + row_gap,
            ),
            (metric_order[1], "Negative"): (
                left_label_width + side_padding + cell_width + col_gap,
                top_label_height + top_padding + cell_height + row_gap,
            ),
        }

        for idx, sign_group in enumerate(SIGN_GROUPS):
            x0 = left_label_width + side_padding + idx * (cell_width + col_gap)
            text_bbox = draw.textbbox((0, 0), sign_group, font=col_font)
            text_width = text_bbox[2] - text_bbox[0]
            draw.text(
                (x0 + (cell_width - text_width) / 2, 18),
                sign_group,
                fill="#333333",
                font=col_font,
            )

        for metric_name in metric_order:
            y0 = positions[(metric_name, "Positive")][1]
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
            for sign_group in SIGN_GROUPS:
                image = importance_images_by_metric[metric_name][sign_group]
                x0, y0 = positions[(metric_name, sign_group)]
                x = x0 + (cell_width - image.width) // 2
                y = y0 + (cell_height - image.height) // 2
                canvas.paste(image, (x, y))

        canvas.save(output_path)
    finally:
        for image in images:
            image.close()
    return output_path


def build_shap_outputs(
    model: XGBRegressor,
    X_reference: pd.DataFrame,
    metric_config: dict[str, object],
    sign_group: str,
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
            "Effect_Sign": sign_group,
            "feature_column": FEATURE_COLUMNS,
            "feature_name": [FEATURE_LABELS[col] for col in FEATURE_COLUMNS],
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
            "shap_sample_size": len(X_sample),
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return shap_importance_df, X_sample, shap_values


def build_overlay_shap_compare_figure(
    metric_config: dict[str, object],
    shap_plot_payloads: dict[str, dict[str, object]],
    shap_tables: dict[str, pd.DataFrame],
) -> Path:
    metric_title = get_metric_display_name(metric_config)
    plot_left = 0.27
    plot_right = 0.96
    title_fontsize = 36
    axis_label_fontsize = 24
    tick_fontsize = 24
    group_label_fontsize = 32

    consensus_order = (
        pd.concat(
            [
                table[["feature_column", "mean_abs_shap"]].copy()
                for table in shap_tables.values()
            ],
            ignore_index=True,
        )
        .groupby("feature_column", as_index=False)["mean_abs_shap"]
        .mean()
        .sort_values("mean_abs_shap", ascending=False)["feature_column"]
        .tolist()
    )
    feature_indices = [FEATURE_COLUMNS.index(feature) for feature in consensus_order]

    shap_limit = 0.0
    mean_abs_limit = 0.0
    for sign_group in SIGN_GROUPS:
        shap_values = np.asarray(shap_plot_payloads[sign_group]["shap_values"], dtype=float)
        if shap_values.size:
            shap_limit = max(shap_limit, float(np.nanmax(np.abs(shap_values[:, feature_indices]))))
        table = shap_tables[sign_group]
        if not table.empty:
            mean_abs_limit = max(mean_abs_limit, float(table["mean_abs_shap"].max()))
    shap_limit = shap_limit * 1.05 if shap_limit > 0 else 1.0
    mean_abs_limit = mean_abs_limit * 1.15 if mean_abs_limit > 0 else 1.0

    fig, axes = plt.subplots(2, 1, figsize=(11.2, 13.2), sharex=True)
    bar_color = "#D9D9D9"

    for idx, sign_group in enumerate(SIGN_GROUPS):
        ax = axes[idx]
        X_plot = rename_feature_frame(shap_plot_payloads[sign_group]["X_sample"][consensus_order])
        shap_values_plot = np.asarray(shap_plot_payloads[sign_group]["shap_values"], dtype=float)[:, feature_indices]

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
        if idx < len(SIGN_GROUPS) - 1:
            ax.set_xlabel("", fontsize=axis_label_fontsize)
        else:
            ax.set_xlabel("SHAP value", fontsize=axis_label_fontsize)
        ax.tick_params(axis="x", bottom=True, labelbottom=True, pad=2, labelsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        add_group_corner_label(ax, sign_group, fontsize=group_label_fontsize)

        ax_top = ax.twiny()
        ax_top.set_zorder(0)
        ax.set_zorder(1)
        ax.patch.set_alpha(0.0)
        ax_top.patch.set_alpha(0.0)
        ax_top.set_xlim(0.0, mean_abs_limit)
        ax_top.set_ylim(ax.get_ylim())

        bar_df = (
            shap_tables[sign_group]
            .set_index("feature_column")
            .loc[list(reversed(consensus_order))]
            .reset_index()
        )
        ax_top.barh(
            y=np.arange(len(bar_df)),
            width=bar_df["mean_abs_shap"].to_numpy(dtype=float),
            left=0.0,
            height=0.76,
            color=bar_color,
            alpha=0.75,
            edgecolor="none",
            zorder=-1,
        )
        if idx == 0:
            ax_top.set_xlabel("Mean(|SHAP value|)", fontsize=axis_label_fontsize, labelpad=10)
        else:
            ax_top.set_xlabel("")
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

    fig.suptitle(metric_title, fontsize=title_fontsize, x=(plot_left + plot_right) / 2.0, y=0.99)
    fig.subplots_adjust(left=plot_left, right=plot_right, top=0.87, bottom=0.16, hspace=0.32)
    add_feature_value_colorbar(
        fig,
        left=plot_left,
        right=plot_right,
        bottom=0.05,
        height=0.03,
        label_fontsize=24,
        tick_fontsize=22,
    )
    output_path = OUTPUT_DIR / f"xgboost_{metric_config['slug']}_positive_negative_shap_overlay_compare_shared_x.png"
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def cleanup_legacy_plot_outputs(metric_slug: str, sign_group: str) -> None:
    group_slug = sign_group_slug(sign_group)
    legacy_paths = [
        OUTPUT_DIR / f"xgboost_{metric_slug}_{group_slug}_feature_importance.png",
        OUTPUT_DIR / f"xgboost_{metric_slug}_{group_slug}_train_prediction_scatter.png",
        OUTPUT_DIR / f"xgboost_{metric_slug}_{group_slug}_test_prediction_scatter.png",
        OUTPUT_DIR / f"xgboost_{metric_slug}_{group_slug}_shap_beeswarm.png",
        OUTPUT_DIR / f"xgboost_{metric_slug}_{group_slug}_shap_bar.png",
        OUTPUT_DIR / f"xgboost_{metric_slug}_{group_slug}_panel_2x2.png",
        OUTPUT_DIR / f"xgboost_{metric_slug}_{group_slug}_panel_3x2.png",
        OUTPUT_DIR / f"xgboost_{metric_slug}_{group_slug}_train_test_prediction_panel_1x2.png",
        OUTPUT_DIR / f"xgboost_{metric_slug}_{group_slug}_shap_panel_1x2.png",
    ]
    for path in legacy_paths:
        if path.exists():
            path.unlink()


def cleanup_metric_level_shap_compare_outputs(metric_slug: str) -> None:
    compare_paths = [
        OUTPUT_DIR / f"xgboost_{metric_slug}_positive_negative_shap_overlay_compare_shared_x.png",
        OUTPUT_DIR / f"xgboost_{metric_slug}_positive_negative_train_test_prediction_panel_2x2.png",
        OUTPUT_DIR / f"xgboost_{metric_slug}_hfi_effect_positive_negative_summary.xlsx",
    ]
    for path in compare_paths:
        if path.exists():
            path.unlink()


def train_model_for_group(
    metric_config: dict[str, object],
    sign_group: str,
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
        f"{metric_config['name']} / {sign_group}: train_samples={len(y_train):,}, "
        f"test_samples={len(y_test):,}, features={', '.join(FEATURE_LABELS[feature] for feature in FEATURE_COLUMNS)}"
    )

    model = create_model(metric_config, sign_group)
    model.fit(X_train, y_train, verbose=False)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    metrics_df = pd.DataFrame(
        [
            {
                "metric_name": metric_config["name"],
                "Effect_Sign": sign_group,
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

    importance_df = build_feature_importance_table(model, metric_config, sign_group)
    shap_importance_df, shap_X_sample, shap_values = build_shap_outputs(
        model,
        X_test,
        metric_config,
        sign_group,
    )

    train_r2_value = float(metrics_df.at[0, "train_r2"])
    test_r2_value = float(metrics_df.at[0, "test_r2"])
    train_prediction_image = build_prediction_scatter_image(
        y_train,
        train_pred,
        metric_config,
        sign_group,
        split_label="Train",
        r2_value=train_r2_value,
    )
    test_prediction_image = build_prediction_scatter_image(
        y_test,
        test_pred,
        metric_config,
        sign_group,
        split_label="Test",
        r2_value=test_r2_value,
    )
    importance_image = build_feature_importance_image(importance_df, metric_config, sign_group)
    cleanup_legacy_plot_outputs(str(metric_config["slug"]), sign_group)

    print(
        f"{metric_config['name']} / {sign_group}: test_r2={metrics_df.at[0, 'test_r2']:.4f}, "
        f"test_rmse={metrics_df.at[0, 'test_rmse']:.6e}"
    )
    print(f"{metric_config['name']} / {sign_group}: gain importance image prepared")

    shap_plot_payload = {
        "Effect_Sign": sign_group,
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
    output_xlsx = OUTPUT_DIR / f"xgboost_{metric_config['slug']}_hfi_effect_positive_negative_summary.xlsx"
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        inventory_df.to_excel(writer, sheet_name="file_inventory", index=False)
        alignment_df.to_excel(writer, sheet_name="alignment_summary", index=False)
        metrics_df.to_excel(writer, sheet_name="model_metrics", index=False)
        for sign_group, table in importance_tables.items():
            table.to_excel(writer, sheet_name=f"{sign_group}_importance", index=False)
        for sign_group, table in shap_tables.items():
            table.to_excel(writer, sheet_name=f"{sign_group}_shap", index=False)
    return output_xlsx


def main() -> None:
    print("=== XGBoost Modeling by Effect Sign ===")
    importance_images_by_metric: dict[str, dict[str, Image.Image]] = {}
    for metric_config in METRIC_CONFIGS:
        print(f"\n=== Processing {metric_config['name']} ===")
        pairs, inventory_df = discover_file_pairs(metric_config)

        all_alignment: list[pd.DataFrame] = []
        all_metrics: list[pd.DataFrame] = []
        importance_tables: dict[str, pd.DataFrame] = {}
        shap_tables: dict[str, pd.DataFrame] = {}
        shap_plot_payloads: dict[str, dict[str, object]] = {}
        prediction_images_by_group: dict[str, dict[str, Image.Image]] = {}

        cleanup_metric_level_shap_compare_outputs(str(metric_config["slug"]))

        for sign_group in SIGN_GROUPS:
            print(f"\n--- Building dataset for {metric_config['name']} / {sign_group} ---")
            dataset, alignment_df = build_dataset_for_sign(sign_group, pairs, metric_config)
            metrics_df, importance_df, shap_importance_df, shap_plot_payload, prediction_images, importance_image = (
                train_model_for_group(
                    metric_config,
                    sign_group,
                    dataset,
                )
            )
            all_alignment.append(alignment_df)
            all_metrics.append(metrics_df)
            importance_tables[sign_group] = importance_df
            shap_tables[sign_group] = shap_importance_df
            shap_plot_payloads[sign_group] = shap_plot_payload
            prediction_images_by_group[sign_group] = prediction_images
            importance_images_by_metric.setdefault(str(metric_config["name"]), {})[sign_group] = importance_image

        prediction_compare_path = compose_prediction_grid_figure(
            prediction_images_by_group,
            OUTPUT_DIR / f"xgboost_{metric_config['slug']}_positive_negative_train_test_prediction_panel_2x2.png",
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
        OUTPUT_DIR / "xgboost_gain_importance_panel_2x2_positive_negative.png",
    )
    print(f"Combined gain importance panel saved: {importance_compare_path}")

    print("\nAll XGBoost modeling completed.")


if __name__ == "__main__":
    main()
