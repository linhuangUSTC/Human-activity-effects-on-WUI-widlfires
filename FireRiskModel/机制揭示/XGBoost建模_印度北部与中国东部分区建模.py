#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称:
    使用 XGBoost 分别建模印度北部与中国东部区域中 Human Footprint 对
    Fire frequency 与 Normalized burned area 的影响机制

功能简介:
    1. 使用空间分布局部放大图中的两个区域范围：
       - Northern India: lon 74-91, lat 19-31
       - East-Central China: lon 105-122, lat 24-36
    2. 从 WUI.gdb 中提取两个区域的 GRID_ID，并按 WUI 年份映射到 2003-2022。
    3. 匹配 HFI effect CSV 与 Basic CSV。
    4. 不再区分 Interface / Intermix WUI，而是分别对两个区域建立 XGBoost 回归模型。
    5. 使用 6 个机制变量：GDP、Population density、Road density、Cropland、Built environment、Pasture。
    6. 输出模型表现、gain importance、SHAP 表、train/test 诊断图与 SHAP overlay 对比图。
"""

from __future__ import annotations

import gc
from io import BytesIO
from pathlib import Path

import geopandas as gpd
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

YEAR_START = 2003
YEAR_END = 2022
YEAR_TAG = f"{YEAR_START}_{YEAR_END}"

WUI_GDB_PATH = Path(r"I:\Processing data\gdb集合\WUI.gdb")
WUI_YEARS = [2005, 2010, 2015, 2020]
WUI_LAYER_TEMPLATE = "Grid_Clip_{wui_year}_WUI"
WUI_YEAR_MAPPING = {
    2005: list(range(2003, 2008)),
    2010: list(range(2008, 2013)),
    2015: list(range(2013, 2018)),
    2020: list(range(2018, 2023)),
}

METRIC_CONFIGS = [
    {
        "name": "Fire frequency",
        "slug": "fire_frequency",
        "effect_dir": Path(r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num"),
        "basic_dir": Path(r"I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic"),
    },
    {
        "name": "Normalized burned area",
        "slug": "normalized_burned_area",
        "effect_dir": Path(r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area"),
        "basic_dir": Path(r"I:\Processing data\渔网分类建模\Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic"),
    },
]

REGIONS = [
    {
        "slug": "northern_india",
        "short_slug": "india",
        "label": "Northern India",
        "lon_min": 74.0,
        "lon_max": 91.0,
        "lat_min": 19.0,
        "lat_max": 31.0,
        "color": "#d73027",
    },
    {
        "slug": "east_central_china",
        "short_slug": "china",
        "label": "East-Central China",
        "lon_min": 105.0,
        "lon_max": 122.0,
        "lat_min": 24.0,
        "lat_max": 36.0,
        "color": "#1e90ff",
    },
]
REGION_ORDER = [str(region["label"]) for region in REGIONS]
REGION_BY_LABEL = {str(region["label"]): region for region in REGIONS}

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
MODEL_CONFIG = {
    "objective": "reg:squarederror",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "tree_method": "hist",
    "eval_metric": "rmse",
    "n_estimators": 600,
    "learning_rate": 0.04,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
}
PREDICTION_PLOT_SAMPLE_SIZE = 50000
SHAP_SAMPLE_SIZE = 20000


def normalize_grid_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def normalize_year(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def extract_effect_category_id(csv_path: Path) -> str:
    stem = csv_path.stem
    prefix = "HFI_effect_"
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected effect filename: {csv_path.name}")
    return stem[len(prefix):]


def discover_file_pairs(metric_config: dict[str, object]) -> tuple[list[tuple[str, Path, Path]], pd.DataFrame]:
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

    pairs: list[tuple[str, Path, Path]] = []
    inventory_rows: list[dict[str, object]] = []
    for category_id in matched_ids:
        effect_path = effect_files[category_id]
        basic_path = basic_files[category_id]
        pairs.append((category_id, effect_path, basic_path))
        inventory_rows.append(
            {
                "metric_name": metric_config["name"],
                "category_id": category_id,
                "effect_csv": str(effect_path),
                "basic_csv": str(basic_path),
            }
        )

    print(f"{metric_config['name']} matched categories: {len(pairs)}")
    return pairs, pd.DataFrame(inventory_rows)


def read_region_grid_ids(region: dict[str, object], wui_year: int) -> pd.DataFrame:
    layer = WUI_LAYER_TEMPLATE.format(wui_year=wui_year)
    bbox = (
        float(region["lon_min"]),
        float(region["lat_min"]),
        float(region["lon_max"]),
        float(region["lat_max"]),
    )
    print(f"Reading {region['label']} GRID_ID from {layer} ...")

    def crop_to_region(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if gdf.empty:
            return gdf
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        return gdf.cx[
            float(region["lon_min"]):float(region["lon_max"]),
            float(region["lat_min"]):float(region["lat_max"]),
        ].copy()

    def read_full_layer() -> gpd.GeoDataFrame:
        return gpd.read_file(
            WUI_GDB_PATH,
            layer=layer,
            columns=["GRID_ID"],
            engine="pyogrio",
        )

    fishnet = gpd.read_file(
        WUI_GDB_PATH,
        layer=layer,
        bbox=bbox,
        columns=["GRID_ID"],
        engine="pyogrio",
    )
    if fishnet.empty:
        print(f"bbox read returned empty for {region['label']} / {layer}; reading full layer as fallback...")
        fishnet = read_full_layer()

    if fishnet.empty:
        return pd.DataFrame(columns=["GRID_ID", "wui_year", "region_slug", "region_label"])

    fishnet = crop_to_region(fishnet)
    if fishnet.empty:
        print(f"bbox read produced no cells after lon/lat crop for {region['label']} / {layer}; reading full layer as fallback...")
        fishnet = crop_to_region(read_full_layer())
    if fishnet.empty:
        return pd.DataFrame(columns=["GRID_ID", "wui_year", "region_slug", "region_label"])

    grid_ids = normalize_grid_id(fishnet["GRID_ID"]).dropna().astype(np.int64).drop_duplicates()
    return pd.DataFrame(
        {
            "GRID_ID": grid_ids.to_numpy(dtype=np.int64),
            "wui_year": int(wui_year),
            "region_slug": str(region["slug"]),
            "region_label": str(region["label"]),
        }
    )


def build_region_year_lookup() -> tuple[pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for region in REGIONS:
        for wui_year in WUI_YEARS:
            region_grid = read_region_grid_ids(region, wui_year)
            summary_rows.append(
                {
                    "region_label": str(region["label"]),
                    "wui_year": int(wui_year),
                    "grid_count": int(region_grid["GRID_ID"].nunique()) if not region_grid.empty else 0,
                }
            )
            if region_grid.empty:
                continue

            expanded_parts: list[pd.DataFrame] = []
            for year in WUI_YEAR_MAPPING[wui_year]:
                year_df = region_grid[["GRID_ID", "region_slug", "region_label"]].copy()
                year_df["Year"] = int(year)
                year_df["wui_year"] = int(wui_year)
                expanded_parts.append(year_df)
            parts.append(pd.concat(expanded_parts, ignore_index=True))

    if not parts:
        raise ValueError("No region GRID_ID-Year lookup rows were created.")

    lookup = pd.concat(parts, ignore_index=True)
    lookup["GRID_ID"] = normalize_grid_id(lookup["GRID_ID"]).astype(np.int64)
    lookup["Year"] = normalize_year(lookup["Year"]).astype(np.int64)
    lookup = (
        lookup.sort_values(["Year", "GRID_ID", "region_slug"])
        .drop_duplicates(subset=["Year", "GRID_ID"], keep="first")
        .reset_index(drop=True)
    )
    print(
        "Region lookup loaded: "
        f"rows={len(lookup):,}, unique_GRID_ID={lookup['GRID_ID'].nunique():,}, "
        f"years={lookup['Year'].nunique()}, regions={lookup['region_label'].nunique()}"
    )
    return lookup[["Year", "GRID_ID", "region_slug", "region_label"]], pd.DataFrame(summary_rows)


def read_effect_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        usecols=["GRID_ID", "Year", "HFI", TARGET_COLUMN],
        low_memory=False,
    )
    df["GRID_ID"] = normalize_grid_id(df["GRID_ID"])
    df["Year"] = normalize_year(df["Year"])
    df["HFI_effect"] = pd.to_numeric(df["HFI"], errors="coerce").astype(np.float32)
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").astype(np.float32)
    df = df[
        df["Year"].between(YEAR_START, YEAR_END, inclusive="both")
    ].dropna(subset=["GRID_ID", "Year", "HFI_effect", TARGET_COLUMN]).copy()
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
    df["Year"] = normalize_year(df["Year"])
    for column in BASIC_FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(np.float32)
    df["HFI_basic"] = pd.to_numeric(df["HFI"], errors="coerce").astype(np.float32)
    df = df[
        df["Year"].between(YEAR_START, YEAR_END, inclusive="both")
    ].dropna(subset=["GRID_ID", "Year", "HFI_basic"] + BASIC_FEATURE_COLUMNS).copy()
    return (
        df.groupby(["GRID_ID", "Year"], as_index=False)
        .agg(**{column: (column, "mean") for column in BASIC_FEATURE_COLUMNS}, HFI_basic=("HFI_basic", "mean"))
    )


def build_datasets_by_region(
    pairs: list[tuple[str, Path, Path]],
    metric_config: dict[str, object],
    region_lookup: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    data_parts_by_region: dict[str, list[pd.DataFrame]] = {label: [] for label in REGION_ORDER}
    alignment_rows: list[dict[str, object]] = []

    for category_id, effect_path, basic_path in pairs:
        effect_df = read_effect_csv(effect_path)
        effect_region = effect_df.merge(region_lookup, how="inner", on=["GRID_ID", "Year"], validate="one_to_one")
        if effect_region.empty:
            alignment_rows.append(
                {
                    "metric_name": metric_config["name"],
                    "category_id": category_id,
                    "effect_rows": len(effect_df),
                    "effect_region_rows": 0,
                    "basic_rows": np.nan,
                    "matched_rows": 0,
                    "region_label": "None",
                    "hfi_mean_abs_diff": np.nan,
                    "hfi_max_abs_diff": np.nan,
                }
            )
            del effect_df, effect_region
            gc.collect()
            continue

        basic_df = read_basic_csv(basic_path)
        merged = effect_region.merge(basic_df, how="inner", on=["GRID_ID", "Year"], validate="one_to_one")
        if merged.empty:
            alignment_rows.append(
                {
                    "metric_name": metric_config["name"],
                    "category_id": category_id,
                    "effect_rows": len(effect_df),
                    "effect_region_rows": len(effect_region),
                    "basic_rows": len(basic_df),
                    "matched_rows": 0,
                    "region_label": "None",
                    "hfi_mean_abs_diff": np.nan,
                    "hfi_max_abs_diff": np.nan,
                }
            )
            del effect_df, effect_region, basic_df, merged
            gc.collect()
            continue

        hfi_abs_diff = (merged["HFI_effect"] - merged["HFI_basic"]).abs()
        for region_label in REGION_ORDER:
            region_merged = merged.loc[merged["region_label"] == region_label].copy()
            alignment_rows.append(
                {
                    "metric_name": metric_config["name"],
                    "category_id": category_id,
                    "effect_rows": len(effect_df),
                    "effect_region_rows": len(effect_region),
                    "basic_rows": len(basic_df),
                    "matched_rows": len(region_merged),
                    "region_label": region_label,
                    "hfi_mean_abs_diff": float(hfi_abs_diff.loc[region_merged.index].mean()) if not region_merged.empty else np.nan,
                    "hfi_max_abs_diff": float(hfi_abs_diff.loc[region_merged.index].max()) if not region_merged.empty else np.nan,
                }
            )
            if region_merged.empty:
                continue
            data_parts_by_region[region_label].append(
                region_merged[FEATURE_COLUMNS + [TARGET_COLUMN]].astype(np.float32)
            )

        del effect_df, effect_region, basic_df, merged
        gc.collect()

    datasets_by_region: dict[str, pd.DataFrame] = {}
    for region_label in REGION_ORDER:
        if not data_parts_by_region[region_label]:
            raise ValueError(f"No matched samples available for {metric_config['name']} / {region_label}.")
        dataset = pd.concat(data_parts_by_region[region_label], ignore_index=True)
        datasets_by_region[region_label] = dataset
        print(
            f"{metric_config['name']} / {region_label}: "
            f"samples={len(dataset):,}, target_mean={dataset[TARGET_COLUMN].mean():.6e}"
        )

    return datasets_by_region, pd.DataFrame(alignment_rows)


def create_model() -> XGBRegressor:
    return XGBRegressor(**MODEL_CONFIG)


def get_booster_score(scores: dict[str, float], feature: str, idx: int) -> float:
    return float(scores.get(feature, scores.get(f"f{idx}", 0.0)))


def rename_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={column: FEATURE_LABELS.get(column, column) for column in df.columns})


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


def add_region_corner_label(ax: plt.Axes, region_label: str, fontsize: float = 16) -> None:
    ax.text(
        0.985,
        0.04,
        region_label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=fontsize,
        color="#333333",
    )


def build_feature_importance_table(
    model: XGBRegressor,
    metric_config: dict[str, object],
    region_label: str,
) -> pd.DataFrame:
    booster = model.get_booster()
    gain_scores = booster.get_score(importance_type="gain")
    weight_scores = booster.get_score(importance_type="weight")
    cover_scores = booster.get_score(importance_type="cover")
    rows: list[dict[str, object]] = []
    for idx, feature in enumerate(FEATURE_COLUMNS):
        rows.append(
            {
                "metric_name": metric_config["name"],
                "region_label": region_label,
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
    importance_df["relative_importance_pct"] = (
        importance_df["importance_gain"] / total_gain * 100.0 if total_gain > 0 else 0.0
    )
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
    region_label: str,
) -> Image.Image:
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    plot_df = importance_df.sort_values("importance_gain", ascending=True)
    color = str(REGION_BY_LABEL[region_label]["color"])
    bars = ax.barh(plot_df["feature_name"], plot_df["importance_gain"], color=color, alpha=0.82)
    ax.set_xlabel("XGBoost importance (gain)")
    ax.set_ylabel("")
    ax.set_title(f"{metric_config['name']} - {region_label} feature importance")
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
    region_label: str,
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

    color = str(REGION_BY_LABEL[region_label]["color"])
    fig, ax = plt.subplots(figsize=(6.8, 6.3))
    ax.scatter(y_true_plot, y_pred_plot, s=8, alpha=0.28, color=color, edgecolors="none")
    min_val = float(min(np.min(y_true_plot), np.min(y_pred_plot)))
    max_val = float(max(np.max(y_true_plot), np.max(y_pred_plot)))
    ax.plot([min_val, max_val], [min_val, max_val], color="#4A4A4A", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Observed HFI effect")
    ax.set_ylabel("Predicted HFI effect")
    ax.set_title(f"{metric_config['name']} - {region_label} {split_label} prediction")
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


def build_shap_outputs(
    model: XGBRegressor,
    X_reference: pd.DataFrame,
    metric_config: dict[str, object],
    region_label: str,
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
            "region_label": region_label,
            "feature_column": FEATURE_COLUMNS,
            "feature_name": [FEATURE_LABELS[col] for col in FEATURE_COLUMNS],
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
            "shap_sample_size": len(X_sample),
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return shap_importance_df, X_sample, shap_values


def load_image_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_name in ["arial.ttf", "Arial.ttf"]:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def compose_prediction_grid_figure(
    prediction_images_by_region: dict[str, dict[str, Image.Image]],
    output_path: Path,
) -> Path:
    split_order = ["Train", "Test"]
    images = [
        prediction_images_by_region[region_label][split_label]
        for region_label in REGION_ORDER
        for split_label in split_order
    ]
    if len(images) != 4:
        raise ValueError(f"Expected 4 images for a 2x2 panel, got {len(images)}.")

    try:
        cell_width = max(image.width for image in images)
        cell_height = max(image.height for image in images)
        left_label_width = 300
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
            (REGION_ORDER[0], "Train"): (left_label_width + side_padding, top_padding),
            (REGION_ORDER[0], "Test"): (left_label_width + side_padding + cell_width + col_gap, top_padding),
            (REGION_ORDER[1], "Train"): (left_label_width + side_padding, top_padding + cell_height + row_gap),
            (REGION_ORDER[1], "Test"): (
                left_label_width + side_padding + cell_width + col_gap,
                top_padding + cell_height + row_gap,
            ),
        }

        for region_label in REGION_ORDER:
            row_top = positions[(region_label, "Train")][1]
            row_center_y = row_top + cell_height // 2
            text_bbox = draw.textbbox((0, 0), region_label, font=label_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.text(
                ((left_label_width - text_width) / 2, row_center_y - text_height / 2),
                region_label,
                fill="#333333",
                font=label_font,
            )

        for region_label in REGION_ORDER:
            for split_label in split_order:
                image = prediction_images_by_region[region_label][split_label]
                x0, y0 = positions[(region_label, split_label)]
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
        importance_images_by_metric[metric_name][region_label]
        for metric_name in metric_order
        for region_label in REGION_ORDER
    ]
    if len(images) != 4:
        raise ValueError(f"Expected 4 images for a 2x2 panel, got {len(images)}.")

    try:
        cell_width = max(image.width for image in images)
        cell_height = max(image.height for image in images)
        left_label_width = 360
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
            (metric_order[0], REGION_ORDER[0]): (left_label_width + side_padding, top_label_height + top_padding),
            (metric_order[0], REGION_ORDER[1]): (
                left_label_width + side_padding + cell_width + col_gap,
                top_label_height + top_padding,
            ),
            (metric_order[1], REGION_ORDER[0]): (
                left_label_width + side_padding,
                top_label_height + top_padding + cell_height + row_gap,
            ),
            (metric_order[1], REGION_ORDER[1]): (
                left_label_width + side_padding + cell_width + col_gap,
                top_label_height + top_padding + cell_height + row_gap,
            ),
        }

        for idx, region_label in enumerate(REGION_ORDER):
            x0 = left_label_width + side_padding + idx * (cell_width + col_gap)
            text_bbox = draw.textbbox((0, 0), region_label, font=col_font)
            text_width = text_bbox[2] - text_bbox[0]
            draw.text(
                (x0 + (cell_width - text_width) / 2, 18),
                region_label,
                fill="#333333",
                font=col_font,
            )

        for metric_name in metric_order:
            y0 = positions[(metric_name, REGION_ORDER[0])][1]
            row_center_y = y0 + cell_height // 2
            text_bbox = draw.textbbox((0, 0), metric_name, font=row_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.text(
                ((left_label_width - text_width) / 2, row_center_y - text_height / 2),
                metric_name,
                fill="#333333",
                font=row_font,
            )

        for metric_name in metric_order:
            for region_label in REGION_ORDER:
                image = importance_images_by_metric[metric_name][region_label]
                x0, y0 = positions[(metric_name, region_label)]
                x = x0 + (cell_width - image.width) // 2
                y = y0 + (cell_height - image.height) // 2
                canvas.paste(image, (x, y))

        canvas.save(output_path)
    finally:
        for image in images:
            image.close()
    return output_path


def get_overlay_feature_order(shap_tables: dict[str, pd.DataFrame]) -> list[str]:
    merged = pd.concat(
        [
            table[["feature_column", "mean_abs_shap"]].assign(region_label=region_label)
            for region_label, table in shap_tables.items()
        ],
        ignore_index=True,
    )
    order = (
        merged.groupby("feature_column", as_index=False)["mean_abs_shap"]
        .mean()
        .sort_values("mean_abs_shap", ascending=False)["feature_column"]
        .tolist()
    )
    return order


def build_overlay_shap_compare_figure(
    metric_config: dict[str, object],
    shap_plot_payloads: dict[str, dict[str, object]],
    shap_tables: dict[str, pd.DataFrame],
) -> Path:
    metric_title = str(metric_config["name"])
    plot_left = 0.10
    title_fontsize = 36
    axis_label_fontsize = 24
    tick_fontsize = 24
    region_label_fontsize = 30
    center_label_x = -0.29
    order_top_down = get_overlay_feature_order(shap_tables)
    feature_indices = [FEATURE_COLUMNS.index(feature) for feature in order_top_down]

    shap_limit = 0.0
    mean_abs_limit = 0.0
    for region_label in REGION_ORDER:
        shap_values = np.asarray(shap_plot_payloads[region_label]["shap_values"], dtype=float)
        if shap_values.size:
            shap_limit = max(shap_limit, float(np.nanmax(np.abs(shap_values[:, feature_indices]))))
        table = shap_tables[region_label]
        if not table.empty:
            mean_abs_limit = max(mean_abs_limit, float(table["mean_abs_shap"].max()))
    shap_limit = shap_limit * 1.05 if shap_limit > 0 else 1.0
    mean_abs_limit = mean_abs_limit * 1.15 if mean_abs_limit > 0 else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(20.4, 7.8), sharex=True)

    for idx, region_label in enumerate(REGION_ORDER):
        ax = axes[idx]
        X_plot = rename_feature_frame(shap_plot_payloads[region_label]["X_sample"][order_top_down])
        shap_values_plot = np.asarray(shap_plot_payloads[region_label]["shap_values"], dtype=float)[:, feature_indices]

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
            shap_tables[region_label]
            .set_index("feature_column")
            .loc[list(reversed(order_top_down))]
            .reset_index()
        )
        ax_top.barh(
            y=np.arange(len(bar_df)),
            width=bar_df["mean_abs_shap"].to_numpy(dtype=float),
            left=0.0,
            height=0.76,
            color=str(REGION_BY_LABEL[region_label]["color"]),
            alpha=0.36,
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
                region_label,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=region_label_fontsize,
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
            add_region_corner_label(ax, region_label, fontsize=region_label_fontsize)

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
    output_path = OUTPUT_DIR / f"xgboost_regional_{metric_config['slug']}_shap_overlay_compare_shared_x.png"
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def train_model_for_region(
    metric_config: dict[str, object],
    region_label: str,
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
        f"{metric_config['name']} / {region_label}: train_samples={len(y_train):,}, "
        f"test_samples={len(y_test):,}, features={', '.join(FEATURE_LABELS[feature] for feature in FEATURE_COLUMNS)}"
    )

    model = create_model()
    model.fit(X_train, y_train, verbose=False)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    metrics_df = pd.DataFrame(
        [
            {
                "metric_name": metric_config["name"],
                "region_label": region_label,
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

    importance_df = build_feature_importance_table(model, metric_config, region_label)
    shap_importance_df, shap_X_sample, shap_values = build_shap_outputs(
        model,
        X_test,
        metric_config,
        region_label,
    )

    train_prediction_image = build_prediction_scatter_image(
        y_train,
        train_pred,
        metric_config,
        region_label,
        split_label="Train",
        r2_value=float(metrics_df.at[0, "train_r2"]),
    )
    test_prediction_image = build_prediction_scatter_image(
        y_test,
        test_pred,
        metric_config,
        region_label,
        split_label="Test",
        r2_value=float(metrics_df.at[0, "test_r2"]),
    )
    importance_image = build_feature_importance_image(importance_df, metric_config, region_label)

    print(
        f"{metric_config['name']} / {region_label}: test_r2={metrics_df.at[0, 'test_r2']:.4f}, "
        f"test_rmse={metrics_df.at[0, 'test_rmse']:.6e}"
    )

    shap_plot_payload = {
        "region_label": region_label,
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


def model_parameter_table() -> pd.DataFrame:
    return pd.DataFrame(
        [{"parameter": key, "value": value} for key, value in MODEL_CONFIG.items()]
    )


def write_outputs(
    metric_config: dict[str, object],
    inventory_df: pd.DataFrame,
    region_lookup_summary: pd.DataFrame,
    alignment_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    importance_tables: dict[str, pd.DataFrame],
    shap_tables: dict[str, pd.DataFrame],
) -> Path:
    output_xlsx = OUTPUT_DIR / f"xgboost_regional_{metric_config['slug']}_hfi_effect_summary.xlsx"
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        inventory_df.to_excel(writer, sheet_name="file_inventory", index=False)
        region_lookup_summary.to_excel(writer, sheet_name="region_lookup", index=False)
        alignment_df.to_excel(writer, sheet_name="alignment_summary", index=False)
        metrics_df.to_excel(writer, sheet_name="model_metrics", index=False)
        model_parameter_table().to_excel(writer, sheet_name="model_parameters", index=False)
        for region_label, table in importance_tables.items():
            short_slug = str(REGION_BY_LABEL[region_label]["short_slug"])
            table.to_excel(writer, sheet_name=f"{short_slug}_importance", index=False)
        for region_label, table in shap_tables.items():
            short_slug = str(REGION_BY_LABEL[region_label]["short_slug"])
            table.to_excel(writer, sheet_name=f"{short_slug}_shap", index=False)
    return output_xlsx


def main() -> None:
    print("=== Regional XGBoost Modeling: Northern India vs East-Central China ===")
    print("Using fixed XGBoost parameters because region-specific tuned workbooks are not available.")
    region_lookup, region_lookup_summary = build_region_year_lookup()
    importance_images_by_metric: dict[str, dict[str, Image.Image]] = {}

    for metric_config in METRIC_CONFIGS:
        print(f"\n=== Processing {metric_config['name']} ===")
        pairs, inventory_df = discover_file_pairs(metric_config)
        datasets_by_region, alignment_df = build_datasets_by_region(pairs, metric_config, region_lookup)

        all_metrics: list[pd.DataFrame] = []
        importance_tables: dict[str, pd.DataFrame] = {}
        shap_tables: dict[str, pd.DataFrame] = {}
        shap_plot_payloads: dict[str, dict[str, object]] = {}
        prediction_images_by_region: dict[str, dict[str, Image.Image]] = {}

        for region_label in REGION_ORDER:
            print(f"\n--- Training {metric_config['name']} / {region_label} ---")
            metrics_df, importance_df, shap_importance_df, shap_plot_payload, prediction_images, importance_image = train_model_for_region(
                metric_config,
                region_label,
                datasets_by_region[region_label],
            )
            all_metrics.append(metrics_df)
            importance_tables[region_label] = importance_df
            shap_tables[region_label] = shap_importance_df
            shap_plot_payloads[region_label] = shap_plot_payload
            prediction_images_by_region[region_label] = prediction_images
            importance_images_by_metric.setdefault(str(metric_config["name"]), {})[region_label] = importance_image

        prediction_compare_path = compose_prediction_grid_figure(
            prediction_images_by_region,
            OUTPUT_DIR / f"xgboost_regional_{metric_config['slug']}_train_test_prediction_panel_2x2.png",
        )
        print(f"{metric_config['name']} combined train/test panel saved: {prediction_compare_path.name}")

        overlay_compare_path = build_overlay_shap_compare_figure(metric_config, shap_plot_payloads, shap_tables)
        print(f"{metric_config['name']} shared-x SHAP overlay saved: {overlay_compare_path.name}")

        metrics_all = pd.concat(all_metrics, ignore_index=True)
        output_xlsx = write_outputs(
            metric_config,
            inventory_df,
            region_lookup_summary,
            alignment_df,
            metrics_all,
            importance_tables,
            shap_tables,
        )
        print(f"{metric_config['name']} summary workbook saved to: {output_xlsx}")

    importance_compare_path = compose_importance_grid_figure(
        importance_images_by_metric,
        OUTPUT_DIR / "xgboost_regional_gain_importance_panel_2x2.png",
    )
    print(f"Combined gain importance panel saved: {importance_compare_path}")
    print("\nRegional XGBoost modeling completed.")


if __name__ == "__main__":
    main()
