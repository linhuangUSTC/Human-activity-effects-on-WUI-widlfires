#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称:
    XGBoost建模_变量保留检验.py

功能简介:
    1. 复用 XGBoost建模.py 的数据读取、样本配对与 tuned 参数口径。
    2. 针对 Fire frequency / Normalized burn area 的 Interface 与 Intermix 四套模型，
       分别建立全变量 baseline 模型。
    3. 逐个执行 drop-one 变量检验:
       - 去掉该变量后 test R2 是否下降；
       - 去掉该变量后 SHAP 结构是否发生明显变化。
    4. 依据上述两个标准，输出建议保留/剔除的变量清单。

说明:
    这里的“显著”指建模意义上的重要性判断，不是 p 值统计检验。
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


OUTPUT_DIR = Path(r"B:\WUI\Picture")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_XLSX = OUTPUT_DIR / "xgboost_variable_retention_summary.xlsx"
OUTPUT_TXT = OUTPUT_DIR / "xgboost_variable_retention_summary.txt"

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

TEST_SIZE = 0.20
RANDOM_STATE = 42
SHAP_SAMPLE_SIZE = 5000
MAX_MODEL_ROWS: int | None = None

R2_GAIN_THRESHOLD = 0.002
SHAP_SHARE_SHIFT_THRESHOLD = 0.10
SHAP_RANK_CORR_THRESHOLD = 0.90

BASE_MODEL_CONFIG = {
    "objective": "reg:squarederror",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "tree_method": "hist",
    "eval_metric": "rmse",
}
INTEGER_PARAMETER_NAMES = {"n_estimators", "max_depth", "min_child_weight"}
MODEL_CONFIG_CACHE: dict[tuple[str, str], dict[str, object]] = {}


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


def discover_file_pairs(metric_config: dict[str, object]) -> dict[str, list[tuple[str, Path, Path]]]:
    effect_dir = Path(metric_config["effect_dir"])
    basic_dir = Path(metric_config["basic_dir"])
    effect_files = {extract_effect_category_id(path): path for path in sorted(effect_dir.glob("HFI_effect_*.csv"))}
    basic_files = {path.stem: path for path in sorted(basic_dir.glob("*.csv"))}

    matched_ids = sorted(set(effect_files).intersection(basic_files))
    pairs_by_type: dict[str, list[tuple[str, Path, Path]]] = {"Interface": [], "Intermix": []}
    for category_id in matched_ids:
        wui_type = infer_wui_type(category_id)
        pairs_by_type[wui_type].append((category_id, effect_files[category_id], basic_files[category_id]))
    return pairs_by_type


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
    pairs: list[tuple[str, Path, Path]],
) -> pd.DataFrame:
    data_parts: list[pd.DataFrame] = []
    for _, effect_path, basic_path in pairs:
        effect_df = read_effect_csv(effect_path)
        basic_df = read_basic_csv(basic_path)
        merged = effect_df.merge(basic_df, how="inner", on=["GRID_ID", "Year"], validate="one_to_one")
        if not merged.empty:
            data_parts.append(merged[FEATURE_COLUMNS + [TARGET_COLUMN]].astype(np.float32))
        del effect_df, basic_df, merged
        gc.collect()

    if not data_parts:
        raise ValueError("No matched samples available after merging effect and basic tables.")
    dataset = pd.concat(data_parts, ignore_index=True)
    if MAX_MODEL_ROWS is not None and len(dataset) > MAX_MODEL_ROWS:
        dataset = dataset.sample(n=MAX_MODEL_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)
    return dataset


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
    return model_config


def create_model(metric_config: dict[str, object], wui_type: str) -> XGBRegressor:
    return XGBRegressor(**load_tuned_parameters(metric_config, wui_type))


def sample_shap_indices(n_rows: int) -> np.ndarray:
    if n_rows <= SHAP_SAMPLE_SIZE:
        return np.arange(n_rows, dtype=int)
    rng = np.random.default_rng(RANDOM_STATE)
    return np.sort(rng.choice(n_rows, size=SHAP_SAMPLE_SIZE, replace=False))


def build_split(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    X = dataset[FEATURE_COLUMNS].astype(np.float32).copy()
    y = dataset[TARGET_COLUMN].to_numpy(dtype=np.float32)
    row_indices = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(
        row_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]
    shap_idx = sample_shap_indices(len(X_test))
    return X_train, X_test, y_train, y_test, shap_idx


def build_shap_importance_table(
    model: XGBRegressor,
    X_reference: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_reference, check_additivity=False)
    shap_values = np.asarray(explanation.values, dtype=float)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    table = pd.DataFrame(
        {
            "feature_column": feature_columns,
            "feature_name": [FEATURE_LABELS[col] for col in feature_columns],
            "mean_abs_shap": mean_abs_shap,
        }
    )
    total = float(table["mean_abs_shap"].sum())
    if total > 0:
        table["shap_share"] = table["mean_abs_shap"] / total
    else:
        table["shap_share"] = 0.0
    table["rank"] = table["mean_abs_shap"].rank(method="dense", ascending=False).astype(int)
    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def fit_model_with_features(
    metric_config: dict[str, object],
    wui_type: str,
    X_train_full: pd.DataFrame,
    X_test_full: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    shap_idx: np.ndarray,
    feature_columns: list[str],
) -> tuple[dict[str, float], pd.DataFrame]:
    X_train = X_train_full[feature_columns].copy()
    X_test = X_test_full[feature_columns].copy()
    model = create_model(metric_config, wui_type)
    model.fit(X_train, y_train, verbose=False)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    metrics = {
        "train_r2": float(r2_score(y_train, train_pred)),
        "test_r2": float(r2_score(y_test, test_pred)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_pred))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
        "train_mae": float(mean_absolute_error(y_train, train_pred)),
        "test_mae": float(mean_absolute_error(y_test, test_pred)),
    }

    X_shap = X_test.iloc[shap_idx].reset_index(drop=True)
    shap_table = build_shap_importance_table(model, X_shap, feature_columns)

    del X_train, X_test, X_shap, train_pred, test_pred, model
    gc.collect()
    return metrics, shap_table


def safe_spearman(left: pd.Series, right: pd.Series) -> float:
    if left.nunique(dropna=False) <= 1 or right.nunique(dropna=False) <= 1:
        return np.nan
    value = left.corr(right, method="spearman")
    return float(value) if pd.notna(value) else np.nan


def compare_shap_structure(
    baseline_shap: pd.DataFrame,
    reduced_shap: pd.DataFrame,
    removed_feature: str,
) -> dict[str, object]:
    aligned_baseline = (
        pd.DataFrame({"feature_column": FEATURE_COLUMNS})
        .merge(baseline_shap[["feature_column", "mean_abs_shap", "shap_share"]], on="feature_column", how="left")
        .fillna({"mean_abs_shap": 0.0, "shap_share": 0.0})
    )
    aligned_reduced = (
        pd.DataFrame({"feature_column": FEATURE_COLUMNS})
        .merge(reduced_shap[["feature_column", "mean_abs_shap", "shap_share"]], on="feature_column", how="left")
        .fillna({"mean_abs_shap": 0.0, "shap_share": 0.0})
    )

    baseline_share = aligned_baseline.set_index("feature_column")["shap_share"]
    reduced_share = aligned_reduced.set_index("feature_column")["shap_share"]
    shap_share_shift = float(0.5 * np.abs(baseline_share - reduced_share).sum())

    common_features = [feature for feature in FEATURE_COLUMNS if feature != removed_feature]
    baseline_common = baseline_shap.set_index("feature_column").loc[common_features, "mean_abs_shap"]
    reduced_common = reduced_shap.set_index("feature_column").loc[common_features, "mean_abs_shap"]
    common_rank_spearman = safe_spearman(baseline_common, reduced_common)

    baseline_top_feature = str(baseline_shap.iloc[0]["feature_column"])
    reduced_top_feature = str(reduced_shap.iloc[0]["feature_column"])
    top_feature_changed = baseline_top_feature != reduced_top_feature

    candidate_share = float(baseline_share.get(removed_feature, 0.0))
    candidate_rank = int(
        baseline_shap.loc[baseline_shap["feature_column"] == removed_feature, "rank"].iloc[0]
    )

    return {
        "baseline_top_feature": baseline_top_feature,
        "reduced_top_feature": reduced_top_feature,
        "top_feature_changed": top_feature_changed,
        "candidate_full_shap_share": candidate_share,
        "candidate_full_shap_rank": candidate_rank,
        "shap_share_shift": shap_share_shift,
        "common_rank_spearman": common_rank_spearman,
    }


def build_decision(
    delta_test_r2: float,
    shap_share_shift: float,
    common_rank_spearman: float,
    top_feature_changed: bool,
) -> tuple[bool, bool, bool, str]:
    retain_by_r2 = delta_test_r2 >= R2_GAIN_THRESHOLD
    retain_by_shap = (
        shap_share_shift >= SHAP_SHARE_SHIFT_THRESHOLD
        or top_feature_changed
        or (pd.notna(common_rank_spearman) and common_rank_spearman < SHAP_RANK_CORR_THRESHOLD)
    )
    retain_variable = retain_by_r2 or retain_by_shap

    reasons: list[str] = []
    if retain_by_r2:
        reasons.append(f"delta_test_r2={delta_test_r2:.4f}")
    if shap_share_shift >= SHAP_SHARE_SHIFT_THRESHOLD:
        reasons.append(f"shap_shift={shap_share_shift:.3f}")
    if pd.notna(common_rank_spearman) and common_rank_spearman < SHAP_RANK_CORR_THRESHOLD:
        reasons.append(f"rank_corr={common_rank_spearman:.3f}")
    if top_feature_changed:
        reasons.append("top_feature_changed")
    if not reasons:
        reasons.append("no_material_change")
    return retain_by_r2, retain_by_shap, retain_variable, "; ".join(reasons)


def feature_order_string(shap_table: pd.DataFrame) -> str:
    return " > ".join(FEATURE_LABELS[feature] for feature in shap_table["feature_column"].tolist())


def run_variable_screen_for_model(
    metric_config: dict[str, object],
    wui_type: str,
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train, X_test, y_train, y_test, shap_idx = build_split(dataset)
    baseline_metrics, baseline_shap = fit_model_with_features(
        metric_config,
        wui_type,
        X_train,
        X_test,
        y_train,
        y_test,
        shap_idx,
        FEATURE_COLUMNS,
    )

    baseline_row = pd.DataFrame(
        [
            {
                "metric_name": metric_config["name"],
                "WUI_Type": wui_type,
                "n_total": len(dataset),
                "n_train": len(y_train),
                "n_test": len(y_test),
                "train_r2": baseline_metrics["train_r2"],
                "test_r2": baseline_metrics["test_r2"],
                "train_rmse": baseline_metrics["train_rmse"],
                "test_rmse": baseline_metrics["test_rmse"],
                "train_mae": baseline_metrics["train_mae"],
                "test_mae": baseline_metrics["test_mae"],
                "baseline_top_feature": FEATURE_LABELS[str(baseline_shap.iloc[0]["feature_column"])],
                "baseline_shap_order": feature_order_string(baseline_shap),
            }
        ]
    )

    detail_rows: list[dict[str, object]] = []
    for feature in FEATURE_COLUMNS:
        reduced_features = [column for column in FEATURE_COLUMNS if column != feature]
        reduced_metrics, reduced_shap = fit_model_with_features(
            metric_config,
            wui_type,
            X_train,
            X_test,
            y_train,
            y_test,
            shap_idx,
            reduced_features,
        )

        delta_train_r2 = baseline_metrics["train_r2"] - reduced_metrics["train_r2"]
        delta_test_r2 = baseline_metrics["test_r2"] - reduced_metrics["test_r2"]
        shap_compare = compare_shap_structure(baseline_shap, reduced_shap, feature)
        retain_by_r2, retain_by_shap, retain_variable, decision_reason = build_decision(
            delta_test_r2=delta_test_r2,
            shap_share_shift=float(shap_compare["shap_share_shift"]),
            common_rank_spearman=float(shap_compare["common_rank_spearman"])
            if pd.notna(shap_compare["common_rank_spearman"])
            else np.nan,
            top_feature_changed=bool(shap_compare["top_feature_changed"]),
        )

        detail_rows.append(
            {
                "metric_name": metric_config["name"],
                "WUI_Type": wui_type,
                "feature_column": feature,
                "feature_name": FEATURE_LABELS[feature],
                "baseline_test_r2": baseline_metrics["test_r2"],
                "reduced_test_r2": reduced_metrics["test_r2"],
                "delta_test_r2": delta_test_r2,
                "baseline_train_r2": baseline_metrics["train_r2"],
                "reduced_train_r2": reduced_metrics["train_r2"],
                "delta_train_r2": delta_train_r2,
                "candidate_full_shap_share": shap_compare["candidate_full_shap_share"],
                "candidate_full_shap_rank": shap_compare["candidate_full_shap_rank"],
                "baseline_top_feature": FEATURE_LABELS[str(shap_compare["baseline_top_feature"])],
                "reduced_top_feature": FEATURE_LABELS[str(shap_compare["reduced_top_feature"])],
                "top_feature_changed": shap_compare["top_feature_changed"],
                "shap_share_shift": shap_compare["shap_share_shift"],
                "common_rank_spearman": shap_compare["common_rank_spearman"],
                "retain_by_r2": retain_by_r2,
                "retain_by_shap": retain_by_shap,
                "retain_variable": retain_variable,
                "decision_reason": decision_reason,
            }
        )
        del reduced_metrics, reduced_shap
        gc.collect()

    detail_df = pd.DataFrame(detail_rows).sort_values(
        by=["retain_variable", "delta_test_r2", "shap_share_shift"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    retained_features = detail_df.loc[detail_df["retain_variable"], "feature_name"].tolist()
    dropped_features = detail_df.loc[~detail_df["retain_variable"], "feature_name"].tolist()
    summary_row = pd.DataFrame(
        [
            {
                "metric_name": metric_config["name"],
                "WUI_Type": wui_type,
                "baseline_test_r2": baseline_metrics["test_r2"],
                "baseline_train_r2": baseline_metrics["train_r2"],
                "retained_feature_count": len(retained_features),
                "dropped_feature_count": len(dropped_features),
                "retained_features": "; ".join(retained_features) if retained_features else "None",
                "dropped_features": "; ".join(dropped_features) if dropped_features else "None",
            }
        ]
    )

    del X_train, X_test, y_train, y_test, shap_idx, baseline_shap
    gc.collect()
    return baseline_row, detail_df, summary_row


def build_overall_feature_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in FEATURE_COLUMNS:
        subset = detail_df.loc[detail_df["feature_column"] == feature].copy()
        kept_models = subset.loc[subset["retain_variable"], ["metric_name", "WUI_Type"]].apply(
            lambda row: f"{row['metric_name']} / {row['WUI_Type']}",
            axis=1,
        ).tolist()
        dropped_models = subset.loc[~subset["retain_variable"], ["metric_name", "WUI_Type"]].apply(
            lambda row: f"{row['metric_name']} / {row['WUI_Type']}",
            axis=1,
        ).tolist()
        rows.append(
            {
                "feature_column": feature,
                "feature_name": FEATURE_LABELS[feature],
                "tested_model_count": len(subset),
                "retained_model_count": int(subset["retain_variable"].sum()),
                "dropped_model_count": int((~subset["retain_variable"]).sum()),
                "mean_delta_test_r2": float(subset["delta_test_r2"].mean()),
                "max_delta_test_r2": float(subset["delta_test_r2"].max()),
                "mean_shap_share_shift": float(subset["shap_share_shift"].mean()),
                "max_shap_share_shift": float(subset["shap_share_shift"].max()),
                "kept_in_models": "; ".join(kept_models) if kept_models else "None",
                "dropped_in_models": "; ".join(dropped_models) if dropped_models else "None",
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["retained_model_count", "mean_delta_test_r2", "mean_shap_share_shift"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def write_text_summary(model_summary_df: pd.DataFrame) -> None:
    lines = [
        "XGBoost variable retention screening",
        f"Decision rules: delta_test_r2 >= {R2_GAIN_THRESHOLD:.3f} or shap_share_shift >= {SHAP_SHARE_SHIFT_THRESHOLD:.2f} or common_rank_spearman < {SHAP_RANK_CORR_THRESHOLD:.2f} or top_feature_changed",
        "",
    ]
    for _, row in model_summary_df.iterrows():
        lines.append(f"{row['metric_name']} / {row['WUI_Type']}")
        lines.append(f"  Keep: {row['retained_features']}")
        lines.append(f"  Drop: {row['dropped_features']}")
        lines.append("")
    OUTPUT_TXT.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    decision_rules_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    model_summary_df: pd.DataFrame,
    overall_feature_df: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        decision_rules_df.to_excel(writer, sheet_name="decision_rules", index=False)
        model_summary_df.to_excel(writer, sheet_name="retained_summary", index=False)
        overall_feature_df.to_excel(writer, sheet_name="feature_overall_summary", index=False)
        baseline_df.to_excel(writer, sheet_name="baseline_models", index=False)
        detail_df.to_excel(writer, sheet_name="variable_tests", index=False)
    write_text_summary(model_summary_df)


def main() -> None:
    print("=== XGBoost Variable Retention Screening ===")
    decision_rules_df = pd.DataFrame(
        [
            {
                "test_r2_gain_threshold": R2_GAIN_THRESHOLD,
                "shap_share_shift_threshold": SHAP_SHARE_SHIFT_THRESHOLD,
                "common_rank_spearman_threshold": SHAP_RANK_CORR_THRESHOLD,
                "shap_sample_size": SHAP_SAMPLE_SIZE,
                "test_size": TEST_SIZE,
                "max_model_rows": MAX_MODEL_ROWS if MAX_MODEL_ROWS is not None else "All rows",
                "decision_logic": "retain if R2 improves materially or SHAP structure changes materially",
            }
        ]
    )

    baseline_tables: list[pd.DataFrame] = []
    detail_tables: list[pd.DataFrame] = []
    summary_tables: list[pd.DataFrame] = []

    for metric_config in METRIC_CONFIGS:
        pairs_by_type = discover_file_pairs(metric_config)
        for wui_type in ["Interface", "Intermix"]:
            print(f"Processing {metric_config['name']} / {wui_type} ...")
            dataset = build_dataset_for_wui_type(pairs_by_type[wui_type])
            baseline_df, detail_df, summary_df = run_variable_screen_for_model(metric_config, wui_type, dataset)
            baseline_tables.append(baseline_df)
            detail_tables.append(detail_df)
            summary_tables.append(summary_df)

            keep_list = summary_df.at[0, "retained_features"]
            print(f"  Keep: {keep_list}")
            del dataset, baseline_df, detail_df, summary_df
            gc.collect()

    baseline_all = pd.concat(baseline_tables, ignore_index=True)
    detail_all = pd.concat(detail_tables, ignore_index=True)
    summary_all = pd.concat(summary_tables, ignore_index=True)
    overall_feature_df = build_overall_feature_summary(detail_all)
    write_outputs(decision_rules_df, baseline_all, detail_all, summary_all, overall_feature_df)

    print(f"Summary workbook saved to: {OUTPUT_XLSX}")
    print(f"Retention text summary saved to: {OUTPUT_TXT}")
    print("Completed.")


if __name__ == "__main__":
    main()
