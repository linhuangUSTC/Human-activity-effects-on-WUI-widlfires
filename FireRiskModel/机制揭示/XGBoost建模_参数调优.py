#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称:
    XGBoost 分 WUI 类型建模参数调优版

功能简介:
    1. 沿用 Interface(face_*) 与 Intermix(mix_*) 数据读取、匹配和合并逻辑。
    2. 保留 XGBoost 回归建模、训练集/测试集划分、模型精度评估和特征重要性输出。
    3. 不计算 SHAP，不输出任何 SHAP 图或诊断图片。
    4. 对 Fire frequency / Normalized burn area 的 Interface 与 Intermix 分别调参建模，输出四个调参结果工作簿。
    5. 通过验证集随机搜索 XGBoost 参数，选择验证 RMSE 最低的参数组合，再用训练集重新训练最终模型。
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


OUTPUT_DIR = Path(r"B:\WUI\Picture")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
WUI_TYPE_CODES = {"Interface": 0, "Intermix": 1}
WUI_TYPES = ("Interface", "Intermix")

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
RANDOM_STATE = 42

# 如果数据量很大，调参阶段可以抽样以控制耗时；最终模型仍使用完整训练集。
# 设为 None 表示调参也使用完整训练集。
TUNING_MAX_ROWS = 600_000
N_TUNING_TRIALS = 36
EARLY_STOPPING_ROUNDS = 40
PRINT_TRIAL_DETAILS = False

BASE_MODEL_CONFIG = {
    "objective": "reg:squarederror",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "tree_method": "hist",
    "eval_metric": "rmse",
}

REFERENCE_PARAMS = {
    "n_estimators": 240,
    "max_depth": 5,
    "learning_rate": 0.06,
    "subsample": 0.80,
    "colsample_bytree": 0.85,
    "min_child_weight": 5,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "gamma": 0.0,
}

PARAMETER_SEARCH_SPACE = {
    "n_estimators": [300, 500, 800, 1200, 1600],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.015, 0.02, 0.03, 0.05, 0.07, 0.10],
    "subsample": [0.65, 0.75, 0.85, 0.95],
    "colsample_bytree": [0.65, 0.75, 0.85, 0.95],
    "min_child_weight": [1, 3, 5, 8, 12],
    "reg_alpha": [0.0, 0.02, 0.05, 0.10, 0.30],
    "reg_lambda": [0.7, 1.0, 1.5, 2.0, 3.0],
    "gamma": [0.0, 0.03, 0.08, 0.15, 0.30],
}
INTEGER_PARAMETER_NAMES = {"n_estimators", "max_depth", "min_child_weight"}


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

    pairs_by_type: dict[str, list[tuple[str, Path, Path]]] = {wui_type: [] for wui_type in WUI_TYPES}
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
                "WUI_Type_Code": WUI_TYPE_CODES[wui_type],
                "effect_csv": str(effect_path),
                "basic_csv": str(basic_path),
            }
        )

    inventory_df = pd.DataFrame(inventory_rows)
    type_counts = inventory_df["WUI_Type"].value_counts().to_dict() if not inventory_df.empty else {}
    print(
        f"{metric_config['name']} matched categories: "
        f"Interface={type_counts.get('Interface', 0)}, "
        f"Intermix={type_counts.get('Intermix', 0)}, "
        f"Total={len(matched_ids)}"
    )
    return pairs_by_type, inventory_df


def read_effect_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        usecols=["GRID_ID", "Year", "HFI", TARGET_COLUMN],
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
                    "WUI_Type_Code": WUI_TYPE_CODES[wui_type],
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
                "WUI_Type_Code": WUI_TYPE_CODES[wui_type],
                "effect_rows": len(effect_df),
                "basic_rows": len(basic_df),
                "matched_rows": len(merged),
                "hfi_mean_abs_diff": float(hfi_abs_diff.mean()),
                "hfi_max_abs_diff": float(hfi_abs_diff.max()),
            }
        )

        feature_frame = merged[BASIC_FEATURE_COLUMNS + [TARGET_COLUMN]].astype(np.float32).copy()
        feature_frame["WUI_Type"] = wui_type
        data_parts.append(feature_frame)

        del effect_df, basic_df, merged, feature_frame
        gc.collect()

    if not data_parts:
        raise ValueError(f"No matched samples available for {metric_config['name']} / {wui_type} modeling.")

    dataset = pd.concat(data_parts, ignore_index=True)
    alignment_df = pd.DataFrame(alignment_rows)
    dataset_summary = (
        dataset.groupby("WUI_Type", as_index=False)
        .agg(
            n_samples=(TARGET_COLUMN, "size"),
            target_mean=(TARGET_COLUMN, "mean"),
            target_std=(TARGET_COLUMN, "std"),
            GDP_mean=("GDP", "mean"),
            Population_density_mean=("PopD", "mean"),
            Road_density_mean=("Roadsd", "mean"),
            Cropland_mean=("Cropland_Area_km2", "mean"),
            Built_environment_mean=("GUB_area_km2", "mean"),
            Pasture_mean=("Pasture_Band2_Mean", "mean"),
        )
        .sort_values("WUI_Type")
        .reset_index(drop=True)
    )
    dataset_summary.insert(0, "metric_name", metric_config["name"])
    print(f"{metric_config['name']} / {wui_type} dataset samples={len(dataset):,}")
    return dataset, alignment_df, dataset_summary


def normalize_parameter_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def coerce_parameter_types(params: dict[str, object]) -> dict[str, object]:
    typed_params: dict[str, object] = {}
    for name, value in params.items():
        value = normalize_parameter_value(value)
        if name in INTEGER_PARAMETER_NAMES:
            typed_params[name] = int(round(float(value)))
        else:
            typed_params[name] = float(value)
    return typed_params


def sample_parameter_candidates() -> list[dict[str, object]]:
    rng = np.random.default_rng(RANDOM_STATE)
    candidates = [REFERENCE_PARAMS.copy()]
    seen = {tuple(sorted(REFERENCE_PARAMS.items()))}

    while len(candidates) < N_TUNING_TRIALS + 1:
        params = {
            name: normalize_parameter_value(values[int(rng.integers(0, len(values)))])
            for name, values in PARAMETER_SEARCH_SPACE.items()
        }
        key = tuple(sorted(params.items()))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(coerce_parameter_types(params))

    return candidates


def create_model(params: dict[str, object]) -> XGBRegressor:
    config = BASE_MODEL_CONFIG.copy()
    config.update(params)
    return XGBRegressor(**config)


def fit_with_validation(
    params: dict[str, object],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
) -> XGBRegressor:
    model = create_model(params)
    if EARLY_STOPPING_ROUNDS <= 0:
        model.fit(X_train, y_train, verbose=False)
        return model

    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=False,
        )
        return model
    except TypeError as exc:
        if "early_stopping_rounds" not in str(exc):
            raise

    try:
        params_with_early_stop = params.copy()
        params_with_early_stop["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
        model = create_model(params_with_early_stop)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        return model
    except TypeError:
        model = create_model(params)
        model.fit(X_train, y_train, verbose=False)
        return model


def get_best_iteration(model: XGBRegressor) -> int | None:
    value = getattr(model, "best_iteration", None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_r2": float(r2_score(y_true, y_pred)),
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
    }


def maybe_sample_tuning_pool(
    X: pd.DataFrame,
    y: np.ndarray,
    metric_name: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    if TUNING_MAX_ROWS is None or len(X) <= TUNING_MAX_ROWS:
        return X.reset_index(drop=True), y

    rng = np.random.default_rng(RANDOM_STATE)
    sampled_index = np.sort(rng.choice(len(X), size=TUNING_MAX_ROWS, replace=False))
    print(
        f"{metric_name} tuning sample: {len(sampled_index):,}/{len(X):,} rows "
        f"(TUNING_MAX_ROWS={TUNING_MAX_ROWS:,})"
    )
    return X.iloc[sampled_index].reset_index(drop=True), y[sampled_index]


def build_feature_importance_table(
    model: XGBRegressor,
    metric_config: dict[str, object],
    wui_type: str,
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
                "WUI_Type": wui_type,
                "feature_column": feature,
                "feature_name": FEATURE_LABELS[feature],
                "importance_gain": float(gain_scores.get(feature, gain_scores.get(f"f{idx}", 0.0))),
                "importance_weight": float(weight_scores.get(feature, weight_scores.get(f"f{idx}", 0.0))),
                "importance_cover": float(cover_scores.get(feature, cover_scores.get(f"f{idx}", 0.0))),
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


def run_parameter_search(
    X_tune: pd.DataFrame,
    y_tune: np.ndarray,
    metric_config: dict[str, object],
    wui_type: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_tune,
        y_tune,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    candidates = sample_parameter_candidates()
    trial_rows: list[dict[str, object]] = []

    for trial_id, params in enumerate(candidates):
        model = fit_with_validation(params, X_fit, y_fit, X_valid, y_valid)
        valid_pred = model.predict(X_valid)
        fit_pred = model.predict(X_fit)
        row = {
            "metric_name": metric_config["name"],
            "WUI_Type": wui_type,
            "trial_id": trial_id,
            "trial_type": "reference_original" if trial_id == 0 else "random_search",
            "n_fit": len(y_fit),
            "n_valid": len(y_valid),
            "best_iteration": get_best_iteration(model),
        }
        row.update(params)
        row.update(evaluate_predictions(y_fit, fit_pred, "fit"))
        row.update(evaluate_predictions(y_valid, valid_pred, "valid"))
        trial_rows.append(row)
        if PRINT_TRIAL_DETAILS:
            print(
                f"{metric_config['name']} / {wui_type} trial {trial_id:02d}: "
                f"valid_r2={row['valid_r2']:.4f}, valid_rmse={row['valid_rmse']:.6e}, "
                f"params={params}"
            )

        del model, valid_pred, fit_pred
        gc.collect()

    tuning_df = pd.DataFrame(trial_rows).sort_values(["valid_rmse", "valid_mae"], ascending=[True, True])
    best_row = tuning_df.iloc[0].to_dict()
    best_params = coerce_parameter_types({name: best_row[name] for name in PARAMETER_SEARCH_SPACE})

    best_iteration = best_row.get("best_iteration")
    if pd.notna(best_iteration):
        best_params["n_estimators"] = max(1, int(best_iteration) + 1)

    print(
        f"{metric_config['name']} / {wui_type} best trial={int(best_row['trial_id'])}, "
        f"valid_r2={best_row['valid_r2']:.4f}, valid_rmse={best_row['valid_rmse']:.6e}, "
        f"final_params={best_params}"
    )
    return tuning_df.reset_index(drop=True), best_params


def train_tuned_model(
    dataset: pd.DataFrame,
    metric_config: dict[str, object],
    wui_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        f"{metric_config['name']} / {wui_type} split: train={len(y_train):,}, test={len(y_test):,}, "
        f"features={', '.join(FEATURE_LABELS[feature] for feature in FEATURE_COLUMNS)}"
    )

    tuning_label = f"{metric_config['name']} / {wui_type}"
    X_tune, y_tune = maybe_sample_tuning_pool(X_train, y_train, tuning_label)
    tuning_df, best_params = run_parameter_search(X_tune, y_tune, metric_config, wui_type)

    final_model = create_model(best_params)
    final_model.fit(X_train, y_train, verbose=False)
    train_pred = final_model.predict(X_train)
    test_pred = final_model.predict(X_test)

    baseline_row = tuning_df.loc[tuning_df["trial_type"] == "reference_original"].iloc[0]
    best_trial_row = tuning_df.iloc[0]
    metrics_row = {
        "metric_name": metric_config["name"],
        "WUI_Type": wui_type,
        "model_name": f"{wui_type}_Tuned_XGBoost",
        "n_total": len(dataset),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_tuning_pool": len(y_tune),
        "n_tuning_trials": len(tuning_df),
        "best_trial_id": int(best_trial_row["trial_id"]),
        "baseline_valid_r2": float(baseline_row["valid_r2"]),
        "baseline_valid_rmse": float(baseline_row["valid_rmse"]),
        "best_valid_r2": float(best_trial_row["valid_r2"]),
        "best_valid_rmse": float(best_trial_row["valid_rmse"]),
    }
    metrics_row.update(evaluate_predictions(y_train, train_pred, "train"))
    metrics_row.update(evaluate_predictions(y_test, test_pred, "test"))
    metrics_df = pd.DataFrame([metrics_row])

    importance_df = build_feature_importance_table(final_model, metric_config, wui_type)
    best_params_df = pd.DataFrame(
        [
            {
                "metric_name": metric_config["name"],
                "WUI_Type": wui_type,
                "parameter": parameter,
                "value": value,
            }
            for parameter, value in best_params.items()
        ]
    )

    print(
        f"{metric_config['name']} / {wui_type} tuned model: "
        f"test_r2={metrics_df.at[0, 'test_r2']:.4f}, "
        f"test_rmse={metrics_df.at[0, 'test_rmse']:.6e}"
    )

    del X, y, X_train, X_test, y_train, y_test
    del X_tune, y_tune, train_pred, test_pred, final_model, dataset
    gc.collect()
    return metrics_df, tuning_df, best_params_df, importance_df


def write_outputs(
    metric_config: dict[str, object],
    wui_type: str,
    metrics_df: pd.DataFrame,
    tuning_df: pd.DataFrame,
    best_params_df: pd.DataFrame,
    importance_df: pd.DataFrame,
) -> Path:
    output_xlsx = OUTPUT_DIR / f"xgboost_{metric_config['slug']}_{wui_type.lower()}_hfi_effect_tuned_summary.xlsx"
    metrics_sheet_df = metrics_df.transpose().copy()
    metrics_sheet_df.columns = ["value"]
    metrics_sheet_df.index.name = "metric"
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        metrics_sheet_df.to_excel(writer, sheet_name="model_metrics")
        tuning_df.to_excel(writer, sheet_name="parameter_trials", index=False)
        best_params_df.to_excel(writer, sheet_name="best_parameters", index=False)
        importance_df.to_excel(writer, sheet_name="feature_importance", index=False)
    return output_xlsx


def main() -> None:
    print("=== XGBoost Parameter Tuning by WUI Type ===")
    for metric_config in METRIC_CONFIGS:
        print(f"\n=== Processing {metric_config['name']} ===")
        pairs_by_type, inventory_df = discover_file_pairs(metric_config)
        for wui_type in WUI_TYPES:
            print(f"\n--- Processing {metric_config['name']} / {wui_type} ---")
            dataset, _alignment_df, _dataset_summary = build_dataset_for_wui_type(
                wui_type,
                pairs_by_type[wui_type],
                metric_config,
            )
            metrics_df, tuning_df, best_params_df, importance_df = train_tuned_model(
                dataset,
                metric_config,
                wui_type,
            )
            output_xlsx = write_outputs(
                metric_config,
                wui_type,
                metrics_df,
                tuning_df,
                best_params_df,
                importance_df,
            )
            print(f"{metric_config['name']} / {wui_type} tuning workbook saved to: {output_xlsx}")

    print("\nXGBoost parameter tuning by WUI type completed.")


if __name__ == "__main__":
    main()
