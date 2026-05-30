#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称: Human Footprint 影响组均值极值对应 GDP / Population density 组别一致性分析

功能:
    1) 读取 Fire frequency 与 Normalized burn area 的 HFI 净效应固定效应结果 CSV
    2) 对每个指标分别将其全部分类样本 pooled 后，按 GDP 与 Population density 各自从低到高划分为 20 个全局等样本组
    3) 计算每组 Human Footprint 影响（diff_contribution）的均值
    4) 识别组均值极大值与极小值对应的组别
    5) 分别输出 Fire frequency 与 Normalized burn area 在 Interface / Intermix 下的结果
    6) 输出结果表到同一个 xlsx 工作簿的多个 mode sheet，并输出对应热图

说明:
    - 分组口径为“每个指标内所有分类 pooled 后的 20 个全局等样本组”
    - Human Footprint 影响使用 diff_contribution 的组均值
    - 若多个组并列极大值或极小值，则保留全部并列组
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


OUTPUT_ROOT = Path(r"B:\WUI\Picture")
OUTPUT_XLSX = OUTPUT_ROOT / "Human Footprint_group_mean_extreme_GDP_PopD_by_metric.xlsx"
HEATMAP_OUTPUT_TEMPLATE = "Human Footprint_group_mean_extreme_{output_slug}_heatmap.png"

N_GROUPS = 20
FLOAT_TOL = 1e-6
HEATMAP_CLIMATE_ORDER = ["A", "B", "C", "D"]
HEATMAP_LANDCOVER_ORDER = ["CRP", "FST", "GRS", "SHR", "WET"]
MODE_CONFIGS = [
    {"mode": "face", "label": "Interface", "slug": "interface"},
    {"mode": "mix", "label": "Intermix", "slug": "intermix"},
]
OUTPUT_COLUMNS = [
    "metric_label",
    "category_id",
    "grouping_parameter_label",
    "max_groups",
    "max_group_ranges",
    "max_mean_effect",
    "max_group_sample_ratio",
    "min_groups",
    "min_group_ranges",
    "min_mean_effect",
    "min_group_sample_ratio",
]

DATASETS = [
    {
        "key": "fire_num",
        "metric_label": "Fire frequency",
        "output_slug": "Fire_frequency",
        "effect_dir": r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num",
        "basic_dir": r"I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic",
    },
    {
        "key": "fire_area",
        "metric_label": "Normalized burn area",
        "output_slug": "Normalized_burn_area",
        "effect_dir": r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area",
        "basic_dir": r"I:\Processing data\渔网分类建模\Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic",
    },
]

GROUPING_PARAMETERS = [
    {"column": "GDP", "label": "GDP"},
    {"column": "PopD", "label": "Population density"},
]

EFFECT_PATTERN = re.compile(
    r"^HFI_effect_((face|mix)_([A-D])_([A-Z]+))_(?:fire_num_with_WUI_offset|fire_area)$",
    flags=re.IGNORECASE,
)
BASIC_PATTERN = re.compile(
    r"^((face|mix)_([A-D])_([A-Z]+))_(?:fire_num_with_WUI_offset|fire_area)$",
    flags=re.IGNORECASE,
)


def parse_effect_filename(path: str) -> tuple[str, str, str, str]:
    name = Path(path).stem
    match = EFFECT_PATTERN.match(name)
    if not match:
        raise ValueError(f"Unrecognized effect filename: {name}")
    return match.group(1), match.group(2), match.group(3), match.group(4)


def parse_basic_filename(path: str) -> tuple[str, str, str, str]:
    name = Path(path).stem
    match = BASIC_PATTERN.match(name)
    if not match:
        raise ValueError(f"Unrecognized basic filename: {name}")
    return match.group(1), match.group(2), match.group(3), match.group(4)


def build_file_map(root_dir: str, pattern: str, parser) -> dict[str, dict[str, str]]:
    file_map: dict[str, dict[str, str]] = {}
    for path in sorted(glob.glob(os.path.join(root_dir, pattern))):
        category_id, mode, climate_zone, landcover = parser(path)
        file_map[category_id] = {
            "path": path,
            "category_id": category_id,
            "mode": mode,
            "climate_zone": climate_zone,
            "landcover": landcover,
        }
    if not file_map:
        raise FileNotFoundError(f"No files matched {pattern!r} in: {root_dir}")
    return file_map


def assign_pooled_equal_count_groups(df: pd.DataFrame, parameter_col: str) -> pd.DataFrame:
    grouped = df.copy()
    values = pd.to_numeric(grouped[parameter_col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).any():
        raise ValueError(f"{parameter_col} contains no valid numeric data for pooled grouping.")
    if np.nanmin(values) == np.nanmax(values):
        raise ValueError(f"{parameter_col} is constant; cannot create {N_GROUPS} pooled groups.")
    if len(grouped) < N_GROUPS:
        raise ValueError(
            f"Only {len(grouped)} pooled rows available; cannot split {parameter_col} into {N_GROUPS} groups."
        )

    ranked = grouped[parameter_col].rank(method="first", ascending=True)
    raw_bins = pd.qcut(ranked, q=N_GROUPS, labels=False)
    if raw_bins.isna().any():
        raise ValueError(f"Some {parameter_col} values could not be assigned to pooled equal-count bins.")

    group_col = f"{parameter_col}_group"
    label_col = f"{parameter_col}_group_label"
    lower_col = f"{parameter_col}_group_lower"
    upper_col = f"{parameter_col}_group_upper"
    grouped[group_col] = (raw_bins.astype(int) + 1).astype(int)
    ranges = pd.DataFrame(
        grouped.groupby(group_col, as_index=False)[parameter_col]
        .agg(**{lower_col: "min", upper_col: "max"})
        .sort_values(group_col)
    )
    grouped = grouped.merge(ranges, how="left", on=group_col, validate="many_to_one")
    grouped[label_col] = grouped[group_col].map(lambda x: f"G{x:02d}")
    return grouped


def load_basic_table(basic_path: str, category_id: str) -> pd.DataFrame:
    usecols = ["GRID_ID", "Year", "GDP", "PopD", "HFI"]
    df = pd.read_csv(basic_path, usecols=usecols, low_memory=False)
    df["GRID_ID"] = pd.to_numeric(df["GRID_ID"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["GDP"] = pd.to_numeric(df["GDP"], errors="coerce")
    df["PopD"] = pd.to_numeric(df["PopD"], errors="coerce")
    df["HFI"] = pd.to_numeric(df["HFI"], errors="coerce")
    df = df.dropna(subset=usecols).copy()
    df["GRID_ID"] = df["GRID_ID"].astype(np.int64)
    df["Year"] = df["Year"].astype(np.int16)
    if df.empty:
        raise ValueError(f"No valid basic rows found for {category_id}: {basic_path}")
    if df.duplicated(["GRID_ID", "Year"]).any():
        raise ValueError(f"Duplicate GRID_ID-Year keys found in basic file for {category_id}: {basic_path}")
    return df.sort_values(["Year", "GRID_ID"]).reset_index(drop=True)


def assign_global_groups_to_basic_tables(
    shared_basic_map: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    pooled_parts: list[pd.DataFrame] = []
    for category_id, basic_df in shared_basic_map.items():
        working = basic_df.copy()
        working["category_id"] = category_id
        pooled_parts.append(working)

    if not pooled_parts:
        raise ValueError("No shared basic tables available for pooled grouping.")

    pooled = pd.concat(pooled_parts, ignore_index=True)
    for parameter in GROUPING_PARAMETERS:
        parameter_col = str(parameter["column"])
        pooled = assign_pooled_equal_count_groups(pooled, parameter_col)

    grouped_basic_map: dict[str, pd.DataFrame] = {}
    for category_id, grouped_df in pooled.groupby("category_id", sort=True):
        grouped_basic_map[str(category_id)] = (
            grouped_df.drop(columns=["category_id"])
            .sort_values(["Year", "GRID_ID"])
            .reset_index(drop=True)
        )
    return grouped_basic_map


def load_effect_table(effect_path: str, category_id: str) -> pd.DataFrame:
    df = pd.read_csv(effect_path, usecols=["GRID_ID", "Year", "HFI", "diff_contribution"], low_memory=False)
    df["GRID_ID"] = pd.to_numeric(df["GRID_ID"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["HFI"] = pd.to_numeric(df["HFI"], errors="coerce")
    df["diff_contribution"] = pd.to_numeric(df["diff_contribution"], errors="coerce")
    df = df.dropna(subset=["GRID_ID", "Year", "HFI", "diff_contribution"]).copy()
    df["GRID_ID"] = df["GRID_ID"].astype(np.int64)
    df["Year"] = df["Year"].astype(np.int16)
    if df.empty:
        raise ValueError(f"No valid effect rows found for {category_id}: {effect_path}")
    if df.duplicated(["GRID_ID", "Year"]).any():
        raise ValueError(f"Duplicate GRID_ID-Year keys found in effect file for {category_id}: {effect_path}")
    return df


def merge_effect_with_basic(effect_df: pd.DataFrame, basic_df: pd.DataFrame, category_id: str, metric_label: str) -> pd.DataFrame:
    merged = effect_df.merge(
        basic_df,
        on=["GRID_ID", "Year"],
        how="inner",
        validate="one_to_one",
        suffixes=("_effect", "_basic"),
    )
    if merged.empty:
        raise ValueError(f"{category_id}: no matched rows between effect file and basic file for {metric_label}.")

    max_hfi_diff = float((merged["HFI_effect"] - merged["HFI_basic"]).abs().max())
    if max_hfi_diff > FLOAT_TOL:
        raise ValueError(
            f"{category_id}: HFI mismatch after merging effect and basic tables for {metric_label} "
            f"(max abs diff = {max_hfi_diff:.6g})."
        )

    merged = merged.rename(columns={"HFI_effect": "HFI"})
    keep_columns = [
        "GRID_ID",
        "Year",
        "HFI",
        "diff_contribution",
        "GDP",
        "PopD",
        "GDP_group",
        "GDP_group_label",
        "GDP_group_lower",
        "GDP_group_upper",
        "PopD_group",
        "PopD_group_label",
        "PopD_group_lower",
        "PopD_group_upper",
    ]
    return merged[keep_columns].copy()


def summarize_parameter_groups(merged_df: pd.DataFrame, parameter_col: str) -> pd.DataFrame:
    group_col = f"{parameter_col}_group"
    label_col = f"{parameter_col}_group_label"
    lower_col = f"{parameter_col}_group_lower"
    upper_col = f"{parameter_col}_group_upper"

    summary = (
        merged_df.groupby(group_col, as_index=False)
        .agg(
            group_label=(label_col, "first"),
            group_lower=(lower_col, "min"),
            group_upper=(upper_col, "max"),
            n_samples=("diff_contribution", "size"),
            mean_hfi_effect=("diff_contribution", "mean"),
            median_hfi_effect=("diff_contribution", "median"),
            mean_HFI=("HFI", "mean"),
        )
        .sort_values(group_col)
        .reset_index(drop=True)
    )
    return summary


def format_group_labels(group_values: list[int]) -> str:
    if not group_values:
        return ""
    return "|".join(f"G{int(v):02d}" for v in sorted(group_values))


def parse_group_labels(group_labels: str) -> list[int]:
    if not isinstance(group_labels, str) or not group_labels:
        return []
    return [int(match) for match in re.findall(r"G(\d+)", group_labels)]


def format_group_ranges(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        return ""
    parts = []
    for row in summary_df.itertuples(index=False):
        parts.append(f"G{int(row[0]):02d}[{float(row.group_lower):.6g},{float(row.group_upper):.6g}]")
    return "; ".join(parts)


def summarize_group_mean_extremes(summary_df: pd.DataFrame) -> dict[str, object]:
    group_col = summary_df.columns[0]
    max_mean = float(summary_df["mean_hfi_effect"].max())
    min_mean = float(summary_df["mean_hfi_effect"].min())
    total_samples = int(summary_df["n_samples"].sum())

    max_rows = summary_df.loc[np.isclose(summary_df["mean_hfi_effect"], max_mean, atol=1e-12, rtol=0.0)].copy()
    min_rows = summary_df.loc[np.isclose(summary_df["mean_hfi_effect"], min_mean, atol=1e-12, rtol=0.0)].copy()

    max_groups = sorted(max_rows[group_col].astype(int).unique().tolist())
    min_groups = sorted(min_rows[group_col].astype(int).unique().tolist())
    max_sample_ratio = float(max_rows["n_samples"].sum() / total_samples) if total_samples > 0 else np.nan
    min_sample_ratio = float(min_rows["n_samples"].sum() / total_samples) if total_samples > 0 else np.nan

    return {
        "max_groups": format_group_labels(max_groups),
        "max_group_ranges": format_group_ranges(max_rows[[group_col, "group_lower", "group_upper"]]),
        "max_mean_effect": max_mean,
        "max_group_sample_ratio": max_sample_ratio,
        "max_group_count": int(len(max_groups)),
        "min_groups": format_group_labels(min_groups),
        "min_group_ranges": format_group_ranges(min_rows[[group_col, "group_lower", "group_upper"]]),
        "min_mean_effect": min_mean,
        "min_group_sample_ratio": min_sample_ratio,
        "min_group_count": int(len(min_groups)),
    }


def build_dataset_result_rows(
    dataset: dict[str, str],
    dataset_maps: dict[str, dict[str, dict[str, str]]],
) -> list[dict[str, object]]:
    matched_categories = sorted(set(dataset_maps["effect"].keys()) & set(dataset_maps["basic"].keys()))
    if not matched_categories:
        raise ValueError(f"No matched categories found for {dataset['metric_label']}.")

    all_categories = sorted(set(dataset_maps["effect"].keys()) | set(dataset_maps["basic"].keys()))
    missing_categories = sorted(set(all_categories) - set(matched_categories))
    print(f"{dataset['metric_label']}: matched categories available in effect/basic inputs: {len(matched_categories)}")
    if missing_categories:
        print(f"{dataset['metric_label']}: categories missing from effect or basic input: {', '.join(missing_categories)}")

    shared_basic_raw_map: dict[str, pd.DataFrame] = {}
    for category_id in matched_categories:
        shared_basic_raw_map[category_id] = load_basic_table(
            dataset_maps["basic"][category_id]["path"],
            category_id,
        )
    shared_basic_grouped_map = assign_global_groups_to_basic_tables(shared_basic_raw_map)

    rows: list[dict[str, object]] = []
    for index, category_id in enumerate(matched_categories, start=1):
        meta = dataset_maps["effect"][category_id]
        print(f"{dataset['metric_label']} [{index}/{len(matched_categories)}] Processing {category_id} ...")
        basic_df = shared_basic_grouped_map[category_id]
        effect_df = load_effect_table(dataset_maps["effect"][category_id]["path"], category_id)
        merged_df = merge_effect_with_basic(
            effect_df,
            basic_df,
            category_id,
            str(dataset["metric_label"]),
        )

        for parameter in GROUPING_PARAMETERS:
            parameter_col = str(parameter["column"])
            parameter_label = str(parameter["label"])
            summary_df = summarize_parameter_groups(merged_df, parameter_col)
            extremes = summarize_group_mean_extremes(summary_df)

            rows.append(
                {
                    "metric_key": str(dataset["key"]),
                    "metric_label": str(dataset["metric_label"]),
                    "category_count_in_metric": int(len(matched_categories)),
                    "category_id": category_id,
                    "mode": meta["mode"],
                    "climate_zone": meta["climate_zone"],
                    "landcover": meta["landcover"],
                    "grouping_parameter": parameter_col,
                    "grouping_parameter_label": parameter_label,
                    "grouping_method": f"{N_GROUPS} pooled equal-count groups within {dataset['metric_label']}",
                    "extreme_definition": "max/min of grouped mean diff_contribution",
                    **extremes,
                }
            )
    return rows


def build_heatmap_tables(
    result_df: pd.DataFrame,
    parameter_label: str,
    group_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = result_df.loc[result_df["grouping_parameter_label"] == parameter_label].copy()
    rows: list[dict[str, object]] = []
    for item in subset.itertuples(index=False):
        group_values = parse_group_labels(getattr(item, group_column))
        if not group_values:
            continue
        rows.append(
            {
                "climate_zone": item.climate_zone,
                "landcover": item.landcover,
                "group_value": float(np.median(group_values)),
                "group_label": format_group_labels(sorted(set(group_values))),
            }
        )

    if not rows:
        empty_table = pd.DataFrame(index=HEATMAP_CLIMATE_ORDER, columns=HEATMAP_LANDCOVER_ORDER, dtype=float)
        empty_labels = pd.DataFrame(index=HEATMAP_CLIMATE_ORDER, columns=HEATMAP_LANDCOVER_ORDER, dtype=object)
        return empty_table, empty_labels

    heatmap_df = pd.DataFrame(rows)
    summary = (
        heatmap_df.groupby(["climate_zone", "landcover"], as_index=False)
        .agg(
            heat_value=("group_value", "median"),
            annotation=("group_label", lambda x: format_group_labels(sorted(set(
                value for labels in x for value in parse_group_labels(str(labels))
            )))),
        )
        .sort_values(["climate_zone", "landcover"])
        .reset_index(drop=True)
    )
    value_table = (
        summary.pivot(index="climate_zone", columns="landcover", values="heat_value")
        .reindex(index=HEATMAP_CLIMATE_ORDER, columns=HEATMAP_LANDCOVER_ORDER)
    )
    annotation_table = (
        summary.pivot(index="climate_zone", columns="landcover", values="annotation")
        .reindex(index=HEATMAP_CLIMATE_ORDER, columns=HEATMAP_LANDCOVER_ORDER)
    )
    return value_table, annotation_table


def save_metric_heatmap(result_df: pd.DataFrame, dataset: dict[str, str]) -> Path:
    panel_configs = [
        ("GDP", "max_groups", "GDP max"),
        ("GDP", "min_groups", "GDP min"),
        ("Population density", "max_groups", "Population density max"),
        ("Population density", "min_groups", "Population density min"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="white")
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    cbar_ax = fig.add_axes([0.92, 0.14, 0.02, 0.72])

    for idx, (parameter_label, group_column, panel_title) in enumerate(panel_configs):
        ax = axes[idx // 2, idx % 2]
        value_table, annotation_table = build_heatmap_tables(result_df, parameter_label, group_column)
        show_cbar = idx == len(panel_configs) - 1
        sns.heatmap(
            value_table,
            ax=ax,
            cmap=cmap,
            vmin=1,
            vmax=N_GROUPS,
            annot=annotation_table,
            fmt="",
            annot_kws={"fontsize": 11, "fontname": "Arial"},
            linewidths=0.5,
            linecolor="white",
            cbar=show_cbar,
            cbar_ax=cbar_ax if show_cbar else None,
            cbar_kws={"label": "Group number"} if show_cbar else None,
            square=True,
        )
        na_positions = np.argwhere(value_table.isna().to_numpy())
        for row_idx, col_idx in na_positions:
            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                "NA",
                ha="center",
                va="center",
                fontsize=11,
                fontname="Arial",
                color="#4f4f4f",
            )
        ax.set_title(panel_title, fontsize=14, fontweight="bold", fontname="Arial", pad=10)
        ax.set_xlabel("Landcover", fontsize=12, fontname="Arial")
        ax.set_ylabel("Climate zone" if idx % 2 == 0 else "", fontsize=12, fontname="Arial")
        ax.tick_params(axis="both", labelsize=11)
        if idx % 2 != 0:
            ax.tick_params(axis="y", left=False, labelleft=False)

    fig.suptitle(str(dataset["metric_label"]), fontsize=18, fontweight="bold", fontname="Arial", y=0.97)
    fig.subplots_adjust(left=0.08, right=0.9, bottom=0.08, top=0.9, wspace=0.16, hspace=0.22)
    output_path = OUTPUT_ROOT / HEATMAP_OUTPUT_TEMPLATE.format(output_slug=str(dataset["output_slug"]))
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def save_mode_heatmap(
    result_df: pd.DataFrame,
    dataset: dict[str, str],
    mode_config: dict[str, str],
) -> Path:
    filtered_df = result_df.loc[result_df["mode"] == mode_config["mode"]].copy()
    if filtered_df.empty:
        raise ValueError(f"No rows available for {dataset['metric_label']} - {mode_config['label']}.")

    panel_configs = [
        ("GDP", "max_groups", "GDP max"),
        ("GDP", "min_groups", "GDP min"),
        ("Population density", "max_groups", "Population density max"),
        ("Population density", "min_groups", "Population density min"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="white")
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    cbar_ax = fig.add_axes([0.92, 0.14, 0.02, 0.72])

    for idx, (parameter_label, group_column, panel_title) in enumerate(panel_configs):
        ax = axes[idx // 2, idx % 2]
        value_table, annotation_table = build_heatmap_tables(filtered_df, parameter_label, group_column)
        show_cbar = idx == len(panel_configs) - 1
        sns.heatmap(
            value_table,
            ax=ax,
            cmap=cmap,
            vmin=1,
            vmax=N_GROUPS,
            annot=annotation_table,
            fmt="",
            annot_kws={"fontsize": 11, "fontname": "Arial"},
            linewidths=0.5,
            linecolor="white",
            cbar=show_cbar,
            cbar_ax=cbar_ax if show_cbar else None,
            cbar_kws={"label": "Group number"} if show_cbar else None,
            square=True,
        )
        na_positions = np.argwhere(value_table.isna().to_numpy())
        for row_idx, col_idx in na_positions:
            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                "NA",
                ha="center",
                va="center",
                fontsize=11,
                fontname="Arial",
                color="#4f4f4f",
            )
        ax.set_title(panel_title, fontsize=14, fontweight="bold", fontname="Arial", pad=10)
        ax.set_xlabel("Landcover", fontsize=12, fontname="Arial")
        ax.set_ylabel("Climate zone" if idx % 2 == 0 else "", fontsize=12, fontname="Arial")
        ax.tick_params(axis="both", labelsize=11)
        if idx % 2 != 0:
            ax.tick_params(axis="y", left=False, labelleft=False)

    fig.suptitle(
        f"{dataset['metric_label']} - {mode_config['label']}",
        fontsize=18,
        fontweight="bold",
        fontname="Arial",
        y=0.97,
    )
    fig.subplots_adjust(left=0.08, right=0.9, bottom=0.08, top=0.9, wspace=0.16, hspace=0.22)
    output_path = OUTPUT_ROOT / HEATMAP_OUTPUT_TEMPLATE.format(
        output_slug=f"{dataset['output_slug']}_{mode_config['slug']}"
    )
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_XLSX) as writer:
        for dataset in DATASETS:
            dataset_maps = {
                "effect": build_file_map(str(dataset["effect_dir"]), "HFI_effect_*.csv", parse_effect_filename),
                "basic": build_file_map(str(dataset["basic_dir"]), "*.csv", parse_basic_filename),
            }
            result_rows = build_dataset_result_rows(dataset, dataset_maps)
            result_df = pd.DataFrame(result_rows).sort_values(
                ["mode", "climate_zone", "landcover", "grouping_parameter"]
            ).reset_index(drop=True)
            for mode_config in MODE_CONFIGS:
                mode_df = result_df.loc[result_df["mode"] == mode_config["mode"]].copy()
                if mode_df.empty:
                    continue
                mode_output_df = mode_df[OUTPUT_COLUMNS].copy()
                sheet_name = f"{dataset['metric_label']}_{mode_config['label']}"[:31]
                mode_output_df.to_excel(writer, sheet_name=sheet_name, index=False)
                mode_heatmap_path = save_mode_heatmap(result_df, dataset, mode_config)
                print(f"Saved mode heatmap figure to: {mode_heatmap_path}")

    print(f"\nSaved result workbook to: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
