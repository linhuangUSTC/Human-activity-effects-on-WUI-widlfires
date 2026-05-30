#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称:
    将多种归因参数分成 10 组，并按气候带和 landcover 分别绘制 4x2 组合图

功能:
    1. 读取 Fire frequency 和 Normalized burned area 的 HFI effect CSV，并保留 2003-2022 年全部匹配样本。
    2. 读取对应的 Basic CSV，通过 category_id + Year + GRID_ID 进行匹配。
    3. 从 category_id 解析 WUI 类型、气候带和 landcover。
    4. 对每个气候带(A/B/C/D)和每个 landcover(CRP/FST/GRS/SHR/WET)分别统计绘图。
    5. 每个分类单元输出一张 4x2 图:
       - 4 行: Cropland、Pasture、GDP、Population density
       - 2 列: Fire frequency、Normalized burned area
       - 每个子图内为 Interface / Intermix 的 10 组 HFI effect 均值 + 标准误。

输出:
    - B:\WUI\Picture\Grouped_Human Footprint_effect_by_attribution_Climate_{A/B/C/D}_interface_intermix_2003_2022_10groups_4x2.png
    - B:\WUI\Picture\Grouped_Human Footprint_effect_by_attribution_Landcover_{CRP/FST/GRS/SHR/WET}_interface_intermix_2003_2022_10groups_4x2.png
    - B:\WUI\Picture\Grouped_Human activity_effect_group_ranges_by_climate_landcover_interface_intermix_2003_2022_10groups.xlsx
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, ScalarFormatter


OUTPUT_DIR = Path(r"B:\WUI\Picture")

YEAR_START = 2003
YEAR_END = 2022
YEAR_TAG = f"{YEAR_START}_{YEAR_END}"
YEAR_DESC = f"{YEAR_START}-{YEAR_END}"
N_GROUPS = 10
GROUP_TAG = f"{N_GROUPS}groups"
GROUP_RANGE_OUTPUT_PATH = OUTPUT_DIR / (
    f"Grouped_Human activity_effect_group_ranges_by_climate_landcover_interface_intermix_{YEAR_TAG}_{GROUP_TAG}.xlsx"
)
CHUNK_SIZE = 250_000
KEY_COLUMNS = ["Year", "GRID_ID"]
EFFECT_COLUMN = "diff_contribution"
PARAMETER_COLUMNS = ["Cropland_Area_km2", "Pasture_Band2_Mean", "GDP", "PopD"]

PARAMETER_CONFIGS = [
    {"column": "Cropland_Area_km2", "label": "Cropland", "slug": "Cropland"},
    {"column": "Pasture_Band2_Mean", "label": "Pasture", "slug": "Pasture"},
    {"column": "GDP", "label": "GDP", "slug": "GDP"},
    {"column": "PopD", "label": "Population density", "slug": "PopD"},
]

WUI_SCOPE_CONFIGS = [
    {"slug": "interface", "label": "Interface", "mode": "face", "color": "#2b8cbe"},
    {"slug": "intermix", "label": "Intermix", "mode": "mix", "color": "#d95f0e"},
]

DATASETS = [
    {
        "metric_slug": "Fire_frequency",
        "effect_dir": Path(r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire num"),
        "basic_dir": Path(r"I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic"),
        "effect_suffix": "fire_num_with_WUI_offset",
        "effect_label": "Fire frequency",
    },
    {
        "metric_slug": "Normalized_burn_area",
        "effect_dir": Path(r"B:\WUI\Picture and statistic results\HFI的净效应-固定效应结果\csv\Fire area"),
        "basic_dir": Path(r"I:\Processing data\渔网分类建模\Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic"),
        "effect_suffix": "fire_area",
        "effect_label": "Normalized burned area",
    },
]

CLIMATE_ORDER = ["A", "B", "C", "D"]
LANDCOVER_ORDER = ["CRP", "FST", "GRS", "SHR", "WET"]
CLIMATE_FULL_NAMES = {
    "A": "Tropical climate",
    "B": "Arid climate",
    "C": "Temperate climate",
    "D": "Cold climate",
}
LANDCOVER_FULL_NAMES = {
    "CRP": "Cropland",
    "FST": "Forest",
    "GRS": "Grassland",
    "SHR": "Shrubland",
    "WET": "Wetland",
}


plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 16

AXIS_LABEL_FONT_SIZE = 28
TICK_LABEL_FONT_SIZE = 25
LEGEND_FONT_SIZE = 25
FIGURE_TITLE_FONT_SIZE = 32
FIGURE_WIDTH = 22.0
FIGURE_HEIGHT = 24.0
Y_MAJOR_TICK_BINS = 4


def extract_category_id(path: str, *, is_effect: bool, suffix: str) -> str:
    stem = Path(path).stem
    if is_effect:
        pattern = rf"HFI_effect_((face|mix)_[A-D]_[A-Z]+)_{suffix}$"
    else:
        pattern = rf"((face|mix)_[A-D]_[A-Z]+)_{suffix}$"
    match = re.search(pattern, stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unrecognized file name pattern: {stem}")
    return match.group(1)


def parse_category_id(category_id: str) -> tuple[str, str, str]:
    parts = category_id.split("_")
    if len(parts) != 3:
        raise ValueError(f"Unexpected category_id: {category_id}")
    return parts[0].lower(), parts[1].upper(), parts[2].upper()


def normalize_grid_id(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.round().astype("Int64")


def load_basic_lookup(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=KEY_COLUMNS + PARAMETER_COLUMNS, low_memory=False)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["GRID_ID"] = normalize_grid_id(df["GRID_ID"])
    for parameter_col in PARAMETER_COLUMNS:
        df[parameter_col] = pd.to_numeric(df[parameter_col], errors="coerce")

    df = df[df["Year"].between(YEAR_START, YEAR_END, inclusive="both")]
    df = df.dropna(subset=KEY_COLUMNS)
    df = df.dropna(subset=PARAMETER_COLUMNS, how="all")
    if df.empty:
        return df

    deduped = (
        df.groupby(KEY_COLUMNS, as_index=False)[PARAMETER_COLUMNS]
        .mean()
        .reset_index(drop=True)
    )
    deduped["Year"] = deduped["Year"].astype(np.int64)
    deduped["GRID_ID"] = deduped["GRID_ID"].astype(np.int64)
    return deduped


def load_and_match_dataset(dataset: dict[str, object]) -> pd.DataFrame:
    effect_dir = Path(str(dataset["effect_dir"]))
    basic_dir = Path(str(dataset["basic_dir"]))
    effect_suffix = str(dataset["effect_suffix"])
    effect_files = sorted(glob.glob(str(effect_dir / "HFI_effect_*.csv")))
    basic_files = sorted(glob.glob(str(basic_dir / "*.csv")))
    if not effect_files:
        raise FileNotFoundError(f"No effect CSV files found in: {effect_dir}")
    if not basic_files:
        raise FileNotFoundError(f"No basic CSV files found in: {basic_dir}")

    effect_map: dict[str, str] = {}
    basic_map: dict[str, str] = {}

    for path in effect_files:
        effect_map[extract_category_id(path, is_effect=True, suffix=effect_suffix)] = path
    for path in basic_files:
        try:
            basic_map[extract_category_id(path, is_effect=False, suffix=effect_suffix)] = path
        except ValueError:
            continue

    matched_categories = sorted(set(effect_map).intersection(basic_map))
    if not matched_categories:
        raise ValueError("No matched category IDs found between effect and basic CSVs.")

    parts: list[pd.DataFrame] = []
    for category_id in matched_categories:
        effect_csv = effect_map[category_id]
        basic_csv = basic_map[category_id]
        wui_mode, climate_zone, landcover = parse_category_id(category_id)
        basic_lookup = load_basic_lookup(basic_csv)
        if basic_lookup.empty:
            continue

        for chunk in pd.read_csv(
            effect_csv,
            usecols=KEY_COLUMNS + [EFFECT_COLUMN],
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            chunk["Year"] = pd.to_numeric(chunk["Year"], errors="coerce").astype("Int64")
            chunk["GRID_ID"] = normalize_grid_id(chunk["GRID_ID"])
            chunk[EFFECT_COLUMN] = pd.to_numeric(chunk[EFFECT_COLUMN], errors="coerce")
            chunk = chunk[
                chunk["Year"].between(YEAR_START, YEAR_END, inclusive="both")
            ].dropna(subset=KEY_COLUMNS + [EFFECT_COLUMN])
            if chunk.empty:
                continue

            chunk["Year"] = chunk["Year"].astype(np.int64)
            chunk["GRID_ID"] = chunk["GRID_ID"].astype(np.int64)
            merged = chunk.merge(basic_lookup, how="inner", on=KEY_COLUMNS, validate="many_to_one")
            if merged.empty:
                continue

            kept = merged[PARAMETER_COLUMNS + [EFFECT_COLUMN]].copy()
            for parameter_col in PARAMETER_COLUMNS:
                kept[parameter_col] = kept[parameter_col].astype(np.float32)
            kept[EFFECT_COLUMN] = kept[EFFECT_COLUMN].astype(np.float32)
            kept["metric_slug"] = str(dataset["metric_slug"])
            kept["effect_label"] = str(dataset["effect_label"])
            kept["category_id"] = category_id
            kept["wui_mode"] = wui_mode
            kept["climate_zone"] = climate_zone
            kept["landcover"] = landcover
            parts.append(kept)

    if not parts:
        raise ValueError(f"No matched sample rows found for Years={YEAR_DESC}.")

    matched = pd.concat(parts, ignore_index=True)
    matched = matched.dropna(subset=[EFFECT_COLUMN])
    matched = matched.dropna(subset=PARAMETER_COLUMNS, how="all")
    return matched


def assign_equal_count_groups(df: pd.DataFrame, parameter_col: str, n_groups: int) -> pd.DataFrame:
    grouped = df.copy()
    values = grouped[parameter_col].to_numpy(dtype=float)
    if not np.isfinite(values).any():
        raise ValueError(f"{parameter_col} contains no valid numeric data for binning.")
    if np.nanmin(values) == np.nanmax(values):
        raise ValueError(f"{parameter_col} is constant; cannot create equal-count bins.")
    if len(grouped) < n_groups:
        raise ValueError(f"Only {len(grouped)} rows available; cannot split into {n_groups} groups.")

    ranked = grouped[parameter_col].rank(method="first", ascending=True)
    raw_bins = pd.qcut(ranked, q=n_groups, labels=False)
    if raw_bins.isna().any():
        raise ValueError(f"Some {parameter_col} values could not be assigned to equal-count bins.")

    grouped["parameter_group"] = (raw_bins.astype(int) + 1).astype(int)
    group_ranges = (
        grouped.groupby("parameter_group", as_index=False)[parameter_col]
        .agg(param_bin_lower="min", param_bin_upper="max")
        .sort_values("parameter_group")
        .reset_index(drop=True)
    )
    grouped = grouped.merge(group_ranges, how="left", on="parameter_group", validate="many_to_one")
    return grouped


def extract_group_ranges(grouped: pd.DataFrame) -> pd.DataFrame:
    return (
        grouped[["parameter_group", "param_bin_lower", "param_bin_upper"]]
        .drop_duplicates(subset=["parameter_group"])
        .sort_values("parameter_group")
        .reset_index(drop=True)
    )


def summarize_groups(
    df: pd.DataFrame,
    parameter_col: str,
    group_ranges: pd.DataFrame,
) -> pd.DataFrame:
    group_ranges = group_ranges.copy().sort_values("parameter_group").reset_index(drop=True)
    summary = (
        df.groupby("parameter_group", as_index=False)
        .agg(
            n_samples=(EFFECT_COLUMN, "size"),
            mean_parameter=(parameter_col, "mean"),
            mean_hfi_effect=(EFFECT_COLUMN, "mean"),
            sd_hfi_effect=(EFFECT_COLUMN, "std"),
        )
        .sort_values("parameter_group")
        .reset_index(drop=True)
    )

    full_groups = pd.DataFrame({"parameter_group": np.arange(1, N_GROUPS + 1, dtype=int)})
    full_groups = full_groups.merge(group_ranges, how="left", on="parameter_group")
    completed = full_groups.merge(summary, how="left", on="parameter_group")
    completed["n_samples"] = completed["n_samples"].fillna(0).astype(int)
    completed["se_hfi_effect"] = completed["sd_hfi_effect"] / np.sqrt(completed["n_samples"].clip(lower=1))
    completed["se_hfi_effect"] = completed["se_hfi_effect"].fillna(0.0)
    completed["group_label"] = completed["parameter_group"].apply(lambda x: f"G{x:02d}")
    return completed


def _format_value(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    abs_value = abs(value)
    if abs_value >= 1000:
        return f"{value:.0f}"
    if abs_value >= 10:
        return f"{value:.1f}"
    if abs_value >= 0.1:
        return f"{value:.2f}"
    if abs_value >= 0.01:
        return f"{value:.3f}"
    return f"{value:.2e}"


def _format_range_label(lower: float, upper: float) -> str:
    return f"[{_format_value(lower)}, {_format_value(upper)}]"


def build_group_range_table(
    *,
    grouping_type: str,
    grouping_value: str,
    parameter_config: dict[str, str],
    dataset: dict[str, str | Path],
    group_ranges: pd.DataFrame,
) -> pd.DataFrame:
    table = group_ranges.copy().sort_values("parameter_group").reset_index(drop=True)
    table["group_label"] = table["parameter_group"].apply(lambda x: f"G{x:02d}")
    table["x_axis_range"] = table.apply(
        lambda row: _format_range_label(row["param_bin_lower"], row["param_bin_upper"]),
        axis=1,
    )
    table.insert(0, "metric_slug", str(dataset["metric_slug"]))
    table.insert(0, "metric_label", str(dataset["effect_label"]))
    table.insert(0, "parameter_slug", str(parameter_config["slug"]))
    table.insert(0, "parameter_label", str(parameter_config["label"]))
    table.insert(0, "grouping_value", grouping_value)
    table.insert(0, "grouping_type", grouping_type)
    return table[
        [
            "grouping_type",
            "grouping_value",
            "parameter_label",
            "parameter_slug",
            "metric_label",
            "metric_slug",
            "group_label",
            "parameter_group",
            "param_bin_lower",
            "param_bin_upper",
            "x_axis_range",
        ]
    ]


def build_x_range_labels() -> list[str]:
    return [f"G{i:02d}" for i in range(1, N_GROUPS + 1)]


def draw_metric_interface_intermix_bar_chart(
    ax,
    scope_summaries: dict[str, pd.DataFrame],
    dataset: dict[str, str | Path],
    parameter_config: dict[str, str],
    *,
    show_legend: bool,
    legend_loc: str,
) -> None:
    positions = np.arange(1, N_GROUPS + 1, dtype=float)
    x_labels = build_x_range_labels()
    bar_width = 0.32
    offsets = {
        "interface": -0.18,
        "intermix": 0.18,
    }

    for scope_config in WUI_SCOPE_CONFIGS:
        subset = scope_summaries[str(scope_config["slug"])].sort_values("parameter_group").reset_index(drop=True)
        x = positions + offsets[str(scope_config["slug"])]
        ax.bar(
            x,
            subset["mean_hfi_effect"],
            width=bar_width,
            color=str(scope_config["color"]),
            alpha=0.72,
            edgecolor=str(scope_config["color"]),
            linewidth=0.9,
            yerr=subset["se_hfi_effect"],
            error_kw={
                "ecolor": str(scope_config["color"]),
                "elinewidth": 0.9,
                "capsize": 2.5,
            },
            label=str(scope_config["label"]),
            zorder=4,
        )

    ax.axhline(0.0, color="#7f7f7f", linewidth=0.9, linestyle="--", zorder=1)
    for pos in positions:
        ax.axvline(pos, color="#eeeeee", linewidth=0.65, zorder=0)

    ax.set_xlim(0.45, N_GROUPS + 0.55)
    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels, fontsize=TICK_LABEL_FONT_SIZE, rotation=0, ha="center", va="center")
    ax.set_xlabel(str(parameter_config["label"]), fontsize=AXIS_LABEL_FONT_SIZE, labelpad=10)
    ax.set_ylabel("")
    ax.set_title("")
    ax.tick_params(axis="x", pad=13)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.38)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    formatter = ScalarFormatter(useMathText=False)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=Y_MAJOR_TICK_BINS))
    ax.yaxis.get_offset_text().set_fontsize(TICK_LABEL_FONT_SIZE)

    if show_legend:
        ax.legend(
            loc=legend_loc,
            ncol=1,
            fontsize=LEGEND_FONT_SIZE,
            frameon=False,
            handlelength=1.4,
            labelspacing=0.25,
            borderaxespad=0.35,
        )


def metric_y_axis_label(dataset: dict[str, str | Path]) -> str:
    if str(dataset["metric_slug"]) == "Fire_frequency":
        return "Human activity effects on fire frequency"
    return "Human activity effects on normalized burned area"


def get_category_title(grouping_type: str, grouping_value: str) -> str:
    if grouping_type == "Climate":
        return CLIMATE_FULL_NAMES.get(grouping_value, f"Climate {grouping_value}")
    if grouping_type == "Landcover":
        landcover_name = LANDCOVER_FULL_NAMES.get(grouping_value, grouping_value)
        return f"Land cover-{landcover_name}"
    return grouping_value


def get_legend_loc(grouping_type: str, grouping_value: str) -> str:
    if grouping_type == "Landcover" and grouping_value == "CRP":
        return "lower right"
    if grouping_type == "Landcover" and grouping_value == "FST":
        return "upper right"
    return "lower left"


def add_shared_y_axis_labels(
    fig,
    axes: np.ndarray,
) -> None:
    for col_idx, dataset in enumerate(DATASETS):
        top_box = axes[0, col_idx].get_position()
        bottom_box = axes[-1, col_idx].get_position()
        x = top_box.x0 - 0.040
        y = (top_box.y1 + bottom_box.y0) / 2.0
        fig.text(
            x,
            y,
            metric_y_axis_label(dataset),
            ha="center",
            va="center",
            rotation=90,
            fontsize=AXIS_LABEL_FONT_SIZE,
            color="#222222",
        )


def build_metric_payloads_for_group(
    combined: pd.DataFrame,
    *,
    grouping_type: str,
    grouping_col: str,
    grouping_value: str,
) -> tuple[dict[str, dict[str, dict[str, object]]], list[pd.DataFrame]]:
    metric_payloads_by_parameter: dict[str, dict[str, dict[str, object]]] = {
        str(parameter_config["slug"]): {} for parameter_config in PARAMETER_CONFIGS
    }
    group_range_tables: list[pd.DataFrame] = []

    for dataset in DATASETS:
        metric_df = combined.loc[
            (combined["metric_slug"] == dataset["metric_slug"]) &
            (combined[grouping_col] == grouping_value)
        ].copy()
        if metric_df.empty:
            continue

        for parameter_config in PARAMETER_CONFIGS:
            parameter_col = str(parameter_config["column"])
            metric_param_df = metric_df.dropna(subset=[parameter_col, EFFECT_COLUMN]).copy()
            if metric_param_df.empty:
                continue
            try:
                grouped_all = assign_equal_count_groups(metric_param_df, parameter_col, N_GROUPS)
            except ValueError as exc:
                print(
                    f"Skip {grouping_type}={grouping_value}, {dataset['effect_label']}, "
                    f"{parameter_config['label']}: {exc}"
                )
                continue

            shared_group_ranges = extract_group_ranges(grouped_all)
            scope_summaries: dict[str, pd.DataFrame] = {}
            for scope_config in WUI_SCOPE_CONFIGS:
                scoped_grouped = grouped_all.loc[grouped_all["wui_mode"] == scope_config["mode"]].copy()
                if scoped_grouped.empty:
                    continue
                summary = summarize_groups(scoped_grouped, parameter_col, shared_group_ranges)
                scope_summaries[str(scope_config["slug"])] = summary

            if len(scope_summaries) != len(WUI_SCOPE_CONFIGS):
                continue

            metric_payloads_by_parameter[str(parameter_config["slug"])][str(dataset["metric_slug"])] = {
                "dataset": dataset,
                "scope_summaries": scope_summaries,
                "group_ranges": shared_group_ranges.copy(),
            }
            group_range_tables.append(
                build_group_range_table(
                    grouping_type=grouping_type,
                    grouping_value=grouping_value,
                    parameter_config=parameter_config,
                    dataset=dataset,
                    group_ranges=shared_group_ranges,
                )
            )

    return metric_payloads_by_parameter, group_range_tables


def plot_group_4x2(
    metric_payloads_by_parameter: dict[str, dict[str, dict[str, object]]],
    *,
    grouping_type: str,
    grouping_value: str,
) -> Path | None:
    valid_parameter_configs = [
        parameter_config
        for parameter_config in PARAMETER_CONFIGS
        if len(metric_payloads_by_parameter[str(parameter_config["slug"])]) == len(DATASETS)
    ]
    if not valid_parameter_configs:
        print(f"Skip {grouping_type}={grouping_value}: no complete 4x2 payload.")
        return None

    legend_loc = get_legend_loc(grouping_type, grouping_value)
    category_title = get_category_title(grouping_type, grouping_value)

    fig, axes = plt.subplots(
        len(valid_parameter_configs),
        2,
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
        facecolor="white",
        squeeze=False,
    )

    for row_idx, parameter_config in enumerate(valid_parameter_configs):
        metric_payloads = metric_payloads_by_parameter[str(parameter_config["slug"])]
        for col_idx, dataset in enumerate(DATASETS):
            ax = axes[row_idx, col_idx]
            payload = metric_payloads.get(str(dataset["metric_slug"]))
            if payload is None:
                ax.axis("off")
                continue
            draw_metric_interface_intermix_bar_chart(
                ax,
                payload["scope_summaries"],
                payload["dataset"],
                parameter_config,
                show_legend=(row_idx == 0 and col_idx == 0),
                legend_loc=legend_loc,
            )

    fig.subplots_adjust(bottom=0.075, top=0.985, left=0.100, right=0.99, wspace=0.16, hspace=0.34)
    fig.text(
        0.545,
        1.005,
        category_title,
        ha="center",
        va="bottom",
        fontsize=FIGURE_TITLE_FONT_SIZE,
        color="#222222",
    )
    add_shared_y_axis_labels(fig, axes)
    safe_grouping_type = re.sub(r"[^0-9A-Za-z]+", "_", grouping_type).strip("_")
    safe_grouping_value = re.sub(r"[^0-9A-Za-z]+", "_", grouping_value).strip("_")
    output_path = OUTPUT_DIR / (
        f"Grouped_Human Footprint_effect_by_attribution_{safe_grouping_type}_{safe_grouping_value}_"
        f"interface_intermix_{YEAR_TAG}_{GROUP_TAG}_4x2.png"
    )
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved figure to: {output_path}")
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    matched_parts: list[pd.DataFrame] = []
    for dataset in DATASETS:
        matched_parts.append(load_and_match_dataset(dataset))

    combined = pd.concat(matched_parts, ignore_index=True)
    group_range_tables: list[pd.DataFrame] = []

    group_specs = [
        *[("Climate", "climate_zone", item) for item in CLIMATE_ORDER],
        *[("Landcover", "landcover", item) for item in LANDCOVER_ORDER],
    ]

    for grouping_type, grouping_col, grouping_value in group_specs:
        print(f"\nProcessing {grouping_type}: {grouping_value}")
        payloads, range_tables = build_metric_payloads_for_group(
            combined,
            grouping_type=grouping_type,
            grouping_col=grouping_col,
            grouping_value=grouping_value,
        )
        group_range_tables.extend(range_tables)
        plot_group_4x2(
            payloads,
            grouping_type=grouping_type,
            grouping_value=grouping_value,
        )

    if group_range_tables:
        group_range_table = pd.concat(group_range_tables, ignore_index=True)
        group_range_table.to_excel(GROUP_RANGE_OUTPUT_PATH, index=False)
        print(f"Saved group range table to: {GROUP_RANGE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
