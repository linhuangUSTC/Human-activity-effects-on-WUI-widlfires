#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
程序名称: GLMM 建模报告汇总表生成程序

功能简介:
    1. 批量读取 GLMM 建模结果目录中的 `.txt` 报告文件。
    2. 从报告中提取分类 ID、模型收敛信息、分布族与链接函数、公式、
       样本量、AIC/BIC/logLik、R²、误差指标以及 DHARMa 诊断结果等信息。
    3. 将每个报告整理为一行，汇总输出为 CSV 统计表。

输入:
    - B:\WUI\Picture and statistic results\model construction\GLMM NB 自然样条曲线 Fire num\*.txt
    - B:\WUI\Picture and statistic results\model construction\Fire num NB without HFI\*.txt
    - B:\WUI\Picture and statistic results\model construction\GLMM Ordbeta 自然样条 Normalized fire area\*.txt
    - B:\WUI\Picture and statistic results\model construction\Normalized fire area OrdBeta without HFI\*.txt

输出:
    - B:\WUI\Picture and statistic results\统计表\GLMM_NB_自然样条曲线_Fire_num_汇总.csv
    - B:\WUI\Picture and statistic results\统计表\GLMM_NB_without_Human Footprint_Fire_num_汇总.csv
    - B:\WUI\Picture and statistic results\统计表\GLMM_Ordbeta_自然样条_Normalized_burn_area_汇总.csv
    - B:\WUI\Picture and statistic results\统计表\GLMM_Ordbeta_without_Human Footprint_Normalized_burn_area_汇总.csv
"""

import csv
import re
from pathlib import Path


MODEL_ROOT = Path(r"B:\WUI\Picture and statistic results\model construction")
OUTPUT_DIR = Path(r"B:\WUI\Picture\统计表")

JOBS = [
    {
        "source_dir": MODEL_ROOT / "GLMM NB 自然样条曲线 Fire num",
        "output_csv": OUTPUT_DIR / "GLMM_NB_自然样条曲线_Fire_num_汇总.csv",
    },
    {
        "source_dir": MODEL_ROOT / "Fire num NB without HFI",
        "output_csv": OUTPUT_DIR / "GLMM_NB_without_Human Footprint_Fire_num_汇总.csv",
    },
    {
        "source_dir": MODEL_ROOT / "GLMM Ordbeta 自然样条 Normalized fire area",
        "output_csv": OUTPUT_DIR / "GLMM_Ordbeta_自然样条_Normalized_burn_area_汇总.csv",
    },
    {
        "source_dir": MODEL_ROOT / "Normalized fire area OrdBeta without HFI",
        "output_csv": OUTPUT_DIR / "GLMM_Ordbeta_without_Human Footprint_Normalized_burn_area_汇总.csv",
    },
]

BASE_COLUMNS = [
    "classification_id",
    "source_folder",
    "source_file",
    "report_title",
    "convergence_message",
    "convergence_code",
    "max_gradient",
    "gradient_check",
    "random_effect_variance_ok",
    "random_effect_variance",
    "family",
    "link",
    "formula",
    "data_name",
    "n_obs",
    "n_groups",
    "dispersion_parameter",
    "AIC",
    "BIC",
    "logLik",
    "r2_line_1",
    "r2_line_2",
    "mse",
    "rmse",
    "mae",
    "conditional_model_coefficients_block",
]


def read_text_with_fallback(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="latin-1")


def normalize_space(value: str) -> str:
    return " ".join(value.split())


def get_match(text: str, pattern: str, flags: int = 0) -> str:
    match = re.search(pattern, text, flags)
    if not match:
        return ""
    return match.group(1).strip()


def get_title(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if set(stripped) == {"="}:
            continue
        return stripped
    return ""


def extract_classification_id(path: Path) -> str:
    stem = path.stem
    for suffix in (
        "_fire_num_with_WUI_offset_report",
        "_fire_num_report",
        "_fire_area_report",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    parts = stem.split("_")
    if len(parts) >= 3:
        return "_".join(parts[-3:])
    return stem


def parse_family_and_link(text: str) -> tuple[str, str]:
    match = re.search(r"Family:\s*([^\s]+)\s+\(\s*([^)]+?)\s*\)", text)
    if not match:
        return "", ""
    return match.group(1).strip(), match.group(2).strip()


def parse_formula_and_data(text: str) -> tuple[str, str]:
    match = re.search(r"Formula:\s*(.*?)\nData:\s*(.+)", text, flags=re.DOTALL)
    if not match:
        return "", ""
    formula = normalize_space(match.group(1))
    data_name = match.group(2).splitlines()[0].strip()
    return formula, data_name


def parse_r2_lines(text: str) -> tuple[str, str]:
    lines = [line.strip() for line in re.findall(r"^R2.*$", text, flags=re.MULTILINE)]
    first = lines[0] if len(lines) >= 1 else ""
    second = lines[1] if len(lines) >= 2 else ""
    return first, second


def extract_block(text: str, title: str) -> str:
    pattern = rf"{re.escape(title)}:\n(.*?)(?:\n\s*\n|\Z)"
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def parse_keyed_value(block: str, label: str) -> str:
    if not block:
        return ""
    match = re.search(rf"{re.escape(label)}:[ \t]*(.*)", block)
    if not match:
        return ""
    return match.group(1).strip()


def extract_conditional_model_coefficients_block(text: str) -> str:
    match = re.search(
        r"Dispersion parameter.*?\n\n(Conditional model:\n.*?)(?:\n---|\Z)",
        text,
        flags=re.DOTALL,
    )
    if not match:
        return ""
    return match.group(1).strip()


def parse_report(path: Path, source_folder: str) -> dict[str, str]:
    text = read_text_with_fallback(path).replace("\r\n", "\n")
    family, link = parse_family_and_link(text)
    formula, data_name = parse_formula_and_data(text)
    r2_line_1, r2_line_2 = parse_r2_lines(text)

    uniformity_block = extract_block(text, "Uniformity Test (Kolmogorov-Smirnov)")
    outliers_block = extract_block(text, "Outliers Test")
    dispersion_block = extract_block(text, "Dispersion Test")
    zero_inflation_block = extract_block(text, "Zero Inflation Test")

    row = {
        "classification_id": extract_classification_id(path),
        "source_folder": source_folder,
        "source_file": path.name,
        "report_title": get_title(text),
        "convergence_message": get_match(
            text,
            r"1\. Model Convergence\s*-+\s*(.+)",
            flags=re.DOTALL,
        ).splitlines()[0].strip() if "1. Model Convergence" in text else "",
        "convergence_code": get_match(text, r"收敛码:\s*(.+)"),
        "max_gradient": get_match(text, r"最大梯度:\s*(.+)"),
        "gradient_check": get_match(text, r"梯度检查:\s*(.+)"),
        "random_effect_variance_ok": get_match(text, r"随机效应方差正常:\s*(.+)"),
        "random_effect_variance": get_match(text, r"随机效应方差值:\s*(.+)"),
        "family": family,
        "link": link,
        "formula": formula,
        "data_name": data_name,
        "n_obs": get_match(text, r"Number of obs:\s*([\d,]+)(?=,\s*groups:)"),
        "n_groups": get_match(text, r"Number of obs:\s*[\d,]+,\s*groups:\s*[^,]+,\s*([\d,]+)"),
        "dispersion_parameter": get_match(
            text,
            r"Dispersion parameter for .*? family \(\):\s*(.+)",
        ),
        "AIC": get_match(text, r"^AIC:\s*(.+)$", flags=re.MULTILINE),
        "BIC": get_match(text, r"^BIC:\s*(.+)$", flags=re.MULTILINE),
        "logLik": get_match(text, r"^logLik:\s*(.+)$", flags=re.MULTILINE),
        "r2_line_1": r2_line_1,
        "r2_line_2": r2_line_2,
        "mse": get_match(text, r"^MSE\s*:\s*(.+)$", flags=re.MULTILINE),
        "rmse": get_match(text, r"^RMSE:\s*(.+)$", flags=re.MULTILINE),
        "mae": get_match(text, r"^MAE\s*:\s*(.+)$", flags=re.MULTILINE),
        "conditional_model_coefficients_block": extract_conditional_model_coefficients_block(text),
        "uniformity_statistic": parse_keyed_value(uniformity_block, "Statistic"),
        "uniformity_p_value": parse_keyed_value(uniformity_block, "p-value"),
        "uniformity_interpretation": parse_keyed_value(uniformity_block, "Interpretation"),
        "outliers_expected": parse_keyed_value(outliers_block, "Number of outliers expected"),
        "outliers_observed": parse_keyed_value(outliers_block, "Number of outliers observed"),
        "outliers_p_value": parse_keyed_value(outliers_block, "p-value"),
        "outliers_interpretation": parse_keyed_value(outliers_block, "Interpretation"),
        "dispersion_test_parameter_estimate": parse_keyed_value(dispersion_block, "Parameter estimate"),
        "dispersion_test_p_value": parse_keyed_value(dispersion_block, "p-value"),
        "dispersion_test_interpretation": parse_keyed_value(dispersion_block, "Interpretation"),
        "zero_inflation_expected": parse_keyed_value(zero_inflation_block, "Number of zeros expected"),
        "zero_inflation_observed": parse_keyed_value(zero_inflation_block, "Number of zeros observed"),
        "zero_inflation_p_value": parse_keyed_value(zero_inflation_block, "p-value"),
        "zero_inflation_interpretation": parse_keyed_value(zero_inflation_block, "Interpretation"),
    }
    return row


def build_fieldnames(rows: list[dict[str, str]]) -> list[str]:
    dynamic_fields = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in BASE_COLUMNS
        }
    )
    return BASE_COLUMNS + dynamic_fields


def write_csv(rows: list[dict[str, str]], output_csv: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = build_fieldnames(rows)

    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["classification_id"]):
            writer.writerow(row)


def run_job(source_dir: Path, output_csv: Path) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"目录不存在: {source_dir}")

    txt_files = sorted(source_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"目录下未找到 txt 文件: {source_dir}")

    rows = [parse_report(path, source_dir.name) for path in txt_files]
    write_csv(rows, output_csv)
    print(f"已生成: {output_csv} ({len(rows)} 行)")


def main() -> None:
    for job in JOBS:
        run_job(job["source_dir"], job["output_csv"])


if __name__ == "__main__":
    main()


