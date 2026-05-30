# -*- coding: utf-8 -*-
"""
批量计算火灾频率 (Fire frequency)

规则：
1) 读取指定目录下的所有 CSV 文件
2) Fire_frequency = All_fire_num / All_WUI_area
3) 结果写回原 CSV 文件
"""

from pathlib import Path

import pandas as pd


INPUT_DIR = Path(r"I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic")
OUTPUT_COLUMN = "Fire_frequency"


def process_csv(csv_path: Path) -> None:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df[OUTPUT_COLUMN] = df["All_fire_num"] / df["All_WUI_area"]
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def main() -> None:
    for csv_path in INPUT_DIR.glob("*.csv"):
        process_csv(csv_path)


if __name__ == "__main__":
    main()
