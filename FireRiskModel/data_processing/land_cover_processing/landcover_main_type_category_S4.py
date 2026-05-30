# -*- coding: utf-8 -*-
"""
根据Top1_Value对GRID_ID进行分类

该程序读取landcover_extract_and_statistic_parallel_S3.py的输出CSV文件，
根据Top1_Value的唯一值将GRID_ID进行分类，然后分别存储到新的CSV中的不同列下。
"""

import os
import csv
import argparse

def classify_grid_ids(input_csv, output_csv):
    """
    根据Top1_Value对GRID_ID进行分类
    
    参数:
    input_csv: 输入CSV文件路径
    output_csv: 输出CSV文件路径
    """
    if not os.path.exists(input_csv):
        print(f"错误: 输入文件不存在 - {input_csv}")
        return False
    
    print(f"读取输入文件: {input_csv}")
    
    # 步骤1: 读取数据并提取唯一的Top1_Value
    top1_values = set()
    grid_data = {}
    
    with open(input_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            grid_id = row['GRID_ID'].strip()
            top1_value = row['Top1_Value'].strip()
            
            if not grid_id or not top1_value:
                continue
                
            top1_values.add(top1_value)
            
            if top1_value not in grid_data:
                grid_data[top1_value] = []
            grid_data[top1_value].append(grid_id)
    
    # 步骤2: 对Top1_Value进行排序
    sorted_top1_values = sorted(list(top1_values), key=lambda x: int(x) if x.isdigit() else x)
    
    # 步骤3: 确定最大行数
    max_rows = max(len(grid_data[top1]) for top1 in grid_data)
    
    # 步骤4: 写入输出文件
    print(f"写入输出文件: {output_csv}")
    print(f"找到 {len(top1_values)} 个唯一的Top1_Value")
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        # 构建列名
        fieldnames = [f"Main_lancover_{top1}" for top1 in sorted_top1_values]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # 写入表头
        writer.writeheader()
        
        # 写入数据
        for row_idx in range(max_rows):
            row = {}
            for top1 in sorted_top1_values:
                col_name = f"Main_lancover_{top1}"
                if row_idx < len(grid_data[top1]):
                    row[col_name] = grid_data[top1][row_idx]
                else:
                    row[col_name] = ''  # 空值填充
            writer.writerow(row)
    
    print(f"\n分类完成！")
    print(f"输出文件: {output_csv}")
    print(f"分类情况:")
    for top1 in sorted_top1_values:
        print(f"  - Main_lancover_{top1}: {len(grid_data[top1])} 个GRID_ID")
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='根据Top1_Value对GRID_ID进行分类')
    parser.add_argument('--years', nargs='+', type=int, default=[2005, 2010, 2015, 2020],
                      help='要处理的年份列表（默认：2005 2010 2015 2020）')
    parser.add_argument('-i', '--input', 
                      default=None,
                      help='输入CSV文件路径（如果指定，将忽略年份参数）')
    parser.add_argument('-o', '--output',
                      default=None,
                      help='输出CSV文件路径（如果指定，将忽略年份参数）')
    
    args = parser.parse_args()
    
    # 如果指定了输入输出路径，只处理单个文件
    if args.input is not None and args.output is not None:
        classify_grid_ids(args.input, args.output)
    else:
        # 处理所有指定年份
        for year in args.years:
            print(f"\n=== 开始处理 {year} 年数据 ===")
            input_csv = rf'I:\Processing data\landcover match\Grid_Clip_{year}_WUI_LandCover_Stats.csv'
            output_csv = rf'I:\Processing data\landcover match\Grid_Clip_{year}_WUI_LandCover_Classified.csv'
            classify_grid_ids(input_csv, output_csv)
            print(f"=== {year} 年数据处理完成 ===")
    
    print("\n所有处理任务已完成！")

if __name__ == "__main__":
    main()