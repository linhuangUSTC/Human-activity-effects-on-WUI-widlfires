#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
程序名称: 四年WUI数据合并工具
功能简介:
    1. 处理2005, 2010, 2015, 2020年的WUI数据
    2. 从每个年份的CSV文件中提取特定的列
    3. 重命名列以保持一致性
    4. 为每个数据添加年份标识
    5. 合并所有年份的数据到一个文件
    6. 保存为新的CSV文件以供共线性分析使用

输入文件:
    - 2005年WUI数据: I:/Processing data/Grid_Clip_2005_WUI.csv,包括2010, 2015, 2020年的数据

输出文件:
    - 合并后的数据: I:/Processing data/Artificial influencing factors.csv
    - 包含所有年份的数据，每行数据都有对应的年份标识
"""

import pandas as pd
import os

def process_yearly_data(year):
    """
    处理指定年份的WUI数据
    
    参数:
    year: 年份
    
    返回:
    DataFrame: 处理后的数据
    """
    # 构建文件路径
    file_path = rf"I:\Processing data\Grid_Clip_{year}_WUI.csv"
    print(f"处理年份: {year}, 文件路径: {file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 - {file_path}")
        return None
    
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        print(f"成功读取文件，共 {len(df)} 行")
        
        # 数据已经在输入CSV中计算好，无需再次计算
        print(f"数据已在输入文件中准备就绪，无需额外计算")
        
        # 提取需要的列
        selected_columns = {
            'GRID_ID': 'GRID_ID',
            'Roadsd': 'Roadsd',
            f'GDP_{year}': 'GDP',
            f'PopD_{year}': 'PopD',
            f'HFI_{year}': 'HFI',
            'Interface_WUI_area': 'Interface_WUI_area',
            'Intermix_WUI_area': 'Intermix_WUI_area',
            f'Interface_Fire_num_{year}': 'Interface_Fire_num',
            f'Interface_Fire_area_{year}': 'Interface_Fire_area',
            f'Intermix_Fire_num_{year}': 'Intermix_Fire_num',
            f'Intermix_Fire_area_{year}': 'Intermix_Fire_area',
            'All_WUI_area': 'All_WUI_area',
            f'All_Fire_num_{year}': 'All_fire_num',
            f'All_Fire_area_{year}': 'All_fire_area'
        }
        
        # 检查必要的列是否存在
        for old_col in selected_columns.keys():
            if old_col not in df.columns:
                print(f"错误: 文件中不存在列 - {old_col}")
                return None
        
        # 选择并重命名列
        processed_df = df[list(selected_columns.keys())].rename(columns=selected_columns)
        
        # 添加年份列
        processed_df['Year'] = year
        
        print(f"年份 {year} 数据处理完成")
        return processed_df
        
    except Exception as e:
        print(f"处理年份 {year} 时出错: {e}")
        return None

def main():
    """
    主函数
    """
    print("=== 开始处理WUI数据 ===")
    
    # 定义需要处理的年份
    years = [2005, 2010, 2015, 2020]
    
    # 存储所有年份的数据
    all_data = []
    
    # 处理每个年份的数据
    for year in years:
        processed_data = process_yearly_data(year)
        if processed_data is not None:
            all_data.append(processed_data)
    
    # 检查是否有数据被处理
    if not all_data:
        print("错误: 没有处理任何数据")
        return
    
    # 合并所有年份的数据
    merged_data = pd.concat(all_data, ignore_index=True)
    print(f"所有年份数据合并完成，共 {len(merged_data)} 行")
    
    # 保存到新的CSV文件
    output_file = rf"I:\Processing data\Artificial influencing factors.csv"
    merged_data.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"数据已保存到: {output_file}")
    
    # 显示数据摘要
    print("\n数据摘要:")
    print(f"年份范围: {min(years)} - {max(years)}")
    print(f"总记录数: {len(merged_data)}")
    print("列名:", list(merged_data.columns))
    
    print("\n=== 处理完成 ===")

if __name__ == "__main__":
    main()