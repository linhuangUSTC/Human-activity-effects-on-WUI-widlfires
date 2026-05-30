#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
程序名称: WUI数据合并程序
功能简介: 
    1. 合并Interface WUI和Intermix WUI的CSV数据文件
    2. 计算总火灾数量和总火灾面积
    3. 生成合并后的WUI数据文件
    4. 根据GRID_ID中Interface和Intermix WUI的面积大小判断主要WUI类型
输入文件:
    - Interface WUI数据: I:\Processing data\Grid_Clip_YYYY_interface_WUI.csv
    - Intermix WUI数据: I:\Processing data\Grid_Clip_YYYY_intermix_WUI.csv
    (其中YYYY为处理年份:2005, 2010, 2015, 2020)

输出文件:
    - 合并后的数据: I:\Processing data\Grid_Clip_YYYY_WUI.csv
"""

import pandas as pd
import numpy as np

def merge_wui_data(interface_path, intermix_path, output_path):
    # 读取两个CSV文件
    df_interface = pd.read_csv(interface_path)
    df_intermix = pd.read_csv(intermix_path)
    
    # 定义需要保留两列的变量前缀
    dual_columns_prefixes = ['WUI_area', 'Fire_num', 'Fire_area']
    
    # 为Interface数据添加前缀
    interface_dual_cols = df_interface[['GRID_ID'] + [col for col in df_interface.columns if any(col.startswith(prefix) for prefix in dual_columns_prefixes)]]
    interface_dual_cols = interface_dual_cols.add_prefix('Interface_')
    interface_dual_cols = interface_dual_cols.rename(columns={'Interface_GRID_ID': 'GRID_ID'})
    
    # 为Intermix数据添加前缀
    intermix_dual_cols = df_intermix[['GRID_ID'] + [col for col in df_intermix.columns if any(col.startswith(prefix) for prefix in dual_columns_prefixes)]]
    intermix_dual_cols = intermix_dual_cols.add_prefix('Intermix_')
    intermix_dual_cols = intermix_dual_cols.rename(columns={'Intermix_GRID_ID': 'GRID_ID'})
    
    # 合并需要保留两列的变量
    df_merged = pd.merge(interface_dual_cols, intermix_dual_cols, on='GRID_ID', how='outer')
    
    # 处理只需要保留一列的变量
    # 获取Interface和Intermix中所有需要保留一列的变量
    # 首先获取两个文件中的所有变量
    all_columns = list(df_interface.columns) + [col for col in df_intermix.columns if col not in df_interface.columns]
    
    # 定义需要只保留一列的变量（排除GRID_ID和需要保留两列的变量）
    single_columns = [col for col in all_columns 
                     if col != 'GRID_ID' and not any(col.startswith(prefix) for prefix in dual_columns_prefixes)]
     # 对带年份的列名进行特殊排序（例如PopD_2000, PopD_2005等按年份排序）
    def sort_columns_with_years(columns):
        # 分离带年份和不带年份的列
        columns_with_years = []
        columns_without_years = []
        
        for col in columns:
            parts = col.split('_')
            if len(parts) > 1 and parts[-1].isdigit():
                columns_with_years.append(col)
            else:
                columns_without_years.append(col)
        
        # 对带年份的列按年份排序
        columns_with_years.sort(key=lambda x: int(x.split('_')[-1]))
        
        # 合并两部分
        return columns_without_years + columns_with_years
    
    # 应用排序
    single_columns = sort_columns_with_years(single_columns)
    
    # 创建一个新的数据框来存储合并后的单列变量
    df_single_columns = pd.DataFrame()
    df_single_columns['GRID_ID'] = df_merged['GRID_ID']
    
    # 对于每个只需要保留一列的变量，优先使用Interface的值，如果Interface没有值则使用Intermix的值
    for col in single_columns:
        # 创建一个字典来存储所有GRID_ID的值
        value_dict = {}
        
        # 首先填充Interface的值
        if col in df_interface.columns:
            for _, row in df_interface.iterrows():
                grid_id = row['GRID_ID']
                value = row[col]
                value_dict[grid_id] = value
        
        # 然后用Intermix的值填补空值
        if col in df_intermix.columns:
            for _, row in df_intermix.iterrows():
                grid_id = row['GRID_ID']
                value = row[col]
                # 如果这个GRID_ID不在字典中，或者字典中的值是NaN，则使用Intermix的值
                if grid_id not in value_dict or pd.isna(value_dict[grid_id]):
                    value_dict[grid_id] = value
        
        # 为合并后的每个GRID_ID赋值
        merged_values = []
        for grid_id in df_merged['GRID_ID']:
            if grid_id in value_dict:
                merged_values.append(value_dict[grid_id])
            else:
                merged_values.append(None)
        
        # 将合并后的变量添加到结果数据框
        df_single_columns[col] = merged_values

    # 合并所有数据 - 这是之前缺失的关键代码行
    df_merged = pd.merge(df_merged, df_single_columns, on='GRID_ID', how='outer')

    # 计算WUI总面积（Interface + Intermix）
    df_merged['All_WUI_area'] = df_merged['Interface_WUI_area'].fillna(0) + df_merged['Intermix_WUI_area'].fillna(0)
    
    # 判断主要WUI类型（根据面积大小）
    df_merged['main_WUI'] = 'mix'  # 默认值为'mix'
    # 当Interface面积大于Intermix面积时，设置为'face'
    interface_area = df_merged['Interface_WUI_area'].fillna(0)
    intermix_area = df_merged['Intermix_WUI_area'].fillna(0)
    df_merged.loc[interface_area > intermix_area, 'main_WUI'] = 'face'
    
    # 获取所有年份（从Fire_num或Fire_area列名中提取）
    fire_years = set()
    for col in df_merged.columns:
        if col.startswith('Interface_Fire_num_') or col.startswith('Interface_Fire_area_'):
            year = col.split('_')[-1]
            fire_years.add(int(year))
    fire_years = sorted(fire_years)
    
    # 计算总火灾数量和总火灾面积
    for year in fire_years:
        # 获取Interface和Intermix的火灾数据
        interface_fire_num = df_merged[f'Interface_Fire_num_{year}'].fillna(0)
        interface_fire_area = df_merged[f'Interface_Fire_area_{year}'].fillna(0)
        intermix_fire_num = df_merged[f'Intermix_Fire_num_{year}'].fillna(0)
        intermix_fire_area = df_merged[f'Intermix_Fire_area_{year}'].fillna(0)
        
        # 计算All的火灾数量和过火面积
        # 将空值视为0参与计算，确保即使只有Interface或Intermix数据也能正确计算总和
        total_fire_num = interface_fire_num + intermix_fire_num
        total_fire_area = interface_fire_area + intermix_fire_area
        
        # 将All_Fire_num和All_Fire_area存储到数据框中
        df_merged[f'All_Fire_num_{year}'] = total_fire_num
        df_merged[f'All_Fire_area_{year}'] = total_fire_area
    
    # 保存合并后的数据
    df_merged.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("  合并完成，共%d行%d列数据，已保存到: %s" % (len(df_merged), len(df_merged.columns), output_path))
    
    return df_merged

def main():
    """
    主函数
    """
    print("=== WUI数据合并程序 ===")
    
    # 定义需要处理的年份
    years = [2005, 2010, 2015, 2020]
    
    print("处理的年份列表: %s" % years)
    
    # 循环处理每个年份的数据
    for year in years:
        print("\n" + "="*50)
        print("开始处理年份: %d" % year)
        print("="*50)
        
        # 构建文件路径
        interface_path = r"I:\Processing data\Grid_Clip_%d_interface_WUI.csv" % year
        intermix_path = r"I:\Processing data\Grid_Clip_%d_intermix_WUI.csv" % year
        output_path = r"I:\Processing data\Grid_Clip_%d_WUI.csv" % year
        
        print("Interface文件路径: %s" % interface_path)
        print("Intermix文件路径: %s" % intermix_path)
        print("输出文件路径: %s" % output_path)
        
            
        # 合并数据
        print("开始合并数据...")
        df_merged = merge_wui_data(interface_path, intermix_path, output_path)
            
        # 显示合并结果的基本信息（只显示行数和列数，不显示详细信息）
        print("\n合并结果基本信息:")
        print("  行数: %d" % len(df_merged))
        print("  列数: %d" % len(df_merged.columns))
        print("年份 %d 处理完成！" % year)

    
    print("\n" + "="*50)
    print("所有年份数据处理完成！")
    print("="*50)

if __name__ == "__main__":
    main()