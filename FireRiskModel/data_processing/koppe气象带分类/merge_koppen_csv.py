#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
程序名称: 柯本气象带分类数据合并程序
功能简介: 
    1. 合并Interface和Intermix的柯本气象带分类CSV数据文件
    2. 将气象带代码转换为A/B/C/D/E类别:
        - 1-3为A
        - 4-7为B
        - 8-16为C
        - 17-28为D
        - 29-30为E
        - 其他数值则为NaN
    3. 剔除Interface与Intermix中重复的GRID_ID（取并集，优先保留Interface的值）
    4. 生成合并后的气象带分类数据文件

输入文件:
    - Interface数据: I:\Processing data\Koppe match\Grid_Clip_YYYY_interface_Koppen_Stats.csv
    - Intermix数据: I:\Processing data\Koppe match\Grid_Clip_YYYY_intermix_Koppen_Stats.csv
    (其中YYYY为处理年份:2005, 2010, 2015, 2020)

输出文件:
    - 合并后的数据: I:\Processing data\Koppe match\Grid_Clip_YYYY_Koppen_Stats.csv
"""

import pandas as pd
import numpy as np

def convert_koppen_code_to_category(koppen_code):
    """
    将柯本气象带代码转换为A/B/C/D/E类别
    
    参数:
    koppen_code: 柯本气象带代码（整数）
    
    返回值:
    str: 类别代码（'A', 'B', 'C', 'D', 'E'或None）
    """
    if pd.isna(koppen_code):
        return None
    koppen_code = int(koppen_code)
    if 1 <= koppen_code <= 3:
        return 'A'
    elif 4 <= koppen_code <= 7:
        return 'B'
    elif 8 <= koppen_code <= 16:
        return 'C'
    elif 17 <= koppen_code <= 28:
        return 'D'
    elif 29 <= koppen_code <= 30:
        return 'E'
    else:
        return None

def merge_koppen_data(interface_path, intermix_path, output_path):
    """
    合并两个柯本气象带分类CSV文件并转换气象带代码
    
    参数:
    interface_path: Interface数据的CSV文件路径
    intermix_path: Intermix数据的CSV文件路径
    output_path: 合并后的数据保存路径
    
    返回值:
    pd.DataFrame: 合并并转换后的DataFrame
    
    功能说明:
    1. 读取Interface和Intermix的柯本气象带数据
    2. 将气象带代码转换为A/B/C类别
    3. 合并两个数据集，取并集（优先保留Interface的值）
    4. 保存合并后的数据到CSV文件
    """
    # 读取两个CSV文件
    df_interface = pd.read_csv(interface_path)
    df_intermix = pd.read_csv(intermix_path)
    
    # 转换气象带代码为类别
    df_interface['Koppen_Category'] = df_interface['Koppen_Type_Code'].apply(convert_koppen_code_to_category)
    df_intermix['Koppen_Category'] = df_intermix['Koppen_Type_Code'].apply(convert_koppen_code_to_category)
    
    # 选择需要的列
    interface_cols = ['GRID_ID', 'Koppen_Type_Code', 'Koppen_Category', 'Pixel_Count', 'Percentage']
    intermix_cols = ['GRID_ID', 'Koppen_Type_Code', 'Koppen_Category', 'Pixel_Count', 'Percentage']
    
    df_interface = df_interface[interface_cols]
    df_intermix = df_intermix[intermix_cols]
    
    # 合并数据，取并集（优先保留Interface的值）
    df_merged = pd.merge(df_interface, df_intermix, on='GRID_ID', how='outer', indicator=True)
    
    # 根据合并结果选择数据
    # 如果在Interface中存在，优先使用Interface的值
    # 否则使用Intermix的值
    df_result = pd.DataFrame()
    df_result['GRID_ID'] = df_merged['GRID_ID']
    
    # 选择Interface的列（优先）
    interface_prefix_cols = ['GRID_ID', 'Koppen_Type_Code_x', 'Koppen_Category_x', 'Pixel_Count_x', 'Percentage_x']
    intermix_prefix_cols = ['GRID_ID', 'Koppen_Type_Code_y', 'Koppen_Category_y', 'Pixel_Count_y', 'Percentage_y']
    
    # 创建最终的数据框
    for i, col in enumerate(['Koppen_Type_Code', 'Koppen_Category', 'Pixel_Count', 'Percentage']):
        interface_col = col + '_x'
        intermix_col = col + '_y'
        
        # 优先使用Interface的值，如果Interface为空则使用Intermix的值
        df_result[col] = df_merged[interface_col].combine_first(df_merged[intermix_col])
    
    # 保存合并后的数据
    df_result.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("  合并完成，共%d行%d列数据，已保存到: %s" % (len(df_result), len(df_result.columns), output_path))
    
    return df_result

def main():
    """
    主函数
    """
    print("=== 柯本气象带分类数据合并程序 ===")
    
    # 定义需要处理的年份
    years = [2005, 2010, 2015, 2020]
    
    print("处理的年份列表: %s" % years)
    
    # 定义输入输出目录
    input_output_dir = r"I:\Processing data\Koppe match"
    
    # 循环处理每个年份的数据
    for year in years:
        print("\n" + "="*50)
        print("开始处理年份: %d" % year)
        print("="*50)
        
        # 构建文件路径
        interface_path = r"%s\Grid_Clip_%d_interface_Koppen_Stats.csv" % (input_output_dir, year)
        intermix_path = r"%s\Grid_Clip_%d_intermix_Koppen_Stats.csv" % (input_output_dir, year)
        output_path = r"%s\Grid_Clip_%d_Koppen_Stats.csv" % (input_output_dir, year)
        
        print("Interface文件路径: %s" % interface_path)
        print("Intermix文件路径: %s" % intermix_path)
        print("输出文件路径: %s" % output_path)
        
        try:
            # 检查文件是否存在
            import os
            if os.path.exists(interface_path):
                print("Interface文件存在")
            else:
                print("Interface文件不存在")
                raise FileNotFoundError("Interface文件不存在: %s" % interface_path)
            
            if os.path.exists(intermix_path):
                print("Intermix文件存在")
            else:
                print("Intermix文件不存在")
                raise FileNotFoundError("Intermix文件不存在: %s" % intermix_path)
            
            # 合并数据
            print("开始合并数据...")
            df_merged = merge_koppen_data(interface_path, intermix_path, output_path)
            
            # 显示合并结果的基本信息
            print("\n合并结果基本信息:")
            print("  行数: %d" % len(df_merged))
            print("  列数: %d" % len(df_merged.columns))
            
            # 显示类别分布
            if 'Koppen_Category' in df_merged.columns:
                print("\n类别分布:")
                category_counts = df_merged['Koppen_Category'].value_counts()
                for category, count in category_counts.items():
                    if pd.notna(category):
                        print("  %s: %d" % (category, count))
                    else:
                        print("  None: %d" % count)
            
            print("年份 %d 处理完成！" % year)
        except Exception as e:
            print("\n处理年份 %d 时出错: %s" % (year, str(e)))
            import traceback
            print("错误详情:")
            traceback.print_exc()
            print("继续处理下一个年份...")
            continue
    
    print("\n" + "="*50)
    print("所有年份数据处理完成！")
    print("="*50)

if __name__ == "__main__":
    main()
