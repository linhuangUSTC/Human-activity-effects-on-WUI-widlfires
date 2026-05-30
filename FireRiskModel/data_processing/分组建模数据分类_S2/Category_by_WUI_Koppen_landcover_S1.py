#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
程序名称: GRID_ID分组程序
功能简介:
    1. 读取三个CSV文件中的特定列
    2. 根据main_WUI、Koppen_Category和Top1_Value的唯一组合对GRID_ID进行分组
    3. 为每组GRID_ID创建单独的CSV文件
    4. 将结果输出到指定目录
    5. 支持处理多个年份的数据 (2005, 2010, 2015, 2020)

输入文件格式:
    - WUI数据: I:/Processing data/Grid_Clip_{year}_WUI.csv (包含main_WUI列)
    - Koppen气候分类: I:/Processing data/Koppe match/Grid_Clip_{year}_Koppen_Stats.csv (包含Koppen_Category列)
    - 土地覆盖数据: I:/Processing data/landcover match/Grid_Clip_{year}_WUI_LandCover_Stats.csv (包含Top1_Value列)

输出目录格式:
    - I:/Processing data/渔网分类建模/{year}/

输出文件命名规则:
    - 格式: {main_WUI}_{Koppen_Category}_{Top1_Value}.csv
"""

import pandas as pd
import numpy as np
import os


def group_grid_ids(year):
    """
    分组GRID_ID并创建CSV文件
    
    参数:
    year: int - 要处理的年份 (例如: 2005, 2010, 2015, 2020)
    """
    # 定义文件路径
    wui_file = fr"I:/Processing data/Grid_Clip_{year}_WUI.csv"
    koppen_file = fr"I:/Processing data/Koppe match/Grid_Clip_{year}_Koppen_Stats.csv"
    landcover_file = fr"I:/Processing data/landcover match/Grid_Clip_{year}_WUI_LandCover_Stats.csv"
    output_dir = fr"I:/Processing data/渔网分类建模/{year}"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== GRID_ID分组程序 (多年份版) ===")
    
    try:
        # 读取WUI数据（只需要GRID_ID和main_WUI列）
        print("\n1. 读取WUI数据...")
        df_wui = pd.read_csv(wui_file, usecols=['GRID_ID', 'main_WUI'])
        print(f"   读取完成，共{len(df_wui)}行数据")
        
        # 读取Koppen气候分类数据（只需要GRID_ID和Koppen_Category列）
        print("2. 读取Koppen气候分类数据...")
        df_koppen = pd.read_csv(koppen_file, usecols=['GRID_ID', 'Koppen_Category'])
        print(f"   读取完成，共{len(df_koppen)}行数据")
        
        # 读取土地覆盖数据（需要GRID_ID、Top1_Value和Top2_Value列）
        print("3. 读取土地覆盖数据...")
        df_landcover = pd.read_csv(landcover_file, usecols=['GRID_ID', 'Top1_Value', 'Top2_Value'])
        print(f"   读取完成，共{len(df_landcover)}行数据")
        
        # 合并三个数据集
        print("4. 合并数据集...")
        # 先合并WUI和Koppen数据
        df_merged = pd.merge(df_wui, df_koppen, on='GRID_ID', how='inner')
        # 再合并土地覆盖数据
        df_merged = pd.merge(df_merged, df_landcover, on='GRID_ID', how='inner')
        print(f"   合并完成，共{len(df_merged)}行数据")
        # 将Top1_Value转换为整数类型，确保分组和文件名生成使用相同的数据类型
        # 这样可以避免因数据类型不同导致的分组数量与实际文件数量不一致的问题
        df_merged['Top1_Value'] = df_merged['Top1_Value'].astype(int)
        print("   Top1_Value列已转换为整数类型")
        
        # 根据三列的唯一组合进行分组
        print("5. 根据main_WUI、Koppen_Category和Top1_Value分组...")
        grouped = df_merged.groupby(['main_WUI', 'Koppen_Category', 'Top1_Value'])
        print(f"   共分成{len(grouped)}个组")
        
        # 为每组创建CSV文件
        print("6. 创建分组CSV文件...")
        for (main_wui, koppen_cat, top1_value), group in grouped:
            # 创建文件名
            filename = f"{main_wui}_{koppen_cat}_{int(top1_value)}.csv"
            filepath = os.path.join(output_dir, filename)
            
            # 保存GRID_ID到CSV文件
            group[['GRID_ID']].to_csv(filepath, index=False, encoding='utf-8-sig')
        
        print("\n✅ 所有分组文件创建完成！")
        print(f"📁 输出目录: {output_dir}")
        print(f"📊 总组数: {len(grouped)}")
        
    except Exception as e:
        print(f"\n❌ 处理过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """
    主函数 - 处理多个年份的数据
    """
    # 要处理的年份列表
    years = [2005, 2010, 2015, 2020]
    
    # 遍历所有年份
    for year in years:
        print(f"\n\n========================================")
        print(f"           处理年份: {year}")
        print(f"========================================")
        group_grid_ids(year)


if __name__ == "__main__":
    main()