#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
程序名称: 火灾数据分类汇总统计程序(包含0值)
功能简介:
    1. 读取火灾分类数据汇总文件夹中的所有年份统计文件
    2. 提取每个文件中的分类ID和火灾数量/火灾面积(包括0值)数量
    3. 将四个年份的数据进行汇总,计算每个分类ID的总和
    4. 将汇总结果保存到新的Excel文件中

输入文件:
    - I:/Processing data/渔网分类建模/火灾分类数据汇总/火灾数据统计汇总_*.csv

输出文件:
    - I:/Processing data/渔网分类建模/火灾分类数据汇总/分类建模火灾数据汇总统计_with_zero.xlsx
"""

import pandas as pd
import os
import glob


def main():
    """
    主函数 - 执行火灾数据分类汇总统计(包含0值)
    """
    print("=== 火灾数据分类汇总统计程序(包含0值) ===")
    
    # 定义要处理的年份列表
    wui_years = [2005, 2010, 2015, 2020]
    
    # 定义文件路径
    input_dir = fr"I:\Processing data\渔网分类建模\火灾分类数据汇总"
    output_file = os.path.join(input_dir, "分类建模火灾数据汇总统计_with_zero.xlsx")
    
    try:
        # 1. 读取所有年份的统计文件
        print("1. 读取所有年份的统计文件...")
        all_data = []
        
        for wui_year in wui_years:
            input_file = os.path.join(input_dir, f"火灾数据统计汇总_{wui_year}.csv")
            
            # 读取文件
            df = pd.read_csv(input_file)
            
            # 确保必要的列存在
            required_columns = ['分类ID', 'fire_num所有值数量', 'fire_area所有值数量']
            missing_cols = [c for c in required_columns if c not in df.columns]
            if missing_cols:
                raise ValueError(f"缺少必要列: {missing_cols} in {input_file}")
            
            # 选择需要的列并添加年份列
            df_selected = df[['分类ID', 'fire_num所有值数量', 'fire_area所有值数量']].copy()
            df_selected['年份'] = wui_year
            
            # 重命名数量列以保持一致性
            df_selected = df_selected.rename(columns={
                'fire_num所有值数量': 'fire_num_total_count',
                'fire_area所有值数量': 'fire_area_total_count'
            })
            
            all_data.append(df_selected)
        
        # 2. 合并所有数据
        print("2. 合并所有年份的数据...")
        df_merged = pd.concat(all_data, ignore_index=True)
        
        # 3. 按分类ID汇总数据
        print("3. 按分类ID汇总火灾数据...")
        df_summary = df_merged.groupby('分类ID').agg({
            'fire_num_total_count': 'sum',
            'fire_area_total_count': 'sum',
            '年份': 'count'  # 统计该分类ID出现的年份数量
        }).reset_index()
        
        # 重命名年份列
        df_summary = df_summary.rename(columns={'年份': '覆盖年份数量'})
        
        # 按分类ID排序
        df_summary = df_summary.sort_values('分类ID', ascending=True)
        
        # 4. 保存汇总结果
        print("4. 保存汇总结果...")
        df_summary.to_excel(output_file, index=False, engine='openpyxl')
        print(f"   ✅ 汇总结果已保存: {os.path.basename(output_file)}")
        print("\n✅ 所有处理完成！")
        
    except Exception as e:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
