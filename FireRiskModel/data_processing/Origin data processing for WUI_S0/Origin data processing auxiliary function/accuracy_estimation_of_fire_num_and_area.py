#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
程序名称: WUI面积与火灾面积差值分析
功能简介:
    1. 读取WUI数据和火灾面积数据
    2. 计算WUI_area与Fire_area_year的差值
    3. 找出每个年份和WUI类型下差值最小的10个值
    4. 将结果输出到txt文件中

输入文件:
    - I:/Processing data/Grid_Clip_{year}_{wui_type}_WUI.csv
    其中wui_type为interface或intermix
    year为2005、2010、2015、2020

输出文件:
    - I:/Processing data/最小WUI火灾差值分析.txt

处理逻辑:
    1. 对每种WUI类型(interface, intermix)和年份(2005, 2010, 2015, 2020)进行循环
    2. 读取对应的CSV文件
    3. 计算WUI_area - Fire_area_year的差值
    4. 对差值进行排序，找出最小的10个值
    5. 将结果写入txt文件
"""

import pandas as pd
import os


def main():
    """
    主函数 - 执行WUI面积与火灾面积差值分析
    """
    print("=== WUI面积与火灾面积差值分析程序 ===")
    
    # 定义基本参数
    base_path = r"I:/Processing data"
    wui_types = ["interface", "intermix"]
    wui_years = [2005, 2010, 2015, 2020]
    output_file = os.path.join(base_path, "最小WUI火灾差值分析.txt")
    
    try:
        # 打开输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("WUI面积与火灾面积差值分析结果\n")
            f.write("=" * 80 + "\n\n")
            
            # 循环处理每种WUI类型
            for wui_type in wui_types:
                f.write(f"【{wui_type.capitalize()} WUI 分析】\n")
                f.write("-" * 60 + "\n\n")
                
                # 循环处理每个WUI年份
                for wui_year in wui_years:
                    # 计算火灾年份范围 (WUI年份前后两年)
                    fire_start_year = wui_year - 2
                    fire_end_year = wui_year + 2
                    fire_years = list(range(fire_start_year, fire_end_year + 1))
                    
                    f.write(f"年份: {wui_year} (火灾年份范围: {fire_start_year}-{fire_end_year})\n")
                    f.write("=" * 40 + "\n")
                    
                    # 构建CSV文件路径
                    csv_file = os.path.join(base_path, f"Grid_Clip_{wui_year}_{wui_type}_WUI.csv")
                    
                    if not os.path.exists(csv_file):
                        f.write(f"❌ 文件不存在: {os.path.basename(csv_file)}\n\n")
                        continue
                    
                    # 读取CSV文件
                    print(f"读取文件: {os.path.basename(csv_file)}")
                    df = pd.read_csv(csv_file)
                    
                    # 检查必要的列是否存在
                    if 'WUI_area' not in df.columns:
                        f.write(f"❌ 文件中缺少'WUI_area'列\n\n")
                        continue
                    
                    # 处理每个火灾年份
                    for fire_year in fire_years:
                        fire_area_col = f'Fire_area_{fire_year}'
                        
                        if fire_area_col not in df.columns:
                            f.write(f"  ⚠️  缺少{fire_area_col}列，跳过\n")
                            continue
                        
                        # 计算差值 (WUI_area - Fire_area_year)
                        df['Difference'] = df['WUI_area'] - df[fire_area_col]
                        
                        # 找出最小的10个差值
                        df_sorted = df.nsmallest(10, 'Difference')
                        
                        f.write(f"\n火灾年份 {fire_year} 的最小10个差值：\n")
                        f.write("  GRID_ID      WUI_area      Fire_area      Difference\n")
                        f.write("  " + "-" * 60 + "\n")
                        
                        for index, row in df_sorted.iterrows():
                            f.write(f"  {row['GRID_ID']:<10}  {row['WUI_area']:<12.6f}  {row[fire_area_col]:<12.6f}  {row['Difference']:<12.6f}\n")
                        
                    f.write("\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
        
        print(f"✅ 分析完成！结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"❌ 处理过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()