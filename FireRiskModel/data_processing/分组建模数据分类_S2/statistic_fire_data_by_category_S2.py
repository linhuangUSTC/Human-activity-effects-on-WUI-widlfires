#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
程序名称: 火灾数据分类统计程序
功能简介:
    1. 读取分类目录下的所有CSV文件(每个文件包含一组GRID_ID)
    2. 根据GIRD_ID从Grid_Clip_2005_WUI.csv中提取建模数据
    3. 将火灾数据添加到对应的分类CSV文件中
    4. 统计每个分类中火灾数量和面积的非零值数量
    5. 统计每个分类中火灾数量和面积的所有值数量(包括0值)
    6. 将统计结果保存到汇总CSV文件中

输入文件:
    - 分类文件: I:/Processing data/渔网分类建模/2005/*.csv
    - 火灾数据: I:/Processing data/Grid_Clip_2005_WUI.csv

输出文件:
    - 更新后的分类文件: I:/Processing data/渔网分类建模/2005/*.csv
    - 统计结果: I:/Processing data/渔网分类建模/2005/火灾数据统计汇总.csv

处理逻辑:
    1. 读取所有分类CSV文件
    2. 读取Grid_Clip_2005_WUI.csv中的火灾数据
    3. 为每个分类CSV文件添加火灾数据列
    4. 统计每个分类的非零值数量
    5. 保存统计结果
"""

import pandas as pd
import os
import glob


def main():
    """
    主函数 - 执行火灾数据分类统计
    """
    print("=== 火灾数据分类统计程序 ===")
    
    # 定义要处理的年份列表
    wui_years = [2005, 2010, 2015, 2020]
    
    # 年份与火点年份范围的映射关系
    wui_year_fire_mapping = {
        2005: range(2003, 2008),  # 2005年WUI对应2003-2007年火点
        2010: range(2008, 2013),  # 2010年WUI对应2008-2012年火点
        2015: range(2013, 2018),  # 2015年WUI对应2013-2017年火点
        2020: range(2018, 2023)   # 2020年WUI对应2018-2022年火点
    }
    
    try:
        # 循环处理每个年份
        for wui_year in wui_years:
            print(f"\n===== 处理 {wui_year} 年WUI数据 =====")
            
            # 定义文件路径
            category_dir = fr"I:\Processing data\渔网分类建模\{wui_year}"
            fire_data_file = fr"I:\Processing data\Grid_Clip_{wui_year}_WUI.csv"
            output_stat_dir = fr"I:\Processing data\渔网分类建模\火灾分类数据汇总"
            output_stat_file = os.path.join(output_stat_dir, f"火灾数据统计汇总_{wui_year}.csv")
            
            # 1. 获取所有分类CSV文件
            print("1. 获取所有分类CSV文件...")
            category_files = glob.glob(os.path.join(category_dir, "*.csv"))
            # 排除统计文件（包含年份的统计文件名）
            category_files = [f for f in category_files if not f.endswith(f"火灾数据统计汇总_{wui_year}.csv")]
            print(f"   共找到{len(category_files)}个分类文件")
            
            if len(category_files) == 0:
                print("   ❌ 没有找到分类文件，跳过该年份")
                continue
            
            # 2. 读取火灾数据文件
            print("2. 读取火灾数据文件...")
            print(f"   文件路径: {fire_data_file}")
            
            # 检查文件是否存在
            if not os.path.exists(fire_data_file):
                print(f"   ❌ 火灾数据文件不存在: {fire_data_file}")
                continue
            
            # 定义需要读取的列
            fire_columns = ['GRID_ID']
            # 获取当前WUI年份对应的火点年份范围
            fire_years = wui_year_fire_mapping[wui_year]
            # 添加对应年份的火灾数量和面积列
            for year in fire_years:
                fire_columns.append(f'All_Fire_num_{year}')
                fire_columns.append(f'All_Fire_area_{year}')
                # 添加HFI和PopD列（与野火年份一致）
                fire_columns.append(f'HFI_{year}')
                fire_columns.append(f'PopD_{year}')
            # 添加GDP列（使用wui_year）
            fire_columns.append(f'GDP_{wui_year}')
            # 添加All_WUI_area和Roadsd列
            fire_columns.append('All_WUI_area')
            fire_columns.append('Roadsd')
            
            try:
                # 读取火灾数据
                df_fire = pd.read_csv(fire_data_file, usecols=fire_columns)
                print(f"   读取完成，共{len(df_fire)}行数据")
                print(f"   数据列: {fire_columns}")
                print(f"   处理的火点年份范围: {min(fire_years)}-{max(fire_years)}")
            except Exception as e:
                print(f"   ❌ 读取火灾数据文件出错: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # 3. 处理每个分类文件
            print("3. 处理每个分类文件...")
            stats = []  # 存储当前年份的统计结果
    
            for i, category_file in enumerate(category_files):
                # 获取分类ID（文件名）
                category_id = os.path.basename(category_file).split('.')[0]
                # 读取分类文件
                df_category = pd.read_csv(category_file)
                
                # 检查分类文件是否已经包含所有数据列
                fire_num_columns = [f'All_Fire_num_{year}' for year in fire_years]
                fire_area_columns = [f'All_Fire_area_{year}' for year in fire_years]
                hfi_columns = [f'HFI_{year}' for year in fire_years]
                popd_columns = [f'PopD_{year}' for year in fire_years]
                gdp_columns = [f'GDP_{wui_year}']
                other_columns = ['All_WUI_area', 'Roadsd']
                
                # 检查是否所有数据列都已存在
                has_all_columns = all(col in df_category.columns for col in fire_num_columns + fire_area_columns + hfi_columns + popd_columns + gdp_columns + other_columns)
                
                if has_all_columns:
                    # 直接使用已有的所有数据列
                    df_merged = df_category.copy()
                else:
                    # 合并所有数据
                    df_merged = pd.merge(df_category, df_fire, on='GRID_ID', how='left')
                    # 更新分类文件，添加所有数据列
                    df_merged.to_csv(category_file, index=False, encoding='utf-8-sig')
                
                # 4. 统计非零值数量
                fire_num_nonzero = (df_merged[fire_num_columns].notna() & (df_merged[fire_num_columns] != 0)).sum().sum()
                fire_area_nonzero = (df_merged[fire_area_columns].notna() & (df_merged[fire_area_columns] != 0)).sum().sum()
                
                # 5. 统计所有值数量(包括0值)
                fire_num_all = df_merged[fire_num_columns].notna().sum().sum()
                fire_area_all = df_merged[fire_area_columns].notna().sum().sum()
                
                # 保存统计结果
                stats.append({
                    '分类ID': category_id,
                    'fire_num非零值数量': fire_num_nonzero,
                    'fire_area非零值数量': fire_area_nonzero,
                    'fire_num所有值数量': fire_num_all,
                    'fire_area所有值数量': fire_area_all
                })
    
            # 6. 保存统计汇总文件
            print("\n4. 保存统计汇总文件...")
            df_stats = pd.DataFrame(stats)
            df_stats.to_csv(output_stat_file, index=False, encoding='utf-8-sig')
            print(f"   统计汇总文件已保存: {os.path.basename(output_stat_file)}")
            print(f"   共统计{len(df_stats)}个分类")
    
            print("\n✅ 所有处理完成！")
            print(f"📁 输出目录: {category_dir}")
            print(f"📊 分类文件数量: {len(category_files)}")
            print(f"📈 统计汇总文件: {os.path.basename(output_stat_file)}")
        
    except Exception as e:
        print(f"\n❌ 处理过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
