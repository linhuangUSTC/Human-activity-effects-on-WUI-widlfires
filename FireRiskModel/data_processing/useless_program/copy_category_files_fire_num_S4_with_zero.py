# -*- coding: utf-8 -*-
"""
程序名称：分类数据文件重组工具 (copy_category_files_S4_with_WUI_offset.py)

程序功能：
    根据Excel文件中指定的分类ID列表,遍历多个年份文件夹.
    读取每个分类ID对应的CSV文件,将数据重组为统一的8列格式,
    最终保存到目标目录的单一文件中。

数据重组规则：
    1. 输出文件包含8列:
       - GRID_ID  - All_fire_num - Roadsd
       - PopD - GDP  - HFI - All_WUI_area  - Year

输入路径：
    - Excel文件：I:\Processing data\渔网分类建模\火灾分类数据汇总\分类建模火灾数据汇总统计_with_zero.xlsx
    - 源目录模式：I:\Processing data\渔网分类建模\{}
      ({}会被替换为年份，如2005、2010、2015、2020)
    - 年份范围：2005、2010、2015、2020
      每个年份对应的数据年份范围：- 2005：2003-2007- 2010：2008-2012- 2015：2013-2017- 2020：2018-2022
输出路径：
    - 目标目录：I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据

"""

import os
import pandas as pd


def process_category_data(excel_file, sheet_name, output_suffix):
    """
    处理特定sheet的分类数据
    
    参数：
    excel_file: Excel文件路径
    sheet_name: 要读取的sheet名称
    output_suffix: 输出文件名的后缀
    """
    # 配置参数
    category_id_column = "分类ID"
    years = [2005, 2010, 2015, 2020]
    source_dir_pattern = r"I:\Processing data\渔网分类建模\{}"
    target_dir = r"I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据"
    
    # 年份映射规则：4年数据对应的目标年份范围
    year_mapping = {
        2005: list(range(2003, 2008)),    # 2003-2007
        2010: list(range(2008, 2013)),    # 2008-2012
        2015: list(range(2013, 2018)),    # 2013-2017
        2020: list(range(2018, 2023))     # 2018-2022
    }
    
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    try:
        # 读取Excel文件中的分类ID
        print("正在读取Excel文件中的分类ID...")
        df_excel = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        # 检查分类ID列是否存在
        if category_id_column not in df_excel.columns:
            print(f"错误：Excel文件中不存在列名为'{category_id_column}'的列")
            return
        
        # 获取所有分类ID
        category_ids = df_excel[category_id_column].tolist()
        print(f"共读取到 {len(category_ids)} 个分类ID")
        
        # 遍历每个分类ID
        for category_id in category_ids:
            # 收集所有年份的数据
            all_data = []
            
            # 遍历每个年份
            for year in years:
                # 构建源文件路径
                source_dir = source_dir_pattern.format(year)
                csv_file = os.path.join(source_dir, f"{category_id}.csv")
                
                # 检查文件是否存在
                if os.path.exists(csv_file):
                    # 读取CSV文件
                    df_csv = pd.read_csv(csv_file)
                    all_data.append((year, df_csv))
            
            # 如果找到数据，进行重组
            if all_data:
                # 准备重组后的数据列表
                restructured_data = []
                
                # 处理每个年份的数据
                for base_year, df_year in all_data:
                    # 确定目标年份范围
                    if len(all_data) == 1:
                        # 只有一年的数据 - 用这一年的数据代替所有20年
                        target_years = list(range(2003, 2023))  # 2003-2022年
                    else:
                        # 多个年份的数据 - 按规则映射
                        if base_year not in year_mapping:
                            continue
                        target_years = year_mapping[base_year]
                    
                    # 处理每行数据
                    for _, row in df_year.iterrows():
                        # 获取基础数据（这些数据用基准年份的值）
                        grid_id = row.get('GRID_ID', '')
                        roads = row.get('Roadsd', 0)
                        all_wui_area = row.get('All_WUI_area', 0)
                        
                        # 查找该年的GDP（这些数据用基准年份的值）
                        gdp_col = f"GDP_{base_year}"
                        gdp = row.get(gdp_col, 0)
                        
                        # 为每个目标年份创建一行数据
                        for target_year in target_years:
                            # 获取对应年份的PopD和HFI（优先使用实际年份的值，否则使用基准年份的值）
                            popd = row.get(f"PopD_{target_year}", row.get(f"PopD_{base_year}", 0))
                            hfi = row.get(f"HFI_{target_year}", row.get(f"HFI_{base_year}", 0))
                            
                            # 直接从CSV文件中读取All_fire_num的值（优先使用实际年份的值，否则使用基准年份的值）
                            all_fire_num = row.get(f"All_Fire_num_{target_year}", row.get(f"All_Fire_num_{base_year}", 0))
                            
                            # 添加重组后的数据行
                            restructured_data.append({
                                'GRID_ID': grid_id,
                                'All_fire_num': all_fire_num,
                                'Roadsd': roads,
                                'PopD': popd,
                                'GDP': gdp,
                                'HFI': hfi,
                                'All_WUI_area': all_wui_area,
                                'Year': target_year
                            })
            
            # 如果有重组后的数据，保存到CSV文件
            if restructured_data:
                # 转换为DataFrame
                df_restructured = pd.DataFrame(restructured_data)
                
                print(f"处理分类ID {category_id}: 数据行数 = {len(df_restructured)}")
                
                # 构建目标文件路径，添加指定后缀
                target_file = os.path.join(target_dir, f"{category_id}_{output_suffix}.csv")
                
                # 保存到目标文件，使用写入模式确保覆盖现有文件
                df_restructured.to_csv(target_file, index=False, encoding="utf-8-sig", mode='w')
        
        print("\n处理完成！")
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    # 配置参数
    excel_file = r"I:\Processing data\渔网分类建模\火灾分类数据汇总\分类建模火灾数据汇总统计_with_zero.xlsx"
    
    print("=== 开始处理火灾数量数据 ===")
    print("Sheet: Sheet1")
    print("输出后缀: fire_num_with_WUI_offset")
    print("过滤条件: 保留所有数据，包括0值")
    # 处理火灾数量数据
    process_category_data(
        excel_file=excel_file,
        sheet_name="Sheet1",
        output_suffix="fire_num_with_WUI_offset"
    )
    
    print("\n=== 所有数据处理完成 ===")


if __name__ == "__main__":
    main()
