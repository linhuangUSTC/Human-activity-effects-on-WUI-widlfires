# -*- coding: utf-8 -*-
"""
程序名称：分类数据文件重组工具(基础分类版)

程序功能：
    根据Excel文件中指定的分类ID列表,遍历多个年份文件夹.
    读取每个分类ID对应的CSV文件,将数据重组为统一的8列格式,
    最终保存到目标目录的单一文件中。
    根据字典将分类ID中的缩写转换为对应的数字ID
    直接从CSV文件中读取All_fire_num的值

数据重组规则：
    1. 输出文件包含8列:
       - GRID_ID  - All_fire_num - Roadsd
       - PopD - GDP  - HFI - All_WUI_area  - Year

输入路径：
    - Excel文件:I:\Processing data\渔网分类建模\分类建模数据_Basic\分类建模火灾数据汇总统计_Basic_with_zero.xlsx
    - 源目录模式:I:\Processing data\渔网分类建模\{}
      ({}会被替换为年份,如2005、2010、2015、2020)

输出路径：
    - 目标目录:I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic

"""

import os
import pandas as pd

# 基础分类缩写到数字ID的映射字典
# 来自 statisitc_fire_data_summary_basic_and_level1_S3.py
c_to_lc = {
    "CRP": [10, 11, 12, 20],
    "FST": [51, 52, 61, 62, 71, 72, 81, 82, 91, 92],
    "SHR": [120, 121, 122],
    "GRS": [130],
    "TUD": [140],
    "WET": [181, 182, 183, 184, 185, 186, 187],
    "IMP": [190],
    "BAL": [150, 152, 153, 200, 201, 202],
    "WTR": [210],
    "PSI": [220],
    "NaN": [0, 250]
}


def get_numeric_ids_from_category(category_id):
    try:
        # 分割分类ID，获取缩写部分
        parts = category_id.split('_')
        if len(parts) < 3:
            # 如果格式不符合预期，返回原始ID作为列表
            return [category_id]
        
        # 获取缩写部分（最后一个部分）
        abbreviation = parts[-1]
        
        # 检查缩写是否在映射字典中
        if abbreviation in c_to_lc:
            # 获取对应的数字ID列表
            numeric_ids = c_to_lc[abbreviation]
            
            # 构建完整的分类ID列表
            prefix = '_'.join(parts[:-1])  # 提取前缀（如face_A、mix、B）
            full_ids = [f"{prefix}_{num_id}" for num_id in numeric_ids]
            return full_ids
        else:
            # 如果缩写不在字典中，返回原始ID
            return [category_id]
    except Exception as e:
        print(f"解析分类ID {category_id} 时出错: {e}")
        return [category_id]


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
    target_dir = r"I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic"  # 修改为新的输出目录
    
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
        for i, category_id in enumerate(category_ids):
            # 获取该分类ID对应的所有数字ID
            numeric_ids = get_numeric_ids_from_category(category_id)
            
            # 收集所有年份的数据（来自所有数字ID）
            all_data = []
            found_ids = []
            found_years = []
            
            # 遍历每个数字ID
            for numeric_id in numeric_ids:
                # 遍历每个年份
                for year in years:
                    # 构建源文件路径
                    source_dir = source_dir_pattern.format(year)
                    csv_file = os.path.join(source_dir, f"{numeric_id}.csv")
                    
                    # 检查文件是否存在
                    if os.path.exists(csv_file):
                        # 读取CSV文件
                        df_csv = pd.read_csv(csv_file)
                        all_data.append((year, df_csv, numeric_id))
                        found_ids.append(numeric_id)
                        found_years.append(year)
                    else:
                        pass
            
            # 如果找到数据，进行重组
            if all_data:
                # 准备重组后的数据列表
                restructured_data = []
                
                # 情况1：只有一个数字ID的数据
                if len(set(found_ids)) == 1:
                    # 处理单数字ID情况
                    numeric_id = list(set(found_ids))[0]
                    
                    # 情况1.1：只有一年的数据 - 用这一年的数据代替所有20年
                    if len(set(found_years)) == 1:
                        single_year = list(set(found_years))[0]
                        # 找到对应年份的数据
                        df_single = next(df for year, df, nid in all_data if year == single_year and nid == numeric_id)
                        target_years = list(range(2003, 2023))  # 2003-2022年
                        
                        # 处理每行数据
                        for _, row in df_single.iterrows():
                            # 获取基础数据（这些数据用基准年份的值）
                            grid_id = row['GRID_ID']
                            roads = row['Roadsd']
                            all_wui_area = row['All_WUI_area']
                            
                            # 查找该年的GDP（这些数据用基准年份的值）
                            gdp_col = f"GDP_{single_year}"
                            gdp = row[gdp_col] if gdp_col in row else 0
                            
                            # 为每个目标年份创建一行数据
                            for target_year in target_years:
                                # 检查是否有该目标年份的具体数据，如果没有则使用基准年份的数据
                                popd_col = f"PopD_{target_year}"
                                if popd_col in row:
                                    popd = row[popd_col]
                                else:
                                    popd_col_base = f"PopD_{single_year}"
                                    popd = row[popd_col_base] if popd_col_base in row else 0
                                
                                hfi_col = f"HFI_{target_year}"
                                if hfi_col in row:
                                    hfi = row[hfi_col]
                                else:
                                    hfi_col_base = f"HFI_{single_year}"
                                    hfi = row[hfi_col_base] if hfi_col_base in row else 0
                                
                                # 直接从CSV文件中读取All_fire_num的值
                                fire_num_col = f"All_Fire_num_{target_year}"
                                if fire_num_col in row:
                                    all_fire_num = row[fire_num_col]
                                else:
                                    fire_num_col_base = f"All_Fire_num_{single_year}"
                                    all_fire_num = row[fire_num_col_base] if fire_num_col_base in row else 0
                                
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
                    
                    # 情况1.2：有多个年份的数据 - 按规则映射
                    else:
                        # 处理每个年份的数据块
                        for base_year, df_year, nid in all_data:
                            if nid != numeric_id:
                                continue
                                
                            # 获取该年份对应的目标年份范围
                            if base_year in year_mapping:
                                target_years = year_mapping[base_year]
                            else:
                                continue  # 跳过未知的年份映射
                            
                            # 处理每行数据
                            for _, row in df_year.iterrows():
                                # 获取基础数据（这些数据用基准年份的值）
                                grid_id = row['GRID_ID']
                                roads = row['Roadsd']  # 用基准年份的值
                                all_wui_area = row['All_WUI_area']  # 用基准年份的值
                                
                                # 查找该年的GDP（这些数据用基准年份的值）
                                gdp_col = f"GDP_{base_year}"
                                gdp = row[gdp_col] if gdp_col in row else 0  # 用基准年份的值
                                
                                # 为每个目标年份创建一行数据
                                for target_year in target_years:
                                    # 获取对应年份的PopD和HFI（这些数据用实际年份的值）
                                    popd_col = f"PopD_{target_year}"
                                    popd = row[popd_col] if popd_col in row else 0
                                    
                                    hfi_col = f"HFI_{target_year}"
                                    hfi = row[hfi_col] if hfi_col in row else 0
                                    
                                    # 直接从CSV文件中读取All_fire_num的值
                                    fire_num_col = f"All_Fire_num_{target_year}"
                                    all_fire_num = row[fire_num_col] if fire_num_col in row else 0
                                    
                                    # 添加重组后的数据行
                                    restructured_data.append({
                                        'GRID_ID': grid_id,  # 用基准年份的值
                                        'All_fire_num': all_fire_num,  # 用实际年份的值
                                        'Roadsd': roads,  # 用基准年份的值
                                        'PopD': popd,  # 用实际年份的值
                                        'GDP': gdp,  # 用基准年份的值
                                        'HFI': hfi,  # 用实际年份的值
                                        'All_WUI_area': all_wui_area,  # 用基准年份的值
                                        'Year': target_year
                                    })
                # 情况2：有多个数字ID的数据
                else:
                    # 处理每个年份的数据块
                    for base_year, df_year, nid in all_data:
                        # 获取该年份对应的目标年份范围
                        if base_year in year_mapping:
                            target_years = year_mapping[base_year]
                        else:
                            continue  # 跳过未知的年份映射
                        
                        # 处理每行数据
                        for _, row in df_year.iterrows():
                            # 获取基础数据（这些数据用基准年份的值）
                            grid_id = row['GRID_ID']
                            roads = row['Roadsd']  # 用基准年份的值
                            all_wui_area = row['All_WUI_area']  # 用基准年份的值
                            
                            # 查找该年的GDP（这些数据用基准年份的值）
                            gdp_col = f"GDP_{base_year}"
                            gdp = row[gdp_col] if gdp_col in row else 0  # 用基准年份的值
                            
                            # 为每个目标年份创建一行数据
                            for target_year in target_years:
                                # 获取对应年份的PopD和HFI（这些数据用实际年份的值）
                                popd_col = f"PopD_{target_year}"
                                popd = row[popd_col] if popd_col in row else 0
                                
                                hfi_col = f"HFI_{target_year}"
                                hfi = row[hfi_col] if hfi_col in row else 0
                                
                                # 直接从CSV文件中读取All_fire_num的值
                                fire_num_col = f"All_Fire_num_{target_year}"
                                all_fire_num = row[fire_num_col] if fire_num_col in row else 0
                                
                                # 添加重组后的数据行
                                restructured_data.append({
                                    'GRID_ID': grid_id,  # 用基准年份的值
                                    'All_fire_num': all_fire_num,  # 用实际年份的值
                                    'Roadsd': roads,  # 用基准年份的值
                                    'PopD': popd,  # 用实际年份的值
                                    'GDP': gdp,  # 用基准年份的值
                                    'HFI': hfi,  # 用实际年份的值
                                    'All_WUI_area': all_wui_area,  # 用基准年份的值
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
    # 配置参数 - 使用基础分类版Excel文件
    excel_file = r"I:\Processing data\渔网分类建模\火灾分类数据汇总\分类建模火灾数据汇总统计_Basic_with_zero.xlsx"
    
    print("=== 开始处理火灾频率数据 ===")
    print("Sheet: Sheet1")
    print("输出后缀: fire_num_with_WUI_offset")
    print("过滤条件: 保留所有数据，包括0值")
    # 处理火灾频率数据
    process_category_data(
        excel_file=excel_file,
        sheet_name="Sheet1",  # 使用正确的sheet名称
        output_suffix="fire_num_with_WUI_offset"
    )
    
    print("\n=== 所有数据处理完成 ===")


if __name__ == "__main__":
    main()
