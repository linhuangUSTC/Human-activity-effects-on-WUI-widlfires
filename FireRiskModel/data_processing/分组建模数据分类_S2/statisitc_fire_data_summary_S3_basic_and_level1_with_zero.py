#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
程序名称: 分类建模火灾数据汇总统计(包含0值)
程序功能: 
    1. 读取多个年份(2005, 2010, 2015, 2020)的火灾数据统计汇总CSV文件
    2. 根据两个分类字典(基础分类和Level-1分类)对数据进行分类
    3. 生成保留原始前缀的分类ID(如face_A_CRP, face_B_SHR等)
    4. 统计每个分类的fire_num所有值数量(包含0值，所有年份之和)
    5. 统计每个分类的fire_area所有值数量(包含0值，所有年份之和)
    6. 将统计结果输出到Excel文件

输入文件路径: 
    I:/Processing data/渔网分类建模/火灾分类数据汇总/火灾数据统计汇总_{year}.csv

输出文件路径: 
    基础分类统计: I:/Processing data/渔网分类建模/火灾分类数据汇总/分类建模火灾数据汇总统计_Basic_with_zero.xlsx
    Level-1分类统计: I:/Processing data/渔网分类建模/火灾分类数据汇总/分类建模火灾数据汇总统计_level1_with_zero.xlsx
"""

import pandas as pd
import os

# 字典1：基础分类缩写（C列）→ 对应LC id列表（I列）
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

# 字典2：Level-1分类缩写 → 对应LC id列表
group_to_ids = {
    "RCP": [10, 11, 12],
    "ICP": [20],
    "EBF": [51, 52],
    "DBF": [61, 62],
    "ENF": [71, 72],
    "DNF": [81, 82],
    "MFT": [91, 92],
    "SHR": [120, 121, 122],
    "GRS": [130],
    "LMS": [140],
    "IWL": [181, 182, 183, 184],
    "CWL": [185, 186, 187],
    "IMP": [190],
    "SVG": [150, 152, 153],
    "BAL": [200, 201, 202],
    "WTR": [210],
    "PSI": [220],
    "NaN": [0, 250],
}

# 处理的年份
years = [2005, 2010, 2015, 2020]

# 输入文件路径
input_dir = fr"I:\Processing data\渔网分类建模\火灾分类数据汇总"

# 输出文件路径
output_dir = input_dir
output_file_basic = os.path.join(output_dir, "分类建模火灾数据汇总统计_Basic_with_zero.xlsx")
output_file_level1 = os.path.join(output_dir, "分类建模火灾数据汇总统计_level1_with_zero.xlsx")

def extract_id_info(classification_id):
    """
    从分类ID中提取前缀和LC ID
    分类ID格式：face_A_10、mix_11、B_12等，提取前缀部分和最后一个数字
    """
    try:
        parts = classification_id.split('_')
        if len(parts) < 2:
            return None, None
        
        # 提取前缀（除了最后一个部分）
        prefix = '_'.join(parts[:-1])
        
        # 提取最后一个数字作为LC ID
        lc_id = int(parts[-1])
        
        return prefix, lc_id
    except (ValueError, IndexError):
        return None, None

def get_category_by_lc_id(lc_id, mapping_dict):
    """
    根据LC ID获取对应的分类
    """
    for category, lc_ids in mapping_dict.items():
        if lc_id in lc_ids:
            return category
    return 'OTHER'

def process_all_years_aggregated(mapping_dict):
    """
    处理所有年份的数据并进行汇总统计
    """
    # 存储每个原始分类在各年份的统计数据
    # 键：原始分类ID（保留前缀），值：统计数据
    original_stats = {}
    
    for year in years:
        print(f'正在处理 {year} 年的数据...')
        
        # 构建输入文件路径
        input_file = os.path.join(input_dir, f'火灾数据统计汇总_{year}.csv')
        
        # 读取CSV文件
        df = pd.read_csv(input_file)
            
        # 遍历每行数据并更新统计
        for _, row in df.iterrows():
            original_id = row['分类ID']
            # 使用火灾数量和火灾面积的所有值数量
            fire_freq = row['fire_num所有值数量']
            fire_area = row['fire_area所有值数量']
            
            # 提取前缀和LC ID
            prefix, lc_id = extract_id_info(original_id)
            
            # 根据LC ID获取分类
            if lc_id is not None:
                category = get_category_by_lc_id(lc_id, mapping_dict)
            else:
                category = 'OTHER'
            
            # 生成新的分类ID：前缀 + '_' + 分类缩写
            if prefix is not None:
                new_classification_id = f'{prefix}_{category}'
            else:
                new_classification_id = f'OTHER_{category}'
            
            # 初始化统计数据
            if new_classification_id not in original_stats:
                original_stats[new_classification_id] = {
                    'fire_num_total': 0,
                    'fire_area_total': 0
                }
            
            # 更新统计数据
            original_stats[new_classification_id]['fire_num_total'] += fire_freq
            original_stats[new_classification_id]['fire_area_total'] += fire_area
    
    # 转换为DataFrame
    stats_list = []
    for classification_id, stats in original_stats.items():
        stats_list.append({
            '分类ID': classification_id,
            'fire_num所有值数量': stats['fire_num_total'],
            'fire_area所有值数量': stats['fire_area_total']
        })
    
    return pd.DataFrame(stats_list)

def main():
    """
    主函数
    """
    print("开始处理分类建模火灾数据汇总统计(包含0值)...")
    
    # 处理基础分类（字典1）
    print('\n正在处理基础分类...')
    basic_stats = process_all_years_aggregated(c_to_lc)
    
    # 保存基础分类统计结果到Excel
    with pd.ExcelWriter(output_file_basic, engine='openpyxl') as writer:
        basic_stats.to_excel(writer, index=False, sheet_name='基础分类统计')
    print(f'基础分类统计结果已保存到: {output_file_basic}')
    
    # 处理Level-1分类（字典2）
    print('\n正在处理Level-1分类...')
    level1_stats = process_all_years_aggregated(group_to_ids)
    
    # 保存Level-1分类统计结果到Excel
    with pd.ExcelWriter(output_file_level1, engine='openpyxl') as writer:
        level1_stats.to_excel(writer, index=False, sheet_name='Level-1分类统计')
    print(f'Level-1分类统计结果已保存到: {output_file_level1}')
    
    print('\n所有数据处理完成！')

if __name__ == '__main__':
    main()
