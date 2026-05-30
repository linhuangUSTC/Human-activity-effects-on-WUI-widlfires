# -*- coding: utf-8 -*-
"""
程序名称：按照HFI二十分位分组统计Fire frequency以及Normalized burned area柱状图

程序功能：
    1. 从两个指定路径读取CSV文件
    2. 对每个CSV文件单独处理
    3. 从Fire_num路径直接读取Fire frequency及HFI
    4. 从Fire_area路径直接读取Normalized burned area及HFI
    5. 根据HFI的值将每个文件的数据分成20等份（每组样本数量相同）
    6. 为每个文件绘制20组数据的柱状图,灰色

输入路径：
    - Fire frequency数据: I:\\Processing data\\渔网分类建模\\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic
    - Normalized burned area数据: I:\\Processing data\\渔网分类建模\\Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic

输出：
    - 保存图片到 B:\\WUI\\Picture 目录
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def normalize_column_name(column_name):
    return ''.join(char.lower() for char in str(column_name) if char.isalnum())


def find_matching_column(columns, candidate_names):
    normalized_to_original = {}
    for column in columns:
        normalized_to_original.setdefault(normalize_column_name(column), column)

    for candidate_name in candidate_names:
        matched_column = normalized_to_original.get(normalize_column_name(candidate_name))
        if matched_column:
            return matched_column

    return None

def split_into_quantiles(dataframe, column_name, n_quantiles=20):
    # 移除NaN值
    dataframe = dataframe.dropna(subset=[column_name])
    
    # Calculate quantiles
    quantiles = np.linspace(0, 1, n_quantiles + 1)
    
    # 根据分位数创建分组
    groups = []
    for i in range(n_quantiles):
        # 计算当前分位数的边界
        lower_bound = dataframe[column_name].quantile(quantiles[i])
        upper_bound = dataframe[column_name].quantile(quantiles[i + 1])
        
        # 获取当前分组的数据
        if i == 0:
            # 第一组包含等于下界的值
            group_data = dataframe[(dataframe[column_name] >= lower_bound) & (dataframe[column_name] <= upper_bound)]
        else:
            # 其他组不包含等于下界的值，避免重复
            group_data = dataframe[(dataframe[column_name] > lower_bound) & (dataframe[column_name] <= upper_bound)]
        
        # 计算HFI范围
        hfi_min = group_data[column_name].min()
        hfi_max = group_data[column_name].max()
        hfi_range = (hfi_min, hfi_max)
        
        # 生成组名
        group_name = f"组{i+1}"
        
        groups.append((group_name, group_data, hfi_range))
    
    return groups


def plot_data_by_hfi_groups(groups, csv_filename, data_column, output_suffix, chart_title, y_axis_label):

    if not groups:
        print("错误：没有分组数据")
        return
    
    # Prepare data
    group_names = []
    data_means = []
    hfi_ranges = []
    
    for group_name, group_data, hfi_range in groups:
        group_names.append(group_name)
        # Calculate mean value
        data_mean = group_data[data_column].mean()
        data_means.append(data_mean)
        # Format HFI range
        hfi_range_str = f"{hfi_range[0]:.2f}-{hfi_range[1]:.2f}"
        hfi_ranges.append(hfi_range_str)
    
    # Build short title from first 3 underscore-separated parts, e.g. face_A_CRP
    filename_no_ext = os.path.splitext(csv_filename)[0]
    title_parts = filename_no_ext.split('_')
    short_title = '_'.join(title_parts[:3]) if len(title_parts) >= 3 else filename_no_ext

    # Create bar chart
    plt.figure(figsize=(5, 6))
    bars = plt.bar(
        range(len(groups)),
        data_means,
        width=1.0,
        facecolor='none',
        edgecolor='black',
        linewidth=1.0
    )
    
    # Set x-axis labels
    side_padding = 0.25
    plt.xlim(-0.5 - side_padding, len(groups) - 0.5 + side_padding)
    plt.xticks(range(len(groups)), hfi_ranges, rotation=90)

    # Set title and axis labels
    plt.title(short_title, fontsize=14)
    plt.xlabel('Human Footprint Range', fontsize=12)
    plt.ylabel(y_axis_label, fontsize=12)

    # Remove top and right frame lines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save image to specified path
    output_dir = r"B:\WUI\Picture"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate save filename (remove .csv extension)
    save_filename = f"{os.path.splitext(csv_filename)[0]}_Human Footprint_groups_{output_suffix}_chart.png"
    output_path = os.path.join(output_dir, save_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存到: {output_path}")
    
    # Close plot to release memory
    plt.close()


def process_folder(folder_path, data_column_candidates, output_suffix, chart_title, y_axis_label):
    print(f"\n=== 处理路径: {folder_path} ===")
    # Get all CSV files in folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # Process each CSV file
    for filename in csv_files:
        file_path = os.path.join(folder_path, filename)
        print(f"\n=== 处理文件: {filename} ===")
        
        # Read CSV file
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        hfi_column = find_matching_column(df.columns, ['HFI'])
        data_column = find_matching_column(df.columns, data_column_candidates)

        if hfi_column and data_column:
            data = df[[data_column, hfi_column]].copy()
            data = data.rename(columns={data_column: 'Metric', hfi_column: 'HFI'})
            data['Metric'] = pd.to_numeric(data['Metric'], errors='coerce')
            data['HFI'] = pd.to_numeric(data['HFI'], errors='coerce')
            data = data.dropna(subset=['Metric', 'HFI'])

            print(f"成功读取{y_axis_label}数据列: {data_column}，共 {len(data)} 条记录")
            print(f"{y_axis_label} 范围: {data['Metric'].min():.6f} - {data['Metric'].max():.6f}")
            
            # Group data by HFI
            print(f"\n=== 开始分组{y_axis_label}数据 ===")
            groups = split_into_quantiles(data, 'HFI', n_quantiles=20)
            
            print(f"成功分成 {len(groups)} 组")
            
            # Print group information
            for i, (group_name, group_data, hfi_range) in enumerate(groups):
                print(f"{group_name}: 样本数={len(group_data)}, HFI范围={hfi_range[0]:.2f}-{hfi_range[1]:.2f}, {y_axis_label}均值={group_data['Metric'].mean():.6f}")
            
            # Plot bar chart
            print(f"\n=== 绘制{y_axis_label}柱状图 ===")
            plot_data_by_hfi_groups(
                groups=groups,
                csv_filename=filename,
                data_column='Metric',
                output_suffix=output_suffix,
                chart_title=chart_title,
                y_axis_label=y_axis_label
            )
        else:
            print(f"错误：文件 {filename} 中缺少指标列或HFI列")
            print(f"可用列名: {', '.join(df.columns)}")


def main():
    """
    Main function
    """
    # Define input paths
    fire_num_folder = r"I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic"
    fire_area_folder = r"I:\Processing data\渔网分类建模\Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic"
    
    print("=== 开始处理数据 ===")
    
    # Process Fire frequency data
    print("\n====================================")
    print("=== 处理Fire frequency数据 ===")
    print("====================================")
    process_folder(
        folder_path=fire_num_folder,
        data_column_candidates=[
            'Fire_frequency',
            'Fire frequency',
            'All_fire_frequency',
            'Fire_frequency_per_WUI'
        ],
        output_suffix='Fire_frequency',
        chart_title='Fire Frequency Mean by Human Footprint Groups',
        y_axis_label='Fire frequency'
    )
    
    # Process Normalized burned area data
    print("\n====================================")
    print("=== 处理Normalized burned area数据 ===")
    print("====================================")
    process_folder(
        folder_path=fire_area_folder,
        data_column_candidates=[
            'Normalized_fire_area',
            'Normalized burned area',
            'All_fire_norm_area',
            'Fire_area_density'
        ],
        output_suffix='Normalized_burn_area',
        chart_title='Normalized Burn Area Mean by Human Footprint Groups',
        y_axis_label='Normalized burned area'
    )
    
    print("\n=== 所有文件处理完成 ===")


if __name__ == "__main__":
    main()


