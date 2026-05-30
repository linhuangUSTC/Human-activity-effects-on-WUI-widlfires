# -*- coding: utf-8 -*-
"""
程序名称：按照HFI分组统计Fire密度柱状图

程序功能：
    1. 从两个指定路径读取CSV文件
    2. 对每个CSV文件单独处理
    3. 从Fire_num路径读取All_fire_num、WUI_area和HFI三列数据
    4. 从Fire_area路径读取All_fire_area、WUI_area和HFI三列数据
    5. 计算Fire density：Fire_num/WUI_area 和 Fire_area/WUI_area
    6. 根据HFI的值将每个文件的数据分成n等份（每组样本数量相同）
    7. 为每个文件绘制这10组数据的柱状图,着色

输入路径：
    - Fire_num数据: I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic
    - Fire_area数据: I:\Processing data\渔网分类建模\Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic

输出：
    - 保存图片到 B:\WUI\Picture 目录
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def split_into_quantiles(dataframe, column_name, n_quantiles=10):
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
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(groups)), data_means, color='skyblue')
    
    # Set x-axis labels
    plt.xticks(range(len(groups)), hfi_ranges, rotation=0)
    
    # Add short title inside the plot frame (top center)
    ax = plt.gca()
    ax.text(0.5, 0.98, short_title, transform=ax.transAxes,
            ha='center', va='top', fontsize=14)

    # Set axis labels
    plt.xlabel('Human Footprint Range', fontsize=12)
    plt.ylabel(y_axis_label, fontsize=12)
    
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


def process_folder(folder_path, data_column, output_suffix, chart_title, y_axis_label):
    print(f"\n=== 处理路径: {folder_path} ===")
    # Get all CSV files in folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # Process each CSV file
    for filename in csv_files:
        file_path = os.path.join(folder_path, filename)
        print(f"\n=== 处理文件: {filename} ===")
        
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Process specified data column if it exists
        if data_column in df.columns and 'All_WUI_area' in df.columns:
            # Extract required columns
            data = df[[data_column, 'All_WUI_area', 'HFI']].copy()
            # Remove rows with NaN in HFI or All_WUI_area
            data = data.dropna(subset=['HFI', 'All_WUI_area'])
            # Remove rows with All_WUI_area = 0 to avoid division by zero
            data = data[data['All_WUI_area'] > 0]
            
            # Calculate Fire density
            data['Fire_density'] = data[data_column] / data['All_WUI_area']
            
            print(f"成功读取{chart_title.split()[0]}数据，共 {len(data)} 条记录")
            print(f"Fire density 范围: {data['Fire_density'].min():.2f} - {data['Fire_density'].max():.2f}")
            
            # Group data by HFI
            print(f"\n=== 开始分组{chart_title.split()[0]}数据 ===")
            groups = split_into_quantiles(data, 'HFI', n_quantiles=10)
            
            print(f"成功分成 {len(groups)} 组")
            
            # Print group information
            for i, (group_name, group_data, hfi_range) in enumerate(groups):
                print(f"{group_name}: 样本数={len(group_data)}, HFI范围={hfi_range[0]:.2f}-{hfi_range[1]:.2f}, Fire density均值={group_data['Fire_density'].mean():.2f}")
            
            # Plot bar chart
            print(f"\n=== 绘制{chart_title.split()[0]}柱状图 ===")
            plot_data_by_hfi_groups(
                groups=groups,
                csv_filename=filename,
                data_column='Fire_density',  # 使用计算出的Fire_density列
                output_suffix=output_suffix,
                chart_title=chart_title,
                y_axis_label=y_axis_label
            )
        else:
            print(f"错误：文件 {filename} 中缺少 {data_column} 或 All_WUI_area 列")


def main():
    """
    Main function
    """
    # Define input paths
    fire_num_folder = r"I:\Processing data\渔网分类建模\Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic"
    fire_area_folder = r"I:\Processing data\渔网分类建模\Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic"
    
    print("=== 开始处理数据 ===")
    
    # Process Fire num data (calculate Fire density)
    print("\n====================================")
    print("=== 处理Fire num数据 (计算密度) ===")
    print("====================================")
    process_folder(
        folder_path=fire_num_folder,
        data_column='All_fire_num',
        output_suffix='Fire_density',
        chart_title='Fire Density (Num) Mean by Human Footprint Groups',
        y_axis_label='Fire frequency'
    )
    
    # Process Fire area data (calculate Fire density)
    print("\n====================================")
    print("=== 处理Fire area数据 (计算密度) ===")
    print("====================================")
    process_folder(
        folder_path=fire_area_folder,
        data_column='All_fire_area',
        output_suffix='Fire_density',
        chart_title='Fire Density (Area) Mean by Human Footprint Groups',
        y_axis_label='Normalized burned area'
    )
    
    print("\n=== 所有文件处理完成 ===")


if __name__ == "__main__":
    main()


