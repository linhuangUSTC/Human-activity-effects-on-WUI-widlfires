#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
程序名称: 大洲野火频率与归一化面积金字塔图绘制程序
功能简介: 本程序用于绘制各大洲不同类型WUI的起火频率和归一化过火面积的金字塔图
         1. 支持2003-2022年的火灾数据统计
         2. 中间显示年份(2003-2022)
         3. 左侧为Interface，右侧为Intermix
         4. 每个年份为6个大洲的数据堆叠
         5. 输出1张1*2组图：Fire frequency和Normalized burned area
         6. 输入路径: I:\\Processing data\\Grid_Clip_{wui_year}_WUI.csv
         7. 输出路径: B:\\WUI\\Picture
         8. 注意:火灾数据与WUI数据的对应关系:
            - 2003-2007年 → 2005年WUI数据
            - 2008-2012年 → 2010年WUI数据
            - 2013-2017年 → 2015年WUI数据
            - 2018-2022年 → 2020年WUI数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.gridspec import GridSpec
import os

# 设置英文显示，使用Arial字体
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['xtick.labelsize'] = 19
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20

# 大洲代码与名称的映射
continent_mapping = {
    1: 'Africa',
    2: 'Asia',
    3: 'Oceania',
    4: 'North America',
    6: 'South America',
    8: 'Europe'
}

# 定义大洲的显示顺序和颜色
continent_order = [1, 2, 3, 4, 6, 8]  # Africa, Asia, Oceania, North America, South America, Europe
continent_colors = {
    1: '#E97C73',  # Africa
    2: '#2CB7B0',  # Asia
    3: '#C9DCAE',  # Oceania
    4: '#C9D7EE',  # North America
    6: '#D5C3E8',  # South America
    8: '#F4C9A7'   # Europe
}

# WUI数据年份列表
wui_years = [2005, 2010, 2015, 2020]

# 输出路径 - 使用英文路径
output_dir = r"B:\WUI\Picture"
os.makedirs(output_dir, exist_ok=True)
# 火灾数据与WUI数据的对应关系
fire_year_to_wui_year = {}
for wui_year in wui_years:
    # 每个WUI年份对应5年的火灾数据（前2年到后2年）
    for i in range(-2, 3):
        fire_year = wui_year + i
        fire_year_to_wui_year[fire_year] = wui_year

# 数据类型和WUI类型配置
stats_config = [
    {
        'name': 'Fire Frequency',
        'wui_types': [
            {
                'name': 'Interface WUI',
                'source_field': lambda year: f'Interface_Fire_num_{year}',
                'metric_field': lambda year: f'Interface_Fire_freq_{year}',
                'area_field': 'Interface_WUI_area',
                'filename_suffix': 'interface_wui'
            },
            {
                'name': 'Intermix WUI',
                'source_field': lambda year: f'Intermix_Fire_num_{year}',
                'metric_field': lambda year: f'Intermix_Fire_freq_{year}',
                'area_field': 'Intermix_WUI_area',
                'filename_suffix': 'intermix_wui'
            }
        ],
        'ylabel': 'Fire frequency',
        'data_type_suffix': 'fire_frequency'
    },
    {
        'name': 'Normalized burned area',
        'wui_types': [
            {
                'name': 'Interface WUI',
                'source_field': lambda year: f'Interface_Fire_area_{year}',
                'metric_field': lambda year: f'Interface_Fire_norm_area_{year}',
                'area_field': 'Interface_WUI_area',
                'filename_suffix': 'interface_wui'
            },
            {
                'name': 'Intermix WUI',
                'source_field': lambda year: f'Intermix_Fire_area_{year}',
                'metric_field': lambda year: f'Intermix_Fire_norm_area_{year}',
                'area_field': 'Intermix_WUI_area',
                'filename_suffix': 'intermix_wui'
            }
        ],
        'ylabel': 'Normalized burned area',
        'data_type_suffix': 'normalized_burned_area'
    }
]


def extract_annual_data(df, base_year):
    """
    从Grid_Clip数据中提取年际数据
    """
    # 计算该Grid_Clip包含的5个年份
    data_years = list(range(base_year - 2, base_year + 3))
    
    all_annual_data = []
    
    for fire_year in data_years:
        # 获取对应的WUI年份
        wui_year = fire_year_to_wui_year.get(fire_year)
        if wui_year is None:
            print(f"No WUI year found for fire year {fire_year}")
            continue
        
        # 获取所有需要的字段
        required_fields = ['Continent']
        for stat in stats_config:
            for wui_type in stat['wui_types']:
                required_fields.append(wui_type['source_field'](fire_year))
                required_fields.append(wui_type['area_field'])
        required_fields = list(dict.fromkeys(required_fields))
        
        # 检查所有字段是否存在
        missing_fields = [field for field in required_fields if field not in df.columns]
        if missing_fields:
            print(f"Fields not found for fire year {fire_year}: {missing_fields}")
            continue
        
        # 创建临时数据框并按大洲汇总原始火灾次数/面积与WUI面积
        temp_df = df[required_fields].copy()
        continent_sums = temp_df.groupby('Continent', as_index=False).sum(numeric_only=True)
        continent_stats = continent_sums[['Continent']].copy()

        # 基于真实字段计算指标:
        # Fire frequency = Fire_num / WUI_area
        # Normalized burned area = Fire_area / WUI_area
        for stat in stats_config:
            for wui_type in stat['wui_types']:
                source_col = wui_type['source_field'](fire_year)
                area_col = wui_type['area_field']
                metric_col = wui_type['metric_field'](fire_year)
                with np.errstate(divide='ignore', invalid='ignore'):
                    continent_stats[metric_col] = np.where(
                        continent_sums[area_col] > 0,
                        continent_sums[source_col] / continent_sums[area_col],
                        np.nan,
                    )
        
        # 添加年份信息
        continent_stats['Year'] = fire_year
        continent_stats['WUI_Year'] = wui_year
        
        all_annual_data.append(continent_stats)
    
    if not all_annual_data:
        return None
    
    return pd.concat(all_annual_data, ignore_index=True)


def plot_stacked_barchart_by_year(all_data, stat_config):
    """
    按年份绘制堆叠柱状图
    """
    for wui_type in stat_config['wui_types']:
        # 准备数据
        # 创建包含所有年份的完整数据框
        all_years = list(range(2003, 2023))
        full_year_data = pd.DataFrame({'Year': all_years})
        
        # 收集每个年份、每个大洲的数据
        year_continent_data = []
        
        for year in all_years:
            year_data = all_data[all_data['Year'] == year].copy()
            
            # 创建包含所有大洲的完整数据
            full_continent_data = pd.DataFrame({'Continent': continent_order})
            
            # 合并数据
            merged_data = full_continent_data.merge(
                year_data[['Continent', wui_type['metric_field'](year)]],
                on='Continent',
                how='left'
            )
            
            # 替换NaN值为0
            merged_data.fillna(0, inplace=True)
            
            # 重命名列
            merged_data.rename(columns={wui_type['metric_field'](year): 'Value'}, inplace=True)
            
            # 添加年份信息
            merged_data['Year'] = year
            
            year_continent_data.append(merged_data)
        
        # 合并所有数据
        stacked_data = pd.concat(year_continent_data, ignore_index=True)
        
        # 准备绘图数据
        years = all_years
        continent_values = {}
        
        for continent in continent_order:
            continent_values[continent] = stacked_data[
                stacked_data['Continent'] == continent
            ].sort_values('Year')['Value'].tolist()
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(15 * 1.3, 8 * 0.8))
        
        # 绘制堆叠柱状图
        bottom = np.zeros(len(years))
        
        for continent in continent_order:
            values = continent_values[continent]
            ax.bar(years, values, bottom=bottom, 
                   color=continent_colors[continent], 
                   label=continent_mapping[continent],
                   width=0.76)
            bottom += values
        
        # 设置图表属性
        ax.set_xlabel('Year')
        ax.set_ylabel(stat_config['ylabel'])
        ax.set_title(wui_type["name"].replace(" WUI", ""))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        
        # 设置x轴标签
        ax.set_xticks(years)
        ax.set_xticklabels(years, rotation=0, ha='center', fontsize=20)
        ax.set_xlim(years[0] - 0.48, years[-1] + 0.48)
        
        # 设置科学计数法
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        # 添加图例并调整位置
        ax.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99), fontsize=14)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        output_filename = f"{stat_config['data_type_suffix']}_{wui_type['filename_suffix']}_stacked_by_year.jpg"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=1200, bbox_inches='tight', format='jpg')
        print(f"Chart saved to: {output_path}")
        
        plt.close()


def _format_abs_tick(value, _position):
    """Display both sides of the pyramid x-axis as positive values."""
    value = abs(value)
    if value == 0:
        return "0"
    if value < 0.001:
        return f"{value:.1e}"
    if value < 0.01:
        return f"{value:.4f}"
    if value < 1:
        return f"{value:.3f}"
    return f"{value:.1f}"


def _collect_pyramid_values(all_data, wui_type, years):
    """
    获取指定WUI类型在各年份、各大洲上的指标值矩阵。
    返回格式: {continent_code: np.ndarray([year1_value, year2_value, ...])}
    """
    continent_values = {}
    for continent in continent_order:
        values = []
        for year in years:
            metric_col = wui_type['metric_field'](year)
            year_data = all_data[
                (all_data['Year'] == year) &
                (all_data['Continent'] == continent)
            ]
            if metric_col in all_data.columns and not year_data.empty:
                value = pd.to_numeric(year_data[metric_col], errors='coerce').fillna(0).iloc[0]
            else:
                value = 0
            values.append(value)
        continent_values[continent] = np.asarray(values, dtype=float)
    return continent_values


def _draw_pyramid_panel(fig, grid_spec, all_data, stat_config):
    """
    在指定GridSpec位置绘制单个指标的Interface/Intermix镜像金字塔图。
    """
    years = list(range(2003, 2023))
    y_positions = np.arange(len(years))
    interface_type, intermix_type = stat_config['wui_types']

    interface_values = _collect_pyramid_values(all_data, interface_type, years)
    intermix_values = _collect_pyramid_values(all_data, intermix_type, years)

    interface_totals = np.zeros(len(years))
    intermix_totals = np.zeros(len(years))
    for continent in continent_order:
        interface_totals += interface_values[continent]
        intermix_totals += intermix_values[continent]

    axis_limit = max(interface_totals.max(), intermix_totals.max())
    if not np.isfinite(axis_limit) or axis_limit <= 0:
        axis_limit = 1
    axis_limit *= 1.12

    panel_spec = grid_spec.subgridspec(
        1,
        3,
        width_ratios=[1, 0.18, 1],
        wspace=0.02,
    )
    ax_left = fig.add_subplot(panel_spec[0, 0])
    ax_year = fig.add_subplot(panel_spec[0, 1], sharey=ax_left)
    ax_right = fig.add_subplot(panel_spec[0, 2], sharey=ax_left)

    left_stack = np.zeros(len(years))
    right_stack = np.zeros(len(years))
    bar_height = 0.648

    for continent in continent_order:
        color = continent_colors[continent]
        left_values = interface_values[continent]
        right_values = intermix_values[continent]

        ax_left.barh(
            y_positions,
            left_values,
            left=left_stack,
            height=bar_height,
            color=color,
            edgecolor='white',
            linewidth=0.35,
        )
        ax_right.barh(
            y_positions,
            right_values,
            left=right_stack,
            height=bar_height,
            color=color,
            edgecolor='white',
            linewidth=0.35,
        )

        left_stack += left_values
        right_stack += right_values

    ax_left.set_xlim(axis_limit, 0)
    ax_right.set_xlim(0, axis_limit)

    for ax in (ax_left, ax_right):
        ax.set_ylim(y_positions[0] - 0.8, y_positions[-1] + 0.8)
        ax.set_yticks([])
        ax.xaxis.set_major_formatter(FuncFormatter(_format_abs_tick))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis='x', labelsize=24)
        ax.grid(axis='x', color='#D9D9D9', linestyle='-', linewidth=0.9, alpha=0.75)
        ax.set_axisbelow(True)
        for spine in ['left', 'right', 'top']:
            ax.spines[spine].set_visible(False)

    ax_left.axvline(0, color='black', linewidth=1.0, zorder=3)
    ax_right.axvline(0, color='black', linewidth=1.0, zorder=3)

    ax_year.set_xlim(0, 1)
    ax_year.set_ylim(y_positions[0] - 0.8, y_positions[-1] + 0.8)
    ax_year.set_xticks([])
    ax_year.set_yticks([])
    for spine in ax_year.spines.values():
        spine.set_visible(False)
    for y_pos, year in zip(y_positions, years):
        ax_year.text(
            0.5,
            y_pos,
            str(year),
            ha='center',
            va='center',
            fontsize=24,
            color='black',
            zorder=4,
        )

    ax_left.set_title('Interface', fontsize=28, pad=6)
    ax_right.set_title('Intermix', fontsize=28, pad=6)

    return ax_left, ax_year, ax_right


def plot_pyramid_chart_grouped(all_data):
    """
    将Fire frequency与Normalized burned area绘制成1*2组图。
    """
    fig = plt.figure(figsize=(24.8, 17.6))
    outer_spec = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1, 1],
        wspace=0.14,
    )

    for index, stat_config in enumerate(stats_config):
        _draw_pyramid_panel(fig, outer_spec[0, index], all_data, stat_config)

    legend_handles = [
        Patch(facecolor=continent_colors[continent], label=continent_mapping[continent])
        for continent in continent_order
    ]
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.08, 0.094, 0.84, 0.05),
        ncol=len(legend_handles),
        mode='expand',
        frameon=False,
        fontsize=23,
        handlelength=1.55,
        handletextpad=0.55,
        columnspacing=1.05,
        borderaxespad=0,
    )

    fig.text(0.255, 0.06, stats_config[0]['ylabel'], ha='center', va='bottom', fontsize=34)
    fig.text(0.745, 0.06, stats_config[1]['ylabel'], ha='center', va='bottom', fontsize=34)

    fig.subplots_adjust(left=0.05, right=0.985, top=0.91, bottom=0.16)

    output_filename = "fire_frequency_and_normalized_burned_area_interface_intermix_pyramid_by_year.jpg"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=1200, bbox_inches='tight', format='jpg')
    print(f"Chart saved to: {output_path}")

    plt.close(fig)


def main():
    """
    主函数
    """
    print("=== Continent Fire Frequency & Normalized Burned Area Pyramid Chart Program ===")
    
    # 加载所有年份的火灾数据
    all_data = []
    for wui_year in wui_years:
        csv_path = f"I:\\Processing data\\Grid_Clip_{wui_year}_WUI.csv"
        
        try:
            # 加载数据
            df = pd.read_csv(csv_path)
            
            # 过滤掉Continent为0的行
            df = df[df['Continent'] != 0]
            
            # 将Continent代码为5的数据合并到代码为3的Oceania中
            df.loc[df['Continent'] == 5, 'Continent'] = 3
            
            # 提取数据
            annual_data = extract_annual_data(df, wui_year)
            if annual_data is not None:
                all_data.append(annual_data)
            
            print(f"Loaded fire data for base year {wui_year}")
        except Exception as e:
            print(f"Error loading fire data for base year {wui_year}: {e}")
    
    if not all_data:
        print("Failed to load any fire data")
        return
    
    # 合并所有数据
    all_data = pd.concat(all_data, ignore_index=True)
    
    # 去除重复的年份-大洲组合
    all_data = all_data.drop_duplicates(subset=['Year', 'Continent'], keep='first')
    
    # 绘制1*2组图
    print("\nProcessing grouped pyramid chart...")
    plot_pyramid_chart_grouped(all_data)

    print("\nAll pyramid chart processing completed!")


if __name__ == "__main__":
    main()
