#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
程序名称: 火灾频率与面积密度空间分布绘制程序
功能简介: 本程序用于计算并绘制全球范围内不同年份的Interface和Intermix火灾频率及火灾面积密度空间分布图
         1. 火灾频率计算方式:Fire_num_年份 / WUI_area(单位:fires/km²)
         2. 火灾面积密度计算方式:Fire_area_年份 / WUI_area(单位:km²/km²)
         3. 年份映射规则：
            - 2003-2007年数据对应2005年的WUI_area
            - 2008-2012年数据对应2010年的WUI_area
            - 2013-2017年数据对应2015年的WUI_area
            - 2018-2022年数据对应2020年的WUI_area
         4. 使用与原程序一致的颜色映射(YlOrRd)和布局样式
         5. 输出高分辨率(1200dpi)的JPG格式地图

数据输入:
    - 全球地图边界数据: I:/Data_huanglin/map/world continent boundary/continent.shp
    - Interface WUI Fire Risk数据: I:/Processing data/Interface_WUI_Fire_Risk.gdb
    - Intermix WUI Fire Risk数据: I:/Processing data/Intermix_WUI_Fire_Risk.gdb

输出文件:
    - 火灾频率分布图: B:/WUI/Picture/{Interface/Intermix}_Fire_Frequency_YYYY.jpg
    - 火灾面积密度分布图: B:/WUI/Picture/{Interface/Intermix}_Fire_Area_Density_YYYY.jpg
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import numpy as np
import pandas as pd

# 设置英文显示，使用Arial字体
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 定义数据路径
continent_shp_path = r"I:\Data_huanglin\map\world continent boundary\continent.shp"

# 定义要处理的数据配置 - 保持与原程序一致的路径结构
fire_data_configs = [
    {
        "type": "Interface",
        "gdb_path": r"I:\Processing data\Interface_WUI_Fire_Risk.gdb",
        "years": [2005, 2010, 2015, 2020],  # 处理所有Grid_Clip年份
        "layer_prefix": "Grid_Clip_"
    },
    {
        "type": "Intermix",
        "gdb_path": r"I:\Processing data\Intermix_WUI_Fire_Risk.gdb",
        "years": [2005, 2010, 2015, 2020],  # 处理所有Grid_Clip年份
        "layer_prefix": "Grid_Clip_"
    }
]

# 定义输出路径
output_dir = r"B:\WUI\Picture"  # 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 固定图内布局，避免不同colorbar标签长度导致保存时裁切出不同图幅尺寸
MAP_AX_POSITION = [0.07, 0.12, 0.74, 0.76]
CBAR_AX_POSITION = [0.84, 0.34, 0.018, 0.36]

# 加载全球地图数据
print(f"Loading global map data: {continent_shp_path}")
continent = gpd.read_file(continent_shp_path)
# 显式转换为WGS1984坐标系统
continent = continent.to_crs("EPSG:4326")
print(f"Successfully loaded global map with {len(continent)} features")
print(f"Map data CRS: {continent.crs}")

# 循环处理不同类型和Grid_Clip年份的数据
for config in fire_data_configs:
    data_type = config["type"]
    gdb_path = config["gdb_path"]
    grid_clip_years = config["years"]
    layer_prefix = config["layer_prefix"]
    
    for grid_year in grid_clip_years:
        # 计算当前Grid_Clip包含的5个数据年份 (例如2020 -> 2018-2022)
        data_years = list(range(grid_year - 2, grid_year + 3))  # 前2年到后2年
        
        # 构建渔网图层名称 - 保持与原程序一致的格式
        fishnet_layer = f"{layer_prefix}{grid_year}_{data_type.lower()}_WUI"
        
        print(f"\n{'='*70}")
        print(f"Processing {data_type} fire frequency data for Grid_Clip year {grid_year}")
        print(f"This Grid_Clip contains data for years: {data_years}")
        print(f"{'='*70}")
        
        try:
            # 加载渔网数据
            print(f"Loading fishnet data layer: {fishnet_layer}")
            print(f"Data source: {gdb_path}")
            fishnet = gpd.read_file(gdb_path, layer=fishnet_layer)  # 加载完整数据
            print(f"Successfully loaded fishnet data with {len(fishnet)} features")
            print(f"Fishnet data fields: {list(fishnet.columns[:15])}...")  # 只显示前15个字段
            print(f"Fishnet data original CRS: {fishnet.crs}")
            
            # 调试：检查数据基本统计信息
            print(f"Debug - First 3 rows:")
            print(fishnet.head(3))
            print(f"Debug - WUI_area sample values: {list(fishnet['WUI_area'].head(10))}")
            
            # 添加详细的字段信息
            print("\nDetailed field information:")
            for i, column in enumerate(fishnet.columns):
                print(f"  [{i+1}] Field name: '{column}'")
            
            # 转换渔网数据的CRS为WGS1984
            print("Unifying coordinate systems...")
            fishnet = fishnet.to_crs("EPSG:4326")
            print(f"Fishnet data CRS after conversion: {fishnet.crs}")
            
            # 为每个数据年份创建图形
            for data_year in data_years:
                print(f"\n--- Processing data for year {data_year} ---")
                
                # 处理火点数量数据 - 格式：Fire_num_年份
                fire_num_field = f"Fire_num_{data_year}"
                
                # 检查火点数量字段是否存在
                if fire_num_field not in fishnet.columns:
                    print(f"Warning: Fire number field {fire_num_field} not found in the data")
                    print(f"Available fields: {list(fishnet.columns)}")
                    continue
                
                # 检查WUI_area字段是否存在
                if 'WUI_area' not in fishnet.columns:
                    print(f"Warning: WUI_area field not found in the data")
                    print(f"Available fields: {list(fishnet.columns)}")
                    continue
                
                # 创建临时副本用于计算火灾频率
                fishnet_freq = fishnet.copy()
                
                # 处理过火面积数据 - 尝试三种可能的字段名格式
                fire_area_field = None
                possible_fields = [
                    str(data_year),  # 直接使用年份数字作为字段名
                    f"Fire_area_{data_year}",  # 使用Fire_area_年份格式
                    f"过火面积{data_year}"  # 使用中文格式：过火面积年份
                ]
                
                # 检查哪个格式存在
                for field in possible_fields:
                    if field in fishnet.columns:
                        fire_area_field = field
                        print(f"Found burned area field: {field}")
                        break
                
                # 检查火点面积字段是否存在
                if fire_area_field is None:
                    print(f"Warning: Burned area field not found for year {data_year}")
                    # 跳过火灾面积密度计算，但继续火灾频率计算
                
                # 计算火灾频率：Fire_num / WUI_area
                # WUI_area应该只有0或正值，无需使用绝对值
                fishnet_freq['Fire_frequency'] = np.nan
                valid_mask = fishnet_freq['WUI_area'] > 0
                fishnet_freq.loc[valid_mask, 'Fire_frequency'] = fishnet_freq.loc[valid_mask, fire_num_field] / fishnet_freq.loc[valid_mask, 'WUI_area']
                
                # 计算火灾面积密度：Fire_area / WUI_area
                if fire_area_field is not None:
                    fishnet_freq['Fire_area_density'] = np.nan
                    fishnet_freq.loc[valid_mask, 'Fire_area_density'] = fishnet_freq.loc[valid_mask, fire_area_field] / fishnet_freq.loc[valid_mask, 'WUI_area']
                    # 限制火灾面积密度的最大值为1
                    fishnet_freq['Fire_area_density'] = fishnet_freq['Fire_area_density'].clip(upper=1)
                
                # 计算非NaN值的范围
                valid_freq = fishnet_freq['Fire_frequency'].dropna()
                if len(valid_freq) == 0:
                    print(f"Warning: No valid Fire_frequency data for year {data_year}")
                    plt.close(fig)
                    continue
                
                fire_freq_min = valid_freq.min()
                fire_freq_max = valid_freq.max()
                print(f"Fire frequency range: {fire_freq_min:.2f} to {fire_freq_max:.2f}")
                
                # 计算火灾面积密度的非NaN值范围（如果有数据）
                fire_density_min = None
                fire_density_max = None
                if fire_area_field is not None:
                    valid_density = fishnet_freq['Fire_area_density'].dropna()
                    if len(valid_density) > 0:
                        fire_density_min = valid_density.min()
                        fire_density_max = valid_density.max()
                        print(f"Fire area density range: {fire_density_min:.2f} to {fire_density_max:.2f}")

                # 创建图形
                fig, ax = plt.subplots(figsize=(15, 10))

                # 绘制全球地图底图，颜色为浅灰色
                print("Drawing global map base layer...")
                continent.plot(ax=ax, color='#f0f0f0', edgecolor='#d0d0d0', linewidth=0.5)

                # 配置火灾频率的颜色映射
                print(f"Configuring fire frequency color mapping for year {data_year}...")

                # 创建临时副本用于绘图，只处理非零值
                fishnet_plot = fishnet_freq.copy()
                
                # 将0值和空值都设置为NaN，使它们与背景颜色一致
                fishnet_plot['Fire_frequency'] = fishnet_plot['Fire_frequency'].replace(0, np.nan).fillna(np.nan)
                
                # 使用与原程序完全相同的颜色映射设置
                full_cmap = plt.get_cmap('YlOrRd', 10)  # 将YlOrRd分成10份，从黄色到红色
                # 直接从原始颜色映射中截取中间6份（索引4-9，共6个颜色），确保红色在最大值
                colors = [full_cmap(i) for i in range(4, 10)]
                cmap_fire_freq = mcolors.ListedColormap(colors)  # 使用ListedColormap创建分层颜色映射
                
                # 创建边界，将数据范围均匀分成6个区间
                boundaries = np.linspace(fire_freq_min, fire_freq_max, 7)  # 7个边界点定义6个区间
                norm_fire_freq = mcolors.BoundaryNorm(boundaries, cmap_fire_freq.N)  # 使用BoundaryNorm实现分层映射
                
                # 创建ScalarMappable对象用于颜色条
                sm_fire_freq = plt.cm.ScalarMappable(cmap=cmap_fire_freq, norm=norm_fire_freq)
                sm_fire_freq.set_array([])
                
                # 设置颜色条标签格式为两位小数
                from matplotlib.ticker import FormatStrFormatter
                formatter_fire_freq = FormatStrFormatter('%.2f')

                # 绘制火灾频率渔网，使用与原程序相同的参数
                print("Drawing fire frequency fishnet data...")
                fishnet_plot.plot(ax=ax, column='Fire_frequency', cmap=cmap_fire_freq, norm=norm_fire_freq, 
                                 edgecolor='none', linewidth=0)  # 与原程序保持一致
                print("Fire frequency fishnet data drawing completed")

                # 手动创建和配置颜色条
                print("Creating custom colorbar...")
                cbar = plt.colorbar(sm_fire_freq, ax=ax, 
                                  shrink=0.465,  # 与原程序保持一致的尺寸
                                  aspect=20,
                                  fraction=0.02,
                                  pad=0.02,
                                  orientation='vertical',
                                  label=f'{data_type} Fire Frequency {data_year} (fires/km²)')
                
                # 设置颜色条标签线朝里
                cbar.ax.tick_params(direction='in')
                # 设置颜色条标签字号
                cbar.ax.tick_params(labelsize=14)
                # 设置颜色条标题字号
                cbar.ax.yaxis.label.set_size(16)
                # 应用两位小数格式化器
                cbar.ax.yaxis.set_major_formatter(formatter_fire_freq)

                # 在地图左上角添加年份文本（无框）
                ax.text(0.02, 0.92, f'{data_year}', transform=ax.transAxes, fontsize=20, 
                       verticalalignment='top')

                # 设置坐标轴标签
                ax.set_xlabel('Longitude', fontsize=16)
                ax.set_ylabel('Latitude', fontsize=16)

                # 设置坐标轴刻度字体大小
                ax.tick_params(axis='both', which='major', labelsize=16)

                # 设置纬度显示范围
                ax.set_ylim(-60, 75)

                # 设置网格
                ax.grid(True, linestyle='--', alpha=0.5)

                # 使用固定布局，保证Fire frequency和Normalized burned area图幅与内部位置一致
                ax.set_position(MAP_AX_POSITION)
                cbar.ax.set_position(CBAR_AX_POSITION)

                # 保存火灾频率图形
                output_filename = f"{data_type}_Fire_Frequency_{data_year}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                fig.savefig(output_path, dpi=1200, format='jpg')
                print(f"Fire frequency figure successfully saved to: {output_path}")

                # 关闭图形，释放内存
                plt.close(fig)
                
                # ============== 火灾面积密度绘图 ==============
                if fire_area_field is not None:
                    print(f"\n--- Processing Fire Area Density data for year {data_year} ---")
                    
                    # 检查是否有有效数据
                    valid_density = fishnet_freq['Fire_area_density'].dropna()
                    if len(valid_density) == 0:
                        print(f"Warning: No valid Fire_area_density data for year {data_year}")
                        continue
                    
                    # 创建图形
                    fig, ax = plt.subplots(figsize=(15, 10))

                    # 绘制全球地图底图，颜色为浅灰色
                    print("Drawing global map base layer...")
                    continent.plot(ax=ax, color='#f0f0f0', edgecolor='#d0d0d0', linewidth=0.5)

                    # 配置火灾面积密度的颜色映射
                    print(f"Configuring fire area density color mapping for year {data_year}...")

                    # 创建临时副本用于绘图，只处理非零值
                    fishnet_plot = fishnet_freq.copy()
                    
                    # 将0值和空值都设置为NaN，使它们与背景颜色一致
                    fishnet_plot['Fire_area_density'] = fishnet_plot['Fire_area_density'].replace(0, np.nan).fillna(np.nan)
                    
                    # 使用与火灾频率相同的颜色映射设置
                    full_cmap = plt.get_cmap('YlOrRd', 10)  # 将YlOrRd分成10份，从黄色到红色
                    # 直接从原始颜色映射中截取中间6份（索引4-9，共6个颜色），确保红色在最大值
                    colors = [full_cmap(i) for i in range(4, 10)]
                    cmap_fire_density = mcolors.ListedColormap(colors)  # 使用ListedColormap创建分层颜色映射
                    
                    # 创建边界，将数据范围均匀分成6个区间
                    boundaries = np.linspace(fire_density_min, fire_density_max, 7)  # 7个边界点定义6个区间
                    norm_fire_density = mcolors.BoundaryNorm(boundaries, cmap_fire_density.N)  # 使用BoundaryNorm实现分层映射
                    
                    # 创建ScalarMappable对象用于颜色条
                    sm_fire_density = plt.cm.ScalarMappable(cmap=cmap_fire_density, norm=norm_fire_density)
                    sm_fire_density.set_array([])
                    
                    # 设置颜色条标签格式为两位小数
                    formatter_fire_density = FormatStrFormatter('%.2f')

                    # 绘制火灾面积密度渔网，使用与火灾频率相同的参数
                    print("Drawing fire area density fishnet data...")
                    fishnet_plot.plot(ax=ax, column='Fire_area_density', cmap=cmap_fire_density, norm=norm_fire_density, 
                                     edgecolor='none', linewidth=0)  # 与原程序保持一致
                    print("Fire area density fishnet data drawing completed")

                    # 手动创建和配置颜色条
                    print("Creating custom colorbar...")
                    cbar = plt.colorbar(sm_fire_density, ax=ax, 
                                      shrink=0.465,  # 与原程序保持一致的尺寸
                                      aspect=20,
                                      fraction=0.02,
                                      pad=0.02,
                                      orientation='vertical',
                                      label=f'{data_type} Fire Area Density {data_year} (km²/km²)')
                    
                    # 设置颜色条标签线朝里
                    cbar.ax.tick_params(direction='in')
                    # 设置颜色条标签字号
                    cbar.ax.tick_params(labelsize=14)
                    # 设置颜色条标题字号
                    cbar.ax.yaxis.label.set_size(16)
                    # 应用两位小数格式化器
                    cbar.ax.yaxis.set_major_formatter(formatter_fire_density)

                    # 在地图左上角添加年份文本（无框）
                    ax.text(0.02, 0.92, f'{data_year}', transform=ax.transAxes, fontsize=20, 
                           verticalalignment='top')

                    # 设置坐标轴标签
                    ax.set_xlabel('Longitude', fontsize=16)
                    ax.set_ylabel('Latitude', fontsize=16)

                    # 设置坐标轴刻度字体大小
                    ax.tick_params(axis='both', which='major', labelsize=16)

                    # 设置纬度显示范围
                    ax.set_ylim(-60, 75)

                    # 设置网格
                    ax.grid(True, linestyle='--', alpha=0.5)

                    # 使用固定布局，保证Fire frequency和Normalized burned area图幅与内部位置一致
                    ax.set_position(MAP_AX_POSITION)
                    cbar.ax.set_position(CBAR_AX_POSITION)

                    # 保存火灾面积密度图形
                    output_filename = f"{data_type}_Fire_Area_Density_{data_year}.jpg"
                    output_path = os.path.join(output_dir, output_filename)
                    fig.savefig(output_path, dpi=1200, format='jpg')
                    print(f"Fire area density figure successfully saved to: {output_path}")

                    # 关闭图形，释放内存
                    plt.close(fig)
                
        except Exception as e:
            print(f"Error processing {data_type} fire frequency data for Grid_Clip year {grid_year}: {str(e)}")
            continue

print("\nAll fire frequency data processing completed!")
