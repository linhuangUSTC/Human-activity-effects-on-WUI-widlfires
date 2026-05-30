#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
程序名称: 总火灾频率与面积密度空间分布绘制程序
功能简介: 本程序用于计算并绘制全球范围内不同年份的Interface和Intermix总火灾频率及总火灾面积密度空间分布图
         1. 合并Interface和Intermix数据(GRID_ID的并集)
         2. 计算每个GRID_ID的总火点数(Fire_num总和)
         3. 计算每个GRID_ID的总过火面积(Fire_area总和)
         4. 计算每个GRID_ID的总WUI面积(WUI_area总和)
         5. 总火灾频率计算:总Fire_num / 总WUI_area(单位:fires/km²)
         6. 总火灾面积密度计算:总Fire_area / 总WUI_area(单位:km²/km²)
         7. 年份映射规则：
            - 2003-2007年数据对应2005年的WUI_area
            - 2008-2012年数据对应2010年的WUI_area
            - 2013-2017年数据对应2015年的WUI_area
            - 2018-2022年数据对应2020年的WUI_area
         8. 使用与原程序一致的颜色映射(YlOrRd)和布局样式
         9. 输出高分辨率(1200dpi)的JPG格式地图

数据输入:
    - 全球地图边界数据: I:/Data_huanglin/map/world continent boundary/continent.shp
    - Interface WUI Fire Risk数据: I:/Processing data/gdb集合/Interface_WUI_Fire_Risk.gdb
    - Intermix WUI Fire Risk数据: I:/Processing data/gdb集合/Intermix_WUI_Fire_Risk.gdb

输出文件:
    - 总火灾频率分布图: B:/WUI/Picture/Fire_frequency_YYYY.jpg
    - 总火灾面积密度分布图: B:/WUI/Picture/Normalized_burned_area_YYYY.jpg
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, FuncFormatter

# 设置英文显示，使用Arial字体
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 定义数据路径
continent_shp_path = r"I:\Data_huanglin\map\world continent boundary\continent.shp"

# 定义要处理的数据配置
interface_config = {
    "type": "Interface",
    "gdb_path": r"I:\Processing data\gdb集合\Interface_WUI_Fire_Risk.gdb",
    "years": [2005, 2010, 2015, 2020],  # 处理所有Grid_Clip年份
    "layer_prefix": "Grid_Clip_"
}

intermix_config = {
    "type": "Intermix",
    "gdb_path": r"I:\Processing data\gdb集合\Intermix_WUI_Fire_Risk.gdb",
    "years": [2005, 2010, 2015, 2020],  # 处理所有Grid_Clip年份
    "layer_prefix": "Grid_Clip_"
}

# 定义输出路径 - 与原程序保持一致
output_dir = r"B:\WUI\Picture"
# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 固定横向画布和图内布局，图框比例匹配经纬度范围，避免地图被拉伸压扁
FIG_SIZE = (15, 5.8)
MAP_AX_POSITION = [0.045, 0.13, 0.82, 0.795]
CBAR_AX_POSITION = [0.895, 0.13, 0.018, 0.795]


def log10p_transform(values):
    """Color-mapping transform for positive Fire frequency values only."""
    return np.log10(1.0 + values)


def inverse_log10p_transform(values):
    """Inverse transform used to show original Fire frequency values on the colorbar."""
    return np.power(10.0, values) - 1.0


def format_original_fire_frequency(value, _position):
    """Display original Fire frequency values on a transformed colorbar."""
    original_value = inverse_log10p_transform(value)
    return f"{original_value:.1f}"

# 加载全球地图数据
print(f"Loading global map data: {continent_shp_path}")
continent = gpd.read_file(continent_shp_path)
# 显式转换为WGS1984坐标系统
continent = continent.to_crs("EPSG:4326")
print(f"Successfully loaded global map with {len(continent)} features")
print(f"Map data CRS: {continent.crs}")

# 循环处理不同的Grid_Clip年份
for year in interface_config["years"]:
    print(f"\n{'='*80}")
    print(f"Processing Total Fire Frequency and Density for Grid_Clip year {year}")
    print(f"{'='*80}")
    
    try:
        # 计算当前Grid_Clip包含的5个数据年份 (例如2020 -> 2018-2022)
        data_years = list(range(year - 2, year + 3))  # 前2年到后2年
        print(f"This Grid_Clip contains data for years: {data_years}")
        
        # 加载Interface WUI数据
        interface_fishnet_layer = f"{interface_config['layer_prefix']}{year}_{interface_config['type'].lower()}_WUI"
        print(f"Loading Interface fishnet data layer: {interface_fishnet_layer}")
        interface_fishnet = gpd.read_file(interface_config['gdb_path'], layer=interface_fishnet_layer)
        print(f"Successfully loaded Interface fishnet data with {len(interface_fishnet)} features")
        
        # 加载Intermix WUI数据
        intermix_fishnet_layer = f"{intermix_config['layer_prefix']}{year}_{intermix_config['type'].lower()}_WUI"
        print(f"Loading Intermix fishnet data layer: {intermix_fishnet_layer}")
        intermix_fishnet = gpd.read_file(intermix_config['gdb_path'], layer=intermix_fishnet_layer)
        print(f"Successfully loaded Intermix fishnet data with {len(intermix_fishnet)} features")
        
        # 转换CRS为WGS1984
        print("Unifying coordinate systems...")
        interface_fishnet = interface_fishnet.to_crs("EPSG:4326")
        intermix_fishnet = intermix_fishnet.to_crs("EPSG:4326")
        print(f"Interface fishnet CRS: {interface_fishnet.crs}")
        print(f"Intermix fishnet CRS: {intermix_fishnet.crs}")
        
        # 确保两个数据框都有唯一标识字段
        if 'GRID_ID' not in interface_fishnet.columns:
            interface_fishnet['GRID_ID'] = interface_fishnet.geometry.apply(lambda x: x.centroid.wkt)
        if 'GRID_ID' not in intermix_fishnet.columns:
            intermix_fishnet['GRID_ID'] = intermix_fishnet.geometry.apply(lambda x: x.centroid.wkt)
        
        # 合并两个数据集
        print("Merging Interface and Intermix fishnet data...")
        total_fishnet = gpd.GeoDataFrame(
            pd.concat([interface_fishnet, intermix_fishnet], ignore_index=True),
            crs="EPSG:4326"
        )
        print(f"Total fishnet data has {len(total_fishnet)} features after merging")
        
        # 为每个数据年份创建图形
        for data_year in data_years:
            print(f"\n--- Processing Total Fire data for year {data_year} ---")
            
            # 处理火点数量数据 - 格式：Fire_num_年份
            fire_num_field = f"Fire_num_{data_year}"
            
            # 处理过火面积数据 - 尝试三种可能的字段名格式
            fire_area_field = None
            possible_fields = [
                str(data_year),  # 直接使用年份数字作为字段名
                f"Fire_area_{data_year}",  # 使用Fire_area_年份格式
                f"过火面积{data_year}"  # 使用中文格式：过火面积年份
            ]
            
            # 检查哪个格式存在
            for field in possible_fields:
                if field in interface_fishnet.columns:
                    fire_area_field = field
                    print(f"Found burned area field: {field}")
                    break
            
            # 检查字段是否存在
            if fire_num_field not in interface_fishnet.columns or fire_num_field not in intermix_fishnet.columns:
                print(f"Warning: Fire number field {fire_num_field} not found in both datasets")
                continue
            
            if fire_area_field is None:
                print(f"Warning: Burned area field not found for year {data_year}")
                # 跳过火灾面积密度计算，但继续火灾频率计算
            
            # 按GRID_ID分组，计算每个网格的总和
            print("Calculating total values by GRID_ID...")
            
            # 定义需要聚合的字段
            agg_fields = {
                'geometry': 'first',
                fire_num_field: 'sum',
                'WUI_area': 'sum'
            }
            
            # 如果有火灾面积字段，也加入聚合
            if fire_area_field is not None:
                agg_fields[fire_area_field] = 'sum'
            
            # 分组并聚合数据
            total_grid_data = total_fishnet.groupby('GRID_ID').agg(agg_fields).reset_index()
            
            # 创建新的GeoDataFrame
            total_grid_gdf = gpd.GeoDataFrame(total_grid_data, geometry='geometry', crs="EPSG:4326")
            print(f"Total grid data has {len(total_grid_gdf)} unique GRID_IDs")
            
            # 计算总火灾频率：总Fire_num / 总WUI_area
            print("Calculating Total Fire Frequency...")
            total_grid_gdf['Total_Fire_Frequency'] = np.nan
            valid_mask = total_grid_gdf['WUI_area'] > 0
            total_grid_gdf.loc[valid_mask, 'Total_Fire_Frequency'] = total_grid_gdf.loc[valid_mask, fire_num_field] / total_grid_gdf.loc[valid_mask, 'WUI_area']
            
            # 计算总火灾面积密度：总Fire_area / 总WUI_area
            if fire_area_field is not None:
                print("Calculating Total Fire Area Density...")
                total_grid_gdf['Total_Fire_Area_Density'] = np.nan
                total_grid_gdf.loc[valid_mask, 'Total_Fire_Area_Density'] = total_grid_gdf.loc[valid_mask, fire_area_field] / total_grid_gdf.loc[valid_mask, 'WUI_area']
                # 限制总火灾面积密度的最大值为1
                total_grid_gdf['Total_Fire_Area_Density'] = total_grid_gdf['Total_Fire_Area_Density'].clip(upper=1)
            
            # 检查是否有有效数据
            valid_freq = total_grid_gdf['Total_Fire_Frequency'].dropna()
            positive_freq = valid_freq[valid_freq > 0]
            if len(positive_freq) == 0:
                print(f"Warning: No valid Total_Fire_Frequency data for year {data_year}")
                continue
            
            # 计算火灾频率的范围
            fire_freq_min = positive_freq.min()
            fire_freq_max = positive_freq.max()
            print(f"Total Fire Frequency range: {fire_freq_min:.2f} to {fire_freq_max:.2f} fires/km²")
            
            # ================== 总火灾频率绘图 ==================
            print(f"\n--- Drawing Total Fire Frequency for year {data_year} ---")
            
            # 创建图形
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            
            # 绘制全球地图底图，颜色为浅灰色
            print("Drawing global map base layer...")
            continent.plot(ax=ax, color='#f0f0f0', edgecolor='#d0d0d0', linewidth=0.5)
            
            # 配置火灾频率的颜色映射
            print(f"Configuring fire frequency color mapping for year {data_year}...")
            
            # 创建临时副本用于绘图，只处理非零值
            fishnet_freq = total_grid_gdf.copy()
            
            # 将0值和空值都设置为NaN，使它们与背景颜色一致
            fishnet_freq['Total_Fire_Frequency'] = fishnet_freq['Total_Fire_Frequency'].replace(0, np.nan).fillna(np.nan)
            fishnet_freq['Total_Fire_Frequency_Log10P'] = np.nan
            positive_mask = fishnet_freq['Total_Fire_Frequency'].gt(0)
            fishnet_freq.loc[positive_mask, 'Total_Fire_Frequency_Log10P'] = log10p_transform(
                fishnet_freq.loc[positive_mask, 'Total_Fire_Frequency']
            )
            
            # 创建颜色映射 - 使用YlOrRd颜色映射，确保红色位于最大值
            full_cmap = plt.get_cmap('YlOrRd', 10)  # 将YlOrRd分成10份，从黄色到红色
            colors = [full_cmap(i) for i in range(4, 10)]  # 截取中间6份
            cmap_fire_freq = mcolors.ListedColormap(colors)  # 使用ListedColormap创建分层颜色映射
            
            # 对Fire frequency正值做log10(1 + x)映射，再在映射空间内等距分层
            transformed_min = log10p_transform(fire_freq_min)
            transformed_max = log10p_transform(fire_freq_max)
            boundaries = np.linspace(transformed_min, transformed_max, 7)  # 7个边界点定义6个区间
            norm_fire_freq = mcolors.BoundaryNorm(boundaries, cmap_fire_freq.N)  # 使用BoundaryNorm实现分层映射
            
            # 创建ScalarMappable对象用于颜色条
            sm_fire_freq = plt.cm.ScalarMappable(cmap=cmap_fire_freq, norm=norm_fire_freq)
            sm_fire_freq.set_array([])
            
            # 颜色条标签仍显示原始Fire frequency值
            formatter_fire_freq = FuncFormatter(format_original_fire_frequency)
            
            # 绘制火灾频率渔网
            print("Drawing Total Fire frequency fishnet data...")
            fishnet_freq.plot(ax=ax, column='Total_Fire_Frequency_Log10P', cmap=cmap_fire_freq, norm=norm_fire_freq, 
                             edgecolor='none', linewidth=0)
            print("Total Fire frequency fishnet data drawing completed")
            
            # 手动创建和配置颜色条，使其高度与地图图框一致
            print("Creating custom colorbar...")
            cbar_ax = fig.add_axes(CBAR_AX_POSITION)
            cbar = fig.colorbar(sm_fire_freq, cax=cbar_ax,
                                orientation='vertical',
                                label='Fire frequency')
            
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
            
            # 设置经纬度显示范围
            ax.set_xlim(-180, 180)
            ax.set_ylim(-60, 75)
            
            # 设置网格
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # 使用固定布局，保证Fire frequency和Normalized burned area图幅与内部位置一致
            ax.set_position(MAP_AX_POSITION)
            ax.set_aspect('equal', adjustable='box')
            
            # 保存火灾频率图形
            output_filename = f"Fire_frequency_{data_year}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            fig.savefig(output_path, dpi=1200, format='jpg')
            print(f"Total Fire Frequency figure successfully saved to: {output_path}")
            
            # 关闭图形，释放内存
            plt.close(fig)
            
            # ================== 总火灾面积密度绘图 ==================
            if fire_area_field is not None:
                print(f"\n--- Processing Total Fire Area Density for year {data_year} ---")
                
                # 检查是否有有效数据
                valid_density = total_grid_gdf['Total_Fire_Area_Density'].dropna()
                if len(valid_density) == 0:
                    print(f"Warning: No valid Total_Fire_Area_Density data for year {data_year}")
                    continue
                
                # 计算火灾面积密度的范围
                fire_density_min = valid_density.min()
                fire_density_max = valid_density.max()
                print(f"Total Fire Area Density range: {fire_density_min:.2f} to {fire_density_max:.2f} km²/km²")
                
                # 创建图形
                fig, ax = plt.subplots(figsize=FIG_SIZE)
                
                # 绘制全球地图底图，颜色为浅灰色
                print("Drawing global map base layer...")
                continent.plot(ax=ax, color='#f0f0f0', edgecolor='#d0d0d0', linewidth=0.5)
                
                # 配置火灾面积密度的颜色映射
                print(f"Configuring fire area density color mapping for year {data_year}...")
                
                # 创建临时副本用于绘图，只处理非零值
                fishnet_density = total_grid_gdf.copy()
                
                # 将0值和空值都设置为NaN，使它们与背景颜色一致
                fishnet_density['Total_Fire_Area_Density'] = fishnet_density['Total_Fire_Area_Density'].replace(0, np.nan).fillna(np.nan)
                
                # 使用与火灾频率相同的颜色映射设置
                full_cmap = plt.get_cmap('YlOrRd', 10)  # 将YlOrRd分成10份，从黄色到红色
                colors = [full_cmap(i) for i in range(4, 10)]  # 截取中间6份
                cmap_fire_density = mcolors.ListedColormap(colors)  # 使用ListedColormap创建分层颜色映射
                
                # 创建边界，将数据范围均匀分成6个区间
                boundaries = np.linspace(fire_density_min, fire_density_max, 7)  # 7个边界点定义6个区间
                norm_fire_density = mcolors.BoundaryNorm(boundaries, cmap_fire_density.N)  # 使用BoundaryNorm实现分层映射
                
                # 创建ScalarMappable对象用于颜色条
                sm_fire_density = plt.cm.ScalarMappable(cmap=cmap_fire_density, norm=norm_fire_density)
                sm_fire_density.set_array([])
                
                # 设置颜色条标签格式为两位小数
                formatter_fire_density = FormatStrFormatter('%.2f')
                
                # 绘制火灾面积密度渔网
                print("Drawing Total Fire Area Density fishnet data...")
                fishnet_density.plot(ax=ax, column='Total_Fire_Area_Density', cmap=cmap_fire_density, norm=norm_fire_density, 
                                     edgecolor='none', linewidth=0)
                print("Total Fire Area Density fishnet data drawing completed")
                
                # 手动创建和配置颜色条，使其高度与地图图框一致
                print("Creating custom colorbar...")
                cbar_ax = fig.add_axes(CBAR_AX_POSITION)
                cbar = fig.colorbar(sm_fire_density, cax=cbar_ax,
                                    orientation='vertical',
                                    label='Normalized burned area')
                
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
                
                # 设置经纬度显示范围
                ax.set_xlim(-180, 180)
                ax.set_ylim(-60, 75)
                
                # 设置网格
                ax.grid(True, linestyle='--', alpha=0.5)
                
                # 使用固定布局，保证Fire frequency和Normalized burned area图幅与内部位置一致
                ax.set_position(MAP_AX_POSITION)
                ax.set_aspect('equal', adjustable='box')
                
                # 保存火灾面积密度图形
                output_filename = f"Normalized_burned_area_{data_year}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                fig.savefig(output_path, dpi=1200, format='jpg')
                print(f"Total Fire Area Density figure successfully saved to: {output_path}")
                
                # 关闭图形，释放内存
                plt.close(fig)
                
    except Exception as e:
        print(f"Error processing Total Fire data for Grid_Clip year {year}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

print("\nAll Total Fire Frequency and Density processing completed!")
