#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
程序名称: WUI总面积空间分布绘制程序
功能简介: 绘制全球范围内2005、2010、2015、2020年的WUI总面积空间分布图

输入文件：
    - 大陆边界数据:I:/Data_huanglin/map/world continent boundary/continent.shp
    - Interface WUI数据:I:/Processing data/gdb集合/Interface_WUI_Fire_Risk.gdb
    - Intermix WUI数据:I:/Processing data/gdb集合/Intermix_WUI_Fire_Risk.gdb

输出文件：
    - 空间分布图:B:/WUI/Picture and statistic results/空间分布图/Total_WUI_area_YYYY.jpg

主要功能：
    1. 计算每个渔网栅格的Interface和Intermix WUI面积总和
    2. 获取所有渔网栅格的并集，确保完整绘制
    3. 使用viridis_r颜色映射和一致的布局样式
    4. 输出高分辨率(1200dpi)的JPG格式地图
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import os

# 设置英文显示，使用Arial字体
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 定义数据路径
continent_shp_path = r"I:\Data_huanglin\map\world continent boundary\continent.shp"

# 定义要处理的数据配置
interface_config = {
    "type": "Interface",
    "gdb_path": r"I:\Processing data\gdb集合\Interface_WUI_Fire_Risk.gdb",
    "years": [2005, 2010, 2015, 2020],
    "layer_prefix": "Grid_Clip_"
}

intermix_config = {
    "type": "Intermix",
    "gdb_path": r"I:\Processing data\gdb集合\Intermix_WUI_Fire_Risk.gdb",
    "years": [2005, 2010, 2015, 2020],
    "layer_prefix": "Grid_Clip_"
}

# 定义输出路径
output_dir = r"B:\WUI\Picture"
# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 加载全球地图数据
print(f"Loading global map data: {continent_shp_path}")
continent = gpd.read_file(continent_shp_path)
# 显式转换为WGS1984坐标系统
continent = continent.to_crs("EPSG:4326")
print(f"Successfully loaded global map with {len(continent)} features")
print(f"Map data CRS: {continent.crs}")

# 循环处理不同年份的数据
for year in interface_config["years"]:
    print(f"\n{'='*50}")
    print(f"Processing Total WUI data for year {year}")
    print(f"{'='*50}")
    
    try:
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
        
        # 合并渔网栅格并计算总面积
        print("Merging fishnet grids and calculating total WUI area...")
        
        # 确保两个数据框都有相同的关键字段
        # 假设关键字段是网格ID或地理坐标
        # 这里使用合并操作来获取所有网格的并集
        total_fishnet = gpd.GeoDataFrame(pd.concat([interface_fishnet, intermix_fishnet], ignore_index=True))
        
        # 按地理坐标或网格ID分组，计算每个网格的总面积
        # 首先确保数据有唯一标识字段
        if 'GRID_ID' not in total_fishnet.columns:
            # 如果没有GRID_ID，尝试使用其他唯一标识字段
            # 或者创建一个基于地理位置的临时标识
            # 这里假设使用网格的面积或其他属性
            total_fishnet['GRID_ID'] = total_fishnet.geometry.apply(lambda x: x.centroid.wkt)
        
        # 分组并计算每个网格的WUI_area总和
        total_wui_area = total_fishnet.groupby('GRID_ID').agg({
            'geometry': 'first',
            'WUI_area': 'sum'
        }).reset_index()
        
        # 创建新的GeoDataFrame
        total_fishnet_gdf = gpd.GeoDataFrame(total_wui_area, geometry='geometry', crs="EPSG:4326")
        print(f"Created total WUI fishnet with {len(total_fishnet_gdf)} features")
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # 绘制全球地图底图，颜色为浅灰色
        print("Drawing global map base layer...")
        continent.plot(ax=ax, color='#f0f0f0', edgecolor='#d0d0d0', linewidth=0.5)
        
        # 配置渔网的颜色映射 - 使用WUI_area字段
        print("Configuring fishnet color mapping...")
        wui_area_min = total_fishnet_gdf['WUI_area'].min()
        wui_area_max = total_fishnet_gdf['WUI_area'].max()
        print(f"Total WUI_area range: {wui_area_min} to {wui_area_max}")
        
        # 创建颜色映射 - 反转颜色映射
        cmap = plt.get_cmap('viridis_r', 10)  # 使用反转的viridis颜色映射，分为10个级别
        norm = mcolors.Normalize(vmin=wui_area_min, vmax=wui_area_max)
        print("Color mapping configuration completed")
        
        # 创建ScalarMappable对象用于颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # 需要设置一个空数组
        
        # 绘制渔网，无框线，使用WUI_area字段进行颜色填充
        print("Drawing total fishnet data...")
        # 先绘制渔网，不直接显示图例
        total_fishnet_gdf.plot(ax=ax, column='WUI_area', cmap=cmap, norm=norm, 
                             edgecolor='none', linewidth=0)  # 不直接显示图例，后续手动创建
        print("Total fishnet data drawing completed")
        
        # 手动创建和配置颜色条，精确控制尺寸
        print("Creating custom colorbar...")
        cbar = plt.colorbar(sm, ax=ax, 
                          shrink=0.465,# 控制颜色条长度，值越小越短
                          aspect=20,  # 控制颜色条的宽高比，值越大越窄
                          fraction=0.02,  # 控制颜色条在图中占用的空间比例
                          pad=0.02,  # 控制颜色条与地图之间的间距
                          orientation='vertical',  # 垂直方向
                          label='Total WUI area (km²)')  # 颜色条标题
        print("Colorbar configuration completed")
        
        # 设置颜色条标签线朝里
        cbar.ax.tick_params(direction='in')
        # 设置颜色条标签字号
        cbar.ax.tick_params(labelsize=14)  # 可根据需要调整字号
        # 设置颜色条标题字号
        cbar.ax.yaxis.label.set_size(16)  # 可根据需要调整字号
        # 移除图形标题
        # ax.set_title('Global Map and Total WUI Fishnet Grid Map', fontsize=16)
        
        # 在地图左上角添加年份文本（无框）
        ax.text(0.02, 0.92, f'{year}', transform=ax.transAxes, fontsize=20, 
               verticalalignment='top')
        
        # 设置坐标轴标签（字体大小增加2号）
        ax.set_xlabel('Longitude', fontsize=16)
        ax.set_ylabel('Latitude', fontsize=16)
        
        # 设置坐标轴刻度字体大小
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        # 设置纬度显示范围为-60到70
        ax.set_ylim(-60, 75)
        
        # 设置网格
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图形
        output_filename = f"Total_WUI_area_{year}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=1200, bbox_inches='tight', format='jpg')
        print(f"Figure successfully saved to: {output_path}")
        
        # 关闭图形，释放内存
        plt.close(fig)
        
    except Exception as e:
        print(f"Error processing Total WUI data for year {year}: {str(e)}")
        continue

print("\nAll processing completed!")
