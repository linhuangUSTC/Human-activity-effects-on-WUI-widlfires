#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
程序名称: WUI面积空间分布绘制程序
功能简介: 本程序用于绘制全球范围内不同年份的WUI(Wildland Urban Interface)面积空间分布图
         1. 支持分别绘制Interface和Intermix两种WUI类型的面积分布
         2. 处理2005、2010、2015、2020四个年份的数据
         3. 使用统一的颜色映射(viridis_r)和布局样式
         4. 输出高分辨率(1200dpi)的JPG格式地图
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# 设置英文显示，使用Arial字体
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 定义数据路径
continent_shp_path = r"I:\Data_huanglin\map\world continent boundary\continent.shp"

# 定义要处理的数据配置
wui_data_configs = [
    {
        "type": "Interface",
        "gdb_path": r"I:\Processing data\gdb集合\Interface_WUI_Fire_Risk.gdb",
        "years": [2005, 2010, 2015, 2020],
        "layer_prefix": "Grid_Clip_"
    },
    {
        "type": "Intermix",
        "gdb_path": r"I:\Processing data\gdb集合\Intermix_WUI_Fire_Risk.gdb",
        "years": [2005, 2010, 2015, 2020],
        "layer_prefix": "Grid_Clip_"
    }
]

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

# 循环处理不同类型和年份的数据
for config in wui_data_configs:
    wui_type = config["type"]
    gdb_path = config["gdb_path"]
    years = config["years"]
    layer_prefix = config["layer_prefix"]
    
    for year in years:
        fishnet_layer = f"{layer_prefix}{year}_{wui_type.lower()}_WUI"
        
        print(f"\n{'='*50}")
        print(f"Processing {wui_type} WUI data for year {year}")
        print(f"{'='*50}")
        
        try:
            # 加载渔网数据（加载所有要素）
            print(f"Loading fishnet data layer: {fishnet_layer} (all features)")
            fishnet = gpd.read_file(gdb_path, layer=fishnet_layer)  # 移除rows参数，加载所有要素
            print(f"Successfully loaded fishnet data with {len(fishnet)} features")
            print(f"Fishnet data fields: {list(fishnet.columns)}")
            print(f"Fishnet data original CRS: {fishnet.crs}")

            # 转换渔网数据的CRS为WGS1984
            print("Unifying coordinate systems...")
            fishnet = fishnet.to_crs("EPSG:4326")
            print(f"Fishnet data CRS after conversion: {fishnet.crs}")

            # 创建图形
            fig, ax = plt.subplots(figsize=(15, 10))

            # 绘制全球地图底图，颜色为浅灰色
            print("Drawing global map base layer...")
            continent.plot(ax=ax, color='#f0f0f0', edgecolor='#d0d0d0', linewidth=0.5)

            # 配置渔网的颜色映射 - 使用WUI_area字段
            print("Configuring fishnet color mapping...")
            wui_area_min = fishnet['WUI_area'].min()
            wui_area_max = fishnet['WUI_area'].max()
            print(f"WUI_area range: {wui_area_min} to {wui_area_max}")

            # 创建颜色映射 - 反转颜色映射
            cmap = plt.get_cmap('viridis_r', 10)  # 使用反转的viridis颜色映射，分为10个级别
            norm = mcolors.Normalize(vmin=wui_area_min, vmax=wui_area_max)
            print("Color mapping configuration completed")

            # 创建ScalarMappable对象用于颜色条
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # 需要设置一个空数组

            # 绘制渔网，无框线，使用WUI_area字段进行颜色填充
            print("Drawing fishnet data...")
            # 先绘制渔网，不直接显示图例
            fishnet.plot(ax=ax, column='WUI_area', cmap=cmap, norm=norm, 
                         edgecolor='none', linewidth=0)  # 不直接显示图例，后续手动创建
            print("Fishnet data drawing completed")

            # 手动创建和配置颜色条，精确控制尺寸
            print("Creating custom colorbar...")
            cbar = plt.colorbar(sm, ax=ax, 
                              shrink=0.465,# 控制颜色条长度，值越小越短
                              aspect=20,  # 控制颜色条的宽高比，值越大越窄
                              fraction=0.02,  # 控制颜色条在图中占用的空间比例
                              pad=0.02,  # 控制颜色条与地图之间的间距
                              orientation='vertical',  # 垂直方向
                              label=f'{wui_type} WUI area (km²)')  # 颜色条标题
            print("Colorbar configuration completed")
            
            # 设置颜色条标签线朝里
            cbar.ax.tick_params(direction='in')
            # 设置颜色条标签字号
            cbar.ax.tick_params(labelsize=14)  # 可根据需要调整字号
            # 设置颜色条标题字号
            cbar.ax.yaxis.label.set_size(16)  # 可根据需要调整字号
            # 移除图形标题
            # ax.set_title('Global Map and Interface WUI Fishnet Grid Map', fontsize=16)

            # 在地图左上角添加年份文本（无框）
            ax.text(0.02, 0.92, f'{year}', transform=ax.transAxes, fontsize=20, 
                   verticalalignment='top')

            # 设置坐标轴标签（字体大小增加2号）
            ax.set_xlabel('Longitude', fontsize=16)
            ax.set_ylabel('Latitude', fontsize=16)

            # 设置坐标轴刻度字体大小
            ax.tick_params(axis='both', which='major', labelsize=16)

            # 设置纬度显示范围为-60到75
            ax.set_ylim(-60, 75)

            # 设置网格
            ax.grid(True, linestyle='--', alpha=0.5)

            # 调整布局
            plt.tight_layout()

            # 保存图形
            output_filename = f"{wui_type}_WUI_area_{year}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, dpi=1200, bbox_inches='tight', format='jpg')
            print(f"Figure successfully saved to: {output_path}")

            # 关闭图形，释放内存
            plt.close(fig)
            
        except Exception as e:
            print(f"Error processing {wui_type} WUI data for year {year}: {str(e)}")
            continue

print("\nAll processing completed!")
