# -*- coding: utf-8 -*-
"""
程序名称：主导WUI类型图片

程序功能：
    1. 从指定路径读取CSV文件，包含2005、2010、2015、2020年的数据
    2. 读取CSV文件中的main_WUI列，获取每个GRID_ID的主导WUI类型（"face"或"mix"）
    3. 从指定GDB文件读取对应年份的渔网栅格数据
    4. 根据main_WUI类型对渔网栅格进行设色
    5. 为每个年份绘制全球渔网地图
    6. 保存图片到指定路径

输入路径：
    - CSV数据: I:\Processing data\Grid_Clip_year_WUI.csv
    - 渔网数据: I:\Processing data\gdb集合\WUI.gdb

输出：
    - 保存图片到 B:\WUI\Picture
"""

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义年份列表
years = [2005, 2010, 2015, 2020]
years_env = os.environ.get("DOMINANT_MAP_YEARS", "").strip()
if years_env:
    years = [int(part.strip()) for part in years_env.split(",") if part.strip()]

# 定义WUI类型颜色映射
wui_type_colors = {
    'face': '#1F77B4',  # 橙色
    'mix':  '#FF7F00'    # 蓝色
}

wui_type_labels = {
    'face': 'Interface',
    'mix': 'Intermix'
}

# 定义输入路径
csv_base_path = r"I:\Processing data\Grid_Clip_{year}_WUI.csv"
gdb_path = r"I:\Processing data\gdb集合\WUI.gdb"
gdb_layer_base = "Grid_Clip_{year}_WUI"

# 定义大陆边界数据路径
continent_shp_path = r"I:\Data_huanglin\map\world continent boundary\continent.shp"

# 定义输出路径
output_dir = r"B:\WUI\Picture"
os.makedirs(output_dir, exist_ok=True)

# 加载全球地图数据
print(f"Loading global map data: {continent_shp_path}")
try:
    continent = gpd.read_file(continent_shp_path)
    # 显式转换为WGS1984坐标系统
    continent = continent.to_crs("EPSG:4326")
    print(f"Successfully loaded global map with {len(continent)} features")
except Exception as e:
    print(f"Error loading continent data: {str(e)}")
    continent = None

# 处理每个年份
for year in years:
    print(f"\n=== 处理 {year} 年数据 ===")
    
    # 构建文件路径
    csv_path = csv_base_path.format(year=year)
    gdb_layer = gdb_layer_base.format(year=year)
    
    try:
        # 读取CSV文件
        print(f"读取CSV文件: {csv_path}")
        csv_data = pd.read_csv(csv_path)
        
        # 检查CSV文件中是否有必要的列
        if 'GRID_ID' not in csv_data.columns or 'main_WUI' not in csv_data.columns:
            print(f"错误：{csv_path} 文件中缺少 GRID_ID 或 main_WUI 列")
            continue
        
        # 过滤出有效的WUI类型
        csv_data = csv_data[csv_data['main_WUI'].isin(['face', 'mix'])]
        
        # 创建GRID_ID到main_WUI的映射
        grid_id_to_wui_type = dict(zip(csv_data['GRID_ID'], csv_data['main_WUI']))
        print(f"成功创建 {len(grid_id_to_wui_type)} 个GRID_ID到WUI类型的映射")
        
        # 读取渔网数据
        print(f"读取渔网数据: {gdb_path}\{gdb_layer}")
        fishnet = gpd.read_file(gdb_path, layer=gdb_layer)
        print(f"成功读取渔网数据，共 {len(fishnet)} 个要素")
        
        # 检查渔网数据中是否有GRID_ID字段
        if 'GRID_ID' not in fishnet.columns:
            print(f"错误：{gdb_layer} 图层中缺少 GRID_ID 字段")
            continue
        
        # 将WUI类型添加到渔网数据
        fishnet['main_WUI'] = fishnet['GRID_ID'].map(grid_id_to_wui_type)
        
        # 统计有类型的要素数量
        has_type_count = sum(fishnet['main_WUI'].notna())
        print(f"成功映射 {has_type_count} 个要素的WUI类型")
        
        # 转换为WGS 84投影
        print("转换为WGS 84投影...")
        fishnet_wgs84 = fishnet.to_crs(epsg=4326)
        
        # 创建绘图
        print("创建绘图...")
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # 绘制全球地图底图，颜色为浅灰色
        print("绘制全球地图底图...")
        if continent is not None:
            continent.plot(ax=ax, color='#f0f0f0', edgecolor='#d0d0d0', linewidth=0.5)
        
        # 绘制背景
        ax.set_facecolor('white')
        
        # 设置坐标轴范围
        ax.set_xlim([-180, 180])
        ax.set_ylim([-60, 75])
        
        # 按WUI类型分组绘制
        for wui_type, color in wui_type_colors.items():
            # 过滤出当前类型的数据
            type_data = fishnet_wgs84[fishnet_wgs84['main_WUI'] == wui_type]
            if not type_data.empty:
                print(f"绘制 {wui_type} 类型，共 {len(type_data)} 个要素")
                type_data.plot(ax=ax, color=color, edgecolor='none', linewidth=0, alpha=0.7)
        
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # 创建图例
        import matplotlib.patches as mpatches
        legend_handles = []
        for wui_type, color in wui_type_colors.items():
            patch = mpatches.Patch(color=color, label=wui_type_labels.get(wui_type, wui_type))
            legend_handles.append(patch)
        ax.legend(
            handles=legend_handles,
            loc='lower left',
            bbox_to_anchor=(0.0, -0.055),
            fontsize=24,
            handlelength=1.0,
            handleheight=0.65,
            borderpad=0.4,
            labelspacing=0.25,
            handletextpad=0.4,
            frameon=False,
        )
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(output_dir, f"Dominant_WUI_Type_{year}.png")
        plt.savefig(output_path, dpi=1200, bbox_inches='tight')
        print(f"图片已保存到: {output_path}")
        
        # 关闭图表
        plt.close()
        
    except Exception as e:
        print(f"处理 {year} 年数据时出错: {str(e)}")

print("\n=== 所有年份数据处理完成 ===")
