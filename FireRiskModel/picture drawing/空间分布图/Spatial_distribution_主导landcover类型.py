# -*- coding: utf-8 -*-
"""
程序名称：主导landcover类型图片

程序功能：
    1. 从指定路径读取CSV文件，包含2005、2010、2015、2020年的数据
    2. 读取CSV文件中的Top1_Value列，根据字典进行反向匹配获取土地覆盖类型
    3. 如果Top1_Value等于210，则读取Top2_Value进行匹配
    4. 将"TUD","IMP","BAL","PSI","NaN"归为others一类
    5. 从指定GDB文件读取对应年份的渔网栅格数据
    6. 根据土地覆盖类型对渔网栅格进行设色
    7. 为每个年份绘制全球渔网地图
    8. 保存图片到指定路径

输入路径：
    - CSV数据: I:\Processing data\landcover match\Grid_Clip_year_WUI_LandCover_Stats.csv
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

# 定义土地覆盖类型颜色映射
landcover_colors = {
    'CRP': '#FFEB3B',  # 黄色
    'FST': '#2E7D32',  # 深绿色
    'SHR': '#A15C4A',  # 灌木色
    'GRS': '#81C784',  # 草绿色
    'WET': '#2196F3',  # 蓝色
    'Other': '#333333'  # 深灰色
}

landcover_labels = {
    'CRP': 'Cropland (CRP)',
    'FST': 'Forest (FST)',
    'SHR': 'Shrubland (SHR)',
    'GRS': 'Grassland (GRS)',
    'WET': 'Wetland (WET)',
    'Other': 'Other'
}

# 定义输入路径
csv_base_path = r"I:\Processing data\landcover match\Grid_Clip_{year}_WUI_LandCover_Stats.csv"
gdb_path = r"I:\Processing data\gdb集合\WUI.gdb"
gdb_layer_base = "Grid_Clip_{year}_WUI"

# 定义大陆边界数据路径
continent_shp_path = r"I:\Data_huanglin\map\world continent boundary\continent.shp"

# 定义输出路径
output_dir = r"B:\WUI\Picture"
os.makedirs(output_dir, exist_ok=True)

# 定义土地覆盖类型字典
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

# 创建反向字典，将代码映射到土地覆盖类型
code_to_lc = {}
for lc, codes in c_to_lc.items():
    for code in codes:
        code_to_lc[code] = lc

# 需要归为Other的类型
other_types = ["TUD", "IMP", "BAL", "PSI", "NaN"]

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
        if 'GRID_ID' not in csv_data.columns or 'Top1_Value' not in csv_data.columns:
            print(f"错误：{csv_path} 文件中缺少 GRID_ID 或 Top1_Value 列")
            continue
        
        # 处理土地覆盖类型
        def get_landcover_type(row):
            # 首先检查Top1_Value
            top1 = row.get('Top1_Value')
            if top1 == 210:  # 如果是WTR，使用Top2_Value
                top2 = row.get('Top2_Value')
                if pd.notna(top2):
                    lc_type = code_to_lc.get(top2, 'Other')
                    # 检查是否需要归为Other
                    if lc_type in other_types:
                        return 'Other'
                    # 检查是否在我们需要的类型中
                    if lc_type in landcover_colors:
                        return lc_type
                    return 'Other'
            else:
                lc_type = code_to_lc.get(top1, 'Other')
                # 检查是否需要归为Other
                if lc_type in other_types:
                    return 'Other'
                # 检查是否在我们需要的类型中
                if lc_type in landcover_colors:
                    return lc_type
                return 'Other'
        
        csv_data['landcover_type'] = csv_data.apply(get_landcover_type, axis=1)
        
        # 创建GRID_ID到landcover_type的映射
        grid_id_to_landcover = dict(zip(csv_data['GRID_ID'], csv_data['landcover_type']))
        print(f"成功创建 {len(grid_id_to_landcover)} 个GRID_ID到土地覆盖类型的映射")
        
        # 读取渔网数据
        print(f"读取渔网数据: {gdb_path}\{gdb_layer}")
        fishnet = gpd.read_file(gdb_path, layer=gdb_layer)
        print(f"成功读取渔网数据，共 {len(fishnet)} 个要素")
        
        # 检查渔网数据中是否有GRID_ID字段
        if 'GRID_ID' not in fishnet.columns:
            print(f"错误：{gdb_layer} 图层中缺少 GRID_ID 字段")
            continue
        
        # 将土地覆盖类型添加到渔网数据
        fishnet['landcover_type'] = fishnet['GRID_ID'].map(grid_id_to_landcover)
        # 填充缺失值为'Other'
        fishnet['landcover_type'] = fishnet['landcover_type'].fillna('Other')
        
        # 统计各类型的要素数量
        type_counts = fishnet['landcover_type'].value_counts()
        print("各土地覆盖类型要素数量:")
        for type_name, count in type_counts.items():
            print(f"{type_name}: {count}")
        
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
        
        # 按土地覆盖类型分组绘制
        for lc_type, color in landcover_colors.items():
            # 过滤出当前类型的数据
            type_data = fishnet_wgs84[fishnet_wgs84['landcover_type'] == lc_type]
            if not type_data.empty:
                print(f"绘制 {lc_type} 类型，共 {len(type_data)} 个要素")
                type_data.plot(ax=ax, color=color, edgecolor='none', linewidth=0, alpha=0.7)
        
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # 创建图例
        import matplotlib.patches as mpatches
        legend_handles = []
        for lc_type, color in landcover_colors.items():
            patch = mpatches.Patch(color=color, label=landcover_labels.get(lc_type, lc_type))
            legend_handles.append(patch)
        ax.legend(
            handles=legend_handles,
            title='Land cover',
            loc='lower left',
            bbox_to_anchor=(0.0, -0.055),
            title_fontsize=24,
            fontsize=24,
            handlelength=1.0,
            handleheight=0.65,
            labelspacing=0.25,
            frameon=False,
        )
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(output_dir, f"Dominant_LandCover_Type_{year}.png")
        plt.savefig(output_path, dpi=1200, bbox_inches='tight')
        print(f"图片已保存到: {output_path}")
        
        # 关闭图表
        plt.close()
        
    except Exception as e:
        print(f"处理 {year} 年数据时出错: {str(e)}")

print("\n=== 所有年份数据处理完成 ===")
