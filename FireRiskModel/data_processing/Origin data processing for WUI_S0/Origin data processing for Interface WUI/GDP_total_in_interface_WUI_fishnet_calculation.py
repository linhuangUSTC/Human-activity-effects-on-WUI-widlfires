# -- coding: utf-8 --
# GDP总量计算程序

import arcpy
from arcpy.sa import *
import os

# 步骤1：定义输入输出路径
gdp_raster_path = r"I:\Data_huanglin\GDP\Global GDP 多分辨率 1990-2022\16741980\rast_gdpTot_1990_2020_30arcsec.tif"
fishnet_gdb_path = r"I:\Processing data\WUI_Fire_Risk.gdb"
output_gdb = fishnet_gdb_path

# 使用WKID创建目标投影
target_wkid = 54034  # World_Cylindrical_Equal_Area
target_projection = arcpy.SpatialReference()
target_projection.factoryCode = target_wkid
target_projection.create()

# 设置处理环境
arcpy.env.workspace = fishnet_gdb_path
arcpy.env.overwriteOutput = True
arcpy.env.parallelProcessingFactor = "100%"

# 时间对应关系：WUI年份→GDP年份（直接对应，不使用范围）
year_mapping = {
    2005: [2005],   # 2005年GDP→2005年WUI
    2010: [2010],   # 2010年GDP→2010年WUI
    2015: [2015],   # 2015年GDP→2015年WUI
    2020: [2020]    # 2020年GDP→2020年WUI
}

# GDP栅格文件的波段与年份对应关系（假设波段顺序为1990,1995,2000,2005,2010,2015,2020）
# 实际使用时可能需要根据栅格文件的实际波段信息调整
gdp_band_year_mapping = {
    1: 1990,
    2: 1995,
    3: 2000,
    4: 2005,
    5: 2010,
    6: 2015,
    7: 2020
}

# 步骤2：读取GDP栅格文件信息
print("步骤2：读取GDP栅格文件信息")

# 获取栅格信息
print("  读取GDP栅格信息...")
gdp_desc = arcpy.Describe(gdp_raster_path)
gdp_sr = gdp_desc.spatialReference
print("  栅格波段数量：%d" % gdp_desc.bandCount)

# 步骤3：先将整个GDP多波段栅格投影到54034坐标系
print("\n步骤3：将整个GDP多波段栅格投影到目标坐标系54034")

# 投影后的完整GDP栅格路径
projected_full_gdp_raster = os.path.join(output_gdb, "gdp_full_projected_54034")

# 创建WGS84空间参考对象
wgs84 = arcpy.SpatialReference(4326)

# 检查投影后的栅格是否已存在，避免重复投影
if arcpy.Exists(projected_full_gdp_raster):
    print("  投影后的完整GDP栅格已存在，直接使用")
else:
    print("  开始投影整个GDP多波段栅格到54034坐标系...")
    arcpy.ProjectRaster_management(
        in_raster=gdp_raster_path,            # in_raster - 整个多波段栅格
        out_raster=projected_full_gdp_raster, # out_raster - 输出到gdb中的栅格
        out_coor_system=target_projection,    # out_coor_system - 目标坐标系
        resampling_type="BILINEAR",           # resampling_type
        cell_size="1000",                    # cell_size
    )
    print("  整个GDP多波段栅格投影完成")

# 步骤4：按照时间顺序处理数据
print("\n步骤4：按照时间顺序处理数据")

# 按WUI年份升序排序
for wui_year in sorted(year_mapping.keys()):
    gdp_years = year_mapping[wui_year]
    print("\n--- 处理WUI年份 %d，对应GDP年份：%s ---" % (wui_year, list(gdp_years)))

    # 获取对应年份的WUI渔网
    wui_fishnet_path = os.path.join(output_gdb, "Grid_Clip_%d_interface_WUI" % wui_year)
    print("  读取WUI渔网：%s" % os.path.basename(wui_fishnet_path))
    # 设置渔网为分析对象
    fishnet_for_analysis = wui_fishnet_path
    # 获取渔网的GRID_ID字段（确认存在）
    zone_field = "GRID_ID"
    
    # 确保人口年份按升序处理
    for gdp_year in sorted(gdp_years):
        print("\n  处理GDP年份 %d → WUI年份 %d" % (gdp_year, wui_year))
        
        # 找到对应年份的波段号
        band_number = None
        for band, year in gdp_band_year_mapping.items():
            if year == gdp_year:
                band_number = band
                break
        
        # 步骤4.1：从已投影的完整栅格中提取特定波段
        print("  步骤4.1：从已投影的完整栅格中提取特定波段")
        # 使用MakeRasterLayer工具创建特定波段的栅格图层
        band_layer_name = "temp_band_layer_%d" % band_number
        arcpy.MakeRasterLayer_management(
            in_raster=projected_full_gdp_raster,
            out_rasterlayer=band_layer_name,
            band_index=str(band_number)
        )
        gdp_raster_for_analysis = band_layer_name
        print("  成功创建波段 %d 的栅格图层" % band_number)
        
        # 步骤4.2：根据渔网裁剪GDP栅格
        print("  步骤4.2：根据WUI渔网范围裁剪GDP栅格")
        clipped_gdp_raster = os.path.join(output_gdb, "clipped_gdp_%d" % gdp_year)
        if arcpy.Exists(clipped_gdp_raster):
            arcpy.Delete_management(clipped_gdp_raster)
        arcpy.env.extent = arcpy.Describe(fishnet_for_analysis).extent
        arcpy.Clip_management(gdp_raster_for_analysis, "#", clipped_gdp_raster, fishnet_for_analysis, "#", "NONE")
        
        # 步骤4.3：计算渔网内GDP总量
        print("  步骤4.3：计算WUI渔网内GDP总量")
        temp_zonal_table = os.path.join(output_gdb, "temp_zonal_stats_gdp_%d" % gdp_year)
        if arcpy.Exists(temp_zonal_table):
            arcpy.Delete_management(temp_zonal_table)
        
        # 清理可能存在的旧字段
        gdp_total_field = "GDP_%d" % gdp_year
        if gdp_total_field in [f.name for f in arcpy.ListFields(fishnet_for_analysis)]:
            arcpy.DeleteField_management(fishnet_for_analysis, gdp_total_field)
        
        # 执行分区统计并连接结果
        arcpy.sa.ZonalStatisticsAsTable(fishnet_for_analysis, zone_field, clipped_gdp_raster, temp_zonal_table, "DATA", "SUM")
        arcpy.JoinField_management(fishnet_for_analysis, zone_field, temp_zonal_table, zone_field, ["SUM"])
        arcpy.AlterField_management(fishnet_for_analysis, "SUM", gdp_total_field, gdp_total_field)
        
        # 步骤4.4：添加统一的GDP字段（使用最新的GDP数据）
        print("  步骤4.4：更新统一的GDP字段")
        
        # 检查是否存在GDP字段，如果不存在则创建
        if "GDP" not in [f.name for f in arcpy.ListFields(fishnet_for_analysis)]:
            arcpy.AddField_management(fishnet_for_analysis, "GDP", "DOUBLE")
        
        # 更新GDP字段值
        arcpy.CalculateField_management(
            fishnet_for_analysis, 
            "GDP", 
            "!%s! if !%s! is not None else 0" % (gdp_total_field, gdp_total_field), 
            "PYTHON"
        )
        
        # 步骤4.5：清理临时数据
        print("  步骤4.5：清理临时数据")
        # 包含临时创建的栅格图层
        band_layer_name = "temp_band_layer_%d" % band_number
        for temp_data in [temp_zonal_table, clipped_gdp_raster, band_layer_name]:
            if arcpy.Exists(temp_data):
                arcpy.Delete_management(temp_data)
        
        print("  GDP年份 %d 处理完成" % gdp_year)

print("\n所有年份处理完成！")
print("\n生成的字段说明：")
print("- GDP_YYYY（如：GDP_2005）：对应年份的GDP总量")
print("- GDP：更新后的统一GDP字段，使用最新的GDP数据")