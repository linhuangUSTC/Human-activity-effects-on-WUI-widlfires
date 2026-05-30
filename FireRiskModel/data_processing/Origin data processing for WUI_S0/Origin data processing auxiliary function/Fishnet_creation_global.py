# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------
# *************************************************************************************

# 程序名称：全球等面积栅格创建
# 优化点：所有年份共用同一渔网（因边界一致），减少重复计算；输入输出路径分离
# 程序功能：
# 1仅创建一次通用等面积栅格（基于WUI边界）；

#
# 输入路径：I:\Data_huanglin\WUI global map\NS\WUI shp
# 输出路径：I:\Processing data\WUI_Fire_Risk.gdb
# 输出文件：
# - 通用等面积栅格：GlobalGrid_5min.shp
# *************************************************************************************
# -------------------------------------------------------------------------------------

import arcpy
import os  # 用于路径处理

# ---------------------
# 定义输入和输出路径（用户指定）
# ---------------------
input_dir = r"I:\Data_huanglin\WUI global map\NS\WUI shp"  # 输入文件所在文件夹
output_dir = r"I:\Processing data"  # 输出文件保存文件夹
# 输出GDB路径（统一存储所有结果）
output_gdb = os.path.join(output_dir, "WUI_Fire_Risk.gdb")

# 环境设置
arcpy.env.workspace=output_gdb
arcpy.env.parallelProcessingFactor = "100%"
arcpy.env.overwriteOutput = True  # 允许覆盖输出文件

# 通用渔网路径（保存到输出文件夹）
common_fishnet = os.path.join(output_gdb, "GlobalGrid_5min")

# ---------------------
# 第一步：创建通用渔网（仅一次，基于参考年份WUI边界）
# ---------------------
print("===== 开始创建通用渔网（仅执行一次） =====")
ref_year = 2005

# 参考年份输入WUI路径（拼接输入文件夹和文件名）
ref_input_shp =os.path.join(input_dir, "Time_series_{0}_Interface_WUI.shp".format(ref_year))

# 参考年份投影后WUI路径（保存到输出文件夹）
ref_output_shp = os.path.join(output_gdb, "Time_series_{0}_Interface_WUI_equal_area".format(ref_year))
# 投影参考年份WUI到等面积坐标系
desc_ref = arcpy.Describe(ref_input_shp)
ref_spatial_ref = desc_ref.spatialReference
print(ref_spatial_ref)
if ref_spatial_ref.factoryCode != 54034:
    arcpy.management.Project(ref_input_shp, ref_output_shp, arcpy.SpatialReference(54034))
    print("参考年份{0}数据已投影到等面积坐标系，保存至输出文件夹".format(ref_year))
else:
    ref_output_shp = ref_input_shp  # 已为目标坐标系，直接使用

# 获取统一边界（所有年份共用）
desc_ref = arcpy.Describe(ref_output_shp)
extent = desc_ref.extent
origin_coord = "{0} {1}".format(extent.XMin, extent.YMin)  # 左下角起点
y_axis_coord = "{0} {1}".format(extent.XMin, extent.YMin + 1)  # Y轴方向点
corner_coord = "{0} {1}".format(extent.XMax, extent.YMax)  # 右上角终点

# 创建通用渔网（保存到输出文件夹）
arcpy.management.CreateFishnet(
    out_feature_class=common_fishnet,
    origin_coord=origin_coord,
    y_axis_coord=y_axis_coord,
    cell_width=9270,
    cell_height=9270,
    number_rows="",
    number_columns="",
    corner_coord=corner_coord,
    labels="NO_LABELS",
    template=ref_output_shp,
    geometry_type="POLYGON"
)

# 动态取 OID 字段名（可能是 'OBJECTID' 或 'FID'）
oid_field = arcpy.Describe(common_fishnet).OIDFieldName

# 给通用渔网添加GRID_ID（唯一标识）
arcpy.AddField_management(common_fishnet, "GRID_ID", "LONG")
arcpy.CalculateField_management(
    in_table=common_fishnet,
    field="GRID_ID",
    expression="!{}!".format(oid_field),
    expression_type="PYTHON_9.3"
)
print("通用渔网创建完成，保存至：{0}（含GRID_ID）".format(common_fishnet))
