# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------
# *************************************************************************************

# 程序名称：全球等面积栅格创建与多年份WUI数据处理（优化版）
# 优化点：所有年份共用同一渔网（因边界一致），减少重复计算；输入输出路径分离
# 程序功能：
# 1. 仅创建一次通用等面积栅格（基于统一边界）；
# 2. 批量处理多年份WUI数据，复用同一渔网；
# 3. 各年份分别与通用渔网相交，输出对应有效栅格。
#
# 输入路径：I:\Data_huanglin\WUI global map\NS\WUI shp
# 输出路径：I:\Processing data\WUI_Fire_Risk.gdb
# 输入文件：Time_series_2005/2010/2015/2020_Interface_WUI.shp
# 输出文件：
# - 各年份投影后WUI：Time_series_年份_Interface_WUI_equal_area.shp
# - 通用等面积栅格：GlobalGrid_5min.shp
# - 各年份有效栅格：Grid_Clip_年份_interface_WUI.shp
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
if not arcpy.Exists(output_gdb):
    arcpy.CreateFileGDB_management(output_dir, "WUI_Fire_Risk1.gdb")

# 环境设置
arcpy.env.workspace = input_dir
arcpy.env.parallelProcessingFactor = "100%"
arcpy.env.overwriteOutput = True  # 允许覆盖输出文件

# 定义处理年份
years = [2005,2010,2015,2020]
# 通用渔网路径（存储于GDB中）
common_fishnet = os.path.join(output_gdb, "GlobalGrid_5min_interface_WUI")

# ---------------------
# 第一步：创建通用渔网（仅一次，基于参考年份WUI边界）
# ---------------------
print("===== 开始创建通用渔网（仅执行一次） =====")
ref_year = years[0]  # 以第一个年份为参考获取边界

# 参考年份输入WUI路径（拼接输入文件夹和文件名）
ref_input_shp = "Time_series_{0}_Interface_WUI.shp".format(ref_year)

# 参考年份投影后WUI路径（存储到GDB中）
ref_output_shp = os.path.join(output_gdb, "Time_series_{0}_Interface_WUI_equal_area".format(ref_year))

# 投影参考年份WUI到等面积坐标系
desc_ref = arcpy.Describe(ref_input_shp)
ref_spatial_ref = desc_ref.spatialReference
print(ref_spatial_ref)
if ref_spatial_ref.factoryCode != 54034:
    arcpy.management.Project(ref_input_shp, ref_output_shp, arcpy.SpatialReference(54034))
    print("参考年份{0}数据已投影到等面积坐标系，保存至GDB".format(ref_year))
else:
    ref_output_shp = ref_input_shp  # 已为目标坐标系，直接使用

# 获取统一边界（所有年份共用）
desc_ref = arcpy.Describe(ref_output_shp)
extent = desc_ref.extent
origin_coord = "{0} {1}".format(extent.XMin, extent.YMin)  # 左下角起点
y_axis_coord = "{0} {1}".format(extent.XMin, extent.YMin + 1)  # Y轴方向点
corner_coord = "{0} {1}".format(extent.XMax, extent.YMax)  # 右上角终点

# 创建通用渔网（存储到GDB中）
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

# 给通用渔网添加GRID_ID（唯一标识）
arcpy.AddField_management(common_fishnet, "GRID_ID", "LONG")
arcpy.CalculateField_management(
    in_table=common_fishnet,
    field="GRID_ID",
    expression="!OID!",
    expression_type="PYTHON"
)
print("通用渔网创建完成，保存至GDB：{0}（含GRID_ID）".format(common_fishnet))

# ---------------------
# 第二步：循环处理各年份WUI数据（输入从input_dir，输出到GDB）
# ---------------------
for year in years:
    print("\n===== 开始处理 {0} 年数据 =====".format(year))

    # 1. 拼接当前年份的文件路径（输入为shapefile，输出为GDB要素类）
    input_shp = os.path.join(input_dir, "Time_series_{0}_Interface_WUI.shp".format(year))  # 输入shapefile路径
    output_shp = os.path.join(output_gdb, "Time_series_{0}_Interface_WUI_equal_area".format(year))  # 投影后GDB要素类
    spatial_join_output = os.path.join(output_gdb, "Fishnet_SpatialJoin_WUI_{0}".format(year))  # 空间连接GDB要素类
    final_output = os.path.join(output_gdb, "Grid_Clip_{0}_interface_WUI".format(year))  # 最终有效栅格GDB要素类

    # 2. 投影当前年份WUI数据
    desc = arcpy.Describe(input_shp)
    spatial_ref = desc.spatialReference
    print("{0}年WUI原始坐标系代码：{1}".format(year, spatial_ref.factoryCode))

    if spatial_ref.factoryCode != 54034:
        arcpy.management.Project(input_shp, output_shp, arcpy.SpatialReference(54034))
        print("{0}年数据已投影到等面积坐标系，保存至GDB".format(year))
    else:
        output_shp = input_shp  # 已为目标坐标系，直接使用
        print("{0}年数据已是等面积坐标系，无需转换".format(year))

    # 3. 复用通用渔网，与当前年份WUI进行空间连接
    arcpy.SpatialJoin_analysis(
        target_features=common_fishnet,  # 复用通用渔网（GDB中）
        join_features=output_shp,  # 当前年份WUI（投影后，GDB或shapefile）
        out_feature_class=spatial_join_output,
        join_operation="JOIN_ONE_TO_ONE",
        join_type="KEEP_ALL",
        match_option="INTERSECT"
    )

    # 4. 筛选有效渔网（与WUI重叠）并保存
    arcpy.MakeFeatureLayer_management(spatial_join_output, "temp_sj_layer")
    arcpy.SelectLayerByAttribute_management(
        "temp_sj_layer",
        "NEW_SELECTION",
        "Join_Count > 0"
    )
    arcpy.CopyFeatures_management("temp_sj_layer", final_output)
    print("{0}年处理完成，有效栅格保存至GDB：{1}".format(year, final_output))

print("\n===== 所有年份数据处理完毕 =====")
print("输入文件路径：{0}".format(input_dir))
print("输出GDB路径：{0}".format(output_gdb))
print("通用渔网路径（GDB内）：{0}".format(common_fishnet))