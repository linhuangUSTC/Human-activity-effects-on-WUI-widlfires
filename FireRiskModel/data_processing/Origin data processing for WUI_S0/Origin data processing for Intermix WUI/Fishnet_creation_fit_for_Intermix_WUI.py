# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------
# *************************************************************************************

# 程序名称：全球等面积栅格创建与多年份WUI数据处理（GDB存储优化版）
# 优化点：所有输出存储于统一GDB，管理更集中；保留原有功能逻辑
# 程序功能：
# 1. 仅创建一次通用等面积栅格（基于统一边界）；
# 2. 批量处理多年份WUI数据，复用同一渔网；
# 3. 各年份分别与通用渔网相交，输出对应有效栅格至GDB。
#
# 输入路径：I:\Data_huanglin\WUI global map\Nature Sustainability-Wildfire risk for global wildland–urban intermix areas\WUI shp
# 输出路径：I:\Processing data\WUI（含GDB文件）
# 输入文件：Time_series_2005/2010/2015/2020_Intermix_WUI.shp
# 输出文件（存储于WUI＿Fire_Risk.gdb）：
# - 通用等面积栅格：GlobalGrid_5min
# - 各年份投影后WUI：Time_series_年份_Intermix_WUI_equal_area
# - 各年份空间连接结果：Fishnet_SpatialJoin_WUI_年份
# - 各年份有效栅格：Grid_Clip_年份_intermix_WUI
# *************************************************************************************
# -------------------------------------------------------------------------------------

import arcpy
import os  # 用于路径处理
import math

# ---------------------
# 定义输入和输出路径（用户指定）
# ---------------------
input_dir = r"E:\data-hl\Origin data" # 输入文件所在文件夹
output_dir = r"E:\data-hl\Processing data"  # 输出文件夹（GDB将创建于此）

# 输出GDB路径（统一存储所有结果）
output_gdb = os.path.join(output_dir, "WUI_Fire_Risk.gdb")
if not arcpy.Exists(output_gdb):
    arcpy.CreateFileGDB_management(output_dir, "WUI_Fire_Risk.gdb")

# 环境设置
arcpy.env.workspace = input_dir
arcpy.env.parallelProcessingFactor = "100%"
arcpy.env.overwriteOutput = True  # 允许覆盖输出文件

# 定义处理年份
years = [2005, 2010, 2015, 2020]
# 通用渔网路径（存储于GDB中）
common_fishnet = os.path.join(output_gdb, "GlobalGrid_5min")

# ---------------------
# 第一步：创建通用渔网（仅一次，基于参考年份WUI边界）
# ---------------------
print("===== 开始创建通用渔网（仅执行一次） =====")
ref_year = years[0]  # 以第一个年份为参考获取边界

# 参考年份输入WUI路径（拼接输入文件夹和文件名）
ref_input_shp = "Time_series_{0}_Intermix_WUI.shp".format(ref_year)

# 参考年份投影后WUI路径（存储到GDB中）
ref_output_shp = os.path.join(output_gdb, "Time_series_{0}_Intermix_WUI_equal_area".format(ref_year))

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
    input_shp = os.path.join(input_dir, "Time_series_{0}_Intermix_WUI.shp".format(year))  # 输入shapefile路径
    output_shp = os.path.join(output_gdb, "Time_series_{0}_Intermix_WUI_equal_area".format(year))  # 投影后GDB要素类
    final_output = os.path.join(output_gdb, "Grid_Clip_{0}_intermix_WUI".format(year))  # 最终有效栅格GDB要素类

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

    # 3. 基于分带选择的内存友好流程：仅将与WUI相交的渔网单元追加输出
    # 创建输出（空要素类，结构同渔网）
    if arcpy.Exists(final_output):
        arcpy.management.Delete(final_output)
    arcpy.MakeFeatureLayer_management(common_fishnet, "fishnet_lyr_all")
    # 复制空结构作为输出
    arcpy.SelectLayerByAttribute_management("fishnet_lyr_all", "NEW_SELECTION", "1 = 2")
    arcpy.CopyFeatures_management("fishnet_lyr_all", final_output)
    arcpy.SelectLayerByAttribute_management("fishnet_lyr_all", "CLEAR_SELECTION")

    # 图层和空间索引
    arcpy.MakeFeatureLayer_management(common_fishnet, "fishnet_lyr")
    arcpy.MakeFeatureLayer_management(output_shp, "wui_lyr")
    try:
        arcpy.AddSpatialIndex_management("fishnet_lyr")
    except Exception:
        pass
    try:
        arcpy.AddSpatialIndex_management("wui_lyr")
    except Exception:
        pass

    # 计算单元高度以对齐分带
    cell_height_val = None
    with arcpy.da.SearchCursor(common_fishnet, ["SHAPE@"], spatial_reference=arcpy.Describe(common_fishnet).spatialReference) as cur:
        for row in cur:
            cell_height_val = row[0].extent.height
            break
    if not cell_height_val:
        raise RuntimeError("无法获取渔网单元高度")

    # 获取渔网总体范围
    fish_desc = arcpy.Describe(common_fishnet)
    fish_extent = fish_desc.extent
    xmin, xmax = fish_extent.XMin, fish_extent.XMax
    ymin, ymax = fish_extent.YMin, fish_extent.YMax

    # 自适应分带高度（对齐到单元高度的整数倍）
    total_height = ymax - ymin
    # 目标带数（可依据内存情况调整），保证至少2带
    target_bands = 20
    approx_band_h = max(total_height / float(target_bands), cell_height_val)
    band_h = math.ceil(approx_band_h / float(cell_height_val)) * float(cell_height_val)

    sr = fish_desc.spatialReference

    # 逐带处理，避免一次性相交导致内存溢出
    cur_y = ymin
    band_idx = 0
    while cur_y < ymax - 1e-6:
        band_idx += 1
        top_y = min(cur_y + band_h, ymax)
        # 构造带状多边形
        band_array = arcpy.Array([
            arcpy.Point(xmin, cur_y),
            arcpy.Point(xmax, cur_y),
            arcpy.Point(xmax, top_y),
            arcpy.Point(xmin, top_y),
            arcpy.Point(xmin, cur_y)
        ])
        band_geom = arcpy.Polygon(band_array, sr)

        # 按带限制候选范围
        arcpy.SelectLayerByAttribute_management("fishnet_lyr", "CLEAR_SELECTION")
        arcpy.SelectLayerByAttribute_management("wui_lyr", "CLEAR_SELECTION")

        # 使用“中心点在带内”确保每个网格只落入一个带，避免重复
        arcpy.SelectLayerByLocation_management("fishnet_lyr", "HAVE_THEIR_CENTER_IN", band_geom, selection_type="NEW_SELECTION")
        fish_count = int(arcpy.GetCount_management("fishnet_lyr").getOutput(0))
        if fish_count == 0:
            cur_y = top_y
            continue

        arcpy.SelectLayerByLocation_management("wui_lyr", "INTERSECT", band_geom, selection_type="NEW_SELECTION")
        wui_count = int(arcpy.GetCount_management("wui_lyr").getOutput(0))
        if wui_count == 0:
            cur_y = top_y
            continue

        # 在当前带内，将与WUI相交的渔网单元选出
        arcpy.SelectLayerByLocation_management("fishnet_lyr", "INTERSECT", "wui_lyr", selection_type="SUBSET_SELECTION")
        sel_count = int(arcpy.GetCount_management("fishnet_lyr").getOutput(0))
        if sel_count > 0:
            arcpy.Append_management("fishnet_lyr", final_output, "NO_TEST")
            print("  带 {0}: 追加 {1} 个单元".format(band_idx, sel_count))

        # 下一带
        cur_y = top_y

    # 带处理完成后，按 GRID_ID 去重，防止极端情况下仍有重复
    try:
        arcpy.management.DeleteIdentical(final_output, ["GRID_ID"])  # 保留最小OID的一条
    except Exception:
        # 如果DeleteIdentical不可用，可退化为Dissolve
        temp_dissolve = final_output + "_diss"
        arcpy.management.Dissolve(final_output, temp_dissolve, ["GRID_ID"]) 
        arcpy.management.Delete(final_output)
        arcpy.management.CopyFeatures(temp_dissolve, final_output)
        arcpy.management.Delete(temp_dissolve)

    # 清理选择
    arcpy.SelectLayerByAttribute_management("fishnet_lyr", "CLEAR_SELECTION")
    arcpy.SelectLayerByAttribute_management("wui_lyr", "CLEAR_SELECTION")

    print("{0}年处理完成，有效栅格保存至GDB：{1}".format(year, final_output))

print("\n===== 所有年份数据处理完毕 =====")
print("输入文件路径：{0}".format(input_dir))
print("输出GDB路径：{0}".format(output_gdb))
print("通用渔网路径（GDB内）：{0}".format(common_fishnet))
