# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------
# 程序名称：WUI过火面积渔网统计工具
# 功能说明：
# 1. 对WUI（城乡结合部）、过火面积边界进行等面积投影统一（渔网直接用原始数据，不投影）
# 2. 提取WUI范围内的过火面积区域，并拆分跨渔网栅格的过火斑块
# 3. 按GRID_ID统计面积总和，通过连接字段关联到原始渔网，转换为km²单位
# --------------------------------------------------------------------------------

import arcpy
import os

# 环境设置
arcpy.env.overwriteOutput = True
workspace = r"I:\Clip shapefile of CHINA for test"
arcpy.env.workspace = workspace

# 输入数据（直接使用原始渔网，不做投影）
wui_shp = os.path.join(workspace, "China_2005_interface_WUI.shp")
fishnet = os.path.join(workspace, "GlobalGrid_5min_Clip_2005_interface_WUI.shp")  # 原始渔网，不投影
fire_perimeter = os.path.join(workspace, "GFA_v20240409_perimeters_2005.shp")

# 输出GDB及路径
output_gdb = os.path.join(workspace, "FireAnalysisOutput.gdb")
if not arcpy.Exists(output_gdb):
    arcpy.CreateFileGDB_management(workspace, "FireAnalysisOutput.gdb")

# 临时数据路径（删除fishnet_proj，直接用原始fishnet）
wui_proj = os.path.join(output_gdb, "WUI_Proj")  # 投影后的WUI
fire_perim_proj = os.path.join(output_gdb, "Fire_Perim_Proj")  # 投影后的过火面积
wui_fire_intersect = os.path.join(output_gdb, "WUI_Fire_Intersect")  # WUI与过火面积相交结果
split_fire = os.path.join(output_gdb, "Split_Fire_By_Fishnet")  # 拆分后的过火斑块
final_fishnet = os.path.join(output_gdb, "GlobalGrid_With_FireArea")  # 最终结果（带Fire_area字段的原始渔网）
stats_table = os.path.join(output_gdb, "FireArea_Stats")  # 面积统计临时表

# 等面积投影（仅用于WUI和过火面积，渔网不投影）
target_sr = arcpy.SpatialReference(54034)

# ----------------------
# 步骤1：投影WUI数据到等面积坐标系
# ----------------------
print("步骤1：将WUI数据投影到等面积坐标系...")
desc_wui = arcpy.Describe(wui_shp)
if desc_wui.spatialReference.factoryCode != target_sr.factoryCode:
    arcpy.Project_management(wui_shp, wui_proj, target_sr)
    print("WUI数据已投影至等面积坐标系")
else:
    wui_proj = wui_shp
    print("WUI数据已是等面积坐标系，无需投影")

# ----------------------
# 步骤2：投影过火面积数据到等面积坐标系
# ----------------------
print("步骤2：将过火面积数据投影到等面积坐标系...")
desc_fire = arcpy.Describe(fire_perimeter)
if desc_fire.spatialReference.factoryCode != target_sr.factoryCode:
    arcpy.Project_management(fire_perimeter, fire_perim_proj, target_sr)
    print("过火面积数据已投影至等面积坐标系")
else:
    fire_perim_proj = fire_perimeter
    print("过火面积数据已是等面积坐标系，无需投影")

# ----------------------
# 步骤3：WUI与过火面积相交（提取WUI内的过火面积）
# ----------------------
print("步骤3：计算等面积坐标系下WUI与过火面积的相交区域...")
arcpy.Intersect_analysis(
    in_features=[wui_proj, fire_perim_proj],
    out_feature_class=wui_fire_intersect,
    join_attributes="ALL",
    cluster_tolerance="0.001 Meters"
)
print("相交结果已保存：{0}".format(wui_fire_intersect))

# ----------------------
# 步骤4：拆分过火斑块
# ----------------------
print("步骤4：拆分跨栅格的过火斑块，确保每个斑块仅属于一个渔网...")
# 用原始渔网与WUI内过火区相交（不投影渔网，直接使用）
arcpy.Intersect_analysis(
    in_features=[fishnet, wui_fire_intersect],  # 输入原始渔网，无需投影
    out_feature_class=split_fire,
    join_attributes="ALL",
    cluster_tolerance="0.001 Meters"
)
print("过火斑块拆分完成：{0}".format(split_fire))

# ----------------------
# 步骤5：按GRID_ID统计面积（生成stats_table）
# ----------------------
print("步骤5：按GRID_ID统计过火面积总和...")
# 统计split_fire中每个GRID_ID的SHAPE_Area总和（生成SUM_SHAPE_Area字段，全大写）
arcpy.Statistics_analysis(
    in_table=split_fire,
    out_table=stats_table,
    statistics_fields=[["SHAPE_Area", "SUM"]],
    case_field="GRID_ID"
)

# ----------------------
# 步骤6：连接字段（将统计结果关联到原始渔网）
# ----------------------
print("步骤6：将统计结果连接到原始渔网...")
# 1. 复制原始渔网作为最终结果（避免修改原始数据）
arcpy.CopyFeatures_management(fishnet, final_fishnet)

# 2. 新增Fire_area字段（用于存储km²单位的过火面积）
arcpy.AddField_management(final_fishnet, "Fire_area", "DOUBLE")

# 3. 关键：通过GRID_ID连接渔网和统计表（核心连接步骤）
arcpy.JoinField_management(
    in_data=final_fishnet,        # 目标：复制后的原始渔网
    in_field="GRID_ID",           # 渔网的连接关键字段
    join_table=stats_table,       # 源：面积统计临时表
    join_field="GRID_ID",         # 统计表的连接关键字段
    fields=["SUM_SHAPE_Area"]     # 要关联的求和字段（全大写，与统计表一致）
)

# 4. 字段计算器：将SUM_SHAPE_Area（平方米）转换为km²，空值补0
arcpy.CalculateField_management(
    in_table=final_fishnet,
    field="Fire_area",
    expression="!SUM_SHAPE_Area! / 1000000 if !SUM_SHAPE_Area! is not None else 0",
    expression_type="PYTHON"
)

# 5. 删除临时关联字段（SUM_SHAPE_Area不再需要）
if "SUM_SHAPE_Area" in [f.name for f in arcpy.ListFields(final_fishnet)]:
    arcpy.DeleteField_management(final_fishnet, "SUM_SHAPE_Area")

print("连接及单位转换完成！")
print("最终结果路径：{0}".format(final_fishnet))

# ----------------------
# 清理临时文件（可选，保留注释方便调试）
# ----------------------
temp_files = [wui_proj, fire_perim_proj, wui_fire_intersect, split_fire, stats_table]
for temp in temp_files:
    if arcpy.Exists(temp):
      arcpy.Delete_management(temp)

print("\n处理完成!")