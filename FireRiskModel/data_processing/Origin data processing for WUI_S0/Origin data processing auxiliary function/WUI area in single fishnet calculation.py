# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------------
# 程序名称：渔网内WUI面积统计
# 功能：计算每个有效渔网（Grid_Clip_WUI_2005.shp）中重叠的WUI面积
# 输入：
#   - 有效渔网：Grid_Clip_WUI_2005.shp（含GRID_ID唯一标识）
#   - WUI数据：Time_series_2005_Interface_WUI_equal_area.shp（等面积投影）
# 输出：
#   - 带WUI面积的渔网：Grid_Clip_WUI_with_area_2005.shp（新增WUI_Area字段，单位：平方米）
#-------------------------------------------------------------------------------------

import arcpy

# 环境设置
arcpy.env.workspace = r"I:\test"
arcpy.env.overwriteOutput = True
arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(54034)  # 等面积投影

# 输入数据
grid_clip = "GlobalGrid_5min_Clip_2005_interface_WUI.shp"  # 渔网（含GRID_ID，面积~80km²）
wui_equal_area = "China_2005_interface_WUI_equal_area.shp"  # WUI数据

# 输出数据
output_grid = "GlobalGrid_5min_Clip_2005_interface_WUI_with_area_statistic.shp"  # 最终结果
temp_intersect = "temp_intersect.shp"  # 渔网与WUI的重叠区域（关键！）
temp_summary = "temp_summary.dbf"  # 汇总临时表


# ----------------------
# 步骤1：相交分析，提取渔网内部的WUI部分（核心修正）
# 生成“渔网和WUI重叠的区域”，这些区域才是真正属于该渔网的WUI
# ----------------------
print("步骤1：计算渔网与WUI的重叠区域...")
arcpy.Intersect_analysis(
    in_features=[grid_clip, wui_equal_area],  # 输入：渔网 + WUI
    out_feature_class=temp_intersect,         # 输出：重叠区域（仅渔网内部的WUI部分）
    join_attributes="ALL",                    # 保留双方属性（含渔网GRID_ID）
    cluster_tolerance="0.001 Meters"          # 确保微小重叠被捕捉
)


# ----------------------
# 步骤2：计算重叠区域的面积（单位：平方千米）
# 这些面积是WUI在渔网内部的真实部分，不会超过渔网面积
# ----------------------
print("步骤2：计算重叠区域的面积（平方千米）...")
arcpy.AddField_management(temp_intersect, "WUI_Area", "DOUBLE")
arcpy.CalculateField_management(
    temp_intersect,
    "WUI_Area",
    "!shape.area! / 1000000",  # 等面积投影下，shape.area单位是平方米，转平方千米
    "PYTHON"
)


# ----------------------
# 步骤3：按渔网GRID_ID汇总重叠面积（每个渔网的真实WUI面积）
# ----------------------
print("步骤3：按渔网ID汇总总面积...")
arcpy.Statistics_analysis(
    in_table=temp_intersect,
    out_table=temp_summary,
    statistics_fields=[["WUI_Area", "SUM"]],  # 生成SUM_WUI_Area_km2
    case_field="GRID_ID"  # 按渔网ID分组
)

# ----------------------
# 步骤4：先将渔网转为图层，再连接表（核心修正）
# ----------------------
print("步骤4：将面积数据关联到渔网...")
# 1. 将渔网要素类转为临时图层（AddJoin要求输入为图层）
grid_layer = "grid_layer"  # 临时图层名称
arcpy.MakeFeatureLayer_management(grid_clip, grid_layer)

# 2. 用图层与汇总表连接
arcpy.AddJoin_management(
    in_layer_or_view=grid_layer,  # 输入图层（而非要素类路径）
    in_field="GRID_ID",
    join_table=temp_summary,
    join_field="GRID_ID",
    join_type="KEEP_ALL"
)


# ----------------------
# 步骤5：从图层导出最终结果（保留连接的面积字段）
# ----------------------
print("步骤5：导出带面积字段的最终渔网...")
arcpy.CopyFeatures_management(
    in_features=grid_layer,  # 从图层导出（而非原始要素类）
    out_feature_class=output_grid
)

print("完成！最终渔网属性表中新增'WUI_Area'字段（单位：平方千米）")