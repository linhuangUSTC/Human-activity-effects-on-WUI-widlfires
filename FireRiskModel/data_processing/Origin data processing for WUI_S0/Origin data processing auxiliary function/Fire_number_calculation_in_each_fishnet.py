# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------
# 程序名称：WUI区域火点统计分析工具
# 功能描述：
# 1. 提取位于WUI（Wildland-Urban Interface，城乡结合部）范围内的野火起火点；
# 2. 基于预设的渔网栅格，统计每个栅格内的火点数量（无火点则记为0）；
# 3. 将火点数量结果作为新字段（Fire_num）添加到渔网栅格的属性表中，完成空间与属性关联。
#
# 输入数据：
#   - 野火起火点数据：GFA_v20240409_ignitions_2005.shp（点要素，包含2005年野火起火点位置）
#   - WUI范围数据：China_2005_interface_WUI.shp（面要素，2005年城乡结合部范围）
#   - 渔网栅格数据：GlobalGrid_5min_Clip_2005_interface_WUI.shp（面要素，用于统计的网格单元，含GRID_ID唯一标识）
#
# 输出数据：
#   - 中间结果：Fire_ignitions_in_2005_interface_WUI.shp（位于WUI范围内的火点）
#   - 最终结果：GlobalGrid_5min_Clip_2005_with_fire_num.shp（含Fire_num字段的渔网栅格，记录每个栅格内的火点数量）
#
# 处理流程：
#   1. 火点与WUI相交分析 → 提取WUI内的火点；
#   2. 火点与渔网空间连接 → 统计每个栅格火点数量；
#   3. 结果关联与字段添加 → 生成带火点统计的渔网数据。
# -------------------------------------------------------------------------------------

import arcpy

# 环境设置
arcpy.env.overwriteOutput = True  # 允许覆盖输出文件
arcpy.env.workspace = r"I:\Clip shapefile of CHINA for test"  # 工作目录

# 输入数据路径（工作区内文件，直接用文件名）
fire_points = "GFA_v20240409_ignitions_2005.shp"  # 野火起火点
wui_shp = "China_2005_interface_WUI.shp"  # WUI数据
fishnet = "GlobalGrid_5min_Clip_2005_interface_WUI.shp"  # 渔网栅格

# 输出数据路径（工作区内文件，直接用文件名）
fire_in_wui = "Fire_ignitions_in_2005_interface_WUI.shp"  # 相交后的火点
fishnet_with_fire = "GlobalGrid_5min_Clip_2005_with_fire_num.shp"  # 最终结果


# ----------------------
# 步骤1：计算火点与WUI的相交区域（提取WUI内的火点）
# ----------------------
print("步骤1：提取位于WUI内部的火点...")
arcpy.Intersect_analysis(
    in_features=[fire_points, wui_shp],
    out_feature_class=fire_in_wui,
    join_attributes="ALL",
    cluster_tolerance="0.001 Meters"
)

# 检查相交结果是否存在
if not arcpy.Exists(fire_in_wui):
    print("错误：火点与WUI相交分析失败，未生成结果文件！")
    exit()


# ----------------------
# 步骤2：火点与渔网空间连接（直接统计每个渔网的火点数量）
# ----------------------
print("步骤2：空间连接统计渔网内火点数量...")
# 目标：渔网；连接：WUI内火点；匹配规则：火点被渔网包含
arcpy.SpatialJoin_analysis(
    target_features=fishnet,  # 直接用渔网要素（无需先转图层）
    join_features=fire_in_wui,
    out_feature_class=fishnet_with_fire,
    join_operation="JOIN_ONE_TO_ONE",  # 一个渔网对应一条记录
    join_type="KEEP_ALL",  # 保留所有渔网（无火点也保留）
    match_option="CONTAINS",  # 火点完全在渔网内才算有效
    search_radius="",
    distance_field_name=""
)

# ----------------------
# 步骤3：创建fire_num字段并赋值为Join_Count
# ----------------------
print("步骤3：添加fire_num字段并赋值...")
# 1. 添加fire_num字段（长整型，存储火点数量）
arcpy.AddField_management(
    in_table=fishnet_with_fire,
    field_name="fire_num",
    field_type="LONG"  # 火点数量为整数，用长整型
)

# 2. 将Join_Count的值赋给fire_num（ArcGIS空间连接默认计数字段为Join_Count）
arcpy.CalculateField_management(
    in_table=fishnet_with_fire,
    field="fire_num",
    expression="!Join_Cou_1!",  # 直接引用Join_Count字段的值
    expression_type="PYTHON"  # 适配ArcGIS 10.8的Python版本
)
print("处理完成!")
print("属性表中'Fire_num'字段为火点数量（无火点为0）")
