# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------------
# 程序名称：渔网内WUI面积统计
# 功能：计算每个有效渔网中重叠的WUI面积，直接在原始渔网添加WUI_area字段
# 输入（均位于WUI_Fire_Risk.gdb内）：
#   - 有效渔网：Grid_Clip_年份_Intermix_WUI（含GRID_ID）
#   - WUI数据：Time_series_年份_Intermix_WUI_equal_area（等面积投影）
# 输出（位于WUI_Fire_Risk.gdb内）：
#   - 原始渔网新增字段：WUI_area（单位：平方千米）
#-------------------------------------------------------------------------------------

import arcpy
import os

# 定义GDB路径（所有输入输出均在此GDB内）
gdb_path = r"E:\data-hl\Processing data\WUI_Fire_Risk.gdb"

# 环境设置（工作空间设为目标GDB，确保所有操作在GDB内进行）
arcpy.env.workspace = gdb_path
arcpy.env.overwriteOutput = True

# 定义处理年份
years = [2005, 2010, 2015, 2020]

# ---------------------
# 循环处理各年份数据
# ---------------------
for year in years:
    print("\n===== 开始处理 {0} 年数据 =====".format(year))

    # 1. 拼接GDB内的文件路径
    grid_clip = os.path.join(gdb_path, "Grid_Clip_{0}_Intermix_WUI".format(year))  # 原始渔网（GDB内要素类）
    wui_equal_area = os.path.join(gdb_path, "Time_series_{0}_Intermix_WUI_equal_area".format(year))  # WUI数据
    # 临时文件（带年份标识，避免重名，存储在GDB内）
    temp_intersect = os.path.join(gdb_path, "temp_intersect_{0}".format(year))
    temp_summary = os.path.join(gdb_path, "temp_summary_{0}".format(year))

    # ----------------------
    # 步骤1：计算渔网与WUI的重叠区域
    # ----------------------
    print("步骤1：计算渔网与WUI的重叠区域...")
    arcpy.Intersect_analysis(
        in_features=[grid_clip, wui_equal_area],
        out_feature_class=temp_intersect,
        join_attributes="ALL",
        cluster_tolerance="0.001 Meters"
    )

    # ----------------------
    # 步骤2：计算重叠区域面积（平方千米）
    # ----------------------
    print("步骤2：计算重叠区域面积...")
    arcpy.AddField_management(temp_intersect, "TEMP_AREA", "DOUBLE")
    arcpy.CalculateField_management(
        temp_intersect,
        "TEMP_AREA",
        "!shape.area! / 1000000",  # 等面积投影下shape.area为平方米，转平方千米
        "PYTHON"
    )

    # ----------------------
    # 步骤3：按GRID_ID汇总每个渔网的WUI总面积
    # ----------------------
    print("步骤3：按渔网ID汇总面积...")
    arcpy.Statistics_analysis(
        in_table=temp_intersect,
        out_table=temp_summary,
        statistics_fields=[["TEMP_AREA", "SUM"]],
        case_field="GRID_ID"
    )

    # ----------------------
    # 步骤4：在原始渔网添加WUI_area字段并赋值
    # ----------------------
    print("步骤4：添加面积字段到原始渔网...")
    # 添加WUI_area字段（若已存在会被overwriteOutput覆盖）
    arcpy.AddField_management(grid_clip, "WUI_area", "DOUBLE")

    # 将渔网转为图层，连接汇总表
    grid_layer = "grid_layer_{0}".format(year)
    arcpy.MakeFeatureLayer_management(grid_clip, grid_layer)
    arcpy.AddJoin_management(
        in_layer_or_view=grid_layer,
        in_field="GRID_ID",
        join_table=temp_summary,
        join_field="GRID_ID",
        join_type="KEEP_ALL"
    )

    # 将汇总面积赋值到WUI_area字段
    arcpy.CalculateField_management(
        in_table=grid_layer,
        field="WUI_area",
        expression="!temp_summary_{0}.SUM_TEMP_AREA!".format(year),  # 引用汇总表的面积
        expression_type="PYTHON"
    )

    # ----------------------
    # 步骤5：移除连接并清理临时数据
    # ----------------------
    print("步骤5：清理临时数据...")
    arcpy.RemoveJoin_management(grid_layer)  # 移除表连接
    arcpy.Delete_management(temp_intersect)  # 删除临时重叠区域
    arcpy.Delete_management(temp_summary)    # 删除临时汇总表

    print("{0}年完成！原始渔网已新增WUI_area字段".format(year))

print("\n===== 所有年份数据处理完毕 =====")
print("结果位置：{0}".format(gdb_path))
print("每个年份的渔网均新增WUI_area字段（单位：平方千米）")