# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------
# 程序名称：WUI区域多时段火点统计分析工具（多字段版）
# 功能描述：
# 1. 按时间段关联WUI地图（2003-2007→2005、2008-2012→2010、2013-2017→2015、2018-2022→2020）；
# 2. 对每个时间段内的所有火点年份，分别统计栅格火点数量，添加独立字段；
# 3. 所有中间结果和最终字段均存储于统一GDB，管理集中。
#
# 输入数据：
#   - 野火起火点数据：GFA_v20240409_ignitions_年份.shp（2003-2022年）
#   - WUI范围数据：Time_series_2005/2010/2015/2020_Interface_WUI.shp
#   - 有效栅格数据：WUI_Fire_Risk.gdb中的Grid_Clip_年份_interface_WUI（EPSG:54034）
#
# 输出结果（全部存储于WUI_Fire_Risk.gdb）：
#   - 最终结果：各目标年份渔网栅格新增多字段（如Fire_num_2003、Fire_num_2004等）
# -------------------------------------------------------------------------------------

import arcpy
import os

# ------------------------
# 定义路径与时间分组
# ------------------------
output_dir = r"I:\Processing data"  # 输出文件夹
output_gdb = os.path.join(output_dir, "WUI_Fire_Risk.gdb")  # 统一存储GDB

# 时间分组字典：{WUI年份: [对应火点年份范围]}
wui_year_mapping = {
    #2005: range(2003, 2008),  # 2003-2007 → 2005年WUI
    2010: range(2008, 2013),  # 2008-2012 → 2010年WUI
    2015: range(2013, 2018),  # 2013-2017 → 2015年WUI
    2020: range(2018, 2023)  # 2018-2022 → 2020年WUI
}

# 环境设置
arcpy.env.parallelProcessingFactor = "100%"
arcpy.env.overwriteOutput = True
target_sr = arcpy.SpatialReference(54034)  # 目标坐标系（等面积）

# ------------------------
# 循环处理每个WUI目标年份
# ------------------------
sorted_wui_years = sorted(wui_year_mapping.keys())
for wui_year in sorted_wui_years:
    fire_years = wui_year_mapping[wui_year]  # 获取对应火点年份范围
    print("\n===== 开始处理 {0} 年WUI对应的火点时间段 =====".format(wui_year))

    # 1. 定义数据路径
    wui_shp = os.path.join(output_gdb, "Time_series_{0}_Interface_WUI_equal_area".format(wui_year))  # WUI原始数据
    target_grid = os.path.join(output_gdb, "Grid_Clip_{0}_interface_WUI".format(wui_year))  # 目标渔网栅格（GDB内）

    # 2. 对每个火点年份，逐一统计并添加字段
    for fire_year in fire_years:
        print("\n--- 处理 {0} 年火点数据 ---".format(fire_year))

        # 火点数据路径（从前置处理GDB读取）
        processed_gdb = r"I:\Processing data\gdb集合\Fire_Data_Processed.gdb"
        fire_points = os.path.join(processed_gdb, "ignitions_GFA_v20240409_ignitions_{0}_Processed".format(fire_year))

        # 中间结果路径（GDB内要素类）
        fire_in_wui = os.path.join(output_gdb, "Fire_ignitions_in_{0}_Interface_WUI_{1}".format(wui_year, fire_year))
        temp_join = os.path.join(output_gdb, "Temp_Join_{0}_{1}".format(wui_year, fire_year))
        field_name = "Fire_num_{0}".format(fire_year)  # 自定义字段名（如Fire_num_2003）

        # 步骤1：提取WUI范围内的火点（相交分析）
        print("步骤1：提取WUI内火点...")
        arcpy.Intersect_analysis(
            in_features=[fire_points, wui_shp],
            out_feature_class=fire_in_wui,
            join_attributes="ALL",
            cluster_tolerance="0.001 Meters"
        )
        if not arcpy.Exists(fire_in_wui):
            print("错误：{0}年火点与{1}年WUI相交分析失败！".format(fire_year, wui_year))
            continue

        # 步骤2：空间连接统计火点数量
        print("步骤2：空间连接统计火点...")
        arcpy.SpatialJoin_analysis(
            target_features=target_grid,
            join_features=fire_in_wui,
            out_feature_class=temp_join,
            join_operation="JOIN_ONE_TO_ONE",
            join_type="KEEP_ALL",
            match_option="CONTAINS"
        )

        # 步骤3：添加并赋值自定义火点字段
        print("步骤3：添加自定义火点字段...")
        # 若字段已存在，先删除
        if field_name in [f.name for f in arcpy.ListFields(target_grid)]:
            arcpy.DeleteField_management(target_grid, field_name)
        # 添加新字段
        arcpy.AddField_management(target_grid, field_name, "LONG", field_alias="Num_{0}".format(fire_year))
        # 关联并赋值
        arcpy.JoinField_management(
            in_data=target_grid,
            in_field="GRID_ID",
            join_table=temp_join,
            join_field="GRID_ID",
            fields=["Join_Count"]
        )
        arcpy.CalculateField_management(
            in_table=target_grid,
            field=field_name,
            expression="!Join_Count!",
            expression_type="PYTHON"
        )
        # 清理临时字段
        if "Join_Count" in [f.name for f in arcpy.ListFields(target_grid)]:
            arcpy.DeleteField_management(target_grid, "Join_Count")

        # 步骤4：清理临时要素类
        print("步骤4：清理临时数据...")
        arcpy.Delete_management(fire_in_wui)
        arcpy.Delete_management(temp_join)
        print("{0}年火点统计完成，字段 {1} 已添加到 {2} 年渔网栅格".format(fire_year, field_name, wui_year))

    print("\n{0} 年WUI对应的所有火点时间段处理完毕！".format(wui_year))

print("\n===== 所有数据处理完毕 =====")
print("最终结果：{0} 中各年份渔网栅格已新增多字段（如Fire_num_2003等）".format(output_gdb))
print("每个渔网栅格的字段数量与对应时间段的火点年份数一致（如2005年渔网含2003-2007共5个字段）")