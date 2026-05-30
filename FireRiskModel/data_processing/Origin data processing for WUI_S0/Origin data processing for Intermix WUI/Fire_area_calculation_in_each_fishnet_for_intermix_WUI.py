# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------
# 程序名称：WUI过火面积渔网统计工具（矢量核心版）
# 功能说明：
# 1. 基于已有等面积渔网，计算WUI内各年份过火面积
# 2. 核心逻辑：WUI与过火面积相交→拆分跨栅格斑块→按GRID_ID统计求和→字段关联
# 3. 多时段循环，新增“过火面积年份”字段（单位：km²）
# --------------------------------------------------------------------------------

import arcpy
import os

# ------------------------
# 定义路径与时间分组
# ------------------------
output_gdb = r"E:\data-hl\Processing data\WUI_Fire_Risk.gdb"  # 统一存储GDB（含等面积WUI和渔网）

# 时间分组（与原程序一致）
wui_year_mapping = {
    2005: range(2003, 2008),
    2010: range(2008, 2013),
    2015: range(2013, 2018),
    2020: range(2018, 2023)
}

# 环境设置（强制等面积投影，确保其他数据与渔网投影一致）
arcpy.env.workspace = output_gdb
arcpy.env.overwriteOutput = True
target_sr = arcpy.SpatialReference(54034)  # 与渔网、WUI的等面积投影一致

# ------------------------
# 循环处理每个WUI年份
# ------------------------
sorted_wui_years = sorted(wui_year_mapping.keys())
for wui_year in sorted_wui_years:
    fire_years = wui_year_mapping[wui_year]
    print("\n===== 处理 {0} 年WUI对应的过火面积时间段 =====".format(wui_year))

    # 1. 定义核心数据路径（均为等面积投影，渔网直接使用，不额外投影）
    wui_equal_area = os.path.join(output_gdb, "Time_series_{0}_Intermix_WUI_equal_area".format(wui_year))  # 等面积WUI
    target_grid = os.path.join(output_gdb, "Grid_Clip_{0}_intermix_WUI".format(wui_year))  # 已为等面积的渔网

    # 2. 循环处理每个过火年份
    for fire_year in fire_years:
        print("\n--- 处理 {0} 年过火面积 ---".format(fire_year))
        field_name = "Fire_area_{0}".format(fire_year)  # 新增字段名（修改为英文格式）

        # 临时数据路径（从前置处理GDB读取已投影的过火面积数据）
        processed_gdb = r"E:\data-hl\Origin data\Fire_Data_Processed.gdb"
        fire_perimeter = os.path.join(processed_gdb, "perimeters_GFA_v20240409_perimeters_{0}_Processed".format(fire_year))  # 前置处理的过火矢量 
            
        fire_perim_proj = fire_perimeter  # 直接使用已投影的数据
        wui_fire_intersect = os.path.join(output_gdb,"WUI_Fire_Intersect_{0}_{1}".format(wui_year, fire_year))  # WUI内过火区域
        split_fire = os.path.join(output_gdb, "Split_Fire_By_Fishnet_{0}_{1}".format(wui_year, fire_year))  # 拆分后斑块
        stats_table = os.path.join(output_gdb, "FireArea_Stats_{0}_{1}".format(wui_year, fire_year))  # 面积统计临时表（新增）

        # 清理可能存在的临时字段（避免字段冲突）
        existing_fields = [f.name for f in arcpy.ListFields(target_grid)]
        temp_sum_field = "SUM_SHAPE_Area"
        if temp_sum_field in existing_fields:
            try:
                arcpy.DeleteField_management(target_grid, temp_sum_field)
                print("已清理临时字段：{0}".format(temp_sum_field))
            except Exception as e:
                print("警告：清理临时字段失败（可能已被删除）：{0}".format(str(e)))

        # ----------------------
        # 步骤2：WUI与过火面积相交（提取WUI内的过火面积）
        # ----------------------
        print("步骤2：计算WUI与{0}年过火面积的相交区域...".format(fire_year))
        arcpy.Intersect_analysis(
            in_features=[wui_equal_area, fire_perim_proj],  # 等面积WUI + 投影后过火矢量
            out_feature_class=wui_fire_intersect,
            join_attributes="ALL",
            cluster_tolerance="0.001 Meters"
        )
        print("相交计算完成")

        # ----------------------
        # 步骤3：拆分跨栅格的过火斑块（直接使用已有等面积渔网）
        # ----------------------
        print("步骤3：拆分跨栅格的过火斑块...")
        arcpy.Intersect_analysis(
            in_features=[target_grid, wui_fire_intersect],  # 直接用原始等面积渔网，不投影
            out_feature_class=split_fire,
            join_attributes="ALL",
            cluster_tolerance="0.001 Meters"
        )
        print("拆分完成")

        # ----------------------
        # 步骤4：按GRID_ID统计面积总和（替换原空间连接，核心修改）
        # ----------------------
        print("步骤4：统计每个渔网的{0}年过火面积...".format(fire_year))

        # 检查是否有数据需要统计
        count = int(arcpy.GetCount_management(split_fire).getOutput(0))
        if count == 0:
            print("警告：{0}年过火面积在WUI内无数据，跳过统计".format(fire_year))
            # 直接设置字段值为0
            existing_fields = [f.name for f in arcpy.ListFields(target_grid)]
            if field_name in existing_fields:
                arcpy.DeleteField_management(target_grid, field_name)
            arcpy.AddField_management(target_grid, field_name, "DOUBLE", field_alias="Area_{0}".format(fire_year))
            arcpy.CalculateField_management(
                in_table=target_grid,
                field=field_name,
                expression="0",
                expression_type="PYTHON"
            )
            # 清理临时数据（不包含前置处理的原始数据）
            temp_files = [wui_fire_intersect, split_fire, stats_table]
            for temp in temp_files:
                if arcpy.Exists(temp):
                    try:
                        arcpy.Delete_management(temp)
                    except Exception as e:
                        print("警告：清理临时文件失败：{0} - {1}".format(temp, str(e)))
            print("{0}年处理完成！渔网已新增字段：{1}（值为0）".format(fire_year, field_name))
            continue

        # 统计split_fire中每个GRID_ID的SHAPE_Area总和（全大写字段，避免匹配错误）
        arcpy.Statistics_analysis(
            in_table=split_fire,
            out_table=stats_table,
            statistics_fields=[["SHAPE_Area", "SUM"]],  # 原始面积字段为SHAPE_Area（全大写）
            case_field="GRID_ID"  # 按渔网ID分组求和
        )
        print("统计完成")

        # ----------------------
        # 步骤5：添加"过火面积年份"字段并关联统计结果（修改关联逻辑）
        # ----------------------
        print("步骤5：添加{0}字段到渔网...".format(field_name))

        # 1. 检查目标字段是否已存在，如果存在则先删除（避免数据错误）
        existing_fields = [f.name for f in arcpy.ListFields(target_grid)]
        if field_name in existing_fields:
            print("目标字段已存在，先删除旧字段：{0}".format(field_name))
            arcpy.DeleteField_management(target_grid, field_name)

        # 2. 新增字段（与原程序一致：双精度类型，保留别名）
        arcpy.AddField_management(target_grid, field_name, "DOUBLE", field_alias="Area_{0}".format(fire_year))

        # 3. 清理可能存在的临时字段（确保JoinField前字段干净）
        existing_fields = [f.name for f in arcpy.ListFields(target_grid)]
        temp_sum_field = "SUM_SHAPE_Area"
        if temp_sum_field in existing_fields:
            try:
                arcpy.DeleteField_management(target_grid, temp_sum_field)
            except Exception as e:
                print("警告：清理临时字段时出错（可能已被删除）：{0}".format(str(e)))

        # 4. 关键：通过GRID_ID关联统计结果到渔网（替换原temp_join关联）
        arcpy.JoinField_management(
            in_data=target_grid,
            in_field="GRID_ID",
            join_table=stats_table,
            join_field="GRID_ID",
            fields=["SUM_SHAPE_Area"]  # 关联求和字段（全大写，与统计表一致）
        )

        # 5. 计算字段值：平方米→km²，空值补0（修改字段引用，与统计表字段一致）
        arcpy.CalculateField_management(
            in_table=target_grid,
            field=field_name,
            expression="!SUM_SHAPE_Area! / 1000000 if !SUM_SHAPE_Area! is not None else 0",
            expression_type="PYTHON"
        )

        # 6. 删除临时关联字段（SUM_SHAPE_Area不再需要，避免字段冗余）
        existing_fields = [f.name for f in arcpy.ListFields(target_grid)]
        if temp_sum_field in existing_fields:
            arcpy.DeleteField_management(target_grid, temp_sum_field)

        # ----------------------
        # 步骤6：清理临时数据（不包含前置处理的原始数据）
        # ----------------------
        print("步骤6：清理{0}年临时数据...".format(fire_year))
        temp_files = [wui_fire_intersect, split_fire, stats_table]  # 清理相交、拆分、统计表

        for temp in temp_files:
            if arcpy.Exists(temp):
                try:
                    arcpy.Delete_management(temp)
                except Exception as e:
                    print("警告：清理临时文件失败：{0} - {1}".format(temp, str(e)))

        print("{0}年处理完成！渔网已新增字段：{1}".format(fire_year, field_name))

    print("\n{0}年WUI对应的所有过火面积年份处理完毕！".format(wui_year))

print("\n===== 所有数据处理完毕 =====")
print("最终结果位置：{0}".format(output_gdb))