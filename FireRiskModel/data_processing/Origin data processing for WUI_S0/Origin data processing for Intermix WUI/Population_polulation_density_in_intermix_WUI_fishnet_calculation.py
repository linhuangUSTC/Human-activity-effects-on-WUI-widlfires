# -- coding: utf-8 --
# 人口数量与人口密度计算程序（按步骤执行版）

import arcpy
import os

# 步骤1：定义输入输出路径
# 使用已投影的人口栅格数据
population_raster_dir = r"E:\data-hl\Origin data\Population_equal_area_projection.gdb"
fishnet_gdb_path = r"E:\data-hl\Processing data\WUI_Fire_Risk.gdb"
output_gdb = fishnet_gdb_path

# 确保GDB存在
if not arcpy.Exists(output_gdb):
    print("警告：GDB文件不存在，尝试创建...")
    # 如果GDB不存在，尝试创建
    try:
        gdb_dir = os.path.dirname(output_gdb)
        gdb_name = os.path.basename(output_gdb)
        arcpy.CreateFileGDB_management(gdb_dir, gdb_name)
        print("GDB创建成功")
    except Exception as e:
        print("错误：无法创建GDB - {0}".format(str(e)))
        raise

# 已投影的人口栅格使用WKID: 54034 (World_Cylindrical_Equal_Area)，此处不再需要创建目标投影

fishnet_area_km2 = 85.9329  # 保持原有的渔网面积设置

# 设置处理环境
arcpy.env.workspace = fishnet_gdb_path
arcpy.env.overwriteOutput = True
arcpy.env.parallelProcessingFactor = "100%"

# 时间对应关系：WUI年份→人口年份范围
year_mapping = {
    2005: range(2003, 2008),  # 2003-2007年人口→2005年WUI
    2010: range(2008, 2013),  # 2008-2012年人口→2010年WUI
    2015: range(2013, 2018),  # 2013-2017年人口→2015年WUI
    2020: range(2018, 2023)  # 2018-2022年人口→2020年WUI
}

# 步骤2：按照时间顺序处理数据（先WUI年份升序，再人口年份升序）
print("\n步骤2：按照时间顺序处理数据")
# 按WUI年份升序排序
for wui_year in sorted(year_mapping.keys()):
    pop_years = year_mapping[wui_year]
    print("\n--- 处理WUI年份 %d，对应人口年份：%s ---" % (wui_year, list(pop_years)))

    # 获取或创建对应年份的WUI渔网
    wui_fishnet_path = os.path.join(output_gdb, "Grid_Clip_%d_intermix_WUI" % wui_year)
    if not arcpy.Exists(wui_fishnet_path):
        print("  错误：WUI渔网不存在：%s" % wui_fishnet_path)
        continue

    # 确保人口年份按升序处理
    for pop_year in sorted(pop_years):
        print("\n  处理人口年份 %d → WUI年份 %d" % (pop_year, wui_year))

        # 步骤2.1：读取已投影的人口栅格数据
        print("    步骤2.1：读取已投影的人口栅格（年份%d）" % pop_year)
        # 使用已投影的人口栅格名称格式
        pop_raster_for_analysis = os.path.join(population_raster_dir, "population_%d_54034" % pop_year)
        if not arcpy.Exists(pop_raster_for_analysis):
            print("      警告：已投影的人口栅格不存在，跳过")
            continue

        # 步骤2.2：读取WUI渔网（直接使用，不再进行投影验证）
        print("    步骤2.2：读取WUI渔网")
        fishnet_for_analysis = wui_fishnet_path

        # 步骤2.4：根据栅格裁剪投影后的人口栅格
        print("    步骤2.4：根据WUI渔网范围裁剪人口栅格")
        clipped_pop_raster = os.path.join(output_gdb, "clipped_pop_%d" % pop_year)
        if arcpy.Exists(clipped_pop_raster):
            arcpy.Delete_management(clipped_pop_raster)
        arcpy.env.extent = arcpy.Describe(fishnet_for_analysis).extent
        arcpy.Clip_management(pop_raster_for_analysis, "#", clipped_pop_raster, fishnet_for_analysis, "#", "NONE")

        # 步骤2.5：求出对应的人口数量
        print("    步骤2.5：计算WUI渔网内人口数量")
        temp_zonal_table = os.path.join(output_gdb, "temp_zonal_stats_%d" % pop_year)
        if arcpy.Exists(temp_zonal_table):
            arcpy.Delete_management(temp_zonal_table)
        zone_field = "GRID_ID" if "GRID_ID" in [f.name for f in arcpy.ListFields(fishnet_for_analysis)] else "OBJECTID"

        # 清理可能存在的旧字段
        pop_total_field = "Pop_%d" % pop_year
        pop_density_field = "PopD_%d" % pop_year
        for field in [pop_total_field, pop_density_field, "SUM"]:
            if field in [f.name for f in arcpy.ListFields(fishnet_for_analysis)]:
                arcpy.DeleteField_management(fishnet_for_analysis, field)

        # 执行分区统计
        arcpy.sa.ZonalStatisticsAsTable(fishnet_for_analysis, zone_field, clipped_pop_raster, temp_zonal_table, "DATA",
                                        "SUM")

        # 连接统计结果
        arcpy.JoinField_management(fishnet_for_analysis, zone_field, temp_zonal_table, zone_field, ["SUM"])
        arcpy.AlterField_management(fishnet_for_analysis, "SUM", pop_total_field, pop_total_field)

        # 步骤2.6：计算人口密度
        print("    步骤2.6：计算人口密度")
        arcpy.AddField_management(fishnet_for_analysis, pop_density_field, "DOUBLE")
        expression = "(!%s! / %f) if !%s! is not None else 0" % (pop_total_field, fishnet_area_km2, pop_total_field)
        arcpy.CalculateField_management(fishnet_for_analysis, pop_density_field, expression, "PYTHON")

        # 步骤2.7：清理临时数据
        print("    步骤2.7：清理临时数据")
        for temp_data in [temp_zonal_table, clipped_pop_raster]:
            if arcpy.Exists(temp_data):
                arcpy.Delete_management(temp_data)

        print("    人口年份 %d 处理完成" % pop_year)

print("\n所有年份按时间顺序处理完成！")
print("\n生成的字段说明：")
print("- Pop_YYYY（如：Pop_2003）：对应年份的人口总量")
print("- PopD_YYYY（如：PopD_2003）：对应年份的人口密度（人/平方公里）")