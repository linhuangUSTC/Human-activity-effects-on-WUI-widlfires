# -- coding: utf-8 --
# 人类足迹指数(HFI)平均值计算程序
# 注：此程序由于尚未解决的原因，不能在此电脑上运行
# 经过调试可知，主要原因出现在不能正确将human footprint index的tif文件转化成等面积投影（54034）
# 但是程序本身并没有问题，此程序已经在课题组的服务器上正常运行
# 因此，interface WUI 的渔网栅格中，有关hfi的字段储存在单独的渔网中

import arcpy
import os

# 步骤1：定义输入输出路径
# 使用已投影的HFI栅格数据
hfi_raster_dir = r"I:\Processing data\Human_footprint_index_equal_area_projection.gdb"
fishnet_gdb_path = r"I:\Processing data\WUI_Fire_Risk.gdb"
output_gdb = fishnet_gdb_path

# 已投影的HFI栅格使用WKID: 54034 (World_Cylindrical_Equal_Area)，此处不再需要创建目标投影

# 设置处理环境
arcpy.env.workspace = fishnet_gdb_path
arcpy.env.overwriteOutput = True
arcpy.env.parallelProcessingFactor = "100%"

# 时间对应关系：WUI年份→HFI年份范围
year_mapping = {
    2005: range(2003, 2008),  # 2003-2007年HFI→2005年WUI
    2010: range(2008, 2013),  # 2008-2012年HFI→2010年WUI
    2015: range(2013, 2018),  # 2013-2017年HFI→2015年WUI
    2020: range(2018, 2023)  # 2018-2022年HFI→2020年WUI
}

# 步骤2：按照时间顺序处理数据（先WUI年份升序）
print("\n步骤2：按照时间顺序处理数据")
# 按WUI年份升序排序
for wui_year in sorted(year_mapping.keys()):
    hfi_years = year_mapping[wui_year]
    print("\n--- 处理WUI年份 %d，对应HFI年份：%s ---" % (wui_year, hfi_years))

    # 获取或创建对应年份的WUI渔网
    wui_fishnet_path = os.path.join(output_gdb, "Grid_Clip_%d_interface_WUI" % wui_year)
    if not arcpy.Exists(wui_fishnet_path):
        print("  错误：WUI渔网不存在：%s" % wui_fishnet_path)
        continue

    # 处理HFI年份（目前只有2003年）
    for hfi_year in hfi_years:
        print("\n  处理HFI年份 %d → WUI年份 %d" % (hfi_year, wui_year))

        # 步骤2.1：读取已投影的HFI栅格数据
        print("    步骤2.1：读取已投影的HFI栅格（年份%d）" % hfi_year)
        # 使用已投影的HFI栅格名称格式
        hfi_raster_for_analysis = os.path.join(hfi_raster_dir, "HFI_%d_54034" % hfi_year)
        if not arcpy.Exists(hfi_raster_for_analysis):
            print("      警告：已投影的HFI栅格不存在，跳过")
            continue

        # 步骤2.2：读取WUI渔网（直接使用，不再进行投影验证）
        print("    步骤2.2：读取WUI渔网")
        fishnet_for_analysis = wui_fishnet_path

        # 步骤2.4：根据栅格裁剪投影后的HFI栅格
        print("    步骤2.4：根据WUI渔网范围裁剪HFI栅格")
        clipped_hfi_raster = os.path.join(output_gdb, "clipped_hfi_%d" % hfi_year)
        if arcpy.Exists(clipped_hfi_raster):
            arcpy.Delete_management(clipped_hfi_raster)
        arcpy.env.extent = arcpy.Describe(fishnet_for_analysis).extent
        arcpy.Clip_management(hfi_raster_for_analysis, "#", clipped_hfi_raster, fishnet_for_analysis, "#", "NONE")

        # 步骤2.5：计算渔网内HFI平均值
        print("    步骤2.5：计算WUI渔网内HFI平均值")
        temp_zonal_table = os.path.join(output_gdb, "temp_zonal_stats_%d" % hfi_year)
        if arcpy.Exists(temp_zonal_table):
            arcpy.Delete_management(temp_zonal_table)
        zone_field = "GRID_ID" if "GRID_ID" in [f.name for f in arcpy.ListFields(fishnet_for_analysis)] else "OBJECTID"

        # 清理可能存在的旧字段
        hfi_mean_field = "HFI_%d" % hfi_year
        for field in [hfi_mean_field, "MEAN"]:
            if field in [f.name for f in arcpy.ListFields(fishnet_for_analysis)]:
                arcpy.DeleteField_management(fishnet_for_analysis, field)

        # 执行分区统计（使用MEAN统计类型）
        arcpy.sa.ZonalStatisticsAsTable(fishnet_for_analysis, zone_field, clipped_hfi_raster, temp_zonal_table, "DATA",
                                        "MEAN")

        # 连接统计结果
        arcpy.JoinField_management(fishnet_for_analysis, zone_field, temp_zonal_table, zone_field, ["MEAN"])
        arcpy.AlterField_management(fishnet_for_analysis, "MEAN", hfi_mean_field, hfi_mean_field)

        # 步骤2.7：清理临时数据
        print("    步骤2.7：清理临时数据")
        # 只清理必要的临时数据，由于直接使用已投影数据，不需要处理临时投影文件
        for temp_data in [temp_zonal_table, clipped_hfi_raster]:
            if arcpy.Exists(temp_data):
                arcpy.Delete_management(temp_data)

        print("    HFI年份 %d 处理完成" % hfi_year)

print("\n所有年份按时间顺序处理完成！")
print("\n生成的字段说明：")
print("- HFI_YYYY（如：HFI_2003）：对应年份的人类足迹指数平均值")