# -- coding: utf-8 --
# 人类足迹指数(HFI)平均值计算程序

import arcpy
import os

# 步骤1：定义输入输出路径
hfi_raster_dir = r"E:\data-hl\Origin data\tif"
fishnet_gdb_path = r"E:\data-hl\Processing data\WUI_Fire_Risk.gdb"
output_gdb = fishnet_gdb_path

# 使用WKID创建目标投影
target_wkid = 54034  # World_Cylindrical_Equal_Area
target_projection = arcpy.SpatialReference()
target_projection.factoryCode = target_wkid
target_projection.create()

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

        # 步骤2.1：读取HFI栅格
        print("    步骤2.1：读取HFI栅格（年份%d）" % hfi_year)
        hfi_raster_path = os.path.join(hfi_raster_dir, "hfp%d.tif" % hfi_year)
        if not arcpy.Exists(hfi_raster_path):
            print("      警告：HFI栅格不存在，跳过：{0}".format(hfi_raster_path))
            continue

        # 步骤2.2：读取渔网shp
        print("    步骤2.2：读取WUI渔网")
        fishnet_to_use = wui_fishnet_path

        # 检查并校正WUI渔网投影
        fishnet_desc = arcpy.Describe(fishnet_to_use)
        fishnet_sr = fishnet_desc.spatialReference

        # 确保空间参考有效
        if not fishnet_sr.name or fishnet_sr.name.lower() == "unknown":
            print("      警告：渔网空间参考未知，设置为目标投影")
            fishnet_for_analysis = fishnet_to_use  # 直接使用原始渔网，但后续分析使用目标投影
        elif fishnet_sr.factoryCode != target_projection.factoryCode:
            print("      校正WUI渔网投影至目标坐标系")
            temp_fishnet = os.path.join(output_gdb, "temp_fishnet_%d_projected" % wui_year)
            if arcpy.Exists(temp_fishnet):
                arcpy.Delete_management(temp_fishnet)

            # 直接执行投影，不使用try-except
            arcpy.Project_management(fishnet_to_use, temp_fishnet, target_projection)
            fishnet_for_analysis = temp_fishnet
        else:
            fishnet_for_analysis = fishnet_to_use

        # 检查并校正HFI栅格投影
        hfi_desc = arcpy.Describe(hfi_raster_path)
        hfi_sr = hfi_desc.spatialReference

        # 设置像元对齐到原始栅格，避免投影后细微位移
        arcpy.env.snapRaster = hfi_raster_path

        # 确保空间参考有效
        if not hfi_sr.name or hfi_sr.name.lower() == "unknown":
            print("      警告：HFI栅格空间参考未知，假设为WGS84")
            hfi_sr = arcpy.SpatialReference(4326)  # 假设为WGS84

        if hfi_sr.factoryCode != target_projection.factoryCode:
            print("      校正HFI栅格投影至目标坐标系")
            temp_hfi_raster = os.path.join(output_gdb, "temp_hfi_%d_projected" % hfi_year)

            # 先删除可能存在的旧文件
            if arcpy.Exists(temp_hfi_raster):
                print("      删除旧的临时投影文件")
                arcpy.Delete_management(temp_hfi_raster)

            # 使用简化的投影参数
            arcpy.ProjectRaster_management(
                hfi_raster_path,  # in_raster
                temp_hfi_raster,  # out_raster
                target_projection,  # out_coor_system
                "BILINEAR",  # resampling_type
                "1000",  # cell_size（字符串更通用）
                ""  # geographic_transform
            )

            print("      HFI栅格投影成功")
            hfi_raster_for_analysis = temp_hfi_raster
        else:
            hfi_raster_for_analysis = hfi_raster_path

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
        for temp_data in [temp_zonal_table, clipped_hfi_raster]:
            if arcpy.Exists(temp_data):
                arcpy.Delete_management(temp_data)
        if hfi_raster_for_analysis != hfi_raster_path:
            arcpy.Delete_management(hfi_raster_for_analysis)

        # 如果使用了投影后的渔网副本，需要将结果复制回原渔网
        if fishnet_for_analysis != fishnet_to_use:
            print("      将结果字段复制回原始WUI渔网")
            # 清理原始渔网中的旧字段
            if hfi_mean_field in [f.name for f in arcpy.ListFields(wui_fishnet_path)]:
                arcpy.DeleteField_management(wui_fishnet_path, hfi_mean_field)
            # 连接新字段
            arcpy.JoinField_management(wui_fishnet_path, zone_field, fishnet_for_analysis, zone_field, [hfi_mean_field])
            # 删除投影后的临时渔网
            arcpy.Delete_management(fishnet_for_analysis)

        print("    HFI年份 %d 处理完成" % hfi_year)

print("\n所有年份按时间顺序处理完成！")
print("\n生成的字段说明：")
print("- HFI_YYYY（如：HFI_2003）：对应年份的人类足迹指数平均值")