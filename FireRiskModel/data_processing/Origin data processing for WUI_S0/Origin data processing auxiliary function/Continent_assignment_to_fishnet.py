#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
程序名称: 大洲属性赋值给渔网栅格程序
功能简介: 
    1. 将大洲地图的C_ID字段赋值给渔网栅格
    2. 对于每个渔网栅格，确定其主要所属的大洲（面积最大的）
    3. 对于不属于任何大洲的渔网栅格,赋值为0
输入文件:
    - 大洲地图: I:\Data_huanglin\map\world continent boundary\continent.shp
    - 渔网栅格: I:\Processing data\Intermix_WUI_Fire_Risk.gdb\Grid_Clip_{year}_intermix_WUI
    (其中year为处理年份:2005, 2010, 2015, 2020)

输出文件:
    - 带有大洲属性的渔网栅格: 覆盖原始渔网栅格,添加Continent字段
"""

import arcpy
import os

# 设置环境
arcpy.env.overwriteOutput = True
arcpy.env.parallelProcessingFactor = "100%"

# 输入路径
continent_shp_path = r"I:\Data_huanglin\map\world continent boundary\continent.shp"
fishnet_gdb_path = r"I:\Processing data\Intermix_WUI_Fire_Risk.gdb"
arcpy.env.workspace = fishnet_gdb_path

# 设置目标投影WKID 54034
target_wkid = 54034
target_projection = arcpy.SpatialReference()
target_projection.factoryCode = target_wkid
target_projection.create()

# 处理年份列表
wui_years = [2005,2010,2015,2020]

print("开始执行大洲属性赋值程序...")

# 步骤1：投影转换大洲地图
print("步骤1：投影转换大洲地图到WKID: {}".format(target_wkid))
projected_continent_path = os.path.join(fishnet_gdb_path, "continent_projected_{}".format(target_wkid))

# 删除已存在的投影数据
if arcpy.Exists(projected_continent_path):
    arcpy.Delete_management(projected_continent_path)

# 执行投影转换
arcpy.Project_management(continent_shp_path, projected_continent_path, target_projection)
print("大洲地图投影转换完成")

# 步骤2：处理每个年份的渔网栅格
for wui_year in wui_years:
    print("\n处理WUI年份 {} 的渔网栅格".format(wui_year))
    
    # 获取渔网路径
    fishnet_path = os.path.join(fishnet_gdb_path, "Grid_Clip_{}_intermix_WUI".format(wui_year))
    
    # 执行相交分析
    temp_intersect_path = os.path.join(fishnet_gdb_path, "temp_intersect_{}".format(wui_year))
    print("执行渔网栅格与大洲的相交分析...")
    arcpy.Intersect_analysis([fishnet_path, projected_continent_path], temp_intersect_path, "ALL")
    
    # 计算相交区域面积
    print("计算相交区域面积...")
    arcpy.AddField_management(temp_intersect_path, "Area_SqKM", "DOUBLE")
    arcpy.CalculateField_management(
        temp_intersect_path,
        "Area_SqKM",
        "!shape.area@squarekilometers!",
        "PYTHON"
    )
    
    # 执行统计分析
    temp_stats_path = os.path.join(fishnet_gdb_path, "temp_continent_stats_{}".format(wui_year))
    print("统计各大洲在每个渔网中的面积...")
    
    # 使用GRID_ID字段
    grid_id_field = "GRID_ID"
    
    arcpy.Statistics_analysis(
        temp_intersect_path,
        temp_stats_path,
        [["Area_SqKM", "SUM"]],
        [grid_id_field, "C_ID"]
    )
    
    # 找出每个渔网中面积最大的大洲
    print("确定每个渔网内面积比例最大的大洲...")
    fishnet_max_continent = {}
    
    with arcpy.da.SearchCursor(temp_stats_path, [grid_id_field, "C_ID", "SUM_Area_SqKM"]) as cursor:
        for row in cursor:
            fishnet_id = row[0]
            continent_id = row[1]
            area = row[2]
            
            if fishnet_id not in fishnet_max_continent or area > fishnet_max_continent[fishnet_id][1]:
                fishnet_max_continent[fishnet_id] = (continent_id, area)
    
    # 创建结果表
    max_continent_table = os.path.join(fishnet_gdb_path, "temp_max_continent_{}".format(wui_year))
    arcpy.CreateTable_management(fishnet_gdb_path, os.path.basename(max_continent_table))
    arcpy.AddField_management(max_continent_table, grid_id_field, "LONG")
    arcpy.AddField_management(max_continent_table, "Continent", "TEXT")
    
    # 写入结果
    with arcpy.da.InsertCursor(max_continent_table, [grid_id_field, "Continent"]) as cursor:
        for fishnet_id, (continent_id, area) in fishnet_max_continent.items():
            cursor.insertRow([fishnet_id, str(continent_id)])
    
    # 删除已存在的Continent字段
    if "Continent" in [f.name for f in arcpy.ListFields(fishnet_path)]:
        arcpy.DeleteField_management(fishnet_path, "Continent")
    
    # 连接大洲信息到渔网
    print("将大洲ID赋值给渔网栅格...")
    arcpy.JoinField_management(fishnet_path, grid_id_field, max_continent_table, grid_id_field, ["Continent"])
    
    # 处理不属于任何大洲的渔网栅格，赋值为0
    print("处理不属于任何大洲的渔网栅格，赋值为0...")
    arcpy.CalculateField_management(
        fishnet_path,
        "Continent",
        "!Continent! if !Continent! is not None else '0'",
        "PYTHON"
    )
    
    # 清理临时数据
    temp_datasets = [temp_intersect_path, temp_stats_path, max_continent_table]
    for dataset in temp_datasets:
        if arcpy.Exists(dataset):
            arcpy.Delete_management(dataset)
    
    print("WUI年份 {} 的渔网栅格大洲属性赋值完成".format(wui_year))

# 删除临时投影数据
if arcpy.Exists(projected_continent_path):
    arcpy.Delete_management(projected_continent_path)

print("\n所有年份渔网栅格大洲属性赋值完成！")