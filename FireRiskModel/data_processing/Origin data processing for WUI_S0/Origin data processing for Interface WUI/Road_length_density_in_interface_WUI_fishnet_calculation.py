# -- coding: utf-8 --
# 道路长度与道路密度计算程序

import arcpy
import os

# 步骤1：定义输入输出路径
road_feature_path = r"I:\Data_huanglin\Road\gROADS\groads-v1-global-gdb\gROADS_v1.gdb\Global_Roads"
fishnet_gdb_path = r"I:\Processing data\WUI_Fire_Risk.gdb"
output_gdb = fishnet_gdb_path


# 使用WKID创建目标投影
target_wkid = 54034  # World_Cylindrical_Equal_Area
target_projection = arcpy.SpatialReference()
target_projection.factoryCode = target_wkid
target_projection.create()

fishnet_area_km2 = 85.9329  # 保持原有的渔网面积设置

# 设置处理环境
arcpy.env.workspace = fishnet_gdb_path
arcpy.env.overwriteOutput = True
arcpy.env.parallelProcessingFactor = "100%"

# WUI年份列表
wui_years = [2005, 2010, 2015, 2020]

# 步骤2：投影道路数据到目标坐标系
print("步骤2：投影道路数据到目标坐标系")

# 检查道路数据是否存在
if not arcpy.Exists(road_feature_path):
    print("错误：道路数据不存在：%s" % road_feature_path)
    raise Exception("道路数据文件未找到")

# 检查道路数据的投影
road_desc = arcpy.Describe(road_feature_path)
road_sr = road_desc.spatialReference

# 投影后的道路数据路径
projected_road_path = os.path.join(output_gdb, "Global_Roads_projected_54034")

# 如果投影后的道路数据不存在或需要重新投影
if not arcpy.Exists(projected_road_path) or (road_sr.factoryCode != target_projection.factoryCode):
    print("  校正道路数据投影至目标坐标系")
    
    # 先删除可能存在的旧文件
    if arcpy.Exists(projected_road_path):
        print("  删除旧的临时投影文件")
        arcpy.Delete_management(projected_road_path)
    
    # 执行投影
    arcpy.Project_management(
        road_feature_path,         # in_dataset
        projected_road_path,       # out_dataset
        target_projection          # out_coor_system
    )
    print("  道路数据投影成功")
else:
    print("  道路数据已投影至目标坐标系，直接使用")

# 步骤3：处理每个年份的WUI渔网
print("\n步骤3：处理每个年份的WUI渔网")

for wui_year in wui_years:
    print("\n--- 处理WUI年份 %d ---" % wui_year)
    
    # 获取对应年份的WUI渔网
    wui_fishnet_path = os.path.join(output_gdb, "Grid_Clip_%d_interface_WUI" % wui_year)
    if not arcpy.Exists(wui_fishnet_path):
        print("  错误：WUI渔网不存在：%s" % wui_fishnet_path)
        continue
    
    print("  读取WUI渔网：%s" % os.path.basename(wui_fishnet_path))
    
    # 设置渔网为分析对象（用户确认渔网投影正确，无需校正）
    fishnet_for_analysis = wui_fishnet_path
    
    # 获取渔网的GRID_ID字段
    zone_field = "GRID_ID" if "GRID_ID" in [f.name for f in arcpy.ListFields(fishnet_for_analysis)] else "OBJECTID"
    
    # 清理可能存在的旧字段
    roads_field = "Roads"
    roads_density_field = "Roadsd"
    for field in [roads_field, roads_density_field]:
        if field in [f.name for f in arcpy.ListFields(fishnet_for_analysis)]:
            arcpy.DeleteField_management(fishnet_for_analysis, field)
    
    # 步骤3.1：计算每个渔网内的道路长度
    print("  步骤3.1：计算每个渔网内的道路长度")
    
    # 创建临时的相交结果
    temp_intersect_path = os.path.join(output_gdb, "temp_road_intersect_%d" % wui_year)
    if arcpy.Exists(temp_intersect_path):
        arcpy.Delete_management(temp_intersect_path)
    
    # 执行相交分析
    arcpy.Intersect_analysis(
        [projected_road_path, fishnet_for_analysis],  # in_features
        temp_intersect_path,                          # out_feature_class
        "ALL",                                       # join_attributes
        "",                                          # cluster_tolerance
        "LINE"                                      # output_type
    )
    
    # 步骤3.2：添加长度字段并计算
    print("  步骤3.2：计算道路长度并统计")
    
    # 添加长度字段（单位：米）
    arcpy.AddField_management(temp_intersect_path, "LENGTH_M", "DOUBLE")
    arcpy.CalculateField_management(temp_intersect_path, "LENGTH_M", "!shape.length@meters!", "PYTHON")
    
    # 步骤3.3：按渔网统计道路长度
    print("  步骤3.3：按渔网统计总道路长度")
    
    # 创建统计表格
    temp_stats_table = os.path.join(output_gdb, "temp_road_stats_%d" % wui_year)
    if arcpy.Exists(temp_stats_table):
        arcpy.Delete_management(temp_stats_table)
    
    # 按渔网ID分组统计道路长度
    arcpy.Statistics_analysis(
        temp_intersect_path,                          # in_table
        temp_stats_table,                             # out_table
        [['LENGTH_M', 'SUM']],                        # statistics_fields
        zone_field                                    # case_field
    )
    
    # 步骤3.4：将统计结果连接到渔网上
    print("  步骤3.4：将道路长度连接到渔网")
    
    # 添加道路长度字段（单位：公里）
    arcpy.AddField_management(fishnet_for_analysis, roads_field, "DOUBLE")
    
    # 连接统计结果
    arcpy.JoinField_management(
        fishnet_for_analysis,                         # in_data
        zone_field,                                   # in_field
        temp_stats_table,                             # join_table
        zone_field,                                   # join_field
        ['SUM_LENGTH_M']                              # fields
    )
    
    # 将米转换为公里
    arcpy.CalculateField_management(
        fishnet_for_analysis, 
        roads_field, 
        "!SUM_LENGTH_M! / 1000 if !SUM_LENGTH_M! is not None else 0", 
        "PYTHON"
    )
    
    # 步骤3.5：计算道路密度
    print("  步骤3.5：计算道路密度")
    
    # 添加道路密度字段（单位：公里/平方公里）
    arcpy.AddField_management(fishnet_for_analysis, roads_density_field, "DOUBLE")
    
    # 计算密度
    expression = "(!%s! / %f) if !%s! is not None else 0" % (roads_field, fishnet_area_km2, roads_field)
    arcpy.CalculateField_management(fishnet_for_analysis, roads_density_field, expression, "PYTHON")
    
    # 步骤3.6：清理临时数据
    print("  步骤3.6：清理临时数据")
    for temp_data in [temp_intersect_path, temp_stats_table]:
        if arcpy.Exists(temp_data):
            arcpy.Delete_management(temp_data)
    
    # 删除临时连接字段
    if "SUM_LENGTH_M" in [f.name for f in arcpy.ListFields(fishnet_for_analysis)]:
        arcpy.DeleteField_management(fishnet_for_analysis, "SUM_LENGTH_M")
    
    print("  WUI年份 %d 处理完成" % wui_year)

print("\n所有年份处理完成！")
print("\n生成的字段说明：")
print("- Roads：每个渔网内的道路总长度（公里）")
print("- Roadsd：道路密度（公里/平方公里）")