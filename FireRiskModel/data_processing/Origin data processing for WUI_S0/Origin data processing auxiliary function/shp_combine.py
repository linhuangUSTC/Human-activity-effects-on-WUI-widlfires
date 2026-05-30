 #-*- coding: utf-8 -*-
#-------------------------------------------------------------------------------------
#*************************************************************************************

# 程序名称：#interface 和 intermix WUI shp合并
# 创建日期：2025.10
# 程序功能：
# 1. 该程序读取输入的当年interface和intermixWUI
# 2. 将以上输入的WUI合并输出为新的WUI，
# 3. 并使用标识符为interface 和intermix类型的WUI进行标号
#
# 输入：
# - WUI数据：输入的WUI Shapefile
#（例如： Time_series_2005_Interface_WUI.shp，Time_series_2005_Intermix_WUI.shp）

# 输出：
# - WUI投影后的数据：Time_series_2005_WUI_combine.shp
# 此文档无法输出WUI合并以后的数据，因为两个shp合并以后超过了2G

#*************************************************************************************
#-------------------------------------------------------------------------------------

import arcpy

# 启用并行处理
arcpy.env.parallelProcessingFactor = "100%"
# 设置工作空间（确保输入SHP在该路径下）
arcpy.env.workspace = r"I:\test"
arcpy.env.overwriteOutput = True  # 允许覆盖已有文件

# ----------------------
# 1. 定义输入和输出路径
# ----------------------
input_shp1 = "Time_series_2005_Interface_WUI.shp"  # 第一个SHP（来源a）
input_shp2 = "Time_series_2005_Intermix_WUI.shp"   # 第二个SHP（来源b，需替换为你的实际文件名）
output_merged = "Time_series_2005_WUI_combine.shp"       # 合并后的输出文件

# 磁盘临时文件（存放在工作空间下，避免内存溢出）
temp_interface = "temp_interface_wui.shp"  # 临时Interface文件
temp_intermix = "temp_intermix_wui.shp"    # 临时Intermix文件

# ----------------------
# 2. 处理Interface WUI：复制到临时文件→添加类型标识
# ----------------------
# 步骤1：复制原始文件到磁盘临时文件（避免修改原始数据）
arcpy.CopyFeatures_management(input_shp1, temp_interface)


# 步骤2：添加类型字段（WUI_Type，明确标识类型）
arcpy.AddField_management(
    in_table=temp_interface,
    field_name="WUI_Type",
    field_type="TEXT",
    field_length=20  # 足够存储"interface"或"intermix"
)

# 步骤3：给字段赋值"interface"（适配ArcMap 10.8的PYTHON表达式类型）
arcpy.CalculateField_management(
    in_table=temp_interface,
    field="WUI_Type",
    expression="'interface'",  # 文本值用单引号包裹（Python语法）
    expression_type="PYTHON"
)
print("Interface临时文件添加类型标识完成")

# ----------------------
# 3. 处理Intermix WUI：复制到临时文件→添加类型标识
# ----------------------
# 步骤1：复制原始文件到磁盘临时文件
arcpy.CopyFeatures_management(input_shp2, temp_intermix)


# 步骤2：添加相同的类型字段（字段名、类型必须与Interface临时文件一致）
arcpy.AddField_management(
    in_table=temp_intermix,
    field_name="WUI_Type",
    field_type="TEXT",
    field_length=20
)

# 步骤3：给字段赋值"intermix"
arcpy.CalculateField_management(
    in_table=temp_intermix,
    field="WUI_Type",
    expression="'intermix'",
    expression_type="PYTHON"
)
print("Intermix临时文件添加类型标识完成")

# ----------------------
# 4. 合并两个临时文件→输出最终结果
# ----------------------
arcpy.Merge_management(
    inputs=[temp_interface, temp_intermix],  # 合并磁盘临时文件（稳定无内存压力）
    output=output_merged
)
print("合并完成！最终输出文件")