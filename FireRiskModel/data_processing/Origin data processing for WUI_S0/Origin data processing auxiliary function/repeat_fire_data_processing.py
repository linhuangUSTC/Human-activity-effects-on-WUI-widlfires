# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------
# 程序名称：全球火灾数据集处理工具
# 功能说明：
# 1. 处理SHP格式的点火数据和过火周长数据
# 2. 根据fire_ID字段删除重复值
# 3. 仅处理2003-2022年的数据
# 4. 投影转换为EPSG:54034（等面积投影）
# 5. 保存处理后的数据到统一GDB
# --------------------------------------------------------------------------------

import arcpy
import os
import glob
import re

# ------------------------
# 定义路径参数
# ------------------------
data_root = r"I:\Data_huanglin\Wildfire\Update of The Global Fire Atlas of individual fire size, duration, speed and direction"
input_dirs = {
    "ignitions": os.path.join(data_root, "SHP_ignitions"),
    "perimeters": os.path.join(data_root, "SHP_perimeters")
}

output_gdb_folder = r"I:\Processing data\gdb集合"
output_gdb_name = "Fire_Data_Processed.gdb"  # 更通用的GDB名称
output_gdb = os.path.join(output_gdb_folder, output_gdb_name)

# 目标年份范围
TARGET_YEARS = range(2003, 2023)  # 2003-2022年

# 目标投影
TARGET_SR = arcpy.SpatialReference(54034)  # EPSG:54034

# ------------------------
# 环境设置
# ------------------------
arcpy.env.overwriteOutput = True

# ------------------------
# 检查并创建输出目录和GDB
# ------------------------
print("检查输出目录...")
if not os.path.exists(output_gdb_folder):
    os.makedirs(output_gdb_folder)
    print("创建输出目录成功：{0}".format(output_gdb_folder))

print("检查输出GDB...")
if not arcpy.Exists(output_gdb):
    arcpy.CreateFileGDB_management(output_gdb_folder, output_gdb_name)
    print("创建输出GDB成功：{0}".format(output_gdb))

# ------------------------
# 从文件名中提取年份
# ------------------------
def extract_year_from_filename(filename):
    """从文件名中提取年份"""
    # 匹配文件名末尾的4位年份（.shp前）
    match = re.search(r'_(\d{4})\.shp$', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

# ------------------------
# 处理单个目录的SHP文件
# ------------------------
def process_directory(input_dir, data_type):
    """处理单个目录的SHP文件"""
    print("\n===== 处理 {0} 数据目录 =====".format(data_type))
    print("输入目录：{0}".format(input_dir))
    
    # 查找所有SHP文件
    print("查找输入SHP文件...")
    shp_files = glob.glob(os.path.join(input_dir, "*.shp"))
    
    if not shp_files:
        print("警告：在 {0} 目录中未找到SHP文件！".format(data_type))
        return
    
    # 过滤目标年份的数据
    filtered_files = []
    for shp_path in shp_files:
        filename = os.path.basename(shp_path)
        year = extract_year_from_filename(filename)
        if year and year in TARGET_YEARS:
            filtered_files.append(shp_path)
    
    if not filtered_files:
        print("警告：在 {0} 目录中未找到2003-2022年的SHP文件！".format(data_type))
        return
    
    print("找到 {0} 个 {1} 数据SHP文件（2003-2022年）".format(len(filtered_files), data_type))
    for shp in filtered_files:
        print("  - {0}".format(os.path.basename(shp)))
    
    # 循环处理每个SHP文件
    for shp_path in filtered_files:
        shp_name = os.path.basename(shp_path)
        file_name = os.path.splitext(shp_name)[0]
        print("\n----- 处理文件：{0} -----".format(shp_name))
        
        # ------------------------
        # 1. 检查fire_ID字段是否存在
        # ------------------------
        print("检查fire_ID字段...")
        fields = [f.name for f in arcpy.ListFields(shp_path)]
        if "fire_ID" not in fields:
            print("警告：文件 {0} 中未找到fire_ID字段，跳过处理！".format(shp_name))
            continue
        
        # ------------------------
        # 2. 删除重复的fire_ID
        # ------------------------
        print("统计原始记录数...")
        original_count = int(arcpy.GetCount_management(shp_path).getOutput(0))
        print("  原始记录数：{0}".format(original_count))
        
        # 使用DeleteIdentical工具删除重复的fire_ID
        temp_uniques = os.path.join(output_gdb, "Temp_Uniques_{0}_{1}".format(data_type, file_name))
        print("删除重复的fire_ID...")
        arcpy.CopyFeatures_management(shp_path, temp_uniques)
        arcpy.DeleteIdentical_management(temp_uniques, "fire_ID")
        
        print("统计去重后记录数...")
        unique_count = int(arcpy.GetCount_management(temp_uniques).getOutput(0))
        print("  去重后记录数：{0}".format(unique_count))
        print("  去除重复数：{0}".format(original_count - unique_count))
        
        # ------------------------
        # 3. 投影转换
        # ------------------------
        print("检查当前投影...")
        desc = arcpy.Describe(temp_uniques)
        current_sr = desc.spatialReference
        
        output_feature = os.path.join(output_gdb, "{0}_{1}_Processed".format(data_type, file_name))
        
        if current_sr.factoryCode != TARGET_SR.factoryCode:
            print("投影转换为EPSG:54034...")
            arcpy.Project_management(temp_uniques, output_feature, TARGET_SR)
            print("投影转换完成")
        else:
            print("数据已为EPSG:54034投影，直接复制...")
            arcpy.CopyFeatures_management(temp_uniques, output_feature)
        
        # ------------------------
        # 4. 清理临时数据
        # ------------------------
        print("清理临时数据...")
        if arcpy.Exists(temp_uniques):
            arcpy.Delete_management(temp_uniques)
        
        print("文件 {0} 处理完成！".format(shp_name))
        print("  输出位置：{0}".format(output_feature))

# ------------------------
# 主程序
# ------------------------
print("\n===== 全球火灾数据集处理工具 =====")
print("处理年份：2003-2022年")
print("目标投影：EPSG:54034")
print("输出GDB：{0}".format(output_gdb))

# 处理两个数据目录
for data_type, input_dir in input_dirs.items():
    if os.path.exists(input_dir):
        process_directory(input_dir, data_type)
    else:
        print("\n错误：目录不存在：{0}".format(input_dir))

# ------------------------
# 处理完成
# ------------------------
print("\n===== 所有文件处理完成！=====")
print("处理后的数据已保存到：")
print("  {0}".format(output_gdb))
print("\n请在ArcGIS中查看处理结果。")