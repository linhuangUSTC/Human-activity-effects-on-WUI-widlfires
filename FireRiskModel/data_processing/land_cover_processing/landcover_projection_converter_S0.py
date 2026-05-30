# -*- coding: utf-8 -*-
"""
# 地表覆盖数据投影转换模块
# 功能:将WGS84(EPSG:4326)的地表覆盖分块数据批量转换为ESRI 54034投影
# 
# 注意事项：
# 此程序必须在PyCharm中运行,原因如下:
# 
# 主要特性：
# 1. 自动读取指定目录下的所有TIF文件
# 2. 使用Arcpy进行高效的投影转换
# 3. 采用串行处理,一次只处理一个TIF文件,避免内存和许可问题
# 4. 保持与输入结构一致的输出目录结构
# 5. 实时显示转换进度和详细状态信息
# 6. 直接执行,无需命令行参数
# 
# 输入输出配置：
# - 输入目录:I:\Data_huanglin\Landcover\GLC_FCS30D the first global 30m land-cover\land_cover_tif
# - 输出目录:I:\Data_huanglin\Landcover\GLC_FCS30D_projected
"""

import os
import sys
import arcpy

# 目标投影设置
target_projection = arcpy.SpatialReference(54034)  # ESRI:54034
print("设置目标投影：World_Cylindrical_Equal_Area (ESRI:54034)")

# 创建输入投影，假设WGS84
input_projection = arcpy.SpatialReference()
input_projection.factoryCode = 4326  # WGS84
input_projection.create()

# 设置ArcPy环境
arcpy.env.overwriteOutput = True
arcpy.env.parallelProcessingFactor = "0"  # 禁用并行处理以避免许可问题

def find_all_tif_files(directory):
    """查找指定目录及其子目录中的所有TIF文件"""
    tif_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.tif'):
                tif_files.append(os.path.join(root, file))
    return tif_files

def convert_raster(input_path, output_dir, output_filename):
    """使用ArcPy转换单个栅格文件"""
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, output_filename)
    
    # 如果输出文件已存在，跳过转换
    if os.path.exists(output_path):
        print("输出文件已存在，跳过: " + output_path)
        return True
    
    try:
        # 使用ArcPy的ProjectRaster_management进行投影转换
        print("转换中: " + input_path)
        
        arcpy.ProjectRaster_management(
            in_raster=input_path,
            out_raster=output_path,
            out_coor_system=target_projection,
            resampling_type="NEAREST",  # 对于分类数据使用最近邻重采样
            cell_size="29.7700370773504",  # 根据用户验证设置精确分辨率
            geographic_transform=""  # 空字符串表示自动选择
        )
        
        print("转换成功: " + output_path)
        print("所有波段已成功投影转换")
        return True
    except Exception as e:
        print("转换失败: " + input_path)
        print("错误: " + str(e))
        return False

def batch_convert(input_dir, output_base_dir):
    """批量转换所有TIF文件（串行处理）"""
    # 查找所有TIF文件
    tif_files = find_all_tif_files(input_dir)
    total_files = len(tif_files)
    
    if total_files == 0:
        print("未找到TIF文件")
        return {"total": 0, "success": 0, "failed": 0}
    
    print("找到 " + str(total_files) + " 个TIF文件，开始投影转换...")
    
    success_count = 0
    failed_count = 0
    failed_files = []
    
    # 逐个串行处理每个文件（一次只处理一个TIF文件）
    for i, input_path in enumerate(tif_files, 1):
        # 计算相对路径
        rel_path = os.path.relpath(input_path, input_dir)
        # 获取文件名
        filename = os.path.basename(rel_path)
        # 计算输出目录
        output_dir = os.path.join(output_base_dir, os.path.dirname(rel_path))
        
        # 转换文件
        success = convert_raster(input_path, output_dir, filename)
        
        if success:
            success_count += 1
        else:
            failed_count += 1
            failed_files.append(input_path)
        
        # 显示进度
        print("进度: " + str(i) + "/" + str(total_files))
    
    print("投影转换完成！成功: " + str(success_count) + ", 失败: " + str(failed_count))
    if failed_files:
        print("失败文件列表:")
        for file in failed_files:
            print("  - " + file)
    
    return {
        "total": total_files,
        "success": success_count,
        "failed": failed_count,
        "failed_files": failed_files
    }





# 直接执行投影转换
def execute_conversion():
    """直接执行投影转换，无需命令行参数"""
    
    # 设置固定路径
    INPUT_DIR = r'I:\Data_huanglin\Landcover\GLC_FCS30D the first global 30m land-cover\land_cover_tif'
    OUTPUT_DIR = r'I:\Data_huanglin\Landcover\GLC_FCS30D_projected'
    
    print("==================================================")
    print("地表覆盖投影转换工具")
    print("==================================================")
    print("输入目录: {}".format(INPUT_DIR))
    print("输出目录: {}".format(OUTPUT_DIR))
    print("处理方式: 串行处理，逐个转换TIF文件")
    print("=" * 50)
    
    # 验证输入目录是否存在
    if not os.path.exists(INPUT_DIR):
        print("错误：输入目录不存在！")
        print("请检查路径: " + INPUT_DIR)
        sys.exit(1)
    
    # 验证输入目录中是否有TIF文件
    print("正在搜索TIF文件...")
    tif_files = find_all_tif_files(INPUT_DIR)
    
    if not tif_files:
        print("在指定目录中未找到TIF文件，请检查路径。")
        sys.exit(1)
    
    print("找到", len(tif_files), "个TIF文件")
    print("开始投影转换...")
    
    # 设置ArcPy环境
    try:
        arcpy.CheckOutExtension("Spatial")
        print("Spatial扩展许可已激活")
    except Exception as e:
        print("无法checkout Spatial扩展:", str(e))
    
    # 执行投影转换
    print("\n开始批量转换...")
    conversion_results = batch_convert(INPUT_DIR, OUTPUT_DIR)
    
    print("\n" + "=" * 50)
    print("投影转换完成，统计结果：")
    print("  总文件数: " + str(conversion_results["total"]))
    print("  成功转换: " + str(conversion_results["success"]))
    print("  转换失败: " + str(conversion_results["failed"]))
    
    # 如果有失败的文件，输出失败文件列表
    if conversion_results["failed"] > 0:
        print("\n失败文件列表:")
        for file in conversion_results["failed_files"]:
            print("  - " + file)
    
    return conversion_results

# 直接执行转换
execute_conversion()