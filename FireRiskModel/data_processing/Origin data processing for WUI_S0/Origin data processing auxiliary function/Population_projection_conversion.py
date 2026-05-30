# -*- coding: utf-8 -*-
# HFI栅格投影转换程序
# 功能：将2003-2022年HFI原始栅格转换为World_Cylindrical_Equal_Area (WKID: 54034) 投影，并存储到指定GDB中

import arcpy
import os

# 定义输入输出路径
hfi_raster_dir = r"I:\Data_huanglin\Human footprint index\tif"
output_gdb_dir = r"I:\Processing data"
output_gdb_name = "Population_equal_area_projection.gdb"
output_gdb = os.path.join(output_gdb_dir, output_gdb_name)

# 确保输出GDB存在
if not arcpy.Exists(output_gdb):
    print("警告：GDB文件不存在，尝试创建...")
    try:
        arcpy.CreateFileGDB_management(output_gdb_dir, output_gdb_name)
        print("GDB创建成功: " + output_gdb)
    except Exception as e:
        print("错误：无法创建GDB - " + str(e))
        raise

# 使用WKID创建目标投影
print("设置目标投影：World_Cylindrical_Equal_Area (WKID: 54034)")
target_wkid = 54034  # World_Cylindrical_Equal_Area
target_projection = arcpy.SpatialReference()
target_projection.factoryCode = target_wkid
target_projection.create()

# 设置处理环境
arcpy.env.workspace = output_gdb
arcpy.env.overwriteOutput = True
arcpy.env.parallelProcessingFactor = "100%"

# 查找2003-2022年的HFI栅格文件
print("查找HFI栅格文件：" + hfi_raster_dir)
hfi_raster_files = []
for year in range(2003, 2023):
    hfi_file = os.path.join(hfi_raster_dir, "hfp" + str(year) + ".tif")
    if os.path.exists(hfi_file):
        hfi_raster_files.append(hfi_file)
print("找到" + str(len(hfi_raster_files)) + "个HFI栅格文件 (2003-2022年)")

# 处理每个HFI栅格文件
for hfi_raster_path in hfi_raster_files:
    # 提取年份
    raster_name = os.path.basename(hfi_raster_path)
    try:
        # 从文件名提取年份，支持格式如hfp2003.tif
        year = int(raster_name.replace("hfp", "").replace(".tif", ""))
        
        print("\n处理HFI栅格：" + raster_name + " (年份: " + str(year) + ")")
        
        # 输出文件名
        output_raster_name = "HFI_" + str(year) + "_54034"
        output_raster_path = os.path.join(output_gdb, output_raster_name)
        
        # 检查当前投影
        hfi_desc = arcpy.Describe(hfi_raster_path)
        hfi_sr = hfi_desc.spatialReference
    except ValueError:
        print("警告：无法从文件名提取年份，跳过文件: " + raster_name)
        continue
    
    if hasattr(hfi_sr, 'factoryCode') and hfi_sr.factoryCode == target_projection.factoryCode:
        print("  栅格已在目标投影中，直接复制")
        arcpy.CopyRaster_management(hfi_raster_path, output_raster_path)
    else:
        try:
            # 增强投影信息获取逻辑
            if hfi_sr is None:
                print("  警告：栅格缺少坐标系信息，假设为WGS84")
                hfi_sr = arcpy.SpatialReference(4326)  # 假设为WGS84
                projection_name = "WGS84 (假设)"
            else:
                projection_name = "未知"
                wkid = "未知"
                
                # 获取投影Wkid
                if hasattr(hfi_sr, 'factoryCode'):
                    wkid = str(hfi_sr.factoryCode)
                
                # 获取投影名称
                if hasattr(hfi_sr, 'name') and hfi_sr.name:
                    try:
                        if isinstance(hfi_sr.name, str):
                            projection_name = hfi_sr.name
                        else:
                            projection_name = str(hfi_sr.name)
                    except:
                        projection_name = "无法获取名称"
                
                print("  当前投影: " + projection_name + " (WKID: " + wkid + ")")
        except:
            print("  当前投影: 无法获取投影信息")
        print("  执行投影转换到WKID " + str(target_wkid))
        
        try:
            # 执行投影转换，使用与原程序相同的参数设置
            arcpy.ProjectRaster_management(
                in_raster=hfi_raster_path,
                out_raster=output_raster_path,
                out_coor_system=target_projection,
                resampling_type="BILINEAR",
                cell_size="1000"
            )
            print("  投影转换成功: " + output_raster_name)
        except Exception as e:
            try:
                print("  错误：投影转换失败 - " + str(e))
            except UnicodeError:
                print("  错误：投影转换失败 - 无法显示错误信息")
            continue

print("\n所有HFI栅格投影转换处理完成！")
print("投影后的栅格存储在: " + output_gdb)
print("\n文件命名规则：HFI_YYYY_54034")