# -*- coding: utf-8 -*-
"""
TIF文件批量裁剪脚本
但是这个程序在裁剪完tif时,不能自己退出
主要是处理E175的土地覆盖数据,因为E175的覆盖时E175-E180,投影以后就会溢出范围到全球,所以裁剪到E175-E179.99
"""

import os
import arcpy

arcpy.env.overwriteOutput = True

input_folder = r'I:\Data_huanglin\Landcover\E175'

tif_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.tif')]

for tif_file in tif_files:
    input_tif = os.path.join(input_folder, tif_file)
    output_tif = os.path.join(input_folder, tif_file.replace('.tif', '_clipped.tif'))
    
    if arcpy.Exists(output_tif):
        arcpy.Delete_management(output_tif)
    
    desc = arcpy.Describe(input_tif)
    extent = desc.extent
    
    clip_xmin = extent.XMin
    clip_ymin = extent.YMin
    clip_xmax = 179.990000
    clip_ymax = extent.YMax
    
    print("处理文件: {0}".format(tif_file))
    print("裁剪范围: XMin={0}, YMin={1}, XMax={2}, YMax={3}".format(
            clip_xmin, clip_ymin, clip_xmax, clip_ymax))
    
    arcpy.Clip_management(
            in_raster=input_tif,
            rectangle="{0} {1} {2} {3}".format(clip_xmin, clip_ymin, clip_xmax, clip_ymax),
            out_raster=output_tif,
            in_template_dataset=None,
            nodata_value=""
        )
    
    print("裁剪完成: {0}".format(output_tif))
    print("-" * 50)

arcpy.ClearWorkspaceCache_management()
print("所有文件处理完成！")
