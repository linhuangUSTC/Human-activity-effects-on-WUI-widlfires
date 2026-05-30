# -*- coding: utf-8 -*-
"""
TIF文件空间边界提取脚本

功能：
- 遍历指定文件夹中的所有TIF文件
- 使用arcpy获取每个TIF文件的空间边界信息
- 将边界信息保存到CSV文件中

输入：
- TIF文件夹路径:I:\Data_huanglin\Landcover\GLC_FCS30D_projected

输出：
- CSV文件:I:\Data_huanglin\Landcover\GLC_FCS30D_projected\tif_simple.csv
- 包含字段:File_Path, File_Name, MinX, MinY, MaxX, MaxY

使用方法：
由于arcpy的原因,该脚本需要在PyCharm中运行
"""

import os
import csv
import arcpy

def main():
    """主函数"""
    # 设置输入和输出路径
    tif_folder = r'I:\Data_huanglin\Landcover\GLC_FCS30D_projected'
    output_csv = os.path.join(tif_folder, 'tif_simple.csv')
    
    print("输入TIF文件夹: {0}".format(tif_folder))
    print("输出CSV文件: {0}".format(output_csv))
    
    # 查找所有TIF文件
    tif_files = []
    for root, _, files in os.walk(tif_folder):
        for file in files:
            if file.lower().endswith('.tif'):
                tif_files.append(os.path.join(root, file))
    
    print("找到 {0} 个TIF文件".format(len(tif_files)))
    
    # 提取边界信息
    bounds_data = []
    
    for tif_path in tif_files:
        try:
            # 使用arcpy获取栅格文件的边界
            desc = arcpy.Describe(tif_path)
            extent = desc.extent
            bounds = {
                'file_path': tif_path,
                'file_name': os.path.basename(tif_path),
                'minx': extent.XMin,
                'miny': extent.YMin,
                'maxx': extent.XMax,
                'maxy': extent.YMax
            }
            bounds_data.append(bounds)
            print("处理成功: {0}".format(os.path.basename(tif_path)))
        except Exception as e:
            print("处理失败: {0}, 错误: {1}".format(tif_path, str(e)))
    
    # 保存到CSV文件
    print("保存到CSV文件...")
    
    try:
        with open(output_csv, 'wb') as csvfile:  # 使用二进制模式写入
            fieldnames = ['File_Path', 'File_Name', 'MinX', 'MinY', 'MaxX', 'MaxY']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # 写入表头
            writer.writeheader()
            
            # 写入数据
            for data in bounds_data:
                writer.writerow({
                    'File_Path': data['file_path'],
                    'File_Name': data['file_name'],
                    'MinX': data['minx'],
                    'MinY': data['miny'],
                    'MaxX': data['maxx'],
                    'MaxY': data['maxy']
                })
        
        print("保存成功")
        
    except Exception as e:
        print("保存失败: {0}".format(str(e)))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
