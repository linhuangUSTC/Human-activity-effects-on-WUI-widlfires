# -*- coding: utf-8 -*-
"""
# 地表覆盖TIF文件波段检查工具
# 功能:检查指定目录下所有TIF文件的波段数量
# 检查每个投影转化以后的tif是否仍然是23个波段
#
# 输入输出配置：
# - 输入目录: I:\Data_huanglin\Landcover\GLC_FCS30D_projected
# - 输出目录: I:\Data_huanglin\Landcover\GLC_FCS30D_projected
#
# 注意事项：
# 此程序需要ArcPy环境支持
"""

import os
import sys
import csv
import arcpy
import time

# 配置设置
def get_config():
    """获取配置信息"""
    config = {
        "input_dir": r'I:\Data_huanglin\Landcover\GLC_FCS30D_projected',  # 原程序输出目录
        "output_csv": r'I:\Data_huanglin\Landcover\GLC_FCS30D_projected\tif_band_count.csv'  # 输出CSV文件路径
    }
    return config



def find_all_tif_files(directory):
    """查找指定目录及其子目录中的所有TIF文件"""
    tif_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.tif'):
                tif_files.append(os.path.join(root, file))
    return tif_files

def get_band_count(tif_path):
    """获取TIF文件的波段数量"""
    try:
        raster = arcpy.Raster(tif_path)
        return raster.bandCount, ""
    except Exception as e:
        return -1, str(e)

def process_tif_files(tif_files, output_csv):
    """处理所有TIF文件并输出结果到CSV"""
    if not tif_files:
        return
    
    # 准备CSV文件（Python 2使用二进制模式）
    with open(output_csv, 'wb') as csvfile:
        fieldnames = ['文件名', '完整路径', '波段数量', '状态']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # 处理每个TIF文件
        for tif_path in tif_files:
            filename = os.path.basename(tif_path)
            band_count, status = get_band_count(tif_path)
            
            # 写入CSV
            writer.writerow({
                '文件名': filename,
                '完整路径': tif_path,
                '波段数量': band_count,
                '状态': status
            })
    
    return {
        'total': len(tif_files)
    }

def main():
    """主函数"""
    # 记录开始时间
    start_time = time.time()
    
    # 获取配置
    config = get_config()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(config['output_csv'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 查找所有TIF文件
    tif_files = find_all_tif_files(config['input_dir'])
    
    # 处理TIF文件
    process_tif_files(tif_files, config['output_csv'])
    
    # 记录结束时间并打印总运行时间
    end_time = time.time()
    total_time = end_time - start_time
    print("程序总运行时间: {0:.2f}秒".format(total_time))

if __name__ == "__main__":
    main()