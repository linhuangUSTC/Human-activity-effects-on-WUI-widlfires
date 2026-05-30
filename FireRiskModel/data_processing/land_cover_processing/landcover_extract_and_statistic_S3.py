# -*- coding: utf-8 -*-
r"""
地表覆盖统计分析程序

输入路径：
- GDB路径: I:\Processing data\WUI.gdb
- CSV路径: I:\Processing data\landcover match\Grid_Clip_2005_WUI_matches.csv

输出路径：
- 分析结果: 由命令行参数指定

功能：
1. 从GDB中加载渔网单元数据,获取GRID_ID
2. 对不同类型和年份的GRID_ID取并集
3. 匹配CSV文件中的GRID_ID
4. 裁剪对应的TIF文件
5. 统计指定波段的像元值，找出占比最大的前两个值
6. 输出结果到新的CSV文件
"""

import os
import sys
import csv
import numpy as np
import arcpy
import time
from arcpy.sa import ExtractByMask

def _pick_band_from_array(arr, band_index, band_count):
    """
    兼容 RasterToNumPyArray 多波段可能返回：
    - (bands, rows, cols)
    - (rows, cols, bands)
    band_index: 1-based
    """
    if arr.ndim != 3:
        return arr

    bi = max(band_index - 1, 0)

    # 情况1：bands-first
    if arr.shape[0] == band_count:
        if bi >= arr.shape[0]:
            raise ValueError("band_index out of range")
        return arr[bi, :, :]

    # 情况2：bands-last
    if arr.shape[2] == band_count:
        if bi >= arr.shape[2]:
            raise ValueError("band_index out of range")
        return arr[:, :, bi]

    # 无法判断时：尽量按最后一维
    if bi < arr.shape[2]:
        return arr[:, :, bi]
    return arr[:, :, 0]

class LandCoverStatsAnalyzer:
    """地表覆盖统计分析器"""
    
    def __init__(self, gdb_path, csv_path, output_csv, year):
        """初始化分析器
        
        参数:
        gdb_path: GDB文件路径
        csv_path: CSV文件路径
        output_csv: 输出CSV文件路径
        year: 年份
        """
        self.gdb_path = gdb_path
        self.csv_path = csv_path
        self.output_csv = output_csv
        self.year = year
        
        # 确定对应的波段索引
        band_mapping = {2005: 6, 2010: 11, 2015: 16, 2020: 21}
        self.band_index = band_mapping[year]
        
        # 生成图层名称
        self.wui_layer = "Grid_Clip_{}_WUI".format(year)
    
    def load_grid_ids_from_gdb(self):
        """从GDB中加载所有GRID_ID"""
        grid_ids = set()
        
        # 从WUI GDB中加载WUI图层的GRID_ID
        wui_path = os.path.join(self.gdb_path, self.wui_layer)
        with arcpy.da.SearchCursor(wui_path, ['GRID_ID']) as cursor:
            for row in cursor:
                grid_ids.add(str(row[0]))
        
        return grid_ids
    
    def load_tif_mapping_from_csv(self):
        """从CSV中加载GRID_ID与TIF文件的映射关系"""
        grid_tif_mapping = {}
        
        with open(self.csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                grid_id = row['GRID_ID']
                tif_files = row['Matching_TIF_Files'].split('; ')
                
                grid_tif_mapping[grid_id] = tif_files
        
        
        return grid_tif_mapping
    
    def load_fishnet_geometry(self, grid_id):
        """加载指定GRID_ID的渔网几何信息"""
        geometry = None
        
        # 从WUI GDB的WUI图层中查找
        wui_path = os.path.join(self.gdb_path, self.wui_layer)
        with arcpy.da.SearchCursor(wui_path, ['GRID_ID', 'SHAPE@']) as cursor:
            for row in cursor:
                if str(row[0]) == grid_id:
                    geometry = row[1]
                    return geometry
        
        return geometry
    
    def calculate_top_two_classes(self, grid_id, tif_files):
        """计算渔网栅格中像元数量占比最大的前两个值"""
        # 加载渔网几何信息
        geometry = self.load_fishnet_geometry(grid_id)
        
        all_pixels = []
        
        # 处理所有匹配的TIF文件
        for tif_file in tif_files:
            # 裁剪TIF文件到内存工作空间，避免文件锁定问题
            # 使用内存工作空间可以在内存中处理，不需要创建物理文件
            clipped_raster = ExtractByMask(tif_file, geometry)
            
            # 获取栅格的NoData值
            no_data_value = clipped_raster.noDataValue
            
            # 转换为numpy数组，同时获取NoData值
            try:
                raster_array = arcpy.RasterToNumPyArray(clipped_raster)
            except Exception as e:
                print("转换栅格到数组时出错:", e)
                continue
            
            # 获取栅格的波段数
            try:
                ras = arcpy.Raster(tif_file)
                band_count = int(ras.bandCount) if hasattr(ras, 'bandCount') else 1
            except Exception:
                band_count = 1
            
            # 选择特定波段的数据
            try:
                band_data = _pick_band_from_array(raster_array, self.band_index, band_count)
            except Exception as e:
                print("警告：TIF文件波段索引错误，跳过此文件。")
                print("TIF文件：", tif_file)
                print("错误信息：", e)
                continue  # 跳过当前TIF文件，继续处理下一个
            
            # 展平数组并过滤掉NoData值
            pixels = band_data.flatten()
            
            # 过滤NoData值
            if no_data_value is not None:
                # 确保数据类型一致后再比较
                try:
                    valid_pixels = pixels[pixels != float(no_data_value)]
                except (TypeError, ValueError):
                    valid_pixels = pixels[pixels != no_data_value]
            else:
                # 如果没有明确的NoData值，过滤掉0值（根据实际情况调整）
                valid_pixels = pixels[pixels != 0]
            
            # 移除调试信息，减少控制台输出
            
            all_pixels.extend(valid_pixels)
            
            # 内存工作空间的栅格会自动释放，不需要手动删除
            # 移除删除临时栅格的代码，避免删除原始TIF文件
        
        if not all_pixels:
            return [{'value': None, 'percentage': 0}, {'value': None, 'percentage': 0}]
        
        # 转换为numpy数组
        all_pixels_np = np.array(all_pixels)
        
        # 计算各类别像素数量
        unique_values, counts = np.unique(all_pixels_np, return_counts=True)
        
        # 计算总像素数
        total_pixels = len(all_pixels_np)
        
        # 创建值和百分比的列表
        value_percent_list = []
        for value, count in zip(unique_values, counts):
            percentage = (float(count) / total_pixels) * 100
            value_percent_list.append({'value': int(value), 'percentage': round(percentage, 2)})
        
        # 按百分比降序排序
        value_percent_list.sort(key=lambda x: x['percentage'], reverse=True)
        
        # 确保有两个结果
        if len(value_percent_list) >= 2:
            return value_percent_list[:2]
        elif len(value_percent_list) == 1:
            return [value_percent_list[0], {'value': None, 'percentage': 0}]
        else:
            return [{'value': None, 'percentage': 0}, {'value': None, 'percentage': 0}]
    
    def _save_results_to_csv(self, results, is_first_save=False):
        """保存结果到CSV文件
        
        参数:
        results: 要保存的结果列表
        is_first_save: 是否是第一次保存，第一次需要写入表头
        """
        mode = 'wb' if is_first_save else 'ab'  # 使用二进制模式兼容Python 2
        with open(self.output_csv, mode) as csvfile:
            fieldnames = ['GRID_ID', 'Top1_Value', 'Top1_Percentage', 'Top2_Value', 'Top2_Percentage']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if is_first_save:
                writer.writeheader()
            
            for result in results:
                writer.writerow(result)
        
    def process(self):
        """处理所有步骤并输出结果"""
        # 加载GDB中的GRID_ID
        print("正在从GDB加载GRID_ID...")
        gdb_grid_ids = self.load_grid_ids_from_gdb()
        print("从GDB加载到 {} 个GRID_ID".format(len(gdb_grid_ids)))
        
        # 加载CSV中的映射关系
        print("正在从CSV加载映射关系...")
        grid_tif_mapping = self.load_tif_mapping_from_csv()
        print("从CSV加载到 {} 个映射关系".format(len(grid_tif_mapping)))
        
        # 获取匹配的GRID_ID
        matched_grid_ids = gdb_grid_ids.intersection(grid_tif_mapping.keys())
        print("找到 {} 个匹配的GRID_ID".format(len(matched_grid_ids)))
        
        # 将GRID_ID转换为整数并排序
        sorted_grid_ids = sorted([int(grid_id) for grid_id in matched_grid_ids])
        # 再转换回字符串
        sorted_grid_ids_str = [str(grid_id) for grid_id in sorted_grid_ids]
        
        # 处理每个匹配的GRID_ID
        results = []
        processed_count = 0
        first_save_done = False
        batch_start_time = None  # 初始化批次开始时间变量
        
        for grid_id in sorted_grid_ids_str:
            # 记录每100个GRID_ID的开始时间（从第1个开始）
            if processed_count % 100 == 0:
                batch_start_time = time.time()   
                
            processed_count += 1  
               
            tif_files = grid_tif_mapping[grid_id]
            top_two = self.calculate_top_two_classes(grid_id, tif_files)
            
            result = {
                'GRID_ID': grid_id,
                'Top1_Value': top_two[0]['value'],
                'Top1_Percentage': top_two[0]['percentage'],
                'Top2_Value': top_two[1]['value'],
                'Top2_Percentage': top_two[1]['percentage']
            }
            results.append(result)
            
            # 每100个GRID_ID输出一次耗时并保存一次结果到csv中
            if processed_count % 100 == 0 and batch_start_time is not None:
                batch_end_time = time.time()
                elapsed_time = batch_end_time - batch_start_time
                print("已处理 {} 个GRID_ID".format(processed_count))
                print("处理100个GRID_ID耗时: {:.2f}秒".format(elapsed_time))
                self._save_results_to_csv(results, is_first_save=not first_save_done)
                first_save_done = True
                results = []  # 重置结果列表，准备下一批次
        
        # 保存剩余的结果
        if results:
            print("正在保存剩余 {} 个GRID_ID的结果...".format(len(results)))
            self._save_results_to_csv(results, is_first_save=not first_save_done)
        
        print("处理完成！共处理 {} 个GRID_ID".format(processed_count))
        print("结果文件路径: {}".format(self.output_csv))


def main():
    """主函数"""
    # 简单的参数处理，直接读取年份和输出路径
    year = 2005  # 默认年份
    output_csv = r"I:\Processing data\landcover match\Grid_Clip_{}_WUI_LandCover_Stats.csv".format(year)
    
    # 如果提供了年份参数
    if len(sys.argv) > 1:
        year = int(sys.argv[1])
        output_csv = r"I:\Processing data\landcover match\Grid_Clip_{}_WUI_LandCover_Stats.csv".format(year)
    
    # 如果还提供了输出路径参数
    if len(sys.argv) > 2:
        output_csv = sys.argv[2]
    
    # 定义GDB路径 - 直接读取WUI.gdb
    gdb_path = r'I:\Processing data\WUI.gdb'
    
    # 定义CSV路径
    csv_path = r'I:\Processing data\landcover match\Grid_Clip_{}_WUI_matches.csv'.format(year)
    
    # 创建分析器实例
    analyzer = LandCoverStatsAnalyzer(gdb_path, csv_path, output_csv, year)
    
    # 执行分析
    analyzer.process()
    
    print("分析完成！")


if __name__ == "__main__":
    main()