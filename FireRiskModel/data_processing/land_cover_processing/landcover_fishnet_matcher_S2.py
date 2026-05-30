#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版渔网匹配程序
直接使用CSV文件中的空间边界信息进行线性匹配,无需复杂空间索引

功能：
1. 读取TIF文件边界信息CSV
2. 从GDB文件中读取渔网数据图层
3. 执行渔网单元与TIF文件的空间匹配
4. 输出匹配结果到CSV文件

输入：
- TIF边界信息CSV:包含每个TIF文件的空间边界坐标
- Interface WUI Fire Risk GDB:包含Grid_Clip_2005/2010/2015/2020_interface_WUI图层
- Intermix WUI Fire Risk GDB:包含Grid_Clip_2005/2010/2015/2020_intermix_WUI图层

输出：
- 8个CSV文件,分别对应Interface和Intermix类型的2005/2010/2015/2020年份数据
- 输出路径:I:/Processing data/landcover/
- 文件名格式:Grid_Clip_YYYY_type_WUI_matches.csv
- CSV表头使用GRID_ID作为渔网栅格ID列名

"""

import os
import sys
import csv
import time
import argparse
import pandas as pd

# 设置工作目录
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(WORKING_DIR)

class SimpleFishnetMatcher:
    """简化版渔网匹配器，使用线性空间匹配"""
    
    def __init__(self, tif_csv_path, fishnet_path):
        """初始化匹配器
        
        参数:
        tif_csv_path: TIF文件边界信息CSV文件的路径
        fishnet_path: 渔网数据GDB文件的路径
        """
        self.tif_csv_path = tif_csv_path
        self.fishnet_path = fishnet_path
        self.tif_bounds = []  # 存储TIF文件的边界信息
        self.load_tif_bounds()
    
    def load_tif_bounds(self):
        """从CSV文件加载TIF边界信息"""
        print("正在加载TIF边界信息...")
        print("TIF边界CSV路径: {0}".format(self.tif_csv_path))
        try:
            # 检查CSV文件是否存在
            if not os.path.exists(self.tif_csv_path):
                print("TIF边界CSV文件不存在: {0}".format(self.tif_csv_path))
                sys.exit(1)
            
            self.tif_bounds = []
            
            # 读取CSV文件（Python 2.7兼容版本）
            with open(self.tif_csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                
                # 检查CSV字段是否正确
                expected_fields = ['File_Path', 'File_Name', 'MinX', 'MinY', 'MaxX', 'MaxY']
                if not all(field in reader.fieldnames for field in expected_fields):
                    print("CSV文件字段不完整，缺少必要字段")
                    print("期望字段: {0}".format(', '.join(expected_fields)))
                    print("实际字段: {0}".format(', '.join(reader.fieldnames)))
                    return
                
                # 读取CSV数据
                for row in reader:
                    try:
                        bound_info = {
                            'file_path': row['File_Path'],
                            'file_name': row['File_Name'],
                            'minx': float(row['MinX']),
                            'miny': float(row['MinY']),
                            'maxx': float(row['MaxX']),
                            'maxy': float(row['MaxY'])
                        }
                        self.tif_bounds.append(bound_info)
                    except ValueError as e:
                        print("解析TIF边界数据失败: {0}, 错误: {1}".format(row['File_Name'], str(e)))
                    except Exception as e:
                        print("处理TIF边界数据失败: {0}, 错误: {1}".format(row['File_Name'], str(e)))
            
            print("成功加载 {0} 个TIF文件的边界信息".format(len(self.tif_bounds)))
            
            if len(self.tif_bounds) == 0:
                print("警告: 没有成功加载任何TIF文件的边界信息")
            
        except Exception as e:
            print("加载TIF边界信息失败: {0}".format(str(e)))
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def check_bounds_overlap(self, fishnet_bounds, tif_bounds):
        """检查两个边界框是否重叠"""
        fishnet_minx, fishnet_miny, fishnet_maxx, fishnet_maxy = fishnet_bounds
        tif_minx, tif_miny, tif_maxx, tif_maxy = tif_bounds
        
        # 检查是否有重叠
        return not (fishnet_maxx < tif_minx or fishnet_minx > tif_maxx or 
                   fishnet_maxy < tif_miny or fishnet_miny > tif_maxy)
    
    def find_matching_blocks_for_unit(self, fishnet_bounds):
        """为单个渔网单元找到重叠的TIF文件"""
        matching_files = []
        
        for tif_info in self.tif_bounds:
            tif_bounds = (tif_info['minx'], tif_info['miny'], 
                         tif_info['maxx'], tif_info['maxy'])
            
            if self.check_bounds_overlap(fishnet_bounds, tif_bounds):
                matching_files.append({
                    'file_path': tif_info['file_path'],
                    'file_name': tif_info['file_name']
                })
        
        return matching_files
    
    def match_fishnet_to_tif(self, layer_name, output_file=None):
        """执行渔网与TIF文件的匹配"""
        print("开始渔网匹配处理，图层: {0}...".format(layer_name))
        start_time = time.time()
        
        # 读取指定图层的渔网数据
        fishnet_units = self.load_fishnet_units(layer_name)
        
        total_units = len(fishnet_units)
        matched_count = 0
        results = []
        
        print("开始处理 {0} 个渔网单元...".format(total_units))
        
        for i, fishnet_unit in enumerate(fishnet_units, 1):
            # 获取渔网单元的边界
            fishnet_bounds = self.get_fishnet_bounds(fishnet_unit)
            
            # 查找匹配的TIF文件
            matching_tifs = self.find_matching_blocks_for_unit(fishnet_bounds)
            
            # 保存结果
            result = {
                'fishnet_id': self.get_fishnet_id(fishnet_unit),
                'fishnet_bounds': fishnet_bounds,
                'matching_tif_count': len(matching_tifs),
                'matching_tif_files': [tif['file_path'] for tif in matching_tifs]  # 保存完整路径
            }
            results.append(result)
            
            if matching_tifs:
                matched_count += 1
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 输出统计信息
        print("\n" + "=" * 50)
        print("渔网匹配完成！")
        print("总渔网单元数: {0}".format(total_units))
        print("有匹配的单元数: {0}".format(matched_count))
        
        # 计算匹配率（避免除以零）
        if total_units > 0:
            match_rate = (matched_count / total_units) * 100
            avg_speed = total_units / processing_time if processing_time > 0 else 0
        else:
            match_rate = 0
            avg_speed = 0
            
        print("匹配率: {0:.1f}%".format(match_rate))
        print("处理时间: {0:.1f} 秒".format(processing_time))
        print("平均处理速度: {0:.0f} 单元/秒".format(avg_speed))
        
        # 保存结果到文件
        if output_file:
            self.save_results(results, output_file)
        
        return results
    
    def load_fishnet_units(self, layer_name):
        """加载渔网单元数据（从GDB文件中读取指定图层）"""
        print("正在加载渔网数据图层: {0}".format(layer_name))
        fishnet_units = []
        
        try:
            import geopandas as gpd
            import fiona
            
            # 确保fiona支持GDB格式
            if 'OpenFileGDB' not in fiona.supported_drivers:
                print("Fiona不支持GDB格式，请确保已安装GDAL的GDB驱动")
                return fishnet_units
            
            # 读取GDB中的指定图层
            fishnet_gdf = gpd.read_file(self.fishnet_path, layer=layer_name)
            print("成功读取图层: {0}，共 {1} 个单元".format(layer_name, len(fishnet_gdf)))
            
            # 转换为渔网单元列表，每个单元包含几何信息和ID
            for idx, row in fishnet_gdf.iterrows():
                fishnet_unit = {
                    'geometry': row.geometry,
                    'GRID_ID': row.get('GRID_ID', idx),  # 使用GRID_ID字段或索引
                    'attributes': row.to_dict()  # 保存所有属性
                }
                fishnet_units.append(fishnet_unit)
                
        except ImportError as e:
            print("导入错误: {0}".format(e))
            print("请确保已安装geopandas和fiona")
        except Exception as e:
            print("加载渔网数据失败: {0}".format(e))
            print("渔网路径: {0}".format(self.fishnet_path))
            print("图层名称: {0}".format(layer_name))
        
        return fishnet_units
    
    def get_fishnet_bounds(self, fishnet_unit):
        """获取渔网单元的边界坐标"""
        if 'geometry' in fishnet_unit:
            bounds = fishnet_unit['geometry'].bounds
            return (bounds[0], bounds[1], bounds[2], bounds[3])
        return (0, 0, 0, 0)
    
    def get_fishnet_id(self, fishnet_unit):
        """获取渔网单元的ID"""
        return fishnet_unit.get('GRID_ID', "Unknown_ID")
    
    def save_results(self, results, output_file):
        """保存匹配结果到文件"""
        try:
            output_dir = os.path.dirname(output_file)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            with open(output_file, 'w', newline='') as csvfile:  # 添加newline=''参数，避免Windows下的空行问题
                writer = csv.writer(csvfile)
                writer.writerow(['GRID_ID', 'Fishnet_MinX', 'Fishnet_MinY', 
                               'Fishnet_MaxX', 'Fishnet_MaxY', 'Matching_TIF_Count', 
                               'Matching_TIF_Files'])
                
                for result in results:
                    bounds = result['fishnet_bounds']
                    writer.writerow([
                        result['fishnet_id'],
                        bounds[0], bounds[1], bounds[2], bounds[3],
                        result['matching_tif_count'],
                        '; '.join(result['matching_tif_files'])
                    ])
            
            print("结果已保存到: {0}".format(output_file))
            
        except Exception as e:
            print("保存结果失败: {0}".format(str(e)))

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='简化版渔网匹配程序')
    parser.add_argument('--tif_csv', default=r'I:\Data_huanglin\Landcover\GLC_FCS30D_projected\tif_simple.csv', 
                       help='TIF文件边界信息CSV文件路径')
    parser.add_argument('--interface_gdb', default=r'I:\Processing data\Interface_WUI_Fire_Risk.gdb', 
                       help='Interface WUI Fire Risk GDB文件路径')
    parser.add_argument('--intermix_gdb', default=r'I:\Processing data\Intermix_WUI_Fire_Risk.gdb', 
                       help='Intermix WUI Fire Risk GDB文件路径')
    parser.add_argument('--output_dir', default=r'I:\Processing data\landcover match', 
                       help='输出结果目录路径')
    
    args = parser.parse_args()
    
    # 检查TIF CSV文件是否存在
    if not os.path.exists(args.tif_csv):
        print("错误：TIF边界信息CSV文件不存在！")
        print("请检查路径: {0}".format(args.tif_csv))
        sys.exit(1)
    
    # 检查GDB文件是否存在
    for gdb_path in [args.interface_gdb, args.intermix_gdb]:
        if not os.path.exists(gdb_path):
            print("错误：GDB文件不存在！")
            print("请检查路径: {0}".format(gdb_path))
            sys.exit(1)
    
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    print("=" * 60)
    print("简化版地表覆盖渔网匹配工具")
    print("=" * 60)
    print("TIF边界CSV: {0}".format(args.tif_csv))
    print("Interface GDB: {0}".format(args.interface_gdb))
    print("Intermix GDB: {0}".format(args.intermix_gdb))
    print("输出目录: {0}".format(args.output_dir))
    print("匹配方式: 线性空间匹配（无需复杂空间索引）")
    print("=" * 60)
    
    # 定义处理的年份和类型
    years = [2005, 2010, 2015, 2020]
    data_types = [('interface', args.interface_gdb), ('intermix', args.intermix_gdb)]
    
    try:
        # 创建匹配器（使用TIF CSV文件路径）
        matcher = SimpleFishnetMatcher(args.tif_csv, args.interface_gdb)  # 先使用interface_gdb
        
        # 处理所有数据类型和年份
        total_processed = 0
        
        for data_type, gdb_path in data_types:
            # 更新匹配器的GDB路径
            matcher.fishnet_path = gdb_path
            
            for year in years:
                # 生成图层名称
                layer_name = "Grid_Clip_{0}_{1}_WUI".format(year, data_type)
                
                print("\n" + "="*60)
                print("处理图层: {0}".format(layer_name))
                print("="*60)
                
                # 生成输出文件名
                output_file = os.path.join(args.output_dir, "{0}_matches.csv".format(layer_name))
                
                # 执行匹配
                results = matcher.match_fishnet_to_tif(layer_name, output_file)
                total_processed += 1
                
                print("\n图层 {0} 处理完成！".format(layer_name))
        
        print("\n" + "="*60)
        print("所有处理完成！")
        print("共处理 {0} 个图层".format(total_processed))
        print("输出文件已保存到: {0}".format(args.output_dir))
        print("="*60)
        
        # 添加合并功能
        print("\n开始合并各年份的Interface和Intermix结果...")
        for year in [2005, 2010, 2015, 2020]:
            print("\n处理年份: {0}".format(year))
            merge_results_for_year(args.output_dir, year)
        
    except Exception as e:
        print("匹配过程出错: {0}".format(str(e)))
        import traceback
        traceback.print_exc()
        sys.exit(1)


def merge_results_for_year(output_dir, year):
    """
    合并指定年份的Interface和Intermix结果，根据GRID_ID取交集，移除重复值
    
    参数:
    output_dir: 输出结果目录路径
    year: 要合并的年份
    """
    try:
        # 定义输入文件路径
        interface_file = os.path.join(output_dir, "Grid_Clip_{0}_interface_WUI_matches.csv".format(year))
        intermix_file = os.path.join(output_dir, "Grid_Clip_{0}_intermix_WUI_matches.csv".format(year))
        output_file = os.path.join(output_dir, "Grid_Clip_{0}_WUI_matches.csv".format(year))
        
        # 检查输入文件是否存在
        if not os.path.exists(interface_file):
            print("错误：Interface文件不存在: {0}".format(interface_file))
            return
            
        if not os.path.exists(intermix_file):
            print("错误：Intermix文件不存在: {0}".format(intermix_file))
            return
            
        print("读取Interface文件: {0}".format(interface_file))
        print("读取Intermix文件: {0}".format(intermix_file))
        
        # 使用pandas读取两个CSV文件
        import pandas as pd
        
        # 读取Interface文件
        df_interface = pd.read_csv(interface_file)
        print("Interface文件包含 {0} 条记录".format(len(df_interface)))
        
        # 读取Intermix文件
        df_intermix = pd.read_csv(intermix_file)
        print("Intermix文件包含 {0} 条记录".format(len(df_intermix)))
        
        # 合并两个DataFrame，基于GRID_ID取交集
        df_merged = pd.concat([df_interface, df_intermix])
        
        # 移除重复的GRID_ID记录，只保留第一条
        df_merged_unique = df_merged.drop_duplicates(subset=['GRID_ID'], keep='first')
        
        print("合并后共 {0} 条记录".format(len(df_merged)))
        print("去重后共 {0} 条记录".format(len(df_merged_unique)))
        
        # 保存合并后的结果
        df_merged_unique.to_csv(output_file, index=False)
        print("合并结果已保存到: {0}".format(output_file))
        
    except Exception as e:
        print("合并结果失败: {0}".format(str(e)))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()