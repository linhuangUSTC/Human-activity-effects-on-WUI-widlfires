"""
程序名称: 柯本气象带分类统计器 (Koppen Climate Zone Stats Calculator)
主要功能
  1. 读取柯本气象带TIF数据(1991-2020年)
  2. 支持多个年份(2005、2010、2015、2020)的渔网单元数据处理
  3. 自动处理Intermix和Interface数据的去重(取并集)
  4. 提供精确的渔网单元边界裁剪功能
  5. 统计各气象带类型的像素数和占比
  6. 输出包含GRID_ID和气象带占比的CSV文件
输入路径:
  - 柯本气象带TIF: I:\Data_huanglin\Koppen 气象带\koppen_geiger_V2\1991_2020\project\koppen_geiger_0p00833333_Project.tif
  - Intermix GDB: I:\Processing data\Intermix_WUI_Fire_Risk.gdb
  - Interface GDB: I:\Processing data\Interface_WUI_Fire_Risk.gdb
输出路径:
  - 统计结果: I:\Processing data\Koppe match\Grid_Clip_{年份}_Koppen_Stats.csv
"""

import os
import csv
import numpy as np
import rasterio
import rasterio.mask
import fiona
from pyproj import Transformer

INTERMIX_GDB_PATH = r'I:\Processing data\Intermix_WUI_Fire_Risk.gdb'
INTERFACE_GDB_PATH = r'I:\Processing data\Interface_WUI_Fire_Risk.gdb'
KOPPEN_TIF_PATH = r'I:\Data_huanglin\Koppen 气象带\koppen_geiger_V2\1991_2020\project\koppen_geiger_0p00833333_Project.tif'
RESULTS_DIR = r'I:\Processing data\Koppe match'

SUPPORTED_YEARS = [2005, 2010, 2015, 2020]

class KoppenClimateZoneCalculator:
    """
    柯本气象带分类统计模块
    负责处理渔网单元，裁剪到柯本气象带数据，并统计气象带类型占比
    """

    def __init__(self, koppen_tif_path):
        """
        初始化柯本气象带计算器

        Args:
            koppen_tif_path: 柯本气象带TIF文件路径
        """
        self.koppen_tif_path = koppen_tif_path
        self.raster_ds = None
        self.raster_crs = None
        self.raster_nodata = None
        self._load_raster()

    def _load_raster(self):
        """
        加载柯本气象带TIF数据
        """
        print(f"正在加载柯本气象带数据: {self.koppen_tif_path}")

        if not os.path.exists(self.koppen_tif_path):
            raise FileNotFoundError(f"TIF文件不存在: {self.koppen_tif_path}")

        self.raster_ds = rasterio.open(self.koppen_tif_path)
        self.raster_crs = self.raster_ds.crs
        self.raster_nodata = self.raster_ds.nodata

        print(f"TIF属性: 波段数={self.raster_ds.count}, 像素大小={self.raster_ds.res[0]}x{self.raster_ds.res[1]}, 图像尺寸={self.raster_ds.width}x{self.raster_ds.height}, 坐标系={self.raster_crs}")

    def process_layer_with_zonal_stats(self, gdb_path, layer_name):
        """
        使用rasterio和fiona处理图层并计算区域统计

        Parameters:
        gdb_path: GDB文件路径
        layer_name: 图层名称

        Returns:
        list: 处理结果列表
        """
        results = []

        try:
            print(f"\n正在处理图层: {layer_name}")

            if not os.path.exists(gdb_path):
                print(f"警告: GDB文件不存在: {gdb_path}")
                return []
            
            with fiona.open(gdb_path, layer=layer_name) as data_source:
                layer_crs = data_source.crs
                raster_crs = self.raster_ds.crs

                for i, feature in enumerate(data_source):
                    grid_id = feature['properties'].get('GRID_ID')
                    if i % 50000 == 0 and i > 0:
                        print(f"  已处理 {i+1} 个要素")

                    geometry = feature['geometry']

                    if layer_crs != raster_crs:
                        transform = Transformer.from_crs(layer_crs, raster_crs, always_xy=True)
                        geometry = self._transform_geometry(geometry, transform)

                    try:
                        out_image, out_transform = rasterio.mask.mask(
                            self.raster_ds,
                            [geometry],
                            crop=True,
                            all_touched=True
                        )
                    except Exception:
                        continue

                    if out_image.size == 0:
                        continue

                    koppen_data = out_image[0]

                    if self.raster_nodata is not None:
                        valid_mask = koppen_data != self.raster_nodata
                    else:
                        valid_mask = koppen_data != 0

                    valid_pixels = koppen_data[valid_mask]

                    if len(valid_pixels) == 0:
                        continue

                    unique_values, counts = np.unique(valid_pixels, return_counts=True)

                    koppen_stats = {}
                    for value, count in zip(unique_values, counts):
                        koppen_stats[int(value)] = {
                            'pixel_count': int(count),
                            'area': float(count * self.raster_ds.res[0] * abs(self.raster_ds.res[1]))
                        }

                    total_pixels = sum(stats['pixel_count'] for stats in koppen_stats.values())
                    for zone_code, stats in koppen_stats.items():
                        stats['percentage'] = round((stats['pixel_count'] / total_pixels) * 100, 4)

                    if koppen_stats:
                        dominant_zone = max(koppen_stats.items(), key=lambda x: x[1]['pixel_count'])
                        results.append({
                            'grid_id': grid_id,
                            'koppen_stats': {dominant_zone[0]: dominant_zone[1]}
                        })

            print(f"成功处理 {len(results)} 个渔网单元")
            return results

        except Exception as e:
            print(f"处理失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _transform_geometry(self, geometry, transformer):
        """
        转换几何对象的坐标系

        Args:
            geometry: 原始几何对象
            transformer: 坐标转换器

        Returns:
            转换后的几何对象
        """
        def transform_coords(coords):
            if isinstance(coords[0], (list, tuple)):
                return [transform_coords(c) for c in coords]
            else:
                x, y = transformer.transform(coords[0], coords[1])
                return [x, y]

        geom_type = geometry['type']
        coords = geometry['coordinates']

        if geom_type == 'Point':
            x, y = transformer.transform(coords[0], coords[1])
            return {'type': geom_type, 'coordinates': [x, y]}
        elif geom_type == 'LineString':
            return {'type': geom_type, 'coordinates': transform_coords(coords)}
        elif geom_type == 'Polygon':
            return {'type': geom_type, 'coordinates': [transform_coords(ring) for ring in coords]}
        elif geom_type == 'MultiPoint':
            return {'type': geom_type, 'coordinates': transform_coords(coords)}
        elif geom_type == 'MultiLineString':
            return {'type': geom_type, 'coordinates': [transform_coords(line) for line in coords]}
        elif geom_type == 'MultiPolygon':
            return {'type': geom_type, 'coordinates': [[transform_coords(ring) for ring in poly] for poly in coords]}
        else:
            return geometry

def process_year(year, calculator):
    """
    处理指定年份的数据 - 分别处理intermix和interface

    Parameters:
    year: 年份
    calculator: KoppenClimateZoneCalculator实例
    """
    print("\n" + "="*60)
    print(f"开始处理 {year} 年的数据")
    print("="*60)

    interface_layer = f"Grid_Clip_{year}_interface_WUI"
    interface_output_file = os.path.join(RESULTS_DIR, f"Grid_Clip_{year}_interface_Koppen_Stats.csv")
    print(f"Interface输出路径: {interface_output_file}")

    interface_results = calculator.process_layer_with_zonal_stats(INTERFACE_GDB_PATH, interface_layer)
    save_results(interface_results, interface_output_file)

    intermix_layer = f"Grid_Clip_{year}_intermix_WUI"
    intermix_output_file = os.path.join(RESULTS_DIR, f"Grid_Clip_{year}_intermix_Koppen_Stats.csv")
    print(f"Intermix输出路径: {intermix_output_file}")

    intermix_results = calculator.process_layer_with_zonal_stats(INTERMIX_GDB_PATH, intermix_layer)
    save_results(intermix_results, intermix_output_file)

    print(f"{year} 年的数据处理完成")

def save_results(results, output_file):
    """
    将处理结果保存到CSV文件

    Parameters:
    results: 处理结果列表
    output_file: 输出文件路径
    """
    print(f"\n正在保存处理结果到: {output_file}")

    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', encoding='utf-8-sig', newline='') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(['GRID_ID', 'Koppen_Type_Code', 'Pixel_Count', 'Percentage'])

            for result in results:
                grid_id = result['grid_id']
                koppen_stats = result.get('koppen_stats', {})
                if koppen_stats:
                    for type_code, stats in koppen_stats.items():
                        writer.writerow([
                            grid_id,
                            type_code,
                            stats['pixel_count'],
                            stats['percentage']
                        ])

        print("结果保存成功!")

    except Exception as e:
        print(f"保存结果失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    calculator = KoppenClimateZoneCalculator(KOPPEN_TIF_PATH)

    for year in SUPPORTED_YEARS:
        process_year(year, calculator)

    print("\n" + "="*60)
    print("所有年份的数据处理完成!")
    print("="*60)
