# -*- coding: utf-8 -*-
r"""
地表覆盖统计分析程序(ArcPy 并行稳定版)

1) ExtractByMask 结果不再 save 成临时 tif(并行写盘/删除最容易锁冲突)
2) 每个子进程设置独立 scratch/workspace,减少 ArcPy 临时文件互抢
3) 主进程预先读取 GRID_ID -> WKT 几何字典，子进程不再反复扫游标
4) 不再到原始 tif 目录递归删除临时文件（非常危险且会删到别的进程文件)
5) CSV 只由主进程写入（原来的 threading.Lock 每次新建无效）

输入：
- GDB路径: I:\Processing data\WUI.gdb
- CSV路径: I:\Processing data\landcover match\Grid_Clip_2005_WUI_matches.csv

输出：
- 统计CSV: I:\Processing data\landcover match\Grid_Clip_2005_WUI_LandCover_Stats.csv
注:
经检验,输出的主要landcover类型与占比与arcmap的结果一致,可以放心运行
单次运行一个年份,在year = '2005'中修改即可
"""

import os
import sys
import csv
import time
import multiprocessing

import numpy as np
import arcpy
from arcpy.sa import ExtractByMask

# 允许覆盖
arcpy.env.overwriteOutput = True


def _safe_mkdir(p):
    try:
        if not os.path.exists(p):
            os.makedirs(p)
    except Exception:
        pass


def _extent_tuple(ext):
    return (ext.XMin, ext.YMin, ext.XMax, ext.YMax)


def _extent_intersects(a, b):
    # a,b: (xmin,ymin,xmax,ymax)
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def build_grid_wkt_dict(gdb_path, wui_layer):
    """主进程一次性读取 GRID_ID -> (WKT, SR_WKT)"""
    wui_path = os.path.join(gdb_path, wui_layer)
    if not arcpy.Exists(wui_path):
        raise RuntimeError(u"图层不存在: {}".format(wui_path))

    sr = arcpy.Describe(wui_path).spatialReference
    sr_wkt = sr.exportToString() if sr else None

    d = {}
    with arcpy.da.SearchCursor(wui_path, ['GRID_ID', 'SHAPE@WKT']) as cur:
        for gid, wkt in cur:
            if gid is None or wkt is None:
                continue
            d[str(gid)] = (wkt, sr_wkt)
    return d


def load_tif_mapping_from_csv(csv_path):
    """从CSV中加载 GRID_ID -> [tif1, tif2, ...]"""
    m = {}
    # ArcMap/py2 建议用 'rb'
    with open(csv_path, 'rb') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gid = str(row.get('GRID_ID', '')).strip()
            s = row.get('Matching_TIF_Files', None)
            if (not gid) or (s is None):
                continue
            # 兼容 '; ' 或 ';'
            parts = [p.strip() for p in s.replace('; ', ';').split(';') if p.strip()]
            if parts:
                m[gid] = parts
    return m


def pool_initializer(tmp_root):
    """子进程初始化：设置独立 scratch/workspace，避免并行锁冲突"""
    try:
        arcpy.CheckOutExtension("Spatial")
    except Exception:
        pass

    pid = os.getpid()
    pdir = os.path.join(tmp_root, "p_{0}".format(pid))
    _safe_mkdir(pdir)

    arcpy.env.workspace = pdir
    arcpy.env.scratchWorkspace = pdir
    arcpy.env.overwriteOutput = True


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


def calculate_top_two_classes(task):
    """
    子进程处理单个 GRID_ID
    返回：(grid_id, [top1, top2], status)
    """
    (grid_id, band_index, tif_files, wkt, sr_wkt) = task

    # 从WKT恢复几何
    try:
        sr = arcpy.SpatialReference()
        if sr_wkt:
            sr.loadFromString(sr_wkt)
        geom = arcpy.FromWKT(wkt, sr)
    except Exception:
        return (grid_id,
                [{'value': None, 'percentage': 0}, {'value': None, 'percentage': 0}],
                'bad_geometry')

    # 尝试修复几何（失败不致命）
    try:
        geom = geom.buffer(0)
    except Exception:
        pass

    geom_ext = _extent_tuple(geom.extent)

    all_pixels = []

    for tif in tif_files:
        try:
            if not os.path.exists(tif):
                continue

            # 快速相交判断，不相交直接跳过，减少奇怪 999999
            try:
                r_ext = _extent_tuple(arcpy.Describe(tif).extent)
                if not _extent_intersects(geom_ext, r_ext):
                    continue
            except Exception:
                pass

            ras = arcpy.Raster(tif)
            band_count = int(ras.bandCount) if hasattr(ras, 'bandCount') else 1

            # 核心修复：不落地 save，直接内存读取
            clipped = ExtractByMask(ras, geom)

            arr = arcpy.RasterToNumPyArray(clipped)

            band_data = _pick_band_from_array(arr, band_index, band_count)

            pix = band_data.ravel()

            # NoData 过滤
            nd = None
            try:
                nd = clipped.noDataValue
            except Exception:
                nd = None

            if nd is not None:
                try:
                    pix = pix[pix != float(nd)]
                except Exception:
                    pix = pix[pix != nd]
            else:
                # 如果你的 0 是有效类别，请改成你的真实 NoData（如 255 或 -9999），或直接不滤
                pix = pix[pix != 0]

            if pix.size:
                all_pixels.append(pix)

        except arcpy.ExecuteError:
            # 捕获ArcPy执行错误，继续下一张tif
            # msg = arcpy.GetMessages(2)  # 如需打印可打开
            continue
        except Exception:
            continue

    if not all_pixels:
        return (grid_id,
                [{'value': None, 'percentage': 0}, {'value': None, 'percentage': 0}],
                'no_pixels')

    try:
        all_pixels_np = np.concatenate(all_pixels)
    except Exception:
        return (grid_id,
                [{'value': None, 'percentage': 0}, {'value': None, 'percentage': 0}],
                'concat_fail')

    # 统计top2
    u, c = np.unique(all_pixels_np, return_counts=True)
    total = float(c.sum())
    per = (c.astype('float64') / total) * 100.0

    order = np.argsort(-per)
    u = u[order]
    per = per[order]

    top1 = {'value': int(u[0]), 'percentage': round(float(per[0]), 2)}
    if len(u) > 1:
        top2 = {'value': int(u[1]), 'percentage': round(float(per[1]), 2)}
    else:
        top2 = {'value': None, 'percentage': 0}

    return (grid_id, [top1, top2], 'ok')


class LandCoverStatsAnalyzer(object):
    def __init__(self, gdb_path, csv_path, output_csv, year):
        self.gdb_path = gdb_path
        self.csv_path = csv_path
        self.output_csv = output_csv
        self.year = year

        band_mapping = {2005: 6, 2010: 11, 2015: 16, 2020: 21}
        self.band_index = band_mapping[year]
        self.wui_layer = "Grid_Clip_{0}_WUI".format(year)

    def load_grid_ids_from_gdb(self):
        ids = set()
        wui_path = os.path.join(self.gdb_path, self.wui_layer)
        if not arcpy.Exists(wui_path):
            print(u"错误：图层不存在 {0}".format(wui_path))
            return ids

        with arcpy.da.SearchCursor(wui_path, ['GRID_ID']) as cur:
            for (gid,) in cur:
                if gid is None:
                    continue
                ids.add(str(gid))
        return ids

    def process(self):
        # 主进程读取数据
        gdb_ids = self.load_grid_ids_from_gdb()
        map_tif = load_tif_mapping_from_csv(self.csv_path)

        matched = gdb_ids.intersection(map_tif.keys())
        matched_sorted = sorted([int(x) for x in matched])
        matched_sorted = [str(x) for x in matched_sorted]

        # 预构建 GRID_ID -> WKT
        grid_wkt = build_grid_wkt_dict(self.gdb_path, self.wui_layer)

        # 临时目录根目录
        tmp_root = r"C:\Temp\LandCoverProcess"
        _safe_mkdir(tmp_root)

        # 组装任务
        tasks = []
        for gid in matched_sorted:
            if gid not in grid_wkt:
                continue
            wkt, sr_wkt = grid_wkt[gid]
            tasks.append((gid, self.band_index, map_tif[gid], wkt, sr_wkt))

        # 并行参数（你可按机器调整）
        pool_size = max(1, int(multiprocessing.cpu_count() * 0.75))
        print(u"使用进程数: {0}".format(pool_size))

        # 主进程写CSV（避免并行写冲突）
        fieldnames = ['GRID_ID', 'Top1_Value', 'Top1_Percentage',
                      'Top2_Value', 'Top2_Percentage']

        with open(self.output_csv, 'wb') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            processed = 0
            batch_start = time.time()

            pool = multiprocessing.Pool(processes=pool_size,
                                        initializer=pool_initializer,
                                        initargs=(tmp_root,))
            try:
                for gid, top_two, status in pool.imap_unordered(calculate_top_two_classes, tasks):
                    processed += 1

                    row = {
                        'GRID_ID': gid,
                        'Top1_Value': top_two[0]['value'],
                        'Top1_Percentage': top_two[0]['percentage'],
                        'Top2_Value': top_two[1]['value'],
                        'Top2_Percentage': top_two[1]['percentage']
                    }
                    writer.writerow(row)

                    if processed % 2000 == 0:
                        elapsed = time.time() - batch_start
                        print(u"已处理 {0} 个GRID_ID, 最近2000个耗时 {1:.2f} 秒".format(processed, elapsed))
                        batch_start = time.time()
            finally:
                pool.close()
                pool.join()


def main():
    year = 2020
    if len(sys.argv) > 1:
        year = int(sys.argv[1])

    out_csv = r"I:\Processing data\landcover match\Grid_Clip_{0}_WUI_LandCover_Stats.csv".format(year)
    if len(sys.argv) > 2:
        out_csv = sys.argv[2]

    gdb_path = r"I:\Processing data\WUI.gdb"
    csv_path = r"I:\Processing data\landcover match\Grid_Clip_{0}_WUI_matches.csv".format(year)

    analyzer = LandCoverStatsAnalyzer(gdb_path, csv_path, out_csv, year)
    analyzer.process()


if __name__ == "__main__":
    main()
