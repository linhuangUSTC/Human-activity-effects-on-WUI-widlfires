# -- coding: utf-8 --
# --------------------------------------------------------------------------------
# 程序名称：GDB 多要素类全字段批量导出工具
# 功能说明：
# 1. 批量处理指定年份的要素类（Grid_Clip_年份_interface_WUI）
# 2. 导出每个要素类的所有非几何字段到CSV
# 3. 结果保存至I:\Processing data文件夹
# --------------------------------------------------------------------------------

import arcpy
import codecs  # 处理编码，确保中文正常显示
import os  # 处理路径

# ----------------------参数配置（可根据实际情况修改）----------------------
output_gdb = r"I:\Processing data\Intermix_WUI_Fire_Risk.gdb"  # 要素类所在的GDB路径（请确认是否正确）
output_folder = r"I:\Processing data"  # CSV保存文件夹
wui_years = [2005,2010,2015,2020]  # 需要处理的年份列表
# -------------------------------------------------------------------------

# 设置工作空间为目标GDB
arcpy.env.workspace = output_gdb

# 循环处理每个年份的要素类
for year in wui_years:
    # 生成要素类名称（格式：Grid_Clip_年份_interface_WUI）
    fc_name = "Grid_Clip_{0}_intermix_WUI".format(year)

    # 1. 验证要素类是否存在
    if not arcpy.Exists(fc_name):
        print("警告：要素类 {0} 不存在，已跳过".format(fc_name))
        continue  # 跳过不存在的要素类，继续处理下一个年份

    # 2. 获取要素类的所有非几何字段（排除Shape等几何字段，避免导出无效信息）
    all_fields = arcpy.ListFields(fc_name)
    # 筛选字段：排除几何类型字段（Geometry）和系统隐藏字段（如OBJECTID通常保留，按需调整）
    export_fields = [f.name for f in all_fields if f.type != "Geometry"]

    if not export_fields:
        print("警告：要素类 {0} 无有效导出字段，已跳过".format(fc_name))
        continue

    # 3. 定义CSV输出路径（文件名包含年份，便于区分）
    csv_filename = "Grid_Clip_{0}_intermix_WUI.csv".format(year)  # CSV文件名
    csv_path = os.path.join(output_folder, csv_filename)

    # 4. 写入CSV文件
    try:
        # 用utf-8编码打开文件，确保中文正常
        with codecs.open(csv_path, "w", encoding="utf-8") as f:
            # 写入表头（字段名）
            f.write(",".join(export_fields) + "\n")

            # 用SearchCursor读取所有字段数据
            with arcpy.da.SearchCursor(fc_name, export_fields) as cursor:
                for row in cursor:
                    # 处理空值（将None转为空字符串，避免写入"None"）
                    row_data = [str(value) if value is not None else "" for value in row]
                    # 写入一行数据（字段值用逗号分隔）
                    f.write(",".join(row_data) + "\n")

        print("成功导出：{0} → 保存至 {1}".format(fc_name, csv_path))

    except Exception as e:
        print("导出失败 {0}：错误信息：{1}".format(fc_name, str(e)))

print("\n所有年份处理完成！")