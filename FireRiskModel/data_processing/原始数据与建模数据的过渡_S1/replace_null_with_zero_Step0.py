#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
程序名称: 空值替换为0工具
功能简介:
    1. 遍历指定年份的Interface和Intermix WUI原始CSV文件
    2. 读取每个CSV文件中的数据
    3. 统计文件中的空值数量
    4. 将所有空值替换为0
    5. 保存修改后的文件

输入文件:
    - Interface WUI文件: I:/Processing data/Grid_Clip_2005_interface_WUI.csv
    - Intermix WUI文件: I:/Processing data/Grid_Clip_2005_intermix_WUI.csv
    - 支持的年份: 2005, 2010, 2015, 2020

输出文件:
    - 与输入文件同名，直接覆盖原文件
"""

import pandas as pd
import os

def replace_null_with_zero(file_path):
    """
    将CSV文件中的所有空值替换为0
    
    参数:
    file_path: CSV文件路径
    
    返回:
    bool: 处理是否成功
    """
    try:
        print(f"正在读取文件: {file_path}")
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 统计替换前的空值数量
        null_count_before = df.isnull().sum().sum()
        
        # 将所有空值替换为0
        df.fillna(0, inplace=True)
        
        # 保存修改后的文件
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        
        print(f"文件已保存: {file_path}")
        print(f"共替换了 {null_count_before} 个空值")
        
        return True
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return False
    except Exception as e:
        print(f"处理文件时出错: {file_path}, 错误信息: {e}")
        return False

def main():
    """
    主函数
    """
    print("=== 开始将CSV文件中的空值替换为0 ===")
    
    # 定义需要处理的年份
    years = [2005, 2010, 2015, 2020]
    
    # 定义文件类型和路径模板
    file_types = ['interface', 'intermix']
    file_template = r"I:\Processing data\Grid_Clip_%d_%s_WUI.csv"
    
    # 循环处理每个年份和类型的文件
    for year in years:
        print(f"\n开始处理年份: {year}")
        for file_type in file_types:
            print(f"  处理文件类型: {file_type}")
            file_path = file_template % (year, file_type)
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"  文件不存在: {file_path}")
                continue
            
            # 替换空值为0
            success = replace_null_with_zero(file_path)
            
            if success:
                print(f"  {file_type} 类型文件处理完成")
            else:
                print(f"  {file_type} 类型文件处理失败")
        
        print(f"年份 {year} 所有类型文件处理完成")
    
    print("\n=== 所有文件处理完成 ===")

if __name__ == "__main__":
    main()