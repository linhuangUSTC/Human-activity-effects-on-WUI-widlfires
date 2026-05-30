#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算Fire_frequency并存储到数据文件中的程序

计算方法:

# 全局指标
1. Fire_frequency_pop = (All_fire_num / (PopD * 85.9329)) * 1000  # 基于人口密度的火灾频率
2. Fire_frequency_per_WUI = All_fire_num / WUI_area  # 基于WUI面积的火灾频率
3. Fire_area_density = All_fire_area / WUI_area  # 基于WUI面积的归一化火灾面积

# Interface类型指标
4. Interface_Fire_frequency_pop = (Interface_Fire_num / (PopD * 85.9329)) * 1000  # Interface基于人口密度的火灾频率
5. Interface_Fire_frequency_per_WUI = Interface_Fire_num / Interface_WUI_area  # Interface基于WUI面积的火灾频率
6. Interface_Fire_area_density = Interface_Fire_area / Interface_WUI_area  # Interface基于WUI面积的归一化火灾面积

# Intermix类型指标
7. Intermix_Fire_frequency_pop = (Intermix_Fire_num / (PopD * 85.9329)) * 1000  # Intermix基于人口密度的火灾频率
8. Intermix_Fire_frequency_per_WUI = Intermix_Fire_num / Intermix_WUI_area  # Intermix基于WUI面积的火灾频率
9. Intermix_Fire_area_density = Intermix_Fire_area / Intermix_WUI_area  # Intermix基于WUI面积的归一化火灾面积

数据输入与输出路径: I:/Processing data/Artificial influencing factors.csv
"""

import pandas as pd
import os

def calculate_fire_frequency():
    """
    读取数据文件，计算Fire_frequency并保存到原文件
    """
    # 定义文件路径
    file_path = "I:/Processing data/Artificial influencing factors.csv"
    
    print(f"正在读取数据文件: {file_path}")
    
    try:
        # 读取数据
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"成功读取数据，共 {len(df)} 行，{len(df.columns)} 列")
        print(f"数据列名: {list(df.columns)}")
        
        # 计算Fire_frequency
        print("\n开始计算Fire_frequency...")
        
        # 检查是否存在所需列
        required_columns = ['All_fire_num', 'PopD', 'All_WUI_area', 'All_fire_area',
                           'Interface_Fire_num', 'Interface_Fire_area', 'Interface_WUI_area',
                           'Intermix_Fire_num', 'Intermix_Fire_area', 'Intermix_WUI_area']
        for col in required_columns:
            if col not in df.columns:
                print(f"错误: 数据文件中缺少必要的列 '{col}'")
                return False
        
        # 计算Fire_frequency = (All_fire_num / (PopD * 85.9329)) * 1000
        # 处理PopD为0的情况，避免除以零
        print("使用向量计算方法处理大数据集...")
        import numpy as np
        
        # 使用numpy向量计算，提高效率
        popd_array = df['PopD'].values
        fire_num_array = df['All_fire_num'].values
        wui_area_array = df['All_WUI_area'].values
        fire_area_array = df['All_fire_area'].values
        
        # Interface类型数据
        interface_fire_num_array = df['Interface_Fire_num'].values
        interface_fire_area_array = df['Interface_Fire_area'].values
        interface_wui_area_array = df['Interface_WUI_area'].values
        
        # Intermix类型数据
        intermix_fire_num_array = df['Intermix_Fire_num'].values
        intermix_fire_area_array = df['Intermix_Fire_area'].values
        intermix_wui_area_array = df['Intermix_WUI_area'].values
        
        # 创建条件数组，避免除以零
        popd_non_zero = popd_array != 0
        wui_non_zero = wui_area_array != 0
        interface_wui_non_zero = interface_wui_area_array != 0
        intermix_wui_non_zero = intermix_wui_area_array != 0
        
        # 初始化结果数组为NaN，用于表示"假0"（无意义的值）
        fire_frequency_pop = np.full(len(df), np.nan)
        fire_frequency_per_wui = np.full(len(df), np.nan)
        fire_area_density = np.full(len(df), np.nan)
        
        # Interface类型结果数组
        interface_fire_frequency_pop = np.full(len(df), np.nan)
        interface_fire_frequency_per_wui = np.full(len(df), np.nan)
        interface_fire_area_density = np.full(len(df), np.nan)
        
        # Intermix类型结果数组
        intermix_fire_frequency_pop = np.full(len(df), np.nan)
        intermix_fire_frequency_per_wui = np.full(len(df), np.nan)
        intermix_fire_area_density = np.full(len(df), np.nan)
        
        # 全局指标计算
        # 1. Fire_frequency_pop - 基于人口密度的火灾频率
        fire_frequency_pop[popd_non_zero] = (fire_num_array[popd_non_zero] / (popd_array[popd_non_zero] * 85.9329)) * 1000
        
        # 2. Fire_frequency_per_WUI 和 Fire_area_density - 基于WUI面积的指标
        # 当WUI面积大于0时
        fire_frequency_per_wui[wui_non_zero] = fire_num_array[wui_non_zero] / wui_area_array[wui_non_zero]
        fire_area_density[wui_non_zero] = fire_area_array[wui_non_zero] / wui_area_array[wui_non_zero]
        
        # Interface类型指标计算
        # 1. Interface_Fire_frequency_pop - 基于人口密度的火灾频率
        interface_fire_frequency_pop[popd_non_zero] = (interface_fire_num_array[popd_non_zero] / (popd_array[popd_non_zero] * 85.9329)) * 1000
        
        # 2. Interface_Fire_frequency_per_WUI 和 Interface_Fire_area_density - 基于WUI面积的指标
        # 当Interface WUI面积大于0时
        interface_fire_frequency_per_wui[interface_wui_non_zero] = interface_fire_num_array[interface_wui_non_zero] / interface_wui_area_array[interface_wui_non_zero]
        interface_fire_area_density[interface_wui_non_zero] = interface_fire_area_array[interface_wui_non_zero] / interface_wui_area_array[interface_wui_non_zero]
        
        # Intermix类型指标计算
        # 1. Intermix_Fire_frequency_pop - 基于人口密度的火灾频率
        intermix_fire_frequency_pop[popd_non_zero] = (intermix_fire_num_array[popd_non_zero] / (popd_array[popd_non_zero] * 85.9329)) * 1000
        
        # 2. Intermix_Fire_frequency_per_WUI 和 Intermix_Fire_area_density - 基于WUI面积的指标
        # 当Intermix WUI面积大于0时
        intermix_fire_frequency_per_wui[intermix_wui_non_zero] = intermix_fire_num_array[intermix_wui_non_zero] / intermix_wui_area_array[intermix_wui_non_zero]
        intermix_fire_area_density[intermix_wui_non_zero] = intermix_fire_area_array[intermix_wui_non_zero] / intermix_wui_area_array[intermix_wui_non_zero]
        
        # 限制归一化过火面积的最大值为1
        fire_area_density = np.minimum(fire_area_density, 1)
        interface_fire_area_density = np.minimum(interface_fire_area_density, 1)
        intermix_fire_area_density = np.minimum(intermix_fire_area_density, 1)
        
        # 将结果赋值回DataFrame
        df['Fire_frequency_pop'] = fire_frequency_pop
        df['Fire_frequency_per_WUI'] = fire_frequency_per_wui
        df['Fire_area_density'] = fire_area_density
        
        # Interface类型结果赋值
        df['Interface_Fire_frequency_pop'] = interface_fire_frequency_pop
        df['Interface_Fire_frequency_per_WUI'] = interface_fire_frequency_per_wui
        df['Interface_Fire_area_density'] = interface_fire_area_density
        
        # Intermix类型结果赋值
        df['Intermix_Fire_frequency_pop'] = intermix_fire_frequency_pop
        df['Intermix_Fire_frequency_per_WUI'] = intermix_fire_frequency_per_wui
        df['Intermix_Fire_area_density'] = intermix_fire_area_density
        
        print("所有频率和密度计算完成")
        
        # 显示计算结果的基本统计信息
        print("\nFire_frequency_pop（基于人口密度）的统计信息:")
        print(f"最大值: {df['Fire_frequency_pop'].max():.4f}")
        print(f"最小值: {df['Fire_frequency_pop'].min():.4f}")
        print(f"平均值: {df['Fire_frequency_pop'].mean():.4f}")
        print(f"中位数: {df['Fire_frequency_pop'].median():.4f}")
        
        print("\nFire_frequency_per_WUI（基于WUI面积）的统计信息:")
        print(f"最大值: {df['Fire_frequency_per_WUI'].max():.4f}")
        print(f"最小值: {df['Fire_frequency_per_WUI'].min():.4f}")
        print(f"平均值: {df['Fire_frequency_per_WUI'].mean():.4f}")
        print(f"中位数: {df['Fire_frequency_per_WUI'].median():.4f}")
        
        print("\nFire_area_density（基于WUI面积的归一化火灾面积）的统计信息:")
        print(f"最大值: {df['Fire_area_density'].max():.4f}")
        print(f"最小值: {df['Fire_area_density'].min():.4f}")
        print(f"平均值: {df['Fire_area_density'].mean():.4f}")
        print(f"中位数: {df['Fire_area_density'].median():.4f}")
        
        # 保存结果回原文件
        print(f"\n正在保存结果到原文件: {file_path}")
        df.to_csv(file_path, index=False, encoding='utf-8')
        print("保存成功!")
        
        return True
        
    except FileNotFoundError:
        print(f"错误: 文件未找到 - {file_path}")
        return False
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    主函数
    """
    print("=== Fire_frequency 计算程序 ===\n")
    
    success = calculate_fire_frequency()
    
    if success:
        print("\n程序执行成功!")
    else:
        print("\n程序执行失败!")

if __name__ == "__main__":
    main()