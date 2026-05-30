#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
功能：
    使用Tweedie回归模型分析5个独立变量(Roadsd, GDP, PopD, HFI, All_WUI_area)对火灾相关目标变量的解释力
    - 一次性将所有独立变量纳入模型

输入文件：
    - 数据文件:I:/Processing data/Artificial influencing factors.csv
输出文件：
    - 可视化图表:B:/WUI/Picture and statistic results/tweedie_regression_results_*.jpg
    - 分析报告:B:/WUI/Picture and statistic results/tweedie_analysis_summary.txt
目标变量：
    - Fire_frequency_per_WUI:基于WUI面积的火灾频率
    - Fire_area_density:基于WUI面积的归一化火灾面积
    注:这个程序运行需要的内存太大,只能在服务器上运行
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.genmod.families import Tweedie

from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')

# 设置中文字体和图形参数
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
output_dir = r"B:/WUI/Picture and statistic results"
if not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir)
        print(f"✓ 成功创建输出目录: {output_dir}")
    except Exception as e:
        print(f"✗ 创建输出目录失败: {e}")
else:
    print(f"✓ 输出目录已存在: {output_dir}")


def load_data(csv_path):
    """
    加载CSV数据
    """
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        print(f"成功读取数据，共 {len(df)} 行，{len(df.columns)} 列")
        print(f"数据列名: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"文件未找到: {csv_path}")
        return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None


def data_cleaning(df, dependent_vars, independent_vars):
    """
    数据清洗
    """
    print("\n" + "="*60)
    print("数据清洗流程")
    print("="*60)
    
    # 原始数据大小
    original_size = len(df)
    print(f"原始数据大小: {original_size}")
    
    # 1. 只检查我们使用的变量（自变量和因变量）是否有缺失值
    # 这样可以保留只在其他列有缺失值的有效数据
    used_vars = independent_vars + dependent_vars
    df_clean = df.dropna(subset=used_vars)
    print(f"移除{used_vars}中缺失值后: {len(df_clean)} 样本 ({original_size - len(df_clean)} 个被移除)")
    
    # 2. 确保所有使用的变量为数值类型(float32)
    for var in used_vars:
        df_clean[var] = pd.to_numeric(df_clean[var], errors='coerce').astype(np.float32)
    
    # 重新检查转换后是否有新的缺失值
    df_clean = df_clean.dropna(subset=used_vars)
    print(f"转换为数值类型后: {len(df_clean)} 样本")
    
    # 3. 确保因变量为非负数（Tweedie分布支持0值）
    for dep_var in dependent_vars:
        mask = df_clean[dep_var] >= 0
        df_clean = df_clean[mask]
        print(f"验证 {dep_var} (非负数)后: {len(df_clean)} 样本")
    
    return df_clean





def build_tweedie_model(df_clean, dependent_var, independent_vars):
    """
    构建Tweedie回归模型
    """
    print("\n" + "="*60)
    print(f"Tweedie回归模型构建 - {dependent_var}")
    print("="*60)
    
    # 准备数据
    X = sm.add_constant(df_clean[independent_vars])
    y = df_clean[dependent_var]
    
    # 构建并拟合Tweedie回归模型
    # 对于火灾数据，常用的Tweedie指数为1.5（复合泊松-伽马分布）
    # 使用对数链接函数（link_power=0对应对数链接）
    family = Tweedie(var_power=1.5, link=sm.families.links.Log())
    model = sm.GLM(y, X, family=family).fit()
    
    # 计算模型性能指标
    # 创建只有截距的设计矩阵，避免常量项冗余
    null_model = sm.GLM(y, sm.add_constant(np.zeros(len(y))), family=family).fit()
    null_deviance = null_model.deviance
    residual_deviance = model.deviance
    proportional_deviance = (null_deviance - residual_deviance) / null_deviance
    
    print(f"\n模型性能指标：")
    print(f"空模型偏差: {null_deviance:.4f}")
    print(f"残差偏差: {residual_deviance:.4f}")
    print(f"解释偏差比例: {proportional_deviance:.4f} ({proportional_deviance*100:.2f}%)")
    print(f"AIC: {model.aic:.4f}")
    print(f"BIC: {model.bic:.4f}")
    
    return model, proportional_deviance


def plot_model_results(model, dependent_var, output_dir):
    """
    绘制模型结果图
    """
    # 提取模型系数
    coefs = model.params[1:]  # 排除截距
    features = coefs.index
    
    # 创建系数图
    plt.figure(figsize=(12, 8))
    colors = ['#4169E1' if x < 0 else '#4169E1' for x in coefs.values]  # 负系数使用灰色，正系数使用蓝色
    bars = plt.bar(range(len(coefs)), coefs.values, color=colors)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xticks(range(len(coefs)), features, ha='right')
    plt.ylabel('Regression Coefficients', fontsize=16)
    plt.title(f'Tweedie Regression Coefficients - {dependent_var}', fontsize=18, fontweight='bold')
    
    # 根据系数范围自动调节ylim，留出适当边距
    coef_min = coefs.values.min()
    coef_max = coefs.values.max()
    margin = (coef_max - coef_min) * 0.1  # 10%的边距
    plt.ylim(coef_min - margin, coef_max + margin)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        # 仅显示两位小数
        label = f'{height:.2f}'
        # 负值标签放在柱子下方（横轴下方），正值放在柱子上方
        if height < 0:
            # 负值标签位置调整到柱子底部以下
            plt.text(bar.get_x() + bar.get_width()/2., height - 0.01, 
                     label, ha='center', va='top', fontsize=16)
        else:
            # 正值标签位置在柱子顶部以上
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                     label, ha='center', va='bottom', fontsize=16)
    
    plt.tight_layout()
    
    # 保存图表
    filename = f"tweedie_regression_coefficients_{dependent_var.replace(' ', '_').lower()}.jpg"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    print(f"系数图已保存至: {filepath}")
    
    plt.close()


def save_analysis_summary(models, proportional_deviance, output_dir):
    """
    保存分析摘要到文本文件
    """
    filename = "tweedie_analysis_summary.txt"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("TWEEDIE回归分析摘要\n")
        f.write("="*60 + "\n\n")
        
        # 写入每个模型的结果
        for dep_var, model in models.items():
            f.write("="*60 + "\n")
            f.write(f"模型: {dep_var}\n")
            f.write("="*60 + "\n")
            f.write(f"解释偏差比例: {proportional_deviance[dep_var]:.4f} ({proportional_deviance[dep_var]*100:.2f}%)\n")
            f.write(f"AIC: {model.aic:.4f}\n")
            f.write(f"BIC: {model.bic:.4f}\n\n")
            
            f.write("系数表:\n")
            f.write("-"*40 + "\n")
            f.write(model.summary().tables[1].as_text() + "\n\n")
    
    print(f"分析摘要已保存至: {filepath}")


def main():
    """
    主函数
    """
    print("=== 火灾数据Tweedie回归分析程序 ===\n")
    
    # 数据文件路径
    csv_path = r"I:/Processing data/Artificial influencing factors.csv"
    
    # 定义变量
    independent_vars = ['Roadsd', 'GDP', 'PopD', 'HFI', 'All_WUI_area']
    dependent_vars = ['Fire_frequency_per_WUI', 'Fire_area_density']
    
    # 加载数据
    df = load_data(csv_path)
    if df is None:
        print("数据加载失败，程序退出")
        return
    
    # 数据清洗
    df_clean = data_cleaning(df, dependent_vars, independent_vars)
    if len(df_clean) == 0:
        print("清洗后数据为空，程序退出")
        return
    
    # 清理原始数据以释放内存
    del df
    print("原始数据已清理，释放内存")
    
    # 标准化自变量
    print("\n" + "="*60)
    print("自变量标准化")
    print("="*60)
    
    scaler = StandardScaler()
    df_clean[independent_vars] = scaler.fit_transform(df_clean[independent_vars]).astype(np.float32)
    print("自变量标准化完成")
    
    # 清理标准化器以释放内存
    del scaler
    print("标准化器已清理，释放内存")
    
    
    
    # 为每个因变量构建Tweedie回归模型
    models = {}
    proportional_deviance = {}
    
    for dep_var in dependent_vars:
        model, prop_dev = build_tweedie_model(df_clean, dep_var, independent_vars)
        models[dep_var] = model
        proportional_deviance[dep_var] = prop_dev
        
        # 打印模型摘要
        print("\n模型摘要:")
        print(model.summary())
        
        # 绘制结果图
        plot_model_results(model, dep_var, output_dir)
    
    # 保存分析摘要
    save_analysis_summary(models, proportional_deviance, output_dir)
    
    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)


if __name__ == "__main__":
    main()
