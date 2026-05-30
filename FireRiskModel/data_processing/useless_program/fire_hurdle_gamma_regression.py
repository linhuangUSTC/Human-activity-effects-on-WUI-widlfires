#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
功能：
    使用Hurdle Gamma回归模型分析5个独立变量(Roadsd, GDP, PopD, HFI, All_WUI_area)对火灾相关目标变量的解释力
    - 一次性将所有独立变量纳入模型

输入文件：
    - 数据文件:I:/Processing data/Artificial influencing factors.csv
输出文件：
    - 可视化图表:B:/WUI/Picture and statistic results/hurdle_gamma_regression_results_*.jpg
    - 分析报告:B:/WUI/Picture and statistic results/hurdle_gamma_analysis_summary.txt
目标变量：
    - Fire_frequency_per_WUI:基于WUI面积的火灾频率
    - Fire_area_density:基于WUI面积的归一化火灾面积
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.genmod.families import Gamma
from statsmodels.discrete.discrete_model import Logit

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
    
    # 3. 确保因变量为非负数（支持0值）
    for dep_var in dependent_vars:
        mask = df_clean[dep_var] >= 0
        df_clean = df_clean[mask]
        print(f"验证 {dep_var} (非负数)后: {len(df_clean)} 样本")
    
    return df_clean





def build_hurdle_gamma_model(df_clean, dependent_var, independent_vars):
    """
    构建Hurdle Gamma回归模型
    Hurdle模型由两部分组成：
    1. Hurdle部分：使用Logit模型预测是否为零值
    2. Gamma部分：使用Gamma模型预测非零值的分布
    """
    print("\n" + "="*60)
    print(f"Hurdle Gamma回归模型构建 - {dependent_var}")
    print("="*60)
    
    # 准备数据
    X = sm.add_constant(df_clean[independent_vars])
    y = df_clean[dependent_var]
    
    # 1. Hurdle部分：使用Logit模型预测是否发生火灾（是否为非零值）
    print("\n1. 训练Hurdle部分（Logit模型）...")
    # 创建发生指示变量（0表示零值/不发生，1表示非零值/发生）
    # 因此解释时需要注意：Logit系数为正表示更倾向于发生火灾（非零值），系数为负表示更倾向于不发生火灾（零值）
    zero_mask = (y > 0).astype(int)
    # 训练Logit模型
    logit_model = Logit(zero_mask, X).fit()
    print(f"Hurdle模型AIC: {logit_model.aic:.4f}")
    print(f"Hurdle模型BIC: {logit_model.bic:.4f}")
    
    # 2. Gamma部分：使用Gamma模型预测非零值
    print("\n2. 训练Gamma部分（Gamma模型）...")
    # 筛选非零值样本
    non_zero_mask = (y > 0)
    X_non_zero = X[non_zero_mask]
    y_non_zero = y[non_zero_mask]
    print(f"非零值样本数: {len(y_non_zero)} ({len(y_non_zero)/len(y)*100:.2f}%)")
    
    # 构建并拟合Gamma模型
    family = Gamma(link=sm.families.links.Log())
    gamma_model = sm.GLM(y_non_zero, X_non_zero, family=family).fit()
    
    # 计算Gamma模型性能指标
    # 使用纯截距设计矩阵创建空模型，避免共线性问题
    # sm.add_constant(np.zeros(...)) 将创建只有截距项的设计矩阵
    null_gamma_model = sm.GLM(y_non_zero, np.ones((len(y_non_zero), 1)), family=family).fit()
    null_deviance = null_gamma_model.deviance
    residual_deviance = gamma_model.deviance
    proportional_deviance = (null_deviance - residual_deviance) / null_deviance
    
    print(f"\nGamma模型性能指标：")
    print(f"空模型偏差: {null_deviance:.4f}")
    print(f"残差偏差: {residual_deviance:.4f}")
    print(f"解释偏差比例: {proportional_deviance:.4f} ({proportional_deviance*100:.2f}%)")
    print(f"Gamma模型AIC: {gamma_model.aic:.4f}")
    print(f"Gamma模型BIC: {gamma_model.bic:.4f}")
    
    # 3. 组合模型信息
    print(f"\nHurdle Gamma模型总览：")
    print(f"零值比例: {(1 - zero_mask.sum()/len(zero_mask))*100:.2f}%")
    print(f"非零值比例: {zero_mask.sum()/len(zero_mask)*100:.2f}%")
    
    # 返回组合模型结果
    return {
        'logit_model': logit_model,  # Hurdle部分
        'gamma_model': gamma_model,  # Gamma部分
        'zero_mask': zero_mask,      # 零值掩码
        'proportional_deviance': proportional_deviance  # Gamma模型的解释偏差比例
    }, proportional_deviance


def plot_model_results(model_dict, dependent_var, output_dir):
    """
    绘制Hurdle Gamma模型结果图
    """
    # 获取两部分模型
    logit_model = model_dict['logit_model']  # Hurdle部分
    gamma_model = model_dict['gamma_model']  # Gamma部分
    
    # 1. 绘制Hurdle部分（Logit模型）的系数图
    print("\n绘制Hurdle部分系数图...")
    coefs_logit = logit_model.params[1:]  # 排除截距
    features_logit = coefs_logit.index
    
    plt.figure(figsize=(12, 8))
    colors_logit = ['#FF6B6B' if x < 0 else '#4ECDC4' for x in coefs_logit.values]  # 红色和青色
    bars_logit = plt.bar(range(len(coefs_logit)), coefs_logit.values, color=colors_logit)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xticks(range(len(coefs_logit)), features_logit, ha='right')
    plt.ylabel('Logit Regression Coefficients', fontsize=16)
    plt.title(f'Hurdle Part Coefficients - {dependent_var}', fontsize=18, fontweight='bold')
    
    # 根据系数范围自动调节ylim
    coef_min = coefs_logit.values.min()
    coef_max = coefs_logit.values.max()
    margin = (coef_max - coef_min) * 0.1
    plt.ylim(coef_min - margin, coef_max + margin)
    
    # 添加数值标签
    for bar in bars_logit:
        height = bar.get_height()
        label = f'{height:.2f}'
        if height < 0:
            plt.text(bar.get_x() + bar.get_width()/2., height - 0.01,
                     label, ha='center', va='top', fontsize=16)
        else:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     label, ha='center', va='bottom', fontsize=16)
    
    plt.tight_layout()
    
    # 保存Hurdle部分系数图
    filename_logit = f"hurdle_gamma_logit_coefficients_{dependent_var.replace(' ', '_').lower()}.jpg"
    filepath_logit = os.path.join(output_dir, filename_logit)
    plt.savefig(filepath_logit, dpi=600, bbox_inches='tight')
    print(f"Hurdle部分系数图已保存至: {filepath_logit}")
    
    plt.close()
    
    # 2. 绘制Gamma部分的系数图
    print("\n绘制Gamma部分系数图...")
    coefs_gamma = gamma_model.params[1:]  # 排除截距
    features_gamma = coefs_gamma.index
    
    plt.figure(figsize=(12, 8))
    colors_gamma = ['#4169E1'] * len(coefs_gamma)  # 统一使用蓝色
    bars_gamma = plt.bar(range(len(coefs_gamma)), coefs_gamma.values, color=colors_gamma)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xticks(range(len(coefs_gamma)), features_gamma, ha='right')
    plt.ylabel('Gamma Regression Coefficients', fontsize=16)
    plt.title(f'Gamma Part Coefficients - {dependent_var}', fontsize=18, fontweight='bold')
    
    # 根据系数范围自动调节ylim
    coef_min = coefs_gamma.values.min()
    coef_max = coefs_gamma.values.max()
    margin = (coef_max - coef_min) * 0.1
    plt.ylim(coef_min - margin, coef_max + margin)
    
    # 添加数值标签
    for bar in bars_gamma:
        height = bar.get_height()
        label = f'{height:.2f}'
        if height < 0:
            plt.text(bar.get_x() + bar.get_width()/2., height - 0.01,
                     label, ha='center', va='top', fontsize=16)
        else:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     label, ha='center', va='bottom', fontsize=16)
    
    plt.tight_layout()
    
    # 保存Gamma部分系数图
    filename_gamma = f"hurdle_gamma_gamma_coefficients_{dependent_var.replace(' ', '_').lower()}.jpg"
    filepath_gamma = os.path.join(output_dir, filename_gamma)
    plt.savefig(filepath_gamma, dpi=600, bbox_inches='tight')
    print(f"Gamma部分系数图已保存至: {filepath_gamma}")
    
    plt.close()


def save_analysis_summary(models, proportional_deviance, output_dir):
    """
    保存分析摘要到文本文件
    """
    filename = "hurdle_gamma_analysis_summary.txt"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("Hurdle Gamma回归分析摘要\n")
        f.write("="*60 + "\n\n")
        
        # 写入每个模型的结果
        for dep_var, model_dict in models.items():
            logit_model = model_dict['logit_model']  # Hurdle部分
            gamma_model = model_dict['gamma_model']  # Gamma部分
            
            f.write("="*60 + "\n")
            f.write(f"模型: {dep_var}\n")
            f.write("="*60 + "\n\n")
            
            # Hurdle部分（Logit模型）
            f.write("1. Hurdle部分（Logit模型）:\n")
            f.write("-"*40 + "\n")
            f.write(f"AIC: {logit_model.aic:.4f}\n")
            f.write(f"BIC: {logit_model.bic:.4f}\n\n")
            f.write("系数表:\n")
            f.write("-"*40 + "\n")
            f.write(logit_model.summary().tables[1].as_text() + "\n\n")
            
            # Gamma部分
            f.write("2. Gamma部分:\n")
            f.write("-"*40 + "\n")
            f.write(f"解释偏差比例: {proportional_deviance[dep_var]:.4f} ({proportional_deviance[dep_var]*100:.2f}%)\n")
            f.write(f"AIC: {gamma_model.aic:.4f}\n")
            f.write(f"BIC: {gamma_model.bic:.4f}\n\n")
            f.write("系数表:\n")
            f.write("-"*40 + "\n")
            f.write(gamma_model.summary().tables[1].as_text() + "\n\n")
    
    print(f"分析摘要已保存至: {filepath}")


def main():
    """
    主函数
    """
    print("=== 火灾数据Hurdle Gamma回归分析程序 ===\n")
    
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
    

    
    # 为每个因变量构建Hurdle Gamma回归模型
    models = {}
    proportional_deviance = {}
    
    for dep_var in dependent_vars:
        model, prop_dev = build_hurdle_gamma_model(df_clean, dep_var, independent_vars)
        models[dep_var] = model
        proportional_deviance[dep_var] = prop_dev
        
        # 打印模型摘要
        print("\nHurdle部分（Logit模型）摘要:")
        print(model['logit_model'].summary())
        print("\nGamma部分摘要:")
        print(model['gamma_model'].summary())
        
        # 绘制结果图
        plot_model_results(model, dep_var, output_dir)
    
    # 保存分析摘要
    save_analysis_summary(models, proportional_deviance, output_dir)
    
    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)


if __name__ == "__main__":
    main()