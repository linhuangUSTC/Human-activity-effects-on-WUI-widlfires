#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
功能：
    对Tweedie回归模型和零膨胀Gamma回归模型进行Vuong检验，比较模型拟合优度
输入文件：
    - 数据文件:I:/Processing data/Artificial influencing factors.csv
输出文件：
    - Vuong检验结果:B:/WUI/Picture and statistic results/vuong_test_results.txt
目标变量：
    - Fire_frequency_per_WUI:基于WUI面积的火灾频率
    - Fire_area_density:基于WUI面积的归一化火灾面积
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.families import Tweedie, Gamma
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')

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


def data_cleaning(df, dependent_vars):
    """
    数据清洗
    """
    print("\n" + "="*60)
    print("数据清洗流程")
    print("="*60)
    
    # 原始数据大小
    original_size = len(df)
    print(f"原始数据大小: {original_size}")
    
    # 1. 移除缺失值
    df_clean = df.dropna()
    print(f"移除缺失值后: {len(df_clean)} 样本 ({original_size - len(df_clean)} 个被移除)")
    
    # 2. 确保因变量为非负数（Tweedie分布支持0值）
    for dep_var in dependent_vars:
        df_clean[dep_var] = pd.to_numeric(df_clean[dep_var], errors='coerce')
        mask = df_clean[dep_var] >= 0
        df_clean = df_clean[mask]
        print(f"验证 {dep_var} (非负数)后: {len(df_clean)} 样本")
    
    return df_clean


def build_tweedie_model(df_clean, dependent_var, independent_vars):
    """
    构建Tweedie回归模型
    """
    print(f"\n构建Tweedie回归模型 - {dependent_var}")
    
    # 准备数据
    X = sm.add_constant(df_clean[independent_vars])
    y = df_clean[dependent_var]
    
    # 构建并拟合Tweedie回归模型
    family = Tweedie(var_power=1.5, link=sm.families.links.Log())
    model = sm.GLM(y, X, family=family).fit()
    
    return model


def build_zeroinflated_gamma_model(df_clean, dependent_var, independent_vars):
    """
    构建零膨胀Gamma回归模型
    """
    print(f"构建零膨胀Gamma回归模型 - {dependent_var}")
    
    # 准备数据
    X = sm.add_constant(df_clean[independent_vars])
    y = df_clean[dependent_var]
    
    # 构建并拟合零膨胀Gamma回归模型
    family = Gamma(link=sm.families.links.Log())
    model = sm.GLM(y, X, family=family).fit()
    
    return model


def vuong_test(model1, model2, y, X):
    """
    执行Vuong检验比较两个模型的拟合优度
    
    参数:
    model1: 第一个模型 (Tweedie)
    model2: 第二个模型 (零膨胀Gamma)
    y: 因变量
    X: 自变量矩阵（包含常数项）
    
    返回:
    (test_statistic, p_value, interpretation)
    """
    # 计算每个观测的对数似然
    
    # 获取预测值
    mu1 = model1.predict(X)
    mu2 = model2.predict(X)
    
    # 计算模型1 (Tweedie)的对数似然
    # Tweedie对数似然公式：https://en.wikipedia.org/wiki/Tweedie_distribution
    p = model1.family.var_power
    phi = model1.scale
    
    if p == 1:
        # Poisson case
        ll1 = y * np.log(mu1) - mu1 - np.log(np.math.factorial(y))
    elif p == 2:
        # Gamma case
        ll1 = -np.log(phi * mu1) - y / (phi * mu1)
    else:
        # General Tweedie case
        ll1 = (y * mu1**(1 - p) / (phi * (1 - p)) - mu1**(2 - p) / (phi * (2 - p)))
    
    # 计算模型2 (Gamma)的对数似然
    # Gamma对数似然公式
    phi2 = model2.scale
    ll2 = -np.log(phi2 * mu2) - y / (phi2 * mu2)
    
    # 计算对数似然差
    ll_diff = ll1 - ll2
    
    # 计算检验统计量
    n = len(y)
    mean_diff = ll_diff.mean()
    var_diff = ll_diff.var()
    
    # 如果方差为0，说明两个模型完全相同
    if var_diff == 0:
        return 0, 1.0, "两个模型的对数似然完全相同"
    
    test_statistic = np.sqrt(n) * mean_diff / np.sqrt(var_diff)
    
    # 计算p值（双侧检验）
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(np.abs(test_statistic)))
    
    # 解释结果
    if p_value < 0.05:
        if test_statistic > 0:
            interpretation = f"在5%显著性水平下，Tweedie模型显著优于零膨胀Gamma模型 (p={p_value:.4f})"
        else:
            interpretation = f"在5%显著性水平下，零膨胀Gamma模型显著优于Tweedie模型 (p={p_value:.4f})"
    else:
        interpretation = f"在5%显著性水平下，两个模型的拟合优度没有显著差异 (p={p_value:.4f})"
    
    return test_statistic, p_value, interpretation


def main():
    """
    主函数
    """
    print("=== Tweedie与零膨胀Gamma模型Vuong检验比较 ===\n")
    
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
    df_clean = data_cleaning(df, dependent_vars)
    if len(df_clean) == 0:
        print("清洗后数据为空，程序退出")
        return
    
    # 标准化自变量
    print("\n" + "="*60)
    print("自变量标准化")
    print("="*60)
    
    scaler = StandardScaler()
    df_clean[independent_vars] = scaler.fit_transform(df_clean[independent_vars])
    print("自变量标准化完成")
    
    # 执行Vuong检验
    results = {}
    
    for dep_var in dependent_vars:
        print("\n" + "="*80)
        print(f"对 {dep_var} 进行Vuong检验")
        print("="*80)
        
        # 构建两个模型
        tweedie_model = build_tweedie_model(df_clean, dep_var, independent_vars)
        zi_gamma_model = build_zeroinflated_gamma_model(df_clean, dep_var, independent_vars)
        
        # 准备数据
        X = sm.add_constant(df_clean[independent_vars])
        y = df_clean[dep_var]
        
        # 执行Vuong检验
        test_stat, p_value, interpretation = vuong_test(tweedie_model, zi_gamma_model, y, X)
        
        # 存储结果
        results[dep_var] = {
            'test_statistic': test_stat,
            'p_value': p_value,
            'interpretation': interpretation,
            'tweedie_loglike': tweedie_model.llf,
            'zi_gamma_loglike': zi_gamma_model.llf
        }
        
        # 打印结果
        print(f"\nVuong检验结果 - {dep_var}:")
        print(f"检验统计量: {test_stat:.4f}")
        print(f"p值: {p_value:.4f}")
        print(f"Tweedie模型对数似然: {tweedie_model.llf:.4f}")
        print(f"零膨胀Gamma模型对数似然: {zi_gamma_model.llf:.4f}")
        print(f"结论: {interpretation}")
    
    # 保存结果到文件
    output_file = os.path.join(output_dir, "vuong_test_results.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Tweedie与零膨胀Gamma模型Vuong检验结果\n")
        f.write("="*80 + "\n\n")
        
        for dep_var, res in results.items():
            f.write(f"目标变量: {dep_var}\n")
            f.write("-"*60 + "\n")
            f.write(f"Tweedie模型对数似然: {res['tweedie_loglike']:.4f}\n")
            f.write(f"零膨胀Gamma模型对数似然: {res['zi_gamma_loglike']:.4f}\n")
            f.write(f"对数似然差: {res['tweedie_loglike'] - res['zi_gamma_loglike']:.4f}\n")
            f.write(f"Vuong检验统计量: {res['test_statistic']:.4f}\n")
            f.write(f"p值: {res['p_value']:.4f}\n")
            f.write(f"结论: {res['interpretation']}\n")
            f.write("\n" + "="*60 + "\n\n")
    
    print(f"\n" + "="*80)
    print(f"Vuong检验结果已保存至: {output_file}")
    print("="*80)
    print("分析完成！")
    print("="*80)


if __name__ == "__main__":
    main()