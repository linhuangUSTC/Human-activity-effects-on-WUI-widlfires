#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
功能：
    使用零膨胀Gamma回归模型（ZIG）分析5个独立变量(Roadsd, GDP, PopD, HFI, All_WUI_area)对火灾相关目标变量的解释力
    - 一次性将所有独立变量纳入模型
    - 使用联合似然（joint likelihood）进行估计，确保AIC/BIC、标准误、参数可比性的有效性

输入文件：
    - 数据文件:I:/Processing data/Artificial influencing factors.csv
输出文件：
    - 可视化图表:B:/WUI/Picture and statistic results/zeroinflated_gamma_regression_results_*.jpg
    - 分析报告:B:/WUI/Picture and statistic results/zeroinflated_gamma_analysis_summary.txt
目标变量：
    - Fire_frequency_per_WUI:基于WUI面积的火灾频率
    - Fire_area_density:基于WUI面积的归一化火灾面积
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
from scipy import stats
from scipy.special import expit  # ✅(微小改动) 更稳定的logit逆函数

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

# 设置输出目录
output_dir = r"B:\WUI\Picture and statistic results"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)


class ZeroInflatedGamma(GenericLikelihoodModel):
    """
    零膨胀Gamma模型（Zero-Inflated Gamma, ZIG）
    使用联合似然（joint likelihood）进行估计

    模型结构：
    1. 零膨胀部分：使用Logit模型预测零值的概率
    2. Gamma部分：使用Gamma模型预测非零值的分布
    3. 联合似然估计所有参数
    """

    def __init__(self, endog, exog, exog_infl=None, **kwargs):
        """
        初始化模型

        Parameters:
        -----------
        endog : array-like
            因变量（包含零值和非零值）
        exog : array-like
            用于Gamma部分的自变量
        exog_infl : array-like, optional
            用于零膨胀部分的自变量
            如果为None，则使用与Gamma部分相同的自变量
        """
        if exog_infl is None:
            exog_infl = exog

        # ✅(微小改动) 使用 has_constant='add' 更稳妥地添加截距，避免误判第一列是不是常数
        self.exog_gamma = sm.add_constant(exog, has_constant='add')
        self.exog_infl = sm.add_constant(exog_infl, has_constant='add')

        # 调用父类初始化（只使用exog_gamma）
        super().__init__(endog, self.exog_gamma, **kwargs)

        # 设置参数数量
        self.k_infl = self.exog_infl.shape[1]          # 零膨胀部分参数数量
        self.k_gamma = self.exog_gamma.shape[1]        # Gamma部分参数数量
        self.k_params = self.k_infl + self.k_gamma + 1  # 零膨胀参数 + Gamma参数 + Gamma形状参数
        
        # 设置参数名称
        self.param_names = (list(self.exog_infl.columns) + 
                           list(self.exog_gamma.columns) + 
                           ['alpha_log'])

    def _get_param_names(self):
        """
        返回参数名称列表
        """
        return self.param_names
        
    def fit(self, start_params=None, method='bfgs', maxiter=1000, **kwargs):
        """
        拟合模型
        """
        if start_params is None:
            # 零膨胀部分初始参数
            zero_mask = (self.endog == 0).astype(int)
            logit_model = sm.Logit(zero_mask, self.exog_infl)
            try:
                logit_results = logit_model.fit(disp=False)
                start_params_infl = logit_results.params
            except Exception:
                start_params_infl = np.zeros(self.k_infl)

            # Gamma部分初始参数
            non_zero_mask = (self.endog > 0)
            y_non_zero = self.endog[non_zero_mask]
            exog_non_zero = self.exog_gamma[non_zero_mask]
            
            try:
                # 先拟合OLS（log转换）
                log_y = np.log(y_non_zero)
                ols_model = sm.OLS(log_y, exog_non_zero)
                ols_results = ols_model.fit(disp=False)
                start_params_gamma = ols_results.params
            except Exception:
                start_params_gamma = np.zeros(self.k_gamma)

            # Gamma形状参数初始值
            if len(y_non_zero) > 0:
                sample_var = np.var(y_non_zero)
                sample_mean = np.mean(y_non_zero)
                
                try:
                    start_alpha = (sample_mean ** 2) / sample_var
                    if start_alpha <= 0:
                        start_alpha = 1.0
                    start_log_alpha = np.log(start_alpha)
                except Exception:
                    start_log_alpha = np.log(1.0)
            else:
                start_log_alpha = np.log(1.0)

            # 合并所有初始参数
            start_params = np.concatenate([start_params_infl, start_params_gamma, [start_log_alpha]])

        # 拟合模型
        results = super().fit(start_params=start_params, method=method, maxiter=maxiter, **kwargs)
        
        return results

    def loglikeobs(self, params):
        """
        计算每个观察值的对数似然

        params:
            - 前k_infl个参数：零膨胀部分的Logit系数
            - 接下来k_gamma个参数：Gamma部分的系数（log均值）
            - 最后一个参数：Gamma形状参数的对数（确保形状参数为正）
        """
        # 分离参数
        params_infl = params[:self.k_infl]
        params_gamma = params[self.k_infl:-1]
        log_alpha = params[-1]
        alpha = np.exp(log_alpha)  # shape > 0

        # 计算零膨胀部分概率（零值概率）
        eta_infl = np.dot(self.exog_infl, params_infl)
        prob_zero = expit(eta_infl)  # ✅(微小改动) 更数值稳定

        # 计算Gamma部分均值（log link）
        eta_gamma = np.dot(self.exog_gamma, params_gamma)
        mu_gamma = np.exp(eta_gamma)

        y = np.asarray(self.endog)
        loglike = np.zeros_like(y, dtype=np.float64)

        # ✅(微小改动) 统一clip，避免log(0)与极端值数值问题
        eps = 1e-12

        # 零值部分
        zero_mask = (y == 0)
        loglike[zero_mask] = np.log(np.clip(prob_zero[zero_mask], eps, 1.0))

        # 非零值部分
        non_zero_mask = ~zero_mask
        if np.any(non_zero_mask):
            y_non_zero = y[non_zero_mask]
            mu_non_zero = mu_gamma[non_zero_mask]

            # P(Y>0) = 1 - pi
            prob_non_zero = 1.0 - prob_zero[non_zero_mask]
            log_prob_non_zero = np.log(np.clip(prob_non_zero, eps, 1.0))

            # ✅(微小改动) 用 logpdf 替代 pdf 再 log，显著提升数值稳定性
            # Gamma(shape=alpha, scale=mu/alpha) => mean = alpha*scale = mu
            log_gamma = stats.gamma.logpdf(
                y_non_zero,
                a=alpha,
                scale=np.clip(mu_non_zero / alpha, eps, np.inf)
            )

            loglike[non_zero_mask] = log_prob_non_zero + log_gamma

        return loglike

    def fit(self, start_params=None, method='bfgs', maxiter=1000, **kwargs):
        """
        拟合模型
        """
        if start_params is None:
            # 零膨胀部分初始参数
            zero_mask = (self.endog == 0).astype(int)
            logit_model = sm.Logit(zero_mask, self.exog_infl)
            try:
                logit_results = logit_model.fit(disp=False)
                start_params_infl = logit_results.params
            except Exception:
                start_params_infl = np.zeros(self.k_infl)

            # Gamma部分初始参数
            non_zero_mask = (self.endog > 0)
            non_zero_count = np.sum(non_zero_mask)

            if non_zero_count > 0:
                y_non_zero = np.asarray(self.endog)[non_zero_mask]
                X_non_zero = np.asarray(self.exog_gamma)[non_zero_mask]
                gamma_model = sm.GLM(
                    y_non_zero, X_non_zero,
                    family=sm.families.Gamma(sm.families.links.Log())
                )
                try:
                    gamma_results = gamma_model.fit()
                    start_params_gamma = gamma_results.params
                except Exception:
                    start_params_gamma = np.zeros(self.k_gamma)
            else:
                start_params_gamma = np.zeros(self.k_gamma)

            # Gamma形状参数初始值
            if non_zero_count > 0:
                y_nz = np.asarray(self.endog)[non_zero_mask]
                y_var = np.var(y_nz)
                y_mean = np.mean(y_nz)
                if y_mean > 0:
                    alpha_init = y_mean**2 / (y_var + 1e-10)
                else:
                    alpha_init = 1.0
            else:
                alpha_init = 1.0

            start_params = np.concatenate([start_params_infl, start_params_gamma, [np.log(alpha_init)]])

        return super().fit(start_params=start_params, method=method, maxiter=maxiter, **kwargs)


def load_data(csv_path):
    """
    加载CSV数据
    """
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        return df
    except:
        return None


def data_cleaning(df, dependent_vars, independent_vars):
    """
    数据清洗
    """
    used_vars = independent_vars + dependent_vars
    
    # 移除缺失值
    df_clean = df.dropna(subset=used_vars)
    
    # 转换为数值类型
    for var in used_vars:
        df_clean[var] = pd.to_numeric(df_clean[var], errors='coerce').astype(np.float32)
    
    # 移除转换后产生的缺失值
    df_clean = df_clean.dropna(subset=used_vars)
    
    # 确保因变量非负
    for dep_var in dependent_vars:
        df_clean = df_clean[df_clean[dep_var] >= 0]
    
    return df_clean


def build_zeroinflated_gamma_model(df_clean, dependent_var, independent_vars):
    """
    构建零膨胀Gamma回归模型（使用联合似然估计）
    """
    X = df_clean[independent_vars]
    y = df_clean[dependent_var]

    # 训练零膨胀Gamma模型（联合似然估计）
    model = ZeroInflatedGamma(y, X)
    results = model.fit(disp=False)

    # 计算零值比例
    zero_mask = (y == 0)
    zero_ratio = zero_mask.sum() / len(y)

    return {
        'model': results,
        'zero_mask': zero_mask,
        'zero_ratio': zero_ratio,
        'k_infl': model.k_infl,
        'k_gamma': model.k_gamma
    }, results


def plot_model_results(model_dict, dependent_var, output_dir):
    """
    绘制零膨胀Gamma模型结果图
    """
    results = model_dict['model']
    k_infl = model_dict['k_infl']
    k_gamma = model_dict['k_gamma']

    params_infl = results.params[:k_infl]
    params_gamma = results.params[k_infl:-1]

    features_infl = results.model.exog_infl.columns[1:]   # 排除截距
    features_gamma = results.model.exog_gamma.columns[1:]  # 排除截距

    # 1) 零膨胀部分系数图
    coefs_zeroinflated = params_infl[1:]

    plt.figure(figsize=(12, 8))
    colors_zeroinflated = ['#FF6B6B' if x < 0 else '#4ECDC4' for x in coefs_zeroinflated]
    bars_zeroinflated = plt.bar(range(len(coefs_zeroinflated)), coefs_zeroinflated, color=colors_zeroinflated)

    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xticks(range(len(coefs_zeroinflated)), features_infl, ha='right')
    plt.ylabel('Logit Regression Coefficients', fontsize=16)
    plt.title(f'Zero-Inflated Part Coefficients - {dependent_var}', fontsize=18, fontweight='bold')

    coef_min = coefs_zeroinflated.min()
    coef_max = coefs_zeroinflated.max()
    margin = (coef_max - coef_min) * 0.1
    plt.ylim(coef_min - margin, coef_max + margin)

    for bar in bars_zeroinflated:
        height = bar.get_height()
        label = f'{height:.2f}'
        if height < 0:
            plt.text(bar.get_x() + bar.get_width() / 2., height - 0.01, label,
                     ha='center', va='top', fontsize=16)
        else:
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01, label,
                     ha='center', va='bottom', fontsize=16)

    plt.tight_layout()
    filename_zeroinflated = f"zeroinflated_gamma_zeroinflated_coefficients_{dependent_var.replace(' ', '_').lower()}.jpg"
    filepath_zeroinflated = os.path.join(output_dir, filename_zeroinflated)
    plt.savefig(filepath_zeroinflated, dpi=600, bbox_inches='tight')
    plt.close()

    # 2) Gamma部分系数图
    coefs_gamma = params_gamma[1:]

    plt.figure(figsize=(12, 8))
    colors_gamma = ['#4169E1'] * len(coefs_gamma)
    bars_gamma = plt.bar(range(len(coefs_gamma)), coefs_gamma, color=colors_gamma)

    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xticks(range(len(coefs_gamma)), features_gamma, ha='right')
    plt.ylabel('Gamma Regression Coefficients', fontsize=16)
    plt.title(f'Gamma Part Coefficients - {dependent_var}', fontsize=18, fontweight='bold')

    coef_min = coefs_gamma.min()
    coef_max = coefs_gamma.max()
    margin = (coef_max - coef_min) * 0.1
    plt.ylim(coef_min - margin, coef_max + margin)

    for bar in bars_gamma:
        height = bar.get_height()
        label = f'{height:.2f}'
        if height < 0:
            plt.text(bar.get_x() + bar.get_width() / 2., height - 0.01, label,
                     ha='center', va='top', fontsize=16)
        else:
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01, label,
                     ha='center', va='bottom', fontsize=16)

    plt.tight_layout()
    filename_gamma = f"zeroinflated_gamma_gamma_coefficients_{dependent_var.replace(' ', '_').lower()}.jpg"
    filepath_gamma = os.path.join(output_dir, filename_gamma)
    plt.savefig(filepath_gamma, dpi=600, bbox_inches='tight')
    plt.close()


def save_analysis_summary(models, output_dir):
    """
    保存分析摘要到文本文件
    """
    filename = "zeroinflated_gamma_analysis_summary.txt"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("零膨胀Gamma回归分析摘要（联合似然估计）\n")
        f.write("=" * 60 + "\n\n")

        for dep_var, model_dict in models.items():
            results = model_dict['model']
            k_infl = model_dict['k_infl']

            f.write("=" * 60 + "\n")
            f.write(f"模型: {dep_var}\n")
            f.write("=" * 60 + "\n\n")

            f.write("1. 模型整体信息:\n")
            f.write("-" * 40 + "\n")
            f.write(f"对数似然值: {results.llf:.4f}\n")
            f.write(f"AIC: {results.aic:.4f}\n")
            f.write(f"BIC: {results.bic:.4f}\n")
            f.write(f"零值比例: {model_dict['zero_ratio'] * 100:.2f}%\n\n")

            f.write("2. 零膨胀部分（Logit模型）:\n")
            f.write("-" * 40 + "\n")
            f.write("系数表:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'变量':<15} {'系数':>10} {'标准误':>10} {'z值':>10} {'P>|z|':>10}\n")
            f.write("-" * 55 + "\n")
            for i, var_name in enumerate(results.model.exog_infl.columns):
                coef = results.params[i]
                se = results.bse[i]
                z = coef / se
                pval = 2 * (1 - stats.norm.cdf(np.abs(z)))
                f.write(f"{var_name:<15} {coef:10.4f} {se:10.4f} {z:10.4f} {pval:10.4f}\n")
            f.write("\n")

            f.write("3. Gamma部分:\n")
            f.write("-" * 40 + "\n")
            f.write("系数表:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'变量':<15} {'系数':>10} {'标准误':>10} {'z值':>10} {'P>|z|':>10}\n")
            f.write("-" * 55 + "\n")
            for i, var_name in enumerate(results.model.exog_gamma.columns):
                idx = k_infl + i
                coef = results.params[idx]
                se = results.bse[idx]
                z = coef / se
                pval = 2 * (1 - stats.norm.cdf(np.abs(z)))
                f.write(f"{var_name:<15} {coef:10.4f} {se:10.4f} {z:10.4f} {pval:10.4f}\n")

            f.write("\n4. Gamma形状参数:\n")
            f.write("-" * 40 + "\n")
            alpha = np.exp(results.params[-1])
            f.write(f"形状参数 (alpha): {alpha:.4f}\n\n")




def main():
    """
    主函数
    """
    csv_path = r"I:/Processing data/Artificial influencing factors.csv"

    independent_vars = ['Roadsd', 'GDP', 'PopD', 'HFI', 'All_WUI_area']
    dependent_vars = ['Fire_frequency_per_WUI', 'Fire_area_density']

    df = load_data(csv_path)
    if df is None:
        return

    # 处理所有数据
    df_clean = data_cleaning(df, dependent_vars, independent_vars)
    if len(df_clean) == 0:
        return

    del df

    # 自变量标准化
    scaler = StandardScaler()
    df_clean[independent_vars] = scaler.fit_transform(df_clean[independent_vars]).astype(np.float32)
    del scaler

    models = {}
    for dep_var in dependent_vars:
        model_dict, results = build_zeroinflated_gamma_model(df_clean, dep_var, independent_vars)
        models[dep_var] = model_dict
        plot_model_results(model_dict, dep_var, output_dir)

    save_analysis_summary(models, output_dir)


if __name__ == "__main__":
    main()
