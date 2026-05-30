#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
程序名称: WUI人类活动参数相关性分析程序
功能简介: 本程序用于分析WUI(野火-城市界面)相关人类活动参数的Spearman相关系数和VIF
         1. 读取CSV格式的人类活动参数数据
         2. 计算并绘制Spearman相关系数矩阵热力图
         3. 计算Variance Inflation Factor (VIF)指标
         4. 保存高分辨率(600dpi)的相关系数矩阵图形
         5. 保存VIF分析结果到文本文件

数据输入:
    - 人类活动参数数据: I:/Processing data/Artificial influencing factors.csv

输出文件:
    - Spearman相关系数矩阵图: B:/WUI/Picture/spearman_correlation_matrix.jpg
    - VIF分析结果文件: B:/WUI/Picture/vif_analysis_results.txt
    - 控制台输出: 相关系数矩阵和VIF分析结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
import warnings
import os
warnings.filterwarnings('ignore')

# Set font to Arial, size 16
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['axes.unicode_minus'] = False

# Create picture directory if it doesn't exist
picture_dir = r"B:\WUI\Picture"
if not os.path.exists(picture_dir):
    os.makedirs(picture_dir)

def save_plot(fig, filename):
    """
    Save plot to the specified directory with 600 dpi
    """
    file_path = os.path.join(picture_dir, f"{filename}.jpg")
    fig.savefig(file_path, dpi=600, bbox_inches='tight')
    print(f"Plot saved to: {file_path}")

def load_data(csv_path):
    """
    Read CSV data
    """
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        print(f"Successfully read data, total {len(df)} rows, {len(df.columns)} columns")
        print(f"Data column names: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def calculate_vif(df, features):
    """
    Calculate Variance Inflation Factor (VIF) for each feature
    """
    vif = pd.DataFrame()
    vif["Feature"] = features
    vif["VIF"] = [1/(1 - LinearRegression().fit(df[features].drop([col], axis=1), df[col]).score(df[features].drop([col], axis=1), df[col])) for col in features]
    return vif



def plot_spearman_matrix(df, required_columns):
    """
    Plot Spearman correlation coefficient matrix
    """
    # Calculate Spearman correlation coefficient between all parameters
    spearman_corr_matrix = df[required_columns].corr(method='spearman')
    
    # Create correlation coefficient matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(spearman_corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', linewidths=.5, vmin=-1, vmax=1)
    plt.title('Spearman Correlation Coefficient Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    save_plot(plt.gcf(), "spearman_correlation_matrix")
    
    return plt.gcf()





def main():
    """
    Main function
    """
    print("=== WUI Fishnet Grid Data Analysis Program ===\n")
    
    # Read data
    csv_path = r"I:\Processing data\Artificial influencing factors.csv"
    df = load_data(csv_path)
    
    if df is None:
        return
    
    # Define required columns for analysis
    required_columns = ['Roadsd', 'GDP', 'PopD', 'HFI', 'All_WUI_area']
    
    # Check if required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing the following columns: {missing_columns}")
        print(f"Available column names: {list(df.columns)}")
        return
    
    print(f"\nFound all columns: {required_columns}")
    
    # Correlation analysis
    print(f"\n{'='*50}")
    print("Spearman correlation coefficient analysis:")
    
    # Draw Spearman correlation coefficient matrix
    plot_spearman_matrix(df, required_columns)
    plt.close()  # Close the plot window after saving
    
    # Output correlation coefficient matrix data - only for original required columns
    spearman_corr_matrix = df[required_columns].corr(method='spearman')
    print(f"\nSpearman correlation coefficient matrix:")
    print(spearman_corr_matrix.round(2))

    # Calculate VIF for required columns
    print(f"\n{'='*50}")
    print("Variance Inflation Factor (VIF) analysis:")
    vif_results = calculate_vif(df, required_columns)
    print(vif_results.round(2))

    # Save VIF results to TXT file in the same directory as pictures
    vif_file_path = os.path.join(picture_dir, "vif_analysis_results.txt")
    with open(vif_file_path, 'w', encoding='utf-8') as f:
        f.write("Variance Inflation Factor (VIF) Analysis Results\n")
        f.write("="*50 + "\n\n")
        f.write(vif_results.round(4).to_string() + "\n\n")
        f.write("VIF Interpretation:\n")
        f.write("- VIF < 5: No significant multicollinearity\n")
        f.write("- 5 ≤ VIF < 10: Moderate multicollinearity\n")
        f.write("- VIF ≥ 10: Severe multicollinearity\n")
    print(f"\nVIF results saved to: {vif_file_path}")

    print(f"\n{'='*50}")
    print("Analysis completed!")

if __name__ == "__main__":
    main()