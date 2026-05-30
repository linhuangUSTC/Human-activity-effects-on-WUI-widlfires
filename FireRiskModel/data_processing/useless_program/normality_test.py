#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
程序名称: WUI网格数据正态分布检验程序
功能简介: 
    1. 对WUI网格数据进行正态分布检验
    2. 提供D'Agostino统计检验和可视化结果
    3. 生成直方图+拟合正态分布曲线和Q-Q图
    4. 支持多变量批量处理
输入文件:
    - 数据文件: I:/Processing data/Artificial influencing factors.csv
    - 包含GRID_ID及各类人类活动参数
输出文件:
    - 图片文件: B:/WUI/Picture/{变量名}_normal_distribution.jpg
    - 文本报告: B:/WUI/Picture/normality_test_results.txt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import normaltest
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

# Create directory if it doesn't exist
picture_dir = r"B:\WUI\Picture"

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

def check_normality(data, column_name):
    """
    Test data for normal distribution
    """
    # Remove missing values
    clean_data = data.dropna()
    
    print(f"\n=== {column_name} Normal Distribution Test Results ===")
    print(f"Sample size: {len(clean_data)}")
    print(f"Mean: {clean_data.mean():.2f}")
    print(f"Standard deviation: {clean_data.std():.2f}")
    print(f"Minimum: {clean_data.min():.2f}")
    print(f"Maximum: {clean_data.max():.2f}")
    
    # D'Agostino normality test
    dagostino_stat, dagostino_p = normaltest(clean_data)
    print(f"D'Agostino test: statistic={dagostino_stat:.2f}, p-value={dagostino_p:.6f}")
    is_normal = dagostino_p > 0.05
    print(f"Normal distribution determination: {'Yes' if is_normal else 'No'}")
    
    return {
        'is_normal': is_normal,
        'dagostino_p': dagostino_p,
        'mean': clean_data.mean(),
        'std': clean_data.std()
    }

def plot_normality_distribution(data, column_name, stats_info):
    """
    Plot normal distribution test chart (only keep histogram and QQ plot)
    """
    clean_data = data.dropna()
    
    # Create subplots (1 row 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{column_name} Normal Distribution Test', fontsize=16, fontweight='bold')
    
    # 1. Histogram + fitted normal distribution
    axes[0].hist(clean_data, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Fitted normal distribution
    mu, sigma = stats.norm.fit(clean_data)
    x = np.linspace(clean_data.min(), clean_data.max(), 100)
    axes[0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                 label=f'Fitted normal distribution\nμ={mu:.2f}, σ={sigma:.2f}\np-value={stats_info["dagostino_p"]:.6f}')
    axes[0].set_title('Histogram + Fitted Normal Distribution')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Q-Q plot
    stats.probplot(clean_data, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (Normality Test)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    save_plot(fig, f"{column_name}_normal_distribution")
    
    return fig

def save_results_to_txt(results_dict, output_path):
    """
    Save normality test results to a text file
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=== WUI Fishnet Grid Data Normal Distribution Test Results ===\n\n")
            
            for column_name, stats_info in results_dict.items():
                f.write(f"=== {column_name} Normal Distribution Test Results ===\n")
                f.write(f"Sample size: {stats_info['sample_size']}\n")
                f.write(f"Mean: {stats_info['mean']:.2f}\n")
                f.write(f"Standard deviation: {stats_info['std']:.2f}\n")
                f.write(f"Minimum: {stats_info['min']:.2f}\n")
                f.write(f"Maximum: {stats_info['max']:.2f}\n")
                f.write(f"D'Agostino test: statistic={stats_info['dagostino_stat']:.2f}, p-value={stats_info['dagostino_p']:.6f}\n")
                f.write(f"Normal distribution determination: {'Yes' if stats_info['is_normal'] else 'No'}\n\n")
        
        print(f"Results saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving results to text file: {e}")
        return False

def main():
    """
    Main function for normal distribution test
    """
    print("=== WUI Fishnet Grid Data Normal Distribution Test Program ===\n")
    
    # Read data
    csv_path = r"I:\Processing data\Artificial influencing factors.csv"
    df = load_data(csv_path)
    
    if df is None:
        return
    
    # Get all columns except GRID_ID
    all_columns = [col for col in df.columns if col != 'GRID_ID']
    
    print(f"\nColumns to test (excluding GRID_ID): {all_columns}")
    
    # Normal distribution test
    normality_results = {}
    
    # Create a text file to save results
    results_txt_path = os.path.join(picture_dir, "normality_test_results.txt")
    
    for col in all_columns:
        print(f"\n{'='*50}")
        
        # Remove missing values
        clean_data = df[col].dropna()
        sample_size = len(clean_data)
        
        # D'Agostino normality test
        dagostino_stat, dagostino_p = normaltest(clean_data)
        is_normal = dagostino_p > 0.05
        
        # Collect statistics
        stats_info = {
            'is_normal': is_normal,
            'dagostino_p': dagostino_p,
            'dagostino_stat': dagostino_stat,
            'mean': clean_data.mean(),
            'std': clean_data.std(),
            'min': clean_data.min(),
            'max': clean_data.max(),
            'sample_size': sample_size
        }
        
        normality_results[col] = stats_info
        
        # Print results (same format as user requested)
        print(f"=== {col} Normal Distribution Test Results ===")
        print(f"Sample size: {len(clean_data)}")
        print(f"Mean: {clean_data.mean():.2f}")
        print(f"Standard deviation: {clean_data.std():.2f}")
        print(f"Minimum: {clean_data.min():.2f}")
        print(f"Maximum: {clean_data.max():.2f}")
        print(f"D'Agostino test: statistic={dagostino_stat:.2f}, p-value={dagostino_p:.6f}")
        print(f"Normal distribution determination: {'Yes' if is_normal else 'No'}")
        
        # Draw normal distribution test chart
        fig = plot_normality_distribution(df[col], col, stats_info)
        plt.show(block=False)  # Show plot without blocking
        plt.pause(0.5)  # Display briefly to save memory
        plt.close()  # Close the plot window
    
    # Save results to text file
    save_results_to_txt(normality_results, results_txt_path)
    
    print(f"\n{'='*50}")
    print("Normal distribution test completed!")

if __name__ == "__main__":
    main()