#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
功能：
    使用Gamma回归模型分析5个独立变量(Roadsd, GDP, PopD, HFI, All_WUI_area)对多个火灾相关目标变量的解释力
输入文件：
    - 数据文件:I:/Processing data/Artificial influencing factors.csv
输出文件：
    - 可视化图表:B:/WUI/Picture and statistic results/proportional_deviance_barplot_*.jpg
    - 分析报告:B:/WUI/Picture and statistic results/analysis_summary_*.txt
目标变量包括：
    - Fire_frequency_pop:火灾频率
    - All_fire_area:总火灾面积
    - Fire_frequency_per_WUI:基于WUI面积的火灾频率
    - Fire_area_density:基于WUI面积的归一化火灾面积 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.genmod.families import Gamma
import warnings
import os

warnings.filterwarnings('ignore')

# Set font to Arial, size 12
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.unicode_minus'] = False

# Create directory if it doesn't exist
# Set output directory to the user-specified path
output_dir = r"B:\WUI\Picture and statistic results"
if not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir)
        print(f"✓ Successfully created output directory: {output_dir}")
    except Exception as e:
        print(f"✗ Error creating output directory: {e}")
else:
    print(f"✓ Output directory already exists: {output_dir}")
print(f"Current output directory: {output_dir}")
print(f"Is output directory writable: {os.access(output_dir, os.W_OK)}")

def load_data(csv_path):
    """
    Load CSV data from the specified path
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

def data_cleaning(df, dependent_var):
    """
    Clean the data according to the requirements
    """
    print("\n" + "="*60)
    print(f"Data Cleaning Process for {dependent_var}")
    print("="*60)
    
    # Original data size
    original_size = len(df)
    print(f"Original data size: {original_size}")
    
    # 1. Remove samples with missing values
    df_clean = df.dropna()
    print(f"After removing missing values: {len(df_clean)} samples ({original_size - len(df_clean)} removed)")
    
    # 2. Verify dependent variable is positive
    # Check if dependent variable is numeric
    df_clean[dependent_var] = pd.to_numeric(df_clean[dependent_var], errors='coerce')
    # Check positive values
    mask = df_clean[dependent_var] > 0
    df_clean = df_clean[mask]
    print(f"After verifying {dependent_var} (positive values): {len(df_clean)} samples ({original_size - len(df_clean)} removed)")
    

    
    return df_clean

def analyze_data_basics(df_clean, dependent_var):
    """
    Analyze basic data information
    """
    print("\n" + "="*60)
    print(f"Data Basic Information for {dependent_var}")
    print("="*60)
    
    # Sample size
    print(f"Final sample size: {len(df_clean)}")
    
    # Calculate mean and variance
    variables = ['Roadsd', 'GDP', 'PopD', 'HFI', 'All_WUI_area', dependent_var]
    
    print("\nMean and Variance of Variables:")
    print("-"*40)
    print(f"{'Variable':<15} {'Mean':<12} {'Variance':<12} {'Std Dev':<12}")
    print("-"*40)
    
    for var in variables:
        mean_val = df_clean[var].mean()
        var_val = df_clean[var].var()
        std_val = df_clean[var].std()
        print(f"{var:<15} {mean_val:<12.4f} {var_val:<12.4f} {std_val:<12.4f}")
    
    # Get basic statistics for dependent variable
    dep_mean = df_clean[dependent_var].mean()
    dep_var = df_clean[dependent_var].var()
    
    print("\n" + "="*60)
    print(f"Distribution Analysis for {dependent_var}")
    print("="*60)
    print(f"Mean of {dependent_var}: {dep_mean:.4f}")
    print(f"Variance of {dependent_var}: {dep_var:.4f}")
    print(f"Variance/Mean Ratio: {dep_var/dep_mean:.4f}")
    
    return dep_mean, dep_var

def build_null_model(df_clean, dependent_var):
    """
    Build the null model (intercept only)
    """
    print("\n" + "="*60)
    print(f"Building Null Model (Intercept Only) for {dependent_var}")
    print("="*60)
    
    # Null model (only intercept)
    X_null = sm.add_constant(np.ones(len(df_clean)))
    y = df_clean[dependent_var]
    
    # Fit Gamma regression model
    null_model = sm.GLM(y, X_null, family=Gamma()).fit()
    
    # Get null deviance
    null_deviance = null_model.null_deviance
    print(f"Null Model Null Deviance: {null_deviance:.4f}")
    print(f"Null Model Residual Deviance: {null_model.deviance:.4f}")
    print(f"Null Model AIC: {null_model.aic:.4f}")
    print(f"Null Model BIC: {null_model.bic:.4f}")
    
    return null_model, null_deviance

def build_single_var_models(df_clean, dependent_var, null_deviance):
    """
    Build single variable Gamma models for each independent variable
    """
    print("\n" + "="*60)
    print(f"Building Single Variable Gamma Models for {dependent_var}")
    print("="*60)
    
    independent_vars = ['Roadsd', 'GDP', 'PopD', 'HFI', 'All_WUI_area']
    model_results = []
    
    for var in independent_vars:
        print(f"\n{'-'*40}")
        print(f"Model with {var} as independent variable")
        print(f"{'-'*40}")
        
        # Prepare data
        X = sm.add_constant(df_clean[var])
        y = df_clean[dependent_var]
        
        # Fit Gamma regression model
        model = sm.GLM(y, X, family=Gamma()).fit()
        
        # Calculate proportional deviance explained
        proportional_deviance = (null_deviance - model.deviance) / null_deviance
        
        # Get model parameters
        coef = model.params[var]
        std_err = model.bse[var]
        p_value = model.pvalues[var]
        aic = model.aic
        bic = model.bic
        
        # Check significance
        is_significant = p_value < 0.05
        
        # Store results
        model_results.append({
            'variable': var,
            'model': model,
            'null_deviance': null_deviance,
            'residual_deviance': model.deviance,
            'proportional_deviance': proportional_deviance,
            'coef': coef,
            'std_err': std_err,
            'p_value': p_value,
            'is_significant': is_significant,
            'aic': aic,
            'bic': bic
        })
        
        # Print results
        print(f"Proportional Deviance Explained: {proportional_deviance:.4f} ({proportional_deviance*100:.2f}%)")
        print(f"Regression Coefficient ({var}): {coef:.6f}")
        print(f"Standard Error: {std_err:.6f}")
        print(f"p-value: {p_value:.6f} {'*'*3 if is_significant else ''}")
        print(f"AIC: {aic:.4f}")
        print(f"BIC: {bic:.4f}")
        print(f"Significant: {'Yes' if is_significant else 'No'}")
    
    # Sort by proportional deviance explained
    model_results.sort(key=lambda x: x['proportional_deviance'], reverse=True)
    
    return model_results

def visualize_results(model_results, dependent_var):
    """
    Visualize the results of the analysis
    """
    print("\n" + "="*60)
    print(f"Visualizing Results for {dependent_var}")
    print("="*60)
    
    try:
        # Extract data for plotting
        variables = [result['variable'] for result in model_results]
        proportional_deviance = [result['proportional_deviance'] for result in model_results]
        is_significant = [result['is_significant'] for result in model_results]
        
        # Create colors: blue if significant, gray if not
        colors = ['#4169E1' if sig else '#808080' for sig in is_significant]
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(variables)), proportional_deviance, color=colors, alpha=0.7)
        
        # Add labels and title
        plt.xlabel('Independent Variables')
        plt.ylabel('Proportional Deviance Explained')
        plt.xticks(range(len(variables)), variables)
        
        # 自动计算合适的y轴范围
        max_deviance = max(proportional_deviance)
        min_deviance = min(proportional_deviance) if min(proportional_deviance) < 0 else 0
        plt.ylim(min_deviance * 1.1, max_deviance * 1.1)  # 留出10%的缓冲空间
        
        # Add values on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            # For positive values, place text above bar, for negative values, place text below bar
            if height >= 0:
                y_pos = height + 0.001
                va = 'bottom'
            else:
                y_pos = height - 0.001
                va = 'top'
            plt.text(bar.get_x() + bar.get_width()/2., y_pos,
                     f'{height:.2f}', ha='center', va=va, fontsize=15)
        
        # Add legend
        significant_patch = plt.Rectangle((0,0),1,1, color='#4169E1', alpha=0.7)
        not_significant_patch = plt.Rectangle((0,0),1,1, color='#808080', alpha=0.7)
        plt.legend([significant_patch, not_significant_patch], ['Significant (p<0.05)', 'Not Significant'], loc='upper right')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f'proportional_deviance_barplot_{dependent_var.lower()}.jpg')
        print(f"Attempting to save plot to: {plot_path}")
        plt.savefig(plot_path, dpi=600, bbox_inches='tight')
        print(f"✓ Bar plot successfully saved to: {plot_path}")
        
        plt.close()
    except Exception as e:
        print(f"✗ Error during visualization: {e}")
        import traceback
        traceback.print_exc()

def generate_summary_output(model_results, null_model, dep_mean, dep_var, dependent_var):
    """
    Generate a comprehensive summary output
    """
    print("\n" + "="*60)
    print(f"Final Analysis Summary for {dependent_var}")
    print("="*60)
    
    try:
        # Distribution summary
        print(f"\n1. Distribution Analysis for {dependent_var}:")
        print(f"   Mean of {dependent_var}: {dep_mean:.4f}")
        print(f"   Variance of {dependent_var}: {dep_var:.4f}")
        print(f"   Variance/Mean Ratio: {dep_var/dep_mean:.4f}")
        
        # Null model summary
        print(f"\n2. Null Model Results for {dependent_var}:")
        print(f"   Null Deviance: {null_model.null_deviance:.4f}")
        print(f"   Residual Deviance: {null_model.deviance:.4f}")
        print(f"   AIC: {null_model.aic:.4f}")
        print(f"   BIC: {null_model.bic:.4f}")
        
        # Single variable model summary
        print(f"\n3. Single Variable Model Results for {dependent_var} (Sorted by Proportional Deviance Explained):")
        print("   " + "-"*70)
        print("   {:<10} {:<20} {:<12} {:<12} {:<12}".format(
            "Variable", "Prop. Deviance", "Coefficient", "p-value", "Significant"))
        print("   " + "-"*70)
        
        for result in model_results:
            print("   {:<10} {:<20.4f} {:<12.6f} {:<12.6f} {:<12}".format(
                result['variable'],
                result['proportional_deviance'],
                result['coef'],
                result['p_value'],
                "Yes" if result['is_significant'] else "No"))
        
        # Explanatory power ranking
        print(f"\n4. Explanatory Power Ranking for {dependent_var}:")
        for i, result in enumerate(model_results, 1):
            print(f"   {i}. {result['variable']} ({result['proportional_deviance']*100:.2f}% variance explained)")
        
        # Significant variables
        significant_vars = [result['variable'] for result in model_results if result['is_significant']]
        print(f"\n5. Significant Variables (p<0.05) for {dependent_var}: {', '.join(significant_vars) if significant_vars else 'None'}")
        
        # Model recommendation
        print(f"\n6. Model Recommendation for {dependent_var}:")
        print("   Gamma regression model has been used for analysis")
        
        # Best model
        best_model = model_results[0]
        print(f"   Best single variable model: {best_model['variable']} (Proportional Deviance: {best_model['proportional_deviance']:.4f})")
        
        # Write summary to file
        summary_path = os.path.join(output_dir, f'analysis_summary_{dependent_var.lower()}.txt')
        print(f"Attempting to save summary to: {summary_path}")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write(f"FIRE {dependent_var.upper()} GAMMA REGRESSION ANALYSIS SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"1. Distribution Analysis for {dependent_var}:\n")
            f.write(f"   Mean of {dependent_var}: {dep_mean:.4f}\n")
            f.write(f"   Variance of {dependent_var}: {dep_var:.4f}\n")
            f.write(f"   Variance/Mean Ratio: {dep_var/dep_mean:.4f}\n")
            
            f.write(f"\n2. Null Model Results for {dependent_var}:\n")
            f.write(f"   Null Deviance: {null_model.null_deviance:.4f}\n")
            f.write(f"   Residual Deviance: {null_model.deviance:.4f}\n")
            f.write(f"   AIC: {null_model.aic:.4f}\n")
            f.write(f"   BIC: {null_model.bic:.4f}\n")
            
            f.write(f"\n3. Single Variable Model Results for {dependent_var}:\n")
            f.write("   " + "-"*80 + "\n")
            f.write("   {:<10} {:<15} {:<12} {:<12} {:<12} {:<10} {:<10}\n".format(
                "Variable", "Prop. Deviance", "Coefficient", "Std Err", "p-value", "AIC", "BIC"))
            f.write("   " + "-"*80 + "\n")
            
            for result in model_results:
                f.write("   {:<10} {:<15.4f} {:<12.6f} {:<12.6f} {:<12.6f} {:<10.2f} {:<10.2f}\n".format(
                    result['variable'],
                    result['proportional_deviance'],
                    result['coef'],
                    result['std_err'],
                    result['p_value'],
                    result['aic'],
                    result['bic']))
            
            f.write(f"\n4. Explanatory Power Ranking for {dependent_var}:\n")
            for i, result in enumerate(model_results, 1):
                f.write(f"   {i}. {result['variable']} ({result['proportional_deviance']*100:.2f}% variance explained)\n")
            
            f.write(f"\n5. Significant Variables (p<0.05) for {dependent_var}: {', '.join(significant_vars) if significant_vars else 'None'}\n")
            
            f.write(f"\n6. Model Recommendation for {dependent_var}:\n")
            f.write("   Gamma regression model has been used for analysis\n")
            f.write(f"   Best single variable model: {best_model['variable']}\n")
        
        print(f"✓ Detailed summary successfully saved to: {summary_path}")
        
    except Exception as e:
        print(f"✗ Error during summary generation: {e}")
        import traceback
        traceback.print_exc()

def analyze_dependent_var(df, dependent_var):
    """
    Analyze a single dependent variable using Gamma regression
    """
    print(f"\n" + "="*80)
    print(f"ANALYZING: {dependent_var}")
    print(f"="*80)
    
    # Step 2: Data cleaning for this dependent variable
    df_clean = data_cleaning(df, dependent_var)
    if len(df_clean) == 0:
        print(f"No data left after cleaning for {dependent_var}, skipping analysis")
        return
    
    # Step 3: Analyze data basics
    dep_mean, dep_var = analyze_data_basics(df_clean, dependent_var)
    
    # Step 4: Build null model
    null_model, null_deviance = build_null_model(df_clean, dependent_var)
    
    # Step 5: Build single variable models
    model_results = build_single_var_models(df_clean, dependent_var, null_deviance)
    
    # Step 6: Visualize results
    visualize_results(model_results, dependent_var)
    
    # Step 7: Generate summary output
    generate_summary_output(model_results, null_model, dep_mean, dep_var, dependent_var)

def main():
    """
    Main function
    """
    print("=== Fire Frequency and Area Gamma Regression Analysis Program ===\n")
    
    # Define file path (user specified)
    csv_path = "I:/Processing data/Artificial influencing factors.csv"  # 使用正斜杠处理路径
    
    # Step 1: Load data
    df = load_data(csv_path)
    if df is None:
        print("Failed to load data, exiting program")
        return
    

    
    # Analyze dependent variables
    dependent_vars = ['Fire_frequency_pop', 'All_fire_area',  'Fire_frequency_per_WUI', 'Fire_area_density']
    for dep_var in dependent_vars:
        analyze_dependent_var(df, dep_var)
    
    print("\n" + "="*60)
    print("Analysis completed successfully for all variables!")
    print("="*60)

if __name__ == "__main__":
    main()