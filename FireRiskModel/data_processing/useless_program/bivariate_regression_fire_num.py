#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fire Number Negative Binomial Regression Analysis Program

功能：
    使用负二项式回归模型分析5个独立变量(Roadsd, GDP, PopD, HFI, All_WUI_area)对All_fire_num的解释力

输入文件：
    - 数据文件:I:/Processing data/Artificial influencing factors.csv

输出文件：
    - 可视化图表:B:/WUI/Picture and statistic results/proportional_deviance_barplot.jpg
    - 分析报告:B:/WUI/Picture and statistic results/analysis_summary.txt

说明：
    程序会检查数据是否存在过度离散，并根据分析结果提供模型选择建议
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.genmod.families import NegativeBinomial
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

def data_cleaning(df):
    """
    Clean the data according to the requirements
    """
    print("\n" + "="*60)
    print("Data Cleaning Process")
    print("="*60)
    
    # Original data size
    original_size = len(df)
    print(f"Original data size: {original_size}")
    
    # 1. Remove samples with missing values
    df_clean = df.dropna()
    print(f"After removing missing values: {len(df_clean)} samples ({original_size - len(df_clean)} removed)")
    
    # 2. Verify All_fire_num is non-negative integer
    # Check if All_fire_num is integer type
    df_clean['All_fire_num'] = pd.to_numeric(df_clean['All_fire_num'], errors='coerce')
    # Check non-negative integers
    mask = (df_clean['All_fire_num'] >= 0) & (df_clean['All_fire_num'] == df_clean['All_fire_num'].astype(int))
    df_clean = df_clean[mask]
    print(f"After verifying All_fire_num (non-negative integers): {len(df_clean)} samples ({original_size - len(df_clean)} removed)")
    
    # 3. Check and handle outliers in independent variables
    independent_vars = ['Roadsd', 'GDP', 'PopD', 'HFI', 'All_WUI_area']
    
    print("\nOutlier detection for independent variables (using IQR method):")
    
    for var in independent_vars:
        # Calculate IQR
        Q1 = df_clean[var].quantile(0.25)
        Q3 = df_clean[var].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outliers as values outside Q1 - 1.5*IQR to Q3 + 1.5*IQR
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outliers = df_clean[(df_clean[var] < lower_bound) | (df_clean[var] > upper_bound)]
        outlier_count = len(outliers)
        
        print(f"{var}: {outlier_count} outliers ({outlier_count/len(df_clean)*100:.2f}% of total)")
        
        # For now, we'll keep the outliers but note them
        # In a real analysis, you might want to handle them differently
        
    print("\nNote: Outliers were detected but kept for analysis (in a real scenario, consider appropriate handling)")
    
    return df_clean

def analyze_data_basics(df_clean):
    """
    Analyze basic data information
    """
    print("\n" + "="*60)
    print("Data Basic Information")
    print("="*60)
    
    # Sample size
    print(f"Final sample size: {len(df_clean)}")
    
    # Calculate mean and variance
    variables = ['Roadsd', 'GDP', 'PopD', 'HFI', 'All_WUI_area', 'All_fire_num']
    
    print("\nMean and Variance of Variables:")
    print("-"*40)
    print(f"{'Variable':<15} {'Mean':<12} {'Variance':<12} {'Std Dev':<12}")
    print("-"*40)
    
    for var in variables:
        mean_val = df_clean[var].mean()
        var_val = df_clean[var].var()
        std_val = df_clean[var].std()
        print(f"{var:<15} {mean_val:<12.4f} {var_val:<12.4f} {std_val:<12.4f}")
    
    # Check for overdispersion in All_fire_num
    fire_mean = df_clean['All_fire_num'].mean()
    fire_var = df_clean['All_fire_num'].var()
    
    print("\n" + "="*60)
    print("Overdispersion Analysis for All_fire_num")
    print("="*60)
    print(f"Mean of All_fire_num: {fire_mean:.4f}")
    print(f"Variance of All_fire_num: {fire_var:.4f}")
    print(f"Variance/Mean Ratio: {fire_var/fire_mean:.4f}")
    
    if fire_var > fire_mean:
        print("Conclusion: There is overdispersion in All_fire_num (variance > mean)")
        print("Recommendation: Consider using Negative Binomial regression in addition to Poisson")
    else:
        print("Conclusion: No significant overdispersion detected in All_fire_num")
    
    return fire_mean, fire_var

def build_null_model(df_clean):
    """
    Build the null model (intercept only)
    """
    print("\n" + "="*60)
    print("Building Null Model (Intercept Only)")
    print("="*60)
    
    # Null model (only intercept)
    X_null = sm.add_constant(np.ones(len(df_clean)))
    y = df_clean['All_fire_num']
    
    # Fit Negative Binomial regression model
    null_model = sm.GLM(y, X_null, family=NegativeBinomial()).fit()
    
    # Get null deviance
    null_deviance = null_model.null_deviance
    print(f"Null Model Null Deviance: {null_deviance:.4f}")
    print(f"Null Model Residual Deviance: {null_model.deviance:.4f}")
    print(f"Null Model AIC: {null_model.aic:.4f}")
    print(f"Null Model BIC: {null_model.bic:.4f}")
    
    return null_model, null_deviance

def build_single_var_models(df_clean, null_deviance):
    """
    Build single variable Negative Binomial models for each independent variable
    """
    print("\n" + "="*60)
    print("Building Single Variable Negative Binomial Models")
    print("="*60)
    
    independent_vars = ['Roadsd', 'GDP', 'PopD', 'HFI', 'All_WUI_area']
    model_results = []
    
    for var in independent_vars:
        print(f"\n{'-'*40}")
        print(f"Model with {var} as independent variable")
        print(f"{'-'*40}")
        
        # Prepare data
        X = sm.add_constant(df_clean[var])
        y = df_clean['All_fire_num']
        
        # Fit Negative Binomial regression model
        model = sm.GLM(y, X, family=NegativeBinomial()).fit()
        
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

def visualize_results(model_results):
    """
    Visualize the results of the analysis
    """
    print("\n" + "="*60)
    print("Visualizing Results")
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
        plt.ylim(0, 0.2)  # Adjust y-axis based on actual values
        
        # Add values on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=15)
        
        # Add legend
        significant_patch = plt.Rectangle((0,0),1,1, color='#4169E1', alpha=0.7)
        not_significant_patch = plt.Rectangle((0,0),1,1, color='#808080', alpha=0.7)
        plt.legend([significant_patch, not_significant_patch], ['Significant (p<0.05)', 'Not Significant'], loc='upper right')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, 'proportional_deviance_barplot.jpg')
        print(f"Attempting to save plot to: {plot_path}")
        plt.savefig(plot_path, dpi=600, bbox_inches='tight')
        print(f"✓ Bar plot successfully saved to: {plot_path}")
        
        plt.close()
    except Exception as e:
        print(f"✗ Error during visualization: {e}")
        import traceback
        traceback.print_exc()

def generate_summary_output(model_results, null_model, fire_mean, fire_var):
    """
    Generate a comprehensive summary output
    """
    print("\n" + "="*60)
    print("Final Analysis Summary")
    print("="*60)
    
    try:
        # Overdispersion summary
        print("\n1. Overdispersion Analysis:")
        print(f"   Mean of All_fire_num: {fire_mean:.4f}")
        print(f"   Variance of All_fire_num: {fire_var:.4f}")
        print(f"   Variance/Mean Ratio: {fire_var/fire_mean:.4f}")
        if fire_var > fire_mean:
            print("   Conclusion: There is significant overdispersion")
        else:
            print("   Conclusion: No significant overdispersion")
        
        # Null model summary
        print("\n2. Null Model Results:")
        print(f"   Null Deviance: {null_model.null_deviance:.4f}")
        print(f"   Residual Deviance: {null_model.deviance:.4f}")
        print(f"   AIC: {null_model.aic:.4f}")
        print(f"   BIC: {null_model.bic:.4f}")
        
        # Single variable model summary
        print("\n3. Single Variable Model Results (Sorted by Proportional Deviance Explained):")
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
        print("\n4. Explanatory Power Ranking:")
        for i, result in enumerate(model_results, 1):
            print(f"   {i}. {result['variable']} ({result['proportional_deviance']*100:.2f}% variance explained)")
        
        # Significant variables
        significant_vars = [result['variable'] for result in model_results if result['is_significant']]
        print(f"\n5. Significant Variables (p<0.05): {', '.join(significant_vars) if significant_vars else 'None'}")
        
        # Model recommendation
        print("\n6. Model Recommendation:")
        if fire_var > fire_mean:
            print("   Due to overdispersion, consider using Negative Binomial regression models")
        else:
            print("   Poisson regression models are appropriate due to no significant overdispersion")
        
        # Best model
        best_model = model_results[0]
        print(f"   Best single variable model: {best_model['variable']} (Proportional Deviance: {best_model['proportional_deviance']:.4f})")
        
        # Write summary to file
        summary_path = os.path.join(output_dir, 'analysis_summary.txt')
        print(f"Attempting to save summary to: {summary_path}")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("FIRE NUMBER NEGATIVE BINOMIAL REGRESSION ANALYSIS SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            f.write("1. Overdispersion Analysis:\n")
            f.write(f"   Mean of All_fire_num: {fire_mean:.4f}\n")
            f.write(f"   Variance of All_fire_num: {fire_var:.4f}\n")
            f.write(f"   Variance/Mean Ratio: {fire_var/fire_mean:.4f}\n")
            if fire_var > fire_mean:
                f.write("   Conclusion: There is significant overdispersion\n")
            else:
                f.write("   Conclusion: No significant overdispersion\n")
            
            f.write("\n2. Null Model Results:\n")
            f.write(f"   Null Deviance: {null_model.null_deviance:.4f}\n")
            f.write(f"   Residual Deviance: {null_model.deviance:.4f}\n")
            f.write(f"   AIC: {null_model.aic:.4f}\n")
            f.write(f"   BIC: {null_model.bic:.4f}\n")
            
            f.write("\n3. Single Variable Model Results:\n")
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
            
            f.write("\n4. Explanatory Power Ranking:\n")
            for i, result in enumerate(model_results, 1):
                f.write(f"   {i}. {result['variable']} ({result['proportional_deviance']*100:.2f}% variance explained)\n")
            
            f.write(f"\n5. Significant Variables (p<0.05): {', '.join(significant_vars) if significant_vars else 'None'}\n")
            
            f.write("\n6. Model Recommendation:\n")
            if fire_var > fire_mean:
                f.write("   Due to overdispersion, consider using Negative Binomial regression models\n")
            else:
                f.write("   Poisson regression models are appropriate due to no significant overdispersion\n")
            f.write(f"   Best single variable model: {best_model['variable']}\n")
        
        print(f"✓ Detailed summary successfully saved to: {summary_path}")
        
    except Exception as e:
        print(f"✗ Error during summary generation: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Main function
    """
    print("=== Fire Number Negative Binomial Regression Analysis Program ===\n")
    
    # Define file path (user specified)
    csv_path = "I:/Processing data/Artificial influencing factors.csv"  # 使用正斜杠处理路径
    
    # Step 1: Load data
    df = load_data(csv_path)
    if df is None:
        print("Failed to load data, exiting program")
        return
    
    # Step 2: Data cleaning
    df_clean = data_cleaning(df)
    if len(df_clean) == 0:
        print("No data left after cleaning, exiting program")
        return
    
    # Step 3: Analyze data basics
    fire_mean, fire_var = analyze_data_basics(df_clean)
    
    # Step 4: Build null model
    null_model, null_deviance = build_null_model(df_clean)
    
    # Step 5: Build single variable models
    model_results = build_single_var_models(df_clean, null_deviance)
    
    # Step 6: Visualize results
    visualize_results(model_results)
    
    # Step 7: Generate summary output
    generate_summary_output(model_results, null_model, fire_mean, fire_var)
    
    print("\n" + "="*60)
    print("Analysis completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()