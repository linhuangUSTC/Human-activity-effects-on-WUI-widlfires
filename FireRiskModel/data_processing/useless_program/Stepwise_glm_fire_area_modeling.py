#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
程序名称: GLM火灾面积建模程序
功能简介: 本程序用于对火灾面积进行广义线性模型(GLM)建模分析，主要功能包括：
         1. 针对每个分类的火灾数据文件进行Gamma回归建模
         2. 使用向后逐步变量选择法优化模型，选择最优变量组合
         3. 生成详细的模型报告和可视化结果
         4. 保存建模结果和最终选择的变量

主要方法:
         1. 向后逐步变量选择法(backward_stepwise_selection)
         2. Gamma回归模型(使用log链接函数)
         3. 模型性能评估(包括BIC、MSE、MAE等指标)

输入路径:
         - 火灾面积数据: I:\Processing data\渔网分类建模\分类建模数据\Fire_area_norm
        - 数据格式: CSV文件,包含Fire_area_norm等必要列

输出路径:
        - 模型报告和散点图: B:\WUI\Picture
        - 建模结果汇总: I:\Processing data\渔网分类建模\分类建模数据\modeling_final_variables.csv
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Set font to Arial
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

def backward_stepwise_selection(y, X_scaled):
    # Initial variable list (without constant term)
    selected_vars = X_scaled.columns.tolist()
    best_bic = float('inf')
    best_results = None
    selection_history = []
    
    # Add constant term
    X = sm.add_constant(X_scaled[selected_vars])
    
    # Fit initial model
    try:
        # Only use Gamma regression
        model = sm.GLM(y, X, family=sm.families.Gamma(link=sm.families.links.Log()))
        best_results = model.fit()
        best_bic = best_results.bic
        
        # Record initial state
        selection_history.append({
            'step': 0,
            'action': 'Initial Model',
            'removed_var': None,
            'variables': selected_vars.copy(),
            'bic': best_bic
        })
    except Exception as e:
        print(f"Initial model fitting failed: {e}")
        return selected_vars, best_results, best_bic, selection_history
    
    step = 1
    # Continue selection while there are variables
    while len(selected_vars) > 0:
        bic_scores = []
        results_list = []
        
        # Try removing each variable
        for var in selected_vars:
            temp_vars = selected_vars.copy()
            temp_vars.remove(var)
            
            # Add constant term
            X_temp = sm.add_constant(X_scaled[temp_vars])
            
            try:
                # Only use Gamma regression
                temp_model = sm.GLM(y, X_temp, family=sm.families.Gamma(link=sm.families.links.Log()))
                temp_results = temp_model.fit()
                bic_scores.append(temp_results.bic)
                results_list.append((var, temp_results))
            except Exception as e:
                # If fitting fails, set BIC to infinity
                bic_scores.append(float('inf'))
                results_list.append((var, None))
        
        # Find the variable removal that gives the smallest BIC value
        min_bic = min(bic_scores)
        min_index = bic_scores.index(min_bic)
        removed_var, temp_results = results_list[min_index]
        
        # Continue removing if BIC decreases
        if min_bic < best_bic:
            # Record changes before and after variable removal
            selection_history.append({
                'step': step,
                'action': 'Remove Variable',
                'removed_var': removed_var,
                'variables_before': selected_vars.copy(),
                'variables_after': selected_vars.copy(),
                'bic_before': best_bic,
                'bic_after': min_bic,
                'bic_change': min_bic - best_bic
            })
            
            # Update best model and BIC
            best_bic = min_bic
            selected_vars.remove(removed_var)
            best_results = temp_results
            
            # Update variable list in record
            selection_history[-1]['variables_after'] = selected_vars.copy()
            step += 1
        else:
            # Record stopping information
            selection_history.append({
                'step': step,
                'action': 'Stop Selection',
                'removed_var': None,
                'variables': selected_vars.copy(),
                'bic': best_bic,
                'reason': 'Cannot reduce BIC by removing variables'
            })
            # Stop if no variables can be removed to reduce BIC
            break
    
    return selected_vars, best_results, best_bic, selection_history


def process_csv_file(csv_file, output_dir, failure_count):
    try:
        # Read data
        df = pd.read_csv(csv_file)
        
        # Check if required columns exist
        required_cols = ['All_fire_norm_area', 'Roadsd', 'PopD', 'GDP', 'HFI', 'All_WUI_area']
        for col in required_cols:
            if col not in df.columns:
                print(f"File {os.path.basename(csv_file)} is missing required column: {col}")
                failure_count[0] += 1
                return
        
        # Extract file name (without extension)
        file_name = os.path.splitext(os.path.basename(csv_file))[0]
        
        # Dependent variable processing
        y = df['All_fire_norm_area'].copy()
        
        # Replace zero values
        y = y.replace(0, 1e-6)
        
        # Independent variable processing
        X_vars = df[['Roadsd', 'PopD', 'GDP', 'HFI', 'All_WUI_area']]
        
        # Standardize independent variables
        X_scaled = (X_vars - X_vars.mean()) / X_vars.std()
        
        # Try to build Gamma regression model
        model = None
        results = None
        selected_vars = []
        
        # Initial model fitting (without backward selection)
        try:
            X_initial = sm.add_constant(X_scaled)
            model = sm.GLM(y, X_initial, family=sm.families.Gamma(link=sm.families.links.Log()))
            results_initial = model.fit()
            
            # Use backward stepwise selection to select variables
            selected_vars, results, best_bic, selection_history = backward_stepwise_selection(y, X_scaled)
        except Exception as e:
            print(f"Gamma regression failed ({file_name}): {e}")
            failure_count[0] += 1
            return
        
        # If backward selection fails, use initial model
        if results is None:
            results = results_initial
            selected_vars = X_scaled.columns.tolist()
            # Create default selection history
            selection_history = [{
                'step': 0,
                'action': 'Initial Model',
                'removed_var': None,
                'variables': selected_vars,
                'bic': results.bic
            }, {
                'step': 1,
                'action': 'Stop Selection',
                'removed_var': None,
                'variables': selected_vars,
                'bic': results.bic,
                'reason': 'Backward selection failed, using initial model'
            }]
        
        # Generate X using selected variables
        X_selected = sm.add_constant(X_scaled[selected_vars])
        
        # Generate predicted values
        y_pred = results.predict(X_selected)
        
        # Generate model report
        report_file = os.path.join(output_dir, f"{file_name}_model_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"GLM Model Report - {file_name}\n")
            f.write(f"Model Type: Gamma Regression\n")
            f.write(f"Using backward stepwise selection\n")
            f.write(f"Selected Variables: {selected_vars}\n")
            f.write(f"{'='*60}\n\n")
            
            f.write("\n0. Variable Selection Process\n")
            f.write("-" * 40 + "\n")
            f.write(f"Initial Variables: Roadsd, PopD, GDP, HFI, All_WUI_area\n\n")
            
            # Display detailed variable selection history
            for record in selection_history:
                f.write(f"Step {record['step']}: {record['action']}\n")
                
                if record['action'] == 'Initial Model':
                    f.write(f"  - Variables: {record['variables']}\n")
                    f.write(f"  - BIC: {record['bic']:.4f}\n")
                elif record['action'] == 'Remove Variable':
                    f.write(f"  - Removed Variable: {record['removed_var']}\n")
                    f.write(f"  - Variables Before: {record['variables_before']}\n")
                    f.write(f"  - Variables After: {record['variables_after']}\n")
                    f.write(f"  - BIC Before: {record['bic_before']:.4f}\n")
                    f.write(f"  - BIC After: {record['bic_after']:.4f}\n")
                    f.write(f"  - BIC Change: {record['bic_change']:.4f}\n")
                elif record['action'] == 'Stop Selection':
                    f.write(f"  - Final Variables: {record['variables']}\n")
                    f.write(f"  - Final BIC: {record['bic']:.4f}\n")
                    if 'reason' in record:
                        f.write(f"  - Reason: {record['reason']}\n")
                f.write("\n")
            
            f.write("\nVariable Selection Results:\n")
            f.write(f"- Selected Variables: {selected_vars}\n")
            f.write(f"- Excluded Variables: {[var for var in ['Roadsd', 'PopD', 'GDP', 'HFI', 'All_WUI_area'] if var not in selected_vars]}\n")
            f.write(f"- Final BIC: {best_bic:.4f}\n")
            
            f.write("\n1. Model Summary\n")
            f.write("-" * 30 + "\n")
            f.write(str(results.summary()) + "\n\n")
            
            # Calculate performance metrics
            f.write("\n2. Performance Metrics\n")
            f.write("-" * 30 + "\n")
            
            # Log-likelihood
            if hasattr(results, 'llf'):
                f.write(f"Log-Likelihood: {results.llf:.4f}\n")
            
            # AIC and BIC
            if hasattr(results, 'aic'):
                f.write(f"AIC: {results.aic:.4f}\n")
            if hasattr(results, 'bic'):
                f.write(f"BIC: {results.bic:.4f}\n")
            
            # R-squared or Pseudo R-squared
            if hasattr(results, 'rsquared'):
                f.write(f"R-squared: {results.rsquared:.4f}\n")
                f.write(f"Adjusted R-squared: {results.rsquared_adj:.4f}\n")
            elif hasattr(results, 'prsquared'):
                f.write(f"Pseudo R-squared: {results.prsquared:.4f}\n")
            
            # Calculate mean squared error
            mse = np.mean((y - y_pred) ** 2)
            f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
            f.write(f"Root Mean Squared Error (RMSE): {np.sqrt(mse):.4f}\n")
            
            # Calculate mean absolute error
            mae = np.mean(np.abs(y - y_pred))
            f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
        
        # Generate scatter plots
        scatter_file = os.path.join(output_dir, f"{file_name}_scatter.png")
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # True vs Predicted Values
        axes[0].scatter(y, y_pred, alpha=0.5, color='blue')
        axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        axes[0].set_xlabel('True Values (All_fire_norm_area)')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title('True vs Predicted Values')
        axes[0].grid(True)
        
        # Scatter plots of independent variables vs dependent variable - show all selected variables (max 5)
        max_vars_to_show = 5
        for i, var in enumerate(selected_vars):
            if i < max_vars_to_show:  # Show maximum 5 independent variable scatter plots
                axes[i+1].scatter(X_vars[var], y, alpha=0.5, color='purple')
                axes[i+1].set_xlabel(var)
                axes[i+1].set_ylabel('Fire_area_norm')
                axes[i+1].set_title(f'{var} (selected) vs Fire_area_norm')
                axes[i+1].grid(True)
        
        # Hide remaining subplots if fewer variables are selected than max_vars_to_show
        for i in range(len(selected_vars), max_vars_to_show):
            axes[i+1].set_visible(False)
        
        # Adjust layout
        plt.tight_layout()
        plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Successfully processed file: {file_name}")
        print(f"  - Model report: {report_file}")
        print(f"  - Scatter plots: {scatter_file}")
        
        return file_name, selected_vars
        
    except Exception as e:
        print(f"Error processing file {os.path.basename(csv_file)}: {e}")
        return None


def main():
    # Input and output directories
    input_dir = r"I:\Processing data\渔网分类建模\分类建模数据\Fire_area_norm"  # Fire area data directory
    output_dir = r"B:\WUI\Picture"  # Output directory
    
    # Use list to implement reference passing for failure count
    failure_count = [0]
    
    # Create list to collect final variable information for all category IDs
    modeling_results = []
    
    # Check if output directory exists, create if it doesn't
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all CSV files
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in directory {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Process each CSV file
    for csv_file in csv_files:
        csv_path = os.path.join(input_dir, csv_file)
        result = process_csv_file(csv_path, output_dir, failure_count)
        
        # If processing is successful, collect results
        if result is not None:
            category_id, selected_vars = result
            modeling_results.append({
                'Category_ID': category_id,
                'Selected_Variables': selected_vars
            })
    
    print(f"\nAll files processed, number of model fitting failures: {failure_count[0]}")
    print(f"Number of successfully processed files: {len(modeling_results)}")
    
    # Return modeling results for saving to CSV
    return modeling_results


if __name__ == "__main__":
    # Run main function and get modeling results
    results = main()
    
    # If there are results, save them to CSV file
    if results:
        # Convert to DataFrame
        df_results = pd.DataFrame(results)
        
        # Convert Selected_Variables column to string format
        df_results['Selected_Variables'] = df_results['Selected_Variables'].apply(lambda x: ', '.join(x))
        
        # Build output file path
        output_csv_path = r"I:\Processing data\渔网分类建模\分类建模数据\modeling_final_variables.csv"
        
        # Save as CSV file
        df_results.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        
        print(f"\nModeling variable selection results saved to: {output_csv_path}")
        print(f"Contains modeling results for {len(results)} category IDs")
    else:
        print("\nNo modeling results generated")
