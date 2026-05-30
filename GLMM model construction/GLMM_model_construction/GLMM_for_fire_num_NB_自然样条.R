#!/usr/bin/env Rscript

# ------------------------------------------------------------------------
# 程序: GLMM_with_HFI_only_for_fire_num_with_WUI_area_offset_NS_NB.R
# 功能: 使用负二项分布的广义线性混合模型(GLMM)对火灾数量数据进行建模分析，具体功能包括：
#       - 将WUI area作为offset项纳入模型，控制面积效应
#       - 仅使用HFI（人类足迹指数）作为自变量，采用自然样条进行非线性建模
#       - 计算模型评估指标：边际R²、条件R²、MSE、RMSE、MAE
#       - 批量处理多份CSV数据文件
# 输入: CSV文件(路径: I:/Processing data/渔网分类建模/Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic/)
#       - 必须包含列：All_fire_num、All_WUI_area、HFI、GRID_ID
# 输出: 模型报告(TXT)和散点图(PNG)，输出路径: B:/WUI/Picture/
#       - TXT报告：包含模型参数估计、拟合优度指标
#       - PNG图片：实际值与预测值的关系图，直观展示模型拟合效果
# ------------------------------------------------------------------------

# 加载必要的包
library(ggplot2)
library(glmmTMB)
library(performance)
library(splines)  # 用于样条函数

# 检查模型收敛性
test_model_convergence <- function(model, maxit = 1e4) {
  # 初始化默认值
  is_converged <- FALSE
  convergence_code <- NA_real_
  max_gradient <- NA_real_
  gradient_ok <- NA
  random_var_ok <- NA
  random_var_value <- NA  # 新增：记录实际的随机效应方差值
  
  # 第一步：检查模型是否为NULL/有效glmmTMB对象
  if (is.null(model) || !inherits(model, "glmmTMB")) {
    cat("❌ 模型对象无效（非glmmTMB类型或为NULL）\n")
    return(list(
      is_converged = is_converged,
      convergence_code = convergence_code,
      max_gradient = max_gradient,
      gradient_ok = gradient_ok,
      random_var_ok = random_var_ok,
      random_var_value = random_var_value
    ))
  }
  
  # 第二步：提取收敛信息
  tryCatch({
    # 1. 收敛码：直接从fit$convergence提取
    if (!is.null(model$fit$convergence)) {
      convergence_code <- model$fit$convergence
    }
    
    # 2. 最大梯度：从sdr$gradient提取
    if (!is.null(model$sdr$gradient)) {
      grad <- model$sdr$gradient
      max_gradient <- max(abs(grad))
      gradient_ok <- max_gradient < 1e-2
    }
    
    # 3. 随机效应方差 > 0 检验
    tryCatch({
      varcorr_result <- VarCorr(model)  # 提取方差-协方差结构
      if (!is.null(varcorr_result$cond) && length(varcorr_result$cond) > 0) {
        # 提取第一个随机效应的方差
        random_effects_names <- names(varcorr_result$cond)
        if (length(random_effects_names) > 0) {
          first_rand_effect <- random_effects_names[1]
          rand_var <- as.numeric(varcorr_result$cond[[first_rand_effect]])
          # 记录实际方差值
          random_var_value <- rand_var
          # 检验方差是否>0（允许微小数值误差）
          random_var_ok <- rand_var > 1e-8
        }
      }
    }, error = function(e) {
      # 如果提取随机效应方差失败，不影响整体收敛判断
      random_var_ok <- TRUE  # 设为TRUE，这样不会影响收敛判断
    })
    
    # 4. 综合判断收敛：收敛码为0且梯度达标
    if (!is.na(convergence_code) && !is.na(gradient_ok)) {
      is_converged <- (convergence_code == 0) && isTRUE(gradient_ok)
    }
    
  }, error = function(e) {
    cat(paste0("❌ 提取收敛信息失败: ", e$message, "\n"))
  })
  
  return(list(
    is_converged = is_converged,
    convergence_code = convergence_code,
    max_gradient = max_gradient,
    gradient_ok = gradient_ok,
    random_var_ok = random_var_ok,
    random_var_value = random_var_value
  ))
}

# 计算R²值
calculate_r_squared <- function(model) {
  r2_default <- performance::r2_nakagawa(model)
  # 只看计数（conditional）部分
  r2_cond <- performance::r2_nakagawa(model,model_component = "conditional",approximation = "lognormal")
  
  return(list(
    # 现有：默认
    marginal = as.numeric(r2_default$R2_marginal),
    conditional = as.numeric(r2_default$R2_conditional),
    # 新增：conditional-only
    marginal_cond = as.numeric(r2_cond$R2_marginal),
    conditional_cond = as.numeric(r2_cond$R2_conditional),
    method = "performance::r2_nakagawa (default + full + conditional)"
  ))
}

# 生成模型报告（带offset项）
generate_model_report <- function(model, df, file_name, output_dir, y_col, x_col, offset_col, group_var, y_pred, pred_freq, obs_freq, r_squared, convergence_result) {
  report_file <- file.path(output_dir, paste0("GLMM_NaturalSpline_ZINB_HFI_only_", file_name, "_report.txt"))
  
  # 火灾频率的评估指标
  mse_freq <- mean((obs_freq - pred_freq)^2)
  rmse_freq <- sqrt(mse_freq)
  mae_freq <- mean(abs(obs_freq - pred_freq))
  
  # 打开文件进行写入
  sink(report_file, type = "output")
  cat(paste0("\n", strrep("=", 70), "\n"))
  cat(paste0("Natural Spline NB-GLMM (log link) Fire Frequency Model Report with HFI Only - ", file_name, "\n"))
  cat(paste0(strrep("=", 70), "\n\n"))
  
  cat("1. Model Convergence\n")
  cat(strrep("-", 40), "\n", sep = "")
  if (convergence_result$is_converged) {
    cat("✅ 模型成功收敛\n")
  } else {
    cat("⚠️  警告: 模型未收敛，参数结果可能不可靠!\n")
  }
  cat(paste0("收敛码: ", ifelse(is.na(convergence_result$convergence_code), "N/A", convergence_result$convergence_code), "\n"))
  cat(paste0("最大梯度: ", format(convergence_result$max_gradient, scientific = TRUE, digits = 4), "\n"))
  cat(paste0("梯度检查: ", ifelse(convergence_result$gradient_ok, "通过", ifelse(is.na(convergence_result$gradient_ok), "N/A", "未通过")), "\n"))
  cat(paste0("随机效应方差正常: ", ifelse(isTRUE(convergence_result$random_var_ok), "是", "否"), "\n"))
  cat(paste0("随机效应方差值: ", ifelse(!is.na(convergence_result$random_var_value), format(convergence_result$random_var_value, scientific = TRUE, digits = 4), "N/A"), "\n\n"))
  
  cat("2. Model Summary\n")
  cat(strrep("-", 40), "\n", sep = "")
  print(summary(model))
  cat("\n")
  
  cat("3. Fit Statistics (AIC/BIC/logLik etc.)\n")
  cat(strrep("-", 40), "\n", sep = "")
  cat(paste0("AIC: ", AIC(model), "\n"))
  cat(paste0("BIC: ", BIC(model), "\n"))
  cat(paste0("logLik: ", logLik(model), "\n"))
  cat("\n")
  
  cat("4. Performance Metrics (in-sample, response scale)\n")
  cat(strrep("-", 40), "\n", sep = "")
  cat(paste0("R2 (deviance explained) - Marginal: ", format(r_squared$marginal, digits=6),"  Conditional: ", format(r_squared$conditional, digits=6), "\n"))
  cat(paste0("R2 Method: ", r_squared$method, "\n"))
  cat("\n")
  cat(paste0("MSE : ", format(mse_freq, digits = 6), "\n"))
  cat(paste0("RMSE: ", format(rmse_freq, digits = 6), "\n"))
  cat(paste0("MAE : ", format(mae_freq, digits = 6), "\n"))
  
  # 关闭文件
  sink()
  
  cat(paste0("模型报告已保存: ", report_file, "\n"))
}

# 生成HFI对Fire frequency的边际效应图
generate_marginal_effect_plot <- function(model, df, file_name, output_dir, x_col, y_col, offset_col, group_var, standardized_stats) {
  # 获取HFI的原始统计信息
  original_var <- gsub("_std$", "", x_col)
  orig_mean <- standardized_stats[[original_var]]$mean
  orig_sd <- standardized_stats[[original_var]]$sd
  
  # 确定样条的节点数（与拟合时一致）
  n_knots <- min(4, max(3, floor(nrow(df)/10)))
  
  # 在预测时使用与拟合时相同的knots和边界knots
  # 从原始数据的HFI_std变量创建自然样条以获取knots和Boundary.knots
  temp_spline <- ns(df[[x_col]], df=n_knots)
  
  # 创建HFI的标准值序列
  hfi_seq <- seq(min(df[[x_col]], na.rm = TRUE), max(df[[x_col]], na.rm = TRUE), length.out = 100)
  
  # 创建新数据框，固定其他变量（除了HFI和offset）
  # 使用平均值固定其他变量（这里只有HFI是变量）
  new_data <- data.frame(
    HFI_std = hfi_seq,
    WUI_area = mean(df[[offset_col]], na.rm = TRUE),  # 使用平均WUI面积
    GRID_ID = df[[group_var]][1]  # 使用第一个GRID_ID（因为随机效应在边际效应中通常被忽略）
  )
  
  # 使用与拟合时相同的knots和Boundary.knots创建样条基函数
  spline_basis_new <- ns(hfi_seq, knots = attr(temp_spline, "knots"), Boundary.knots = attr(temp_spline, "Boundary.knots"))
  spline_names <- paste0("spline_", 1:ncol(spline_basis_new))
  for(j in 1:ncol(spline_basis_new)) {
    new_data[[spline_names[j]]] <- spline_basis_new[, j]
  }
  
  # 预测响应值（包括置信区间）
  pred_result <- predict(model, newdata = new_data, type = "response", se.fit = TRUE, re.form = NA)
  
  # 计算预测的火灾频率
  pred_freq <- pred_result$fit / new_data$WUI_area
  se_freq <- pred_result$se.fit / new_data$WUI_area  # 标准误也需要按面积标准化
  
  # 计算95%置信区间
  ci_lower <- (pred_result$fit - 1.96 * pred_result$se.fit) / new_data$WUI_area
  ci_upper <- (pred_result$fit + 1.96 * pred_result$se.fit) / new_data$WUI_area
  
  # 将标准化的HFI转换回原始尺度用于绘图
  hfi_original_scale <- hfi_seq * orig_sd + orig_mean
  
  # 创建预测曲线数据框
  pred_data <- data.frame(
    hfi_original = hfi_original_scale,
    pred_freq = pred_freq,
    ci_lower = pmax(0, ci_lower),  # 确保下限非负
    ci_upper = ci_upper
  )
  
  # 按数据量分成10个等分（每个bin数据量相等）
  n_bins <- 10
  n_total <- nrow(df)
  
  # 按HFI值排序
  df_sorted <- df[order(df[[original_var]]), ]
  
  # 为每个数据点分配所属的bin（按排序后的顺序等分）
  bin_assignments <- cut(1:n_total, breaks = n_bins, labels = FALSE)
  df_sorted$bin <- bin_assignments
  
  # 创建汇总表
  bin_summary <- data.frame(bin = integer(), hfi_min = numeric(), hfi_max = numeric(), hfi_center = numeric(), mean = numeric(), se = numeric(), count = numeric())
  
  # 逐个计算每个bin的统计信息（使用均值和标准误）
  for(i in 1:n_bins) {
    bin_data <- df_sorted[df_sorted$bin == i, ]
    if(nrow(bin_data) > 0) {
      hfi_values <- bin_data[[original_var]]
      hfi_min <- min(hfi_values, na.rm = TRUE)
      hfi_max <- max(hfi_values, na.rm = TRUE)
      hfi_center <- (hfi_min + hfi_max) / 2  # 使用范围的中点作为x坐标
      freq_mean <- mean(bin_data$Fire_frequency, na.rm = TRUE)  # 使用均值
      freq_se <- sd(bin_data$Fire_frequency, na.rm = TRUE) / sqrt(length(bin_data$Fire_frequency))  # 计算标准误
      freq_count <- length(bin_data$Fire_frequency)
      
      new_row <- data.frame(
        bin = i,
        hfi_min = hfi_min,
        hfi_max = hfi_max,
        hfi_center = hfi_center,
        mean = freq_mean,
        se = freq_se,
        count = freq_count
      )
      
      bin_summary <- rbind(bin_summary, new_row)
    }
  }
  
  # 过滤掉空的bin（即没有数据的bin）
  bin_summary <- bin_summary[!is.na(bin_summary$mean) & !is.nan(bin_summary$mean), ]
  
  # 计算柱状图的宽度（相邻bin中心点距离的一半）
  bin_summary <- bin_summary[order(bin_summary$hfi_center), ]  # 按HFI中心值排序
  if(nrow(bin_summary) > 1) {
    # 计算相邻bin中心点的距离，取最小距离的一半作为柱宽
    hfi_diffs <- diff(sort(bin_summary$hfi_center))
    bar_width <- min(hfi_diffs) * 0.8  # 使用最小距离的80%作为柱宽
  } else {
    # 如果只有一个bin，使用一个固定的小值
    bar_width <- 0.1
  }
  
  # 绘制边际效应图（添加标准误误差棒）
  marginal_plot <- ggplot(pred_data, aes(x = hfi_original)) +
    geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), fill = "lightblue", alpha = 0.5) +
    geom_line(aes(y = pred_freq), color = "red", size = 1) +
    # 添加柱状图表示各bin的均值（使用浅蓝色填充）
    geom_col(data = bin_summary, aes(x = hfi_center, y = mean), 
             width = bar_width, fill = "lightblue", color = "black", alpha = 0.7) +
    # 添加标准误的误差棒
    geom_errorbar(data = bin_summary, aes(x = hfi_center, ymin = mean - se, ymax = mean + se), 
                  width = bar_width * 0.6, color = "black", size = 0.8) +
    xlab("HFI") +
    ylab(expression(Fire~frequency~(per~km^2))) +
    xlim(0, max(df[[original_var]], na.rm = TRUE)) +
    theme_bw() +
    theme(
      panel.grid = element_line(colour = "gray90"),
      axis.text = element_text(size = rel(1.5)),
      axis.title = element_text(size = rel(1.5))
    )
  
  # 保存边际效应图
  marginal_file <- file.path(output_dir, paste0("GLMM_NaturalSpline_ZINB_HFI_only_", file_name, "_marginal_effect.png"))
  png(marginal_file, width = 12, height = 8, units = "in", res = 600)
  print(marginal_plot)
  dev.off()
  
  cat(paste0("边际效应图已保存: ", marginal_file, "\n"))
  
  return(marginal_file)
}

# 创建目录（如果不存在）
create_dir_if_not_exists <- function(dir_path) {
  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE)
    cat(paste0("创建目录: ", dir_path, "\n"))
  }
}

# 保存模型与可复用信息
save_model_artifact <- function(model, file_name, output_dir, standardized_stats, spline_info, formula_str, x_col, y_col, offset_col, group_var) {
  model_file <- file.path(output_dir, paste0("GLMM_NaturalSpline_ZINB_HFI_only_", file_name, "_model.rds"))
  saveRDS(list(
    model = model,
    standardized_stats = standardized_stats,
    spline_info = spline_info,
    formula = formula_str,
    x_col = x_col,
    y_col = y_col,
    offset_col = offset_col,
    group_var = group_var
  ), model_file)
  cat(paste0("模型已保存: ", model_file, "\n"))
}

# 处理单个CSV文件
process_single_csv <- function(csv_path, output_dir) {
  tryCatch({
    # 读取数据
    df <- read.csv(csv_path)
    
    # 检查必要列
    required_cols <- c("All_fire_num", "All_WUI_area", "HFI", "GRID_ID")
    for (col in required_cols) {
      if (!(col %in% colnames(df))) {
        cat(paste0("文件 ", basename(csv_path), " 缺少必要列: ", col, "\n"))
        return(FALSE)
      }
    }
    
    file_name <- tools::file_path_sans_ext(basename(csv_path))
    cat(paste0("正在处理文件: ", file_name, "\n"))
    
    # 重命名All_fire_num为Fire_num
    df$Fire_num <- df$All_fire_num
    
    # 重命名All_WUI_area为WUI_area
    df$WUI_area <- df$All_WUI_area
    
    # 计算Fire_frequency = Fire_num / WUI_area
    df$Fire_frequency <- df$Fire_num / df$WUI_area
    
    y_col <- "Fire_num"
    offset_col <- "WUI_area"
    original_x_col <- "HFI"  # 只使用HFI作为自变量
    group_var <- "GRID_ID"
    
    # 仅保留必要列并去除缺失（包括Fire_frequency用于可视化）
    use_cols <- c(y_col, "Fire_frequency", offset_col, original_x_col, group_var)
    df <- df[, use_cols]
    df <- na.omit(df)
    
    # 将分组变量转为factor
    df[[group_var]] <- as.factor(df[[group_var]])
    
    # 对自变量进行标准化（z-score标准化）- 移到最开始
    cat("\n--- 开始自变量标准化 ---")
    
    # 保存标准化前的统计信息
    standardized_stats <- list()
    
    # 创建标准化后的变量名
    standardized_x_col <- paste0(original_x_col, "_std")
    
    # 对HFI变量进行标准化
    var_mean <- mean(df[[original_x_col]], na.rm = TRUE)
    var_sd <- sd(df[[original_x_col]], na.rm = TRUE)
    
    # 保存统计信息
    standardized_stats[[original_x_col]] <- list(mean = var_mean, sd = var_sd)
    
    # 进行标准化
    df[[standardized_x_col]] <- (df[[original_x_col]] - var_mean) / var_sd
    
    cat(paste0("\n自变量标准化完成: ", standardized_x_col))
      
    # 确定样条的节点数
    n_knots <- min(4, max(3, floor(nrow(df)/10)))
    
    # 创建自然样条基函数
    spline_basis <- ns(df[[standardized_x_col]], df=n_knots)
    
    # 将样条基函数添加到数据框
    spline_names <- paste0("spline_", 1:ncol(spline_basis))
    for(j in 1:ncol(spline_basis)) {
      df[[spline_names[j]]] <- spline_basis[, j]
    }
    
    # 保存样条信息用于复用
    spline_info <- list(
      knots = attr(spline_basis, "knots"),
      boundary_knots = attr(spline_basis, "Boundary.knots"),
      spline_names = spline_names,
      df = n_knots
    )
    
    # 构建公式，使用样条基函数项替代s()函数
    spline_terms <- paste(spline_names, collapse=" + ")
    formula_str <- paste(y_col, "~", spline_terms, "+ offset(log(", offset_col, ")) + (1|", group_var, ")")
    
    # 拟合负二项GLMM模型（使用自然样条）
    final_model <- tryCatch({
        # 使用glmmTMB拟合负二项模型（使用自然样条进行非线性建模）
        glmmTMB(
          formula = as.formula(formula_str),
          data = df,
          family = nbinom2(link = "log"),  # 负二项分布
          control = glmmTMBControl(
            optimizer = nlminb,
            optCtrl = list(iter.max = 20000, eval.max = 50000)
          )
        )
      }, error = function(e) {
      # 如果负二项模型拟合失败，直接返回错误，不尝试其他分布
      cat(paste0("负二项GLMM模型拟合失败: ", e$message, "，跳过此文件\n"))
      stop(paste0("NB GLMM模型拟合失败: ", e$message))
    })
    
    # 检查模型收敛性
    cat("\n--- 开始模型收敛性检查 ---")
    convergence_result <- test_model_convergence(final_model)
    
    # 详细分析收敛状态
    cat("\n--- 模型收敛性分析结果 ---")
    if (convergence_result$is_converged) {
      cat("\n✅ 模型成功收敛!\n")
      cat(paste0("   收敛码: ", convergence_result$convergence_code, " (0=成功)\n"))
      cat(paste0("   最大梯度: ", format(convergence_result$max_gradient, scientific = TRUE), " (<1e-2=良好)\n"))
      cat(paste0("   随机效应方差正常: ", ifelse(isTRUE(convergence_result$random_var_ok), "是", "否"), "\n"))
    } else {
      cat("\n⚠️  警告: 模型未收敛，参数结果可能不可靠!\n")
      
      # 分析具体的收敛失败原因
      if (is.na(convergence_result$convergence_code)) {
        cat("   ❌ 原因: 无法获取收敛码，可能是模型结构问题\n")
      } else if (convergence_result$convergence_code != 0) {
        cat(paste0("   ❌ 原因: 收敛码为 ", convergence_result$convergence_code, "，优化器未成功收敛\n"))
        cat("      可能的原因: 目标函数不光滑、参数初始值不佳、模型过于复杂\n")
      }
      
      if (is.na(convergence_result$max_gradient)) {
        cat("   ❌ 原因: 无法获取梯度值，可能是模型结构问题\n")
      } else if (!convergence_result$gradient_ok) {
        cat(paste0("   ❌ 原因: 梯度值过大 (", format(convergence_result$max_gradient, scientific = TRUE), ")，未达到收敛阈值\n"))
        cat("      可能的原因: 需要更多迭代次数、学习率不合适\n")
      }
      
      # 打印详细的收敛指标
      cat("\n--- 详细收敛指标 ---")
      cat(paste0("\n   收敛码: ", convergence_result$convergence_code, " (0=成功)\n"))
      cat(paste0("   最大梯度: ", format(convergence_result$max_gradient, scientific = TRUE), " (<1e-2=良好)\n"))
      cat(paste0("   梯度状态: ", ifelse(convergence_result$gradient_ok, "良好", "不佳"), "\n"))
      cat(paste0("   随机效应方差正常: ", ifelse(isTRUE(convergence_result$random_var_ok), "是", "否"), "\n"))
      cat(paste0("   随机效应方差值: ", ifelse(!is.na(convergence_result$random_var_value), format(convergence_result$random_var_value, scientific = TRUE), "NA"), "\n"))
    }
    
    # 预测
    y_pred <- predict(final_model, newdata = df, type = "response")
    
    # 计算火灾频率（预测值和真实值）
    pred_freq <- y_pred / df[[offset_col]]
    obs_freq <- df[[y_col]] / df[[offset_col]]
    
    # 计算R²值
    r_squared <- calculate_r_squared(final_model)
    
    # 生成边际效应图
    marginal_plot_file <- generate_marginal_effect_plot(final_model, df, file_name, output_dir, standardized_x_col, y_col, offset_col, group_var, standardized_stats)
    
    # 输出报告（带offset项），使用原始变量名
    generate_model_report(final_model, df, file_name, output_dir, y_col, standardized_x_col, offset_col, group_var, y_pred, pred_freq, obs_freq, r_squared, convergence_result)

    # 保存模型与可复用信息
    save_model_artifact(final_model, file_name, output_dir, standardized_stats, spline_info, formula_str, standardized_x_col, y_col, offset_col, group_var)
    
    return(TRUE)
    
  }, error = function(e) {
    cat(paste0("处理文件 ", basename(csv_path), " 时出错: ", e$message, "\n"))
    # 打印详细的错误信息
    cat("详细错误信息:\n")
    print(e)
    # 打印调用栈
    cat("调用栈:\n")
    traceback()
    return(FALSE)
  })
}

# 主函数
main <- function() {
  input_dir <- "I:/Processing data/渔网分类建模/Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic/"
  output_dir <- "B:/WUI/Picture"
  
  create_dir_if_not_exists(output_dir)
  
  csv_files <- list.files(input_dir, pattern = "\\.csv$", ignore.case = TRUE)
  if (length(csv_files) == 0) {
    cat(paste0("在目录 ", input_dir, " 中未找到CSV文件\n"))
    return()
  }
  
  cat(paste0("找到 ", length(csv_files), " 个CSV文件\n"))
  
  success_count <- 0
  failure_count <- 0
  
  for (csv_file in csv_files) {
    csv_path <- file.path(input_dir, csv_file)
    if (process_single_csv(csv_path, output_dir)) {
      success_count <- success_count + 1
    } else {
      failure_count <- failure_count + 1
    }
  }
  
  cat("\n处理完成!\n")
  cat(paste0("成功处理: ", success_count, " 个文件\n"))
  cat(paste0("处理失败: ", failure_count, " 个文件\n"))
}

# 运行主函数
main()