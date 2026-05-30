#!/usr/bin/env Rscript

# ------------------------------------------------------------------------
# 程序: GLMM_for_normalized_fire_area_空模型_logit连接.R
# 功能: 使用Ordbeta分布的广义线性混合模型(GLMM)对归一化火灾面积建立空模型，具体功能包括：
#       - 直接使用 Normalized_fire_area 作为响应变量
#       - 不包含 HFI，仅拟合空模型：Normalized_fire_area ~ 1 + (1|GRID_ID)
#       - 计算模型评估指标：边际R²、条件R²、MSE、RMSE、MAE
#       - 批量处理多份CSV数据文件
# 输入: CSV文件(路径: E:/data-hl/Origin data/渔网分类建模/Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic/)
#       - 必须包含列：Normalized_fire_area、GRID_ID
# 输出: 模型报告(TXT)与模型RDS，输出路径: E:/data-hl/Picture/
# ------------------------------------------------------------------------

library(glmmTMB)
library(performance)

test_model_convergence <- function(model, maxit = 1e4) {
  is_converged <- FALSE
  convergence_code <- NA_real_
  max_gradient <- NA_real_
  gradient_ok <- NA
  random_var_ok <- NA
  random_var_value <- NA_real_

  if (is.null(model) || !inherits(model, "glmmTMB")) {
    cat("模型对象无效（非glmmTMB类型或为NULL）\n")
    return(list(
      is_converged = is_converged,
      convergence_code = convergence_code,
      max_gradient = max_gradient,
      gradient_ok = gradient_ok,
      random_var_ok = random_var_ok,
      random_var_value = random_var_value
    ))
  }

  tryCatch({
    if (!is.null(model$fit$convergence)) {
      convergence_code <- model$fit$convergence
    }

    if (!is.null(model$sdr$gradient)) {
      grad <- model$sdr$gradient
      max_gradient <- max(abs(grad))
      gradient_ok <- max_gradient < 1e-2
    }

    tryCatch({
      varcorr_result <- VarCorr(model)
      if (!is.null(varcorr_result$cond) && length(varcorr_result$cond) > 0) {
        random_effects_names <- names(varcorr_result$cond)
        if (length(random_effects_names) > 0) {
          first_rand_effect <- random_effects_names[1]
          rand_var <- as.numeric(varcorr_result$cond[[first_rand_effect]])
          random_var_value <- rand_var
          random_var_ok <- rand_var > 1e-8
        }
      }
    }, error = function(e) {
      random_var_ok <- TRUE
    })

    if (!is.na(convergence_code) && !is.na(gradient_ok)) {
      is_converged <- (convergence_code == 0) && isTRUE(gradient_ok)
    }
  }, error = function(e) {
    cat(paste0("提取收敛信息失败: ", e$message, "\n"))
  })

  list(
    is_converged = is_converged,
    convergence_code = convergence_code,
    max_gradient = max_gradient,
    gradient_ok = gradient_ok,
    random_var_ok = random_var_ok,
    random_var_value = random_var_value
  )
}

calculate_r_squared <- function(model) {
  r2_default <- performance::r2_nakagawa(model)
  r2_cond <- performance::r2_nakagawa(model, model_component = "conditional", approximation = "lognormal")

  list(
    marginal = as.numeric(r2_default$R2_marginal),
    conditional = as.numeric(r2_default$R2_conditional),
    marginal_cond = as.numeric(r2_cond$R2_marginal),
    conditional_cond = as.numeric(r2_cond$R2_conditional),
    method = "performance::r2_nakagawa (default + conditional)"
  )
}

generate_model_report <- function(model, file_name, output_dir, pred_freq, obs_freq, r_squared, convergence_result) {
  report_file <- file.path(output_dir, paste0("GLMM_Null_Ordbeta_without_HFI_", file_name, "_report.txt"))

  mse_freq <- mean((obs_freq - pred_freq)^2)
  rmse_freq <- sqrt(mse_freq)
  mae_freq <- mean(abs(obs_freq - pred_freq))

  sink(report_file, type = "output")
  cat(paste0("\n", strrep("=", 70), "\n"))
  cat(paste0("Null Ordbeta-GLMM (logit link) Normalized Fire Area Model Report without HFI - ", file_name, "\n"))
  cat(paste0(strrep("=", 70), "\n\n"))

  cat("1. Model Convergence\n")
  cat(strrep("-", 40), "\n", sep = "")
  if (convergence_result$is_converged) {
    cat("模型成功收敛\n")
  } else {
    cat("警告: 模型未收敛，参数结果可能不可靠\n")
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
  cat(paste0("R2 (default) - Marginal: ", format(r_squared$marginal, digits = 6), "  Conditional: ", format(r_squared$conditional, digits = 6), "\n"))
  cat(paste0("R2 (conditional, lognormal) - Marginal: ", format(r_squared$marginal_cond, digits = 6), "  Conditional: ", format(r_squared$conditional_cond, digits = 6), "\n"))
  cat("\n")
  cat(paste0("MSE : ", format(mse_freq, digits = 6), "\n"))
  cat(paste0("RMSE: ", format(rmse_freq, digits = 6), "\n"))
  cat(paste0("MAE : ", format(mae_freq, digits = 6), "\n"))

  sink()
  cat(paste0("模型报告已保存: ", report_file, "\n"))
}

create_dir_if_not_exists <- function(dir_path) {
  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE)
    cat(paste0("创建目录: ", dir_path, "\n"))
  }
}

save_model_artifact <- function(model, file_name, output_dir, formula_str, y_col, offset_col, group_var) {
  model_file <- file.path(output_dir, paste0("GLMM_Null_Ordbeta_without_HFI_", file_name, "_model.rds"))
  saveRDS(list(
    model = model,
    standardized_stats = NULL,
    spline_info = NULL,
    formula = formula_str,
    x_col = NA_character_,
    y_col = y_col,
    offset_col = offset_col,
    group_var = group_var,
    model_type = "Ordbeta_null_model",
    response_transform = "direct_logit: Normalized_fire_area"
  ), model_file)
  cat(paste0("模型已保存: ", model_file, "\n"))
}

process_single_csv <- function(csv_path, output_dir) {
  tryCatch({
    df <- read.csv(csv_path)

    required_cols <- c("Normalized_fire_area", "GRID_ID")
    for (col in required_cols) {
      if (!(col %in% colnames(df))) {
        cat(paste0("文件 ", basename(csv_path), " 缺少必要列: ", col, "\n"))
        return(FALSE)
      }
    }

    file_name <- tools::file_path_sans_ext(basename(csv_path))
    cat(paste0("正在处理文件: ", file_name, "\n"))

    y_col <- "Normalized_fire_area"
    offset_col <- NULL
    group_var <- "GRID_ID"

    df <- df[is.finite(df[[y_col]]) & df[[y_col]] >= 0, ]
    use_cols <- c(y_col, group_var)
    df <- df[, use_cols]
    df <- na.omit(df)

    if (nrow(df) < 5) {
      cat(paste0("有效样本过少（去除缺失/非正值后 <5）: ", file_name, "\n"))
      return(FALSE)
    }

    df[[group_var]] <- as.factor(df[[group_var]])

    formula_str <- paste(y_col, "~ 1 + (1|", group_var, ")")

    final_model <- tryCatch({
      glmmTMB(
        formula = as.formula(formula_str),
        data = df,
        family = ordbeta(link = "logit"),
        control = glmmTMBControl(
          optimizer = nlminb,
          optCtrl = list(iter.max = 20000, eval.max = 50000)
        )
      )
    }, error = function(e) {
      cat(paste0("Ordbeta 空模型拟合失败: ", e$message, "，跳过此文件\n"))
      stop(paste0("Ordbeta 空模型拟合失败: ", e$message))
    })

    cat("\n--- 开始模型收敛性检查 ---")
    convergence_result <- test_model_convergence(final_model)

    cat("\n--- 模型收敛性分析结果 ---")
    if (convergence_result$is_converged) {
      cat("\n模型成功收敛\n")
      cat(paste0("   收敛码: ", convergence_result$convergence_code, " (0=成功)\n"))
      cat(paste0("   最大梯度: ", format(convergence_result$max_gradient, scientific = TRUE), " (<1e-2=良好)\n"))
      cat(paste0("   随机效应方差正常: ", ifelse(isTRUE(convergence_result$random_var_ok), "是", "否"), "\n"))
      cat(paste0("   随机效应方差值: ", ifelse(!is.na(convergence_result$random_var_value), format(convergence_result$random_var_value, scientific = TRUE), "NA"), "\n"))
    } else {
      cat("\n警告: 模型未收敛，参数结果可能不可靠\n")
    }

    y_pred <- predict(final_model, newdata = df, type = "response")
    pred_freq <- y_pred
    obs_freq <- df[[y_col]]
    r_squared <- calculate_r_squared(final_model)

    generate_model_report(final_model, file_name, output_dir, pred_freq, obs_freq, r_squared, convergence_result)
    save_model_artifact(final_model, file_name, output_dir, formula_str, y_col, offset_col, group_var)

    TRUE
  }, error = function(e) {
    cat(paste0("处理文件 ", basename(csv_path), " 时出错: ", e$message, "\n"))
    FALSE
  })
}

main <- function() {
  input_dir <- "I:/Processing data/渔网分类建模/Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic"
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

main()
