#!/usr/bin/env Rscript

# ------------------------------------------------------------------------
# 程序: Fire_num_test.R
# 功能: 对因变量进行分布检验并给出模型建议，包括：
#       - 计算均值、方差、零值占比
#       - 根据检验结果建议适合的模型（泊松/负二项）
#       - 对 All_fire_area 给出是否适合 Gamma 回归的建议
#       - 将所有结果输出到CSV文件
# 输入:
#   1) I:/Processing data/渔网分类建模/Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic/
#      - 必须包含列：All_fire_num
#   2) I:/Processing data/渔网分类建模/Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic/
#      - 必须包含列：All_fire_area
# 输出: CSV文件，包含所有检验结果（按目标变量区分）
# ------------------------------------------------------------------------

# 加载必要的包（基础统计即可）

# 设置参数
input_dir_num <- "I:/Processing data/渔网分类建模/Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic/"
input_dir_area <- "I:/Processing data/渔网分类建模/Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic/"
output_file <- "B:/WUI/Picture/Fire_Distribution_Test_Results.csv"

create_dir_if_not_exists <- function(dir_path) {
  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE)
    cat(paste0("Created directory: ", dir_path, "\n"))
  }
}

compute_skewness <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 3) return(NA_real_)
  m <- mean(x)
  s <- sd(x)
  if (is.na(s) || s == 0) return(NA_real_)
  mean((x - m)^3) / (s^3)
}

suggest_count_model <- function(y_values) {
  y_values <- y_values[is.finite(y_values)]
  if (length(y_values) == 0) {
    return(list(suggestion = "No valid data", zero_ratio = NA_real_))
  }
  y_mean <- mean(y_values)
  y_var <- var(y_values)
  y_zero_ratio <- mean(y_values == 0)
  if (!is.finite(y_mean) || y_mean <= 0) {
    suggestion <- "Invalid mean, check data"
  } else if (abs(y_var - y_mean) / y_mean < 0.1) {
    suggestion <- "Variance ≈ Mean, suggest Poisson"
  } else if (y_var > y_mean) {
    suggestion <- "Variance > Mean, suggest Negative Binomial"
  } else {
    suggestion <- "Variance < Mean, suggest checking distribution"
  }
  list(suggestion = suggestion, zero_ratio = y_zero_ratio)
}

suggest_area_model <- function(y_values) {
  y_values <- y_values[is.finite(y_values)]
  if (length(y_values) == 0) {
    return(list(suggestion = "No valid data", gamma_ok = "NA", zero_ratio = NA_real_, skewness = NA_real_))
  }
  y_zero_ratio <- mean(y_values == 0)
  y_min <- min(y_values)
  skew <- compute_skewness(y_values)
  if (y_min < 0) {
    return(list(suggestion = "Negative values present, check data", gamma_ok = "No", zero_ratio = y_zero_ratio, skewness = skew))
  }
  if (y_zero_ratio > 0) {
    return(list(suggestion = "Zeros present, suggest Tweedie or hurdle (Gamma+zero)", gamma_ok = "No", zero_ratio = y_zero_ratio, skewness = skew))
  }
  if (!is.na(skew) && skew >= 1) {
    return(list(suggestion = "Right-skewed positive, suggest Gamma GLM or lognormal", gamma_ok = "Yes", zero_ratio = y_zero_ratio, skewness = skew))
  }
  if (!is.na(skew) && skew >= 0.5) {
    return(list(suggestion = "Mild skew, consider Gamma or log/Box-Cox Gaussian", gamma_ok = "Maybe", zero_ratio = y_zero_ratio, skewness = skew))
  }
  list(suggestion = "Low skew, suggest Gaussian", gamma_ok = "No", zero_ratio = y_zero_ratio, skewness = skew)
}

process_dir <- function(input_dir, target_col, target_type) {
  csv_files <- list.files(input_dir, pattern = "\\.csv$", ignore.case = TRUE)
  if (length(csv_files) == 0) {
    cat(paste0("No CSV files found in ", input_dir, "\n"))
    return(data.frame())
  }
  
  results <- data.frame(
    File_Title = character(),
    Target = character(),
    Sample_Size = integer(),
    Mean = numeric(),
    Variance = numeric(),
    Variance_Mean_Ratio = numeric(),
    Zero_Count = integer(),
    Zero_Ratio = numeric(),
    Skewness = numeric(),
    Model_Suggestion = character(),
    Gamma_Suitable = character(),
    stringsAsFactors = FALSE
  )
  
  for (csv_file in csv_files) {
    csv_path <- file.path(input_dir, csv_file)
    file_name <- tools::file_path_sans_ext(basename(csv_path))
    
    cat(paste0("\n正在处理文件: ", file_name, "\n"))
    
    df <- read.csv(csv_path)
    if (!(target_col %in% colnames(df))) {
      cat(paste0("Missing column ", target_col, " in ", file_name, "\n"))
      next
    }
    
    y_values <- df[[target_col]]
    y_values <- y_values[is.finite(y_values)]
    
    y_total_count <- length(y_values)
    y_mean <- mean(y_values, na.rm = TRUE)
    y_var <- var(y_values, na.rm = TRUE)
    y_var_mean_ratio <- ifelse(is.finite(y_mean) && y_mean != 0, y_var / y_mean, NA_real_)
    y_zero_count <- sum(y_values == 0, na.rm = TRUE)
    y_zero_ratio <- ifelse(y_total_count > 0, y_zero_count / y_total_count, NA_real_)
    y_skew <- compute_skewness(y_values)
    
    if (target_type == "count") {
      s <- suggest_count_model(y_values)
      model_suggestion <- s$suggestion
      gamma_ok <- "Not applicable"
    } else {
      s <- suggest_area_model(y_values)
      model_suggestion <- s$suggestion
      gamma_ok <- s$gamma_ok
    }
    
    result_row <- data.frame(
      File_Title = file_name,
      Target = target_col,
      Sample_Size = y_total_count,
      Mean = round(y_mean, 4),
      Variance = round(y_var, 4),
      Variance_Mean_Ratio = round(y_var_mean_ratio, 4),
      Zero_Count = y_zero_count,
      Zero_Ratio = round(y_zero_ratio, 4),
      Skewness = round(y_skew, 4),
      Model_Suggestion = model_suggestion,
      Gamma_Suitable = gamma_ok
    )
    
    results <- rbind(results, result_row)
    cat(paste0("完成处理文件: ", file_name, "\n"))
  }
  
  results
}

# 创建输出目录
output_dir <- dirname(output_file)
create_dir_if_not_exists(output_dir)

# 写入结果到CSV文件
cat("Writing CSV file...\n")
results_num <- process_dir(input_dir_num, "All_fire_num", "count")
results_area <- process_dir(input_dir_area, "All_fire_area", "area")
results <- rbind(results_num, results_area)

cat(paste0("Results data frame rows: ", nrow(results), "\n"))
cat(paste0("Results data frame columns: ", ncol(results), "\n"))

if (nrow(results) > 0) {
  write.table(results, output_file, sep = ",", row.names = FALSE,
              quote = TRUE, na = "NA", fileEncoding = "UTF-8")
  cat("CSV file writing completed!\n")
  cat(paste0("\nAll test results saved to: ", output_file, "\n"))
} else {
  cat("No results to write to CSV file!\n")
}

print("Test completed!\n")
