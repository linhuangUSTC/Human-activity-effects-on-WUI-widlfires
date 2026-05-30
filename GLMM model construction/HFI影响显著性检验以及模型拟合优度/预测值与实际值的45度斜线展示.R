#!/usr/bin/env Rscript

# ------------------------------------------------------------------------
# 程序: 预测值与实际值的45度斜线展示.R
# 功能:
#   - 读取两个模型目录中的 *.rds
#   - 从 RDS 文件名中提取分类ID（如 face_A_CRP）
#   - 匹配对应 CSV，读取 HFI 并进行预测
#   - 绘制预测值 vs 实际值的一比一图
#   - 在图中打印分类ID、Pearson R²、以及从 report.txt 中读取的 Conditional R²
#   - 输出到 B:/WUI/Picture
# ------------------------------------------------------------------------

library(glmmTMB)
library(ggplot2)
library(splines)

create_dir_if_not_exists <- function(dir_path) {
  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE)
    cat(paste0("创建目录: ", dir_path, "\n"))
  }
}

safe_read_model_bundle <- function(rds_path) {
  obj <- readRDS(rds_path)
  model <- obj
  if (is.list(obj) && !inherits(obj, "glmmTMB") && !is.null(obj$model)) {
    model <- obj$model
  }
  if (!inherits(model, "glmmTMB")) {
    stop("RDS does not contain a glmmTMB model.")
  }
  list(bundle = obj, model = model)
}

get_meta_value <- function(bundle, name, default = NULL) {
  if (is.list(bundle) && !is.null(bundle[[name]])) {
    return(bundle[[name]])
  }
  default
}

extract_class_id <- function(rds_path) {
  base <- tools::file_path_sans_ext(basename(rds_path))
  m <- regexpr("(face|mix)_[^_]+_[^_]+", base, perl = TRUE)
  if (m[1] != -1) {
    return(regmatches(base, m))
  }
  parts <- strsplit(base, "_")[[1]]
  if (length(parts) >= 3) {
    return(paste(parts[1:3], collapse = "_"))
  }
  base
}

extract_key_from_csv <- function(csv_path) {
  base <- tools::file_path_sans_ext(basename(csv_path))
  parts <- strsplit(base, "_")[[1]]
  if (length(parts) < 3) {
    return(NA_character_)
  }
  paste(parts[1:3], collapse = "_")
}

build_spline_basis <- function(x_std, spline_info) {
  if (is.null(spline_info)) {
    stop("Missing spline_info in model bundle.")
  }
  if (!is.null(spline_info$knots) && length(spline_info$knots) > 0) {
    if (!is.null(spline_info$boundary_knots) && length(spline_info$boundary_knots) > 0) {
      return(ns(x_std, knots = spline_info$knots, Boundary.knots = spline_info$boundary_knots))
    }
    return(ns(x_std, knots = spline_info$knots))
  }
  if (!is.null(spline_info$df)) {
    if (!is.null(spline_info$boundary_knots) && length(spline_info$boundary_knots) > 0) {
      return(ns(x_std, df = spline_info$df, Boundary.knots = spline_info$boundary_knots))
    }
    return(ns(x_std, df = spline_info$df))
  }
  stop("Spline info is incomplete (no knots or df).")
}

add_spline_columns <- function(newdata, x_std, spline_info) {
  basis <- build_spline_basis(x_std, spline_info)
  basis <- as.matrix(basis)
  spline_names <- spline_info$spline_names
  if (is.null(spline_names) || length(spline_names) != ncol(basis)) {
    spline_names <- paste0("spline_", seq_len(ncol(basis)))
  }
  for (j in seq_len(ncol(basis))) {
    newdata[[spline_names[j]]] <- basis[, j]
  }
  newdata
}

standardize_predictor <- function(hfi_raw, x_col, standardized_stats) {
  if (!grepl("_std$", x_col)) {
    return(hfi_raw)
  }
  if (is.null(standardized_stats) || is.null(standardized_stats$mean) ||
      is.null(standardized_stats$sd) || !is.finite(standardized_stats$sd) ||
      standardized_stats$sd == 0) {
    stop(paste0("Cannot standardize predictor for ", x_col, ": invalid standardized_stats."))
  }
  (hfi_raw - standardized_stats$mean) / standardized_stats$sd
}

safe_predict_full <- function(model, newdata) {
  tryCatch({
    predict(model, newdata = newdata, type = "response", allow.new.levels = TRUE)
  }, error = function(e) {
    predict(model, newdata = newdata, type = "response")
  })
}

derive_report_path <- function(rds_path) {
  sub("_model\\.rds$", "_report.txt", rds_path, perl = TRUE)
}

extract_conditional_r2 <- function(report_path) {
  if (!file.exists(report_path)) {
    return(NA_real_)
  }
  txt <- tryCatch(paste(readLines(report_path, warn = FALSE, encoding = "UTF-8"), collapse = "\n"),
                  error = function(e) NULL)
  if (is.null(txt) || !nzchar(txt)) {
    return(NA_real_)
  }
  m <- regexec("Conditional:\\s*([0-9eE.+-]+)", txt, perl = TRUE)
  hit <- regmatches(txt, m)[[1]]
  if (length(hit) >= 2) {
    val <- suppressWarnings(as.numeric(hit[2]))
    if (is.finite(val)) {
      return(val)
    }
  }
  NA_real_
}

compute_pearson_r2 <- function(obs, pred) {
  ok <- is.finite(obs) & is.finite(pred)
  obs <- obs[ok]
  pred <- pred[ok]
  if (length(obs) < 2) {
    return(NA_real_)
  }
  if (stats::sd(obs) == 0 || stats::sd(pred) == 0) {
    return(NA_real_)
  }
  r <- suppressWarnings(stats::cor(obs, pred, method = "pearson"))
  if (!is.finite(r)) {
    return(NA_real_)
  }
  r^2
}

format_r2_value <- function(x) {
  if (is.finite(x)) {
    return(format(x, digits = 4))
  }
  "NA"
}

build_plot_limits <- function(obs_value, pred_value) {
  vals <- c(obs_value, pred_value)
  vals <- vals[is.finite(vals)]
  if (length(vals) == 0) {
    return(c(0, 1))
  }
  min_val <- min(vals)
  max_val <- max(vals)
  if (!is.finite(min_val) || !is.finite(max_val)) {
    return(c(0, 1))
  }
  if (min_val == max_val) {
    delta <- if (min_val == 0) 1 else abs(min_val) * 0.1
    return(c(min_val - delta, max_val + delta))
  }
  c(min_val, max_val)
}

make_one_to_one_plot <- function(plot_df, class_id, out_path, observed_label, predicted_label, pearson_r2, conditional_r2) {
  limits <- build_plot_limits(plot_df$obs_value, plot_df$pred_value)
  span <- limits[2] - limits[1]
  if (!is.finite(span) || span <= 0) {
    span <- 1
  }
  class_x <- mean(limits)
  class_y <- limits[2] - span * 0.04
  metric_x <- limits[1] + span * 0.03
  metric_y <- limits[2] - span * 0.12
  metric_label <- paste0(
    "Pearson R²: ", format_r2_value(pearson_r2), "\n",
    "Conditional R²: ", format_r2_value(conditional_r2)
  )

  p <- ggplot(plot_df, aes(x = obs_value, y = pred_value)) +
    geom_point(alpha = 0.5, color = "#2C7FB8", size = 1.4) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red3", linewidth = 0.8) +
    annotate(
      "text",
      x = class_x,
      y = class_y,
      label = class_id,
      size = 5,
      fontface = "bold",
      hjust = 0.5,
      vjust = 1
    ) +
    annotate(
      "text",
      x = metric_x,
      y = metric_y,
      label = metric_label,
      size = 4.2,
      hjust = 0,
      vjust = 1
    ) +
    xlab(observed_label) +
    ylab(predicted_label) +
    theme_bw() +
    theme(panel.grid = element_line(colour = "gray90")) +
    coord_fixed(xlim = limits, ylim = limits)

  ggsave(out_path, p, width = 6, height = 6, dpi = 600)
}

resolve_hfi_stats <- function(standardized_stats_all) {
  if (is.null(standardized_stats_all)) {
    return(NULL)
  }
  if (!is.null(standardized_stats_all$HFI)) {
    return(standardized_stats_all$HFI)
  }
  if (is.list(standardized_stats_all) && length(standardized_stats_all) == 1) {
    return(standardized_stats_all[[1]])
  }
  NULL
}

prepare_response_column <- function(df, config) {
  if (config$response_col %in% colnames(df)) {
    return(df)
  }
  if (!is.null(config$response_builder)) {
    df[[config$response_col]] <- config$response_builder(df)
    return(df)
  }
  stop(paste0("CSV缺少响应列: ", config$response_col))
}

prepare_offset_column <- function(df, offset_col) {
  if (is.null(offset_col) || offset_col == "") {
    return(df)
  }
  if (offset_col %in% colnames(df)) {
    return(df)
  }
  if ("All_WUI_area" %in% colnames(df)) {
    df[[offset_col]] <- df[["All_WUI_area"]]
    return(df)
  }
  stop(paste0("CSV缺少offset列: ", offset_col))
}

prepare_group_column <- function(df, group_var, model) {
  if (!(group_var %in% colnames(df))) {
    if ("GRID_ID" %in% colnames(df)) {
      df[[group_var]] <- df[["GRID_ID"]]
    } else {
      stop(paste0("CSV缺少分组列: ", group_var))
    }
  }

  current_values <- as.character(df[[group_var]])
  mf <- tryCatch(model.frame(model), error = function(e) NULL)
  if (!is.null(mf) && group_var %in% colnames(mf)) {
    model_levels <- levels(as.factor(mf[[group_var]]))
    df[[group_var]] <- factor(current_values, levels = unique(c(model_levels, current_values)))
  } else {
    df[[group_var]] <- factor(current_values)
  }
  df
}

process_single_rds <- function(rds_path, csv_map, output_dir, config) {
  class_id <- extract_class_id(rds_path)
  cat(paste0("正在处理模型: ", basename(rds_path), " | 分类ID: ", class_id, " | 类型: ", config$name, "\n"))

  csv_candidates <- csv_map[[class_id]]
  if (is.null(csv_candidates) || length(csv_candidates) == 0) {
    cat(paste0("未找到匹配的CSV: ", class_id, "\n"))
    return(FALSE)
  }
  csv_path <- csv_candidates[1]
  if (length(csv_candidates) > 1) {
    cat(paste0("匹配到多个CSV，使用第一个: ", basename(csv_path), "\n"))
  }

  rb <- tryCatch(safe_read_model_bundle(rds_path), error = function(e) {
    cat(paste0("读取RDS失败: ", e$message, "\n"))
    NULL
  })
  if (is.null(rb)) {
    return(FALSE)
  }

  bundle <- rb$bundle
  model <- rb$model
  x_col <- get_meta_value(bundle, "x_col", config$default_x_col)
  offset_col <- get_meta_value(bundle, "offset_col", config$default_offset_col)
  group_var <- get_meta_value(bundle, "group_var", config$default_group_var)
  standardized_stats_all <- get_meta_value(bundle, "standardized_stats", NULL)
  spline_info <- get_meta_value(bundle, "spline_info", NULL)

  df <- read.csv(csv_path)
  if (!("HFI" %in% colnames(df))) {
    cat(paste0("CSV缺少HFI列: ", basename(csv_path), "\n"))
    return(FALSE)
  }

  df <- tryCatch({
    df <- prepare_response_column(df, config)
    df <- prepare_offset_column(df, offset_col)
    df <- prepare_group_column(df, group_var, model)
    df
  }, error = function(e) {
    cat(paste0("预处理CSV失败: ", e$message, "\n"))
    NULL
  })
  if (is.null(df)) {
    return(FALSE)
  }

  valid <- is.finite(df[["HFI"]]) &
    is.finite(df[[config$response_col]]) &
    !is.na(df[[group_var]])
  if (!is.null(offset_col) && offset_col != "") {
    valid <- valid & is.finite(df[[offset_col]]) & df[[offset_col]] > 0
  }
  df <- df[valid, , drop = FALSE]
  if (nrow(df) == 0) {
    cat(paste0("无有效数据可用于预测: ", basename(csv_path), "\n"))
    return(FALSE)
  }

  hfi_stats <- resolve_hfi_stats(standardized_stats_all)
  x_input <- tryCatch({
    standardize_predictor(df[["HFI"]], x_col, hfi_stats)
  }, error = function(e) {
    cat(paste0("HFI标准化失败: ", e$message, "\n"))
    NULL
  })
  if (is.null(x_input)) {
    return(FALSE)
  }

  newdata <- data.frame(row_id = seq_len(nrow(df)))
  newdata[[x_col]] <- x_input
  if (!is.null(offset_col) && offset_col != "") {
    newdata[[offset_col]] <- df[[offset_col]]
  }
  newdata[[group_var]] <- df[[group_var]]

  if (!is.null(spline_info)) {
    newdata <- tryCatch({
      add_spline_columns(newdata, x_input, spline_info)
    }, error = function(e) {
      cat(paste0("构建样条基函数失败: ", e$message, "\n"))
      NULL
    })
    if (is.null(newdata)) {
      return(FALSE)
    }
  }
  newdata$row_id <- NULL

  y_pred <- tryCatch({
    safe_predict_full(model, newdata)
  }, error = function(e) {
    cat(paste0("预测失败: ", e$message, "\n"))
    NULL
  })
  if (is.null(y_pred)) {
    return(FALSE)
  }

  pred_value <- as.numeric(y_pred)
  if (isTRUE(config$divide_prediction_by_offset)) {
    pred_value <- pred_value / df[[offset_col]]
  }
  obs_value <- df[[config$response_col]]
  keep <- is.finite(pred_value) & is.finite(obs_value)
  if (!any(keep)) {
    cat(paste0("预测结果无有效值: ", basename(csv_path), "\n"))
    return(FALSE)
  }

  pearson_r2 <- compute_pearson_r2(obs_value[keep], pred_value[keep])
  conditional_r2 <- extract_conditional_r2(derive_report_path(rds_path))
  plot_df <- data.frame(
    obs_value = obs_value[keep],
    pred_value = pred_value[keep]
  )
  out_path <- file.path(output_dir, paste0(config$output_prefix, class_id, ".png"))
  make_one_to_one_plot(
    plot_df = plot_df,
    class_id = class_id,
    out_path = out_path,
    observed_label = config$observed_label,
    predicted_label = config$predicted_label,
    pearson_r2 = pearson_r2,
    conditional_r2 = conditional_r2
  )
  cat(paste0("图片已保存: ", out_path, "\n"))
  TRUE
}

main <- function() {
  output_dir <- "B:/WUI/Picture"
  create_dir_if_not_exists(output_dir)

  configs <- list(
    list(
      name = "NB_Fire_frequency",
      rds_dir = "B:/WUI/Picture and statistic results/model construction/GLMM NB 自然样条曲线 Fire num",
      csv_dir = "I:/Processing data/渔网分类建模/Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic",
      response_col = "Fire_frequency",
      response_builder = function(df) {
        if (all(c("All_fire_num", "All_WUI_area") %in% colnames(df))) {
          return(df$All_fire_num / df$All_WUI_area)
        }
        stop("无法由 All_fire_num 和 All_WUI_area 生成 Fire_frequency")
      },
      default_x_col = "HFI_std",
      default_offset_col = "WUI_area",
      default_group_var = "GRID_ID",
      divide_prediction_by_offset = TRUE,
      observed_label = "Observed Fire_frequency",
      predicted_label = "Predicted Fire_frequency",
      output_prefix = "Predicted_vs_Observed_Fire_frequency_"
    ),
    list(
      name = "Ordbeta_Normalized_fire_area",
      rds_dir = "B:/WUI/Picture and statistic results/model construction/GLMM Ordbeta 自然样条 Normalized fire area",
      csv_dir = "I:/Processing data/渔网分类建模/Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic",
      response_col = "Normalized_fire_area",
      response_builder = NULL,
      default_x_col = "HFI_std",
      default_offset_col = NULL,
      default_group_var = "GRID_ID",
      divide_prediction_by_offset = FALSE,
      observed_label = "Observed Normalized_fire_area",
      predicted_label = "Predicted Normalized_fire_area",
      output_prefix = "Predicted_vs_Observed_Normalized_fire_area_"
    )
  )

  total_success <- 0
  total_failure <- 0

  for (config in configs) {
    rds_files <- list.files(config$rds_dir, pattern = "\\.rds$", full.names = TRUE)
    if (length(rds_files) == 0) {
      cat(paste0("未在目录中找到RDS文件: ", config$rds_dir, "\n"))
      total_failure <- total_failure + 1
      next
    }

    csv_files <- list.files(config$csv_dir, pattern = "\\.csv$", full.names = TRUE)
    if (length(csv_files) == 0) {
      cat(paste0("未在目录中找到CSV文件: ", config$csv_dir, "\n"))
      total_failure <- total_failure + 1
      next
    }

    csv_keys <- sapply(csv_files, extract_key_from_csv)
    valid_csv <- !is.na(csv_keys) & csv_keys != ""
    csv_map <- split(csv_files[valid_csv], csv_keys[valid_csv])

    for (rds_path in rds_files) {
      ok <- tryCatch({
        process_single_rds(rds_path, csv_map, output_dir, config)
      }, error = function(e) {
        cat(paste0("处理失败: ", basename(rds_path), " | ", e$message, "\n"))
        FALSE
      })
      if (isTRUE(ok)) {
        total_success <- total_success + 1
      } else {
        total_failure <- total_failure + 1
      }
    }
  }

  cat("\n处理完成!\n")
  cat(paste0("成功处理: ", total_success, " 个模型\n"))
  cat(paste0("处理失败: ", total_failure, " 个模型\n"))
}

main()
