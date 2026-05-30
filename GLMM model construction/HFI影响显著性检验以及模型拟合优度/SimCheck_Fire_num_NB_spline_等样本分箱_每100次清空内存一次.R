#!/usr/bin/env Rscript

# ------------------------------------------------------------------------
# 程序: SimCheck_Fire_num_NB_spline_等样本分箱.R
# 功能: 使用已拟合的GLMM模型进行多次模拟，比较模拟分布与真实数据分布特征，
#       评估模型能否生成与真实 Fire frequency 数据相似的整体分布（Fire num / WUI area）。
# 模型来源: B:/WUI/Picture and statistic results/model construction/GLMM NB 自然样条曲线 Fire num/*.rds
# 数据来源: I:/Processing data/渔网分类建模/Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic/*.csv
# 匹配方式: 根据CSV文件名中的前三段（如 face_A_WET）匹配RDS文件名
# 输出: B:/WUI/Picture/SimCheck_Fire_num_NB_spline.csv
#       B:/WUI/Picture/PPC_HFI_Fire_frequency_***.png
# ------------------------------------------------------------------------

library(glmmTMB)
library(ggplot2)

create_dir_if_not_exists <- function(dir_path) {
  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE)
    cat(paste0("创建目录: ", dir_path, "\n"))
  }
}

write_csv_utf8_bom <- function(df, path) {
  tmp <- tempfile(fileext = ".csv")
  write.csv(df, tmp, row.names = FALSE, fileEncoding = "UTF-8")
  con_out <- file(path, open = "wb")
  on.exit(close(con_out), add = TRUE)
  writeBin(as.raw(c(0xEF, 0xBB, 0xBF)), con_out)
  con_in <- file(tmp, open = "rb")
  on.exit({
    close(con_in)
    unlink(tmp)
  }, add = TRUE)
  repeat {
    buf <- readBin(con_in, "raw", n = 65536)
    if (length(buf) == 0) break
    writeBin(buf, con_out)
  }
}

extract_key_from_csv <- function(csv_path) {
  base <- tools::file_path_sans_ext(basename(csv_path))
  parts <- strsplit(base, "_")[[1]]
  if (length(parts) < 3) return(NA_character_)
  paste(parts[1:3], collapse = "_")
}

find_key_in_rds <- function(rds_base, keys) {
  hits <- keys[sapply(keys, function(k) grepl(k, rds_base, fixed = TRUE))]
  if (length(hits) == 0) return(list(key = NA_character_, note = "no_match"))
  if (length(hits) > 1) return(list(key = hits[1], note = paste0("multiple_matches:", length(hits))))
  list(key = hits[1], note = "ok")
}

calc_stats <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) {
    return(list(
      zero_ratio = NA_real_,
      mean = NA_real_,
      variance = NA_real_,
      q95 = NA_real_,
      q99 = NA_real_,
      max = NA_real_,
      tail_index = NA_real_
    ))
  }
  q95 <- as.numeric(stats::quantile(x, 0.95, names = FALSE, na.rm = TRUE))
  q99 <- as.numeric(stats::quantile(x, 0.99, names = FALSE, na.rm = TRUE))
  tail_index <- ifelse(is.finite(q95) && q95 > 0, q99 / q95, NA_real_)
  list(
    zero_ratio = mean(x == 0, na.rm = TRUE),
    mean = mean(x, na.rm = TRUE),
    variance = stats::var(x, na.rm = TRUE),
    q95 = q95,
    q99 = q99,
    max = max(x, na.rm = TRUE),
    tail_index = tail_index
  )
}

summarize_sim <- function(sim_values, obs_value) {
  sim_values <- sim_values[is.finite(sim_values)]
  if (length(sim_values) == 0) {
    return(list(mean = NA_real_, q025 = NA_real_, q975 = NA_real_, pct = NA_real_, in95 = NA))
  }
  q025 <- as.numeric(stats::quantile(sim_values, 0.025, names = FALSE, na.rm = TRUE))
  q975 <- as.numeric(stats::quantile(sim_values, 0.975, names = FALSE, na.rm = TRUE))
  pct <- mean(sim_values <= obs_value, na.rm = TRUE)
  list(
    mean = mean(sim_values, na.rm = TRUE),
    q025 = q025,
    q975 = q975,
    pct = pct,
    in95 = !is.na(obs_value) && obs_value >= q025 && obs_value <= q975
  )
}

split_nsim_into_chunks <- function(nsim, chunk_size) {
  nsim <- as.integer(nsim)
  chunk_size <- as.integer(chunk_size)
  if (!is.finite(nsim) || nsim <= 0) return(integer(0))
  if (!is.finite(chunk_size) || chunk_size <= 0 || chunk_size >= nsim) return(nsim)
  n_full <- nsim %/% chunk_size
  remainder <- nsim %% chunk_size
  chunks <- rep(chunk_size, n_full)
  if (remainder > 0) {
    chunks <- c(chunks, remainder)
  }
  chunks
}

build_hfi_bins <- function(hfi, n_bins = 20) {
  hfi <- hfi[is.finite(hfi)]
  if (length(hfi) == 0) return(NULL)
  probs <- seq(0, 1, length.out = n_bins + 1)
  brks <- unique(stats::quantile(hfi, probs, na.rm = TRUE, names = FALSE))
  if (length(brks) < 3) return(NULL)
  list(
    breaks = brks,
    centers = (brks[-1] + brks[-length(brks)]) / 2
  )
}

prepare_ppc_hfi <- function(hfi_raw, y_obs, n_bins = 20) {
  valid <- is.finite(hfi_raw) & is.finite(y_obs)
  if (!any(valid)) return(NULL)
  
  hfi_raw <- hfi_raw[valid]
  y_obs <- y_obs[valid]
  
  bins <- build_hfi_bins(hfi_raw, n_bins = n_bins)
  if (is.null(bins)) return(NULL)
  
  bin_id <- cut(hfi_raw, breaks = bins$breaks, include.lowest = TRUE, labels = FALSE)
  n_bins_use <- length(bins$centers)
  
  obs_bin_means <- rep(NA_real_, n_bins_use)
  obs_bin_se <- rep(NA_real_, n_bins_use)
  for (b in seq_len(n_bins_use)) {
    idx <- which(bin_id == b)
    if (length(idx) > 0) {
      obs_bin_means[b] <- mean(y_obs[idx], na.rm = TRUE)
      obs_bin_se[b] <- stats::sd(y_obs[idx], na.rm = TRUE) / sqrt(length(idx))
    }
  }
  
  list(
    valid = valid,
    bins = bins,
    bin_id = bin_id,
    n_bins_use = n_bins_use,
    obs_bin_means = obs_bin_means,
    obs_bin_se = obs_bin_se
  )
}

calc_sim_bin_means <- function(sim_mat, ppc_info) {
  trimmed_mean_5pct <- function(x) {
    x <- x[is.finite(x)]
    if (length(x) == 0) return(NA_real_)
    mean(x, trim = 0.05, na.rm = TRUE)
  }

  sim_mat <- sim_mat[ppc_info$valid, , drop = FALSE]
  sim_bin_means <- matrix(NA_real_, nrow = ppc_info$n_bins_use, ncol = ncol(sim_mat))
  for (b in seq_len(ppc_info$n_bins_use)) {
    idx <- which(ppc_info$bin_id == b)
    if (length(idx) > 0) {
      sim_bin_means[b, ] <- apply(
        sim_mat[idx, , drop = FALSE],
        2,
        trimmed_mean_5pct
      )
    }
  }
  sim_bin_means
}

plot_ppc_hfi <- function(ppc_info, sim_bin_means, out_path) {
  if (is.null(ppc_info) || is.null(sim_bin_means) || ncol(sim_bin_means) == 0) return(FALSE)
  
  median_line <- apply(sim_bin_means, 1, stats::median, na.rm = TRUE)
  
  smooth_predict <- function(x, y, x_new, spar = 0.8) {
    ok <- is.finite(x) & is.finite(y)
    if (sum(ok) < 4) {
      return(stats::approx(x[ok], y[ok], xout = x_new, rule = 2)$y)
    }
    fit <- tryCatch(stats::smooth.spline(x[ok], y[ok], spar = spar), error = function(e) NULL)
    if (is.null(fit)) {
      return(stats::approx(x[ok], y[ok], xout = x_new, rule = 2)$y)
    }
    as.numeric(stats::predict(fit, x = x_new)$y)
  }
  
  # 仅保留中间90%的模拟曲线（按整体均值排序，剔除5%和95%以外）
  sim_curve_mean <- colMeans(sim_bin_means, na.rm = TRUE)
  keep_low <- stats::quantile(sim_curve_mean, 0.05, na.rm = TRUE)
  keep_high <- stats::quantile(sim_curve_mean, 0.95, na.rm = TRUE)
  keep_idx <- which(sim_curve_mean >= keep_low & sim_curve_mean <= keep_high)
  if (length(keep_idx) == 0) {
    keep_idx <- seq_len(ncol(sim_bin_means))
  }
  sim_keep <- sim_bin_means[, keep_idx, drop = FALSE]
  
  smooth_x <- seq(min(ppc_info$bins$centers), max(ppc_info$bins$centers), length.out = 200)
  sim_smooth_list <- lapply(seq_len(ncol(sim_keep)), function(j) {
    data.frame(
      HFI = smooth_x,
      Fire_frequency = smooth_predict(ppc_info$bins$centers, sim_keep[, j], smooth_x),
      sim = j
    )
  })
  plot_df <- do.call(rbind, sim_smooth_list)
  obs_df <- data.frame(
    HFI = ppc_info$bins$centers,
    Fire_frequency = ppc_info$obs_bin_means
  )
  obs_err_df <- data.frame(
    HFI = ppc_info$bins$centers,
    Fire_frequency = ppc_info$obs_bin_means,
    Lower = ppc_info$obs_bin_means - ppc_info$obs_bin_se,
    Upper = ppc_info$obs_bin_means + ppc_info$obs_bin_se
  )
  
  med_smooth_df <- data.frame(
    HFI = smooth_x,
    Fire_frequency = smooth_predict(ppc_info$bins$centers, median_line, smooth_x)
  )
  obs_smooth_df <- data.frame(
    HFI = smooth_x,
    Fire_frequency = smooth_predict(ppc_info$bins$centers, ppc_info$obs_bin_means, smooth_x)
  )
  
  p <- ggplot(plot_df, aes(x = HFI, y = Fire_frequency, group = sim)) +
    geom_line(color = "grey80", alpha = 0.25, linewidth = 1.0) +
    geom_errorbar(data = obs_err_df, aes(x = HFI, y = Fire_frequency, ymin = Lower, ymax = Upper), color = "red3", width = (max(ppc_info$bins$centers) - min(ppc_info$bins$centers)) * 0.05, linewidth = 0.5, inherit.aes = FALSE) +
    geom_point(data = obs_df, aes(x = HFI, y = Fire_frequency), color = "red3", size = 2, inherit.aes = FALSE) +
    geom_line(data = obs_smooth_df, aes(x = HFI, y = Fire_frequency), color = "red3", linewidth = 0.9, inherit.aes = FALSE) +
    geom_line(data = med_smooth_df, aes(x = HFI, y = Fire_frequency), color = "black", linewidth = 0.9, inherit.aes = FALSE) +
    xlab("HFI") +
    ylab("Fire frequency per km^2") +
    theme_bw() +
    theme(panel.grid = element_line(colour = "gray90"))
  
  ggsave(out_path, p, width = 6, height = 4, dpi = 600)
  TRUE
}

# 配置
rds_dir <- "B:/WUI/Picture and statistic results/model construction/GLMM Tweedie 自然样条 Fire frequency"
csv_dir <- "I:/Processing data/渔网分类建模/Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic"
output_dir <- "B:/WUI/Picture"
output_file <- file.path(output_dir, "SimCheck_Fire_num_NB_spline.csv")

nsim <- 1000
sim_chunk_size <- 100
set.seed(123)

create_dir_if_not_exists(output_dir)

# 加载CSV映射
csv_files <- list.files(csv_dir, pattern = "\\.csv$", full.names = TRUE)
if (length(csv_files) == 0) {
  stop(paste0("No CSV files found in ", csv_dir))
}

csv_keys <- sapply(csv_files, extract_key_from_csv)
valid_idx <- !is.na(csv_keys) & csv_keys != ""
csv_files <- csv_files[valid_idx]
csv_keys <- csv_keys[valid_idx]

# key -> csv path (若重复取第一个，并备注)
key_to_csv <- split(csv_files, csv_keys)
key_list <- names(key_to_csv)

# 读取RDS文件
rds_files <- list.files(rds_dir, pattern = "\\.rds$", full.names = TRUE)
if (length(rds_files) == 0) {
  stop(paste0("No RDS files found in ", rds_dir))
}

results <- data.frame(
  分类ID = character(),
  rds_file = character(),
  csv_file = character(),
  n_obs = integer(),
  nsim = integer(),
  zero_ratio_obs = numeric(),
  zero_ratio_sim_mean = numeric(),
  zero_ratio_sim_q025 = numeric(),
  zero_ratio_sim_q975 = numeric(),
  zero_ratio_obs_pct = numeric(),
  zero_ratio_in95 = logical(),
  mean_obs = numeric(),
  mean_sim_mean = numeric(),
  mean_sim_q025 = numeric(),
  mean_sim_q975 = numeric(),
  mean_obs_pct = numeric(),
  mean_in95 = logical(),
  var_obs = numeric(),
  var_sim_mean = numeric(),
  var_sim_q025 = numeric(),
  var_sim_q975 = numeric(),
  var_obs_pct = numeric(),
  var_in95 = logical(),
  q95_obs = numeric(),
  q95_sim_mean = numeric(),
  q95_sim_q025 = numeric(),
  q95_sim_q975 = numeric(),
  q95_obs_pct = numeric(),
  q95_in95 = logical(),
  q99_obs = numeric(),
  q99_sim_mean = numeric(),
  q99_sim_q025 = numeric(),
  q99_sim_q975 = numeric(),
  q99_obs_pct = numeric(),
  q99_in95 = logical(),
  max_obs = numeric(),
  max_sim_mean = numeric(),
  max_sim_q025 = numeric(),
  max_sim_q975 = numeric(),
  max_obs_pct = numeric(),
  max_in95 = logical(),
  tail_index_obs = numeric(),
  tail_index_sim_mean = numeric(),
  tail_index_sim_q025 = numeric(),
  tail_index_sim_q975 = numeric(),
  tail_index_obs_pct = numeric(),
  tail_index_in95 = logical(),
  note = character(),
  stringsAsFactors = FALSE
)

cat(paste0("开始处理RDS数量: ", length(rds_files), "\n"))
total_rds <- length(rds_files)

for (i in seq_along(rds_files)) {
  rds_path <- rds_files[i]
  note_vec <- character()
  rds_base <- basename(rds_path)
  match_info <- find_key_in_rds(rds_base, key_list)
  key <- match_info$key
  if (match_info$note != "ok") note_vec <- c(note_vec, paste0("key_", match_info$note))
  if (is.na(key)) {
    results <- rbind(results, data.frame(
      分类ID = "",
      rds_file = rds_path,
      csv_file = "",
      n_obs = NA_integer_,
      nsim = nsim,
      zero_ratio_obs = NA_real_,
      zero_ratio_sim_mean = NA_real_,
      zero_ratio_sim_q025 = NA_real_,
      zero_ratio_sim_q975 = NA_real_,
      zero_ratio_obs_pct = NA_real_,
      zero_ratio_in95 = NA,
      mean_obs = NA_real_,
      mean_sim_mean = NA_real_,
      mean_sim_q025 = NA_real_,
      mean_sim_q975 = NA_real_,
      mean_obs_pct = NA_real_,
      mean_in95 = NA,
      var_obs = NA_real_,
      var_sim_mean = NA_real_,
      var_sim_q025 = NA_real_,
      var_sim_q975 = NA_real_,
      var_obs_pct = NA_real_,
      var_in95 = NA,
      q95_obs = NA_real_,
      q95_sim_mean = NA_real_,
      q95_sim_q025 = NA_real_,
      q95_sim_q975 = NA_real_,
      q95_obs_pct = NA_real_,
      q95_in95 = NA,
      q99_obs = NA_real_,
      q99_sim_mean = NA_real_,
      q99_sim_q025 = NA_real_,
      q99_sim_q975 = NA_real_,
      q99_obs_pct = NA_real_,
      q99_in95 = NA,
      max_obs = NA_real_,
      max_sim_mean = NA_real_,
      max_sim_q025 = NA_real_,
      max_sim_q975 = NA_real_,
      max_obs_pct = NA_real_,
      max_in95 = NA,
      tail_index_obs = NA_real_,
      tail_index_sim_mean = NA_real_,
      tail_index_sim_q025 = NA_real_,
      tail_index_sim_q975 = NA_real_,
      tail_index_obs_pct = NA_real_,
      tail_index_in95 = NA,
      note = paste(note_vec, collapse = ";"),
      stringsAsFactors = FALSE
    ))
    cat(paste0("[进度 ", i, "/", total_rds, "] 已处理: ", rds_base, " | note: ", ifelse(length(note_vec) == 0, "ok", paste(note_vec, collapse = ";")), "\n"))
    next
  }
  
  csv_candidates <- key_to_csv[[key]]
  csv_path <- csv_candidates[1]
  if (length(csv_candidates) > 1) note_vec <- c(note_vec, paste0("csv_multiple:", length(csv_candidates)))
  
  # 读取模型
  model_bundle <- readRDS(rds_path)
  model_obj <- model_bundle
  if (is.list(model_bundle) && !inherits(model_bundle, "glmmTMB") && !is.null(model_bundle$model)) {
    model_obj <- model_bundle$model
  }
  if (!inherits(model_obj, "glmmTMB")) {
    note_vec <- c(note_vec, "model_not_glmmTMB")
    cat(paste0("[进度 ", i, "/", total_rds, "] 已处理: ", rds_base, " | note: ", paste(note_vec, collapse = ";"), "\n"))
    next
  }
  
  mf <- tryCatch({
    model.frame(model_obj)
  }, error = function(e) {
    note_vec <<- c(note_vec, paste0("model_frame_error:", conditionMessage(e)))
    NULL
  })
  
  # 读取CSV
  df_raw <- read.csv(csv_path)
  required_cols <- c("Fire_frequency", "HFI")
  df_required <- NULL
  if (!all(required_cols %in% colnames(df_raw))) {
    note_vec <- c(note_vec, "csv_missing_cols")
  } else {
    df_required <- df_raw[, required_cols, drop = FALSE]
  }
  
  # 若没有model.frame，才尝试用CSV直接得到观测响应
  df_fallback <- df_required
  if (!is.null(df_fallback) && is.null(mf)) {
    df_fallback <- df_fallback[
      is.finite(df_fallback[["Fire_frequency"]]) &
      is.finite(df_fallback[["HFI"]]),
      ,
      drop = FALSE
    ]
  }
  
  df_aligned <- NULL
  if (!is.null(df_required) && !is.null(mf) && all(rownames(mf) %in% rownames(df_required))) {
    df_aligned <- df_required[rownames(mf), , drop = FALSE]
  } else if (!is.null(df_required) && !is.null(mf) && nrow(df_required) == nrow(mf)) {
    df_aligned <- df_required
    note_vec <- c(note_vec, "csv_aligned_by_position")
  }
  
  # 真实响应值：优先使用CSV中的Fire_frequency列
  y_obs <- NULL
  if (!is.null(df_aligned)) {
    y_obs <- df_aligned[["Fire_frequency"]]
  }
  if (is.null(y_obs)) {
    y_obs <- tryCatch({
      if (!is.null(mf)) mf[[1]] else NULL
    }, error = function(e) NULL)
  }
  if (is.null(y_obs)) {
    if (!is.null(df_fallback) && nrow(df_fallback) > 0) {
      y_obs <- df_fallback[["Fire_frequency"]]
    } else {
      y_obs <- numeric(0)
    }
  }
  
  obs_stats <- calc_stats(y_obs)
  n_obs <- length(y_obs)
  
  ppc_status <- "not_ready"
  ppc_info <- NULL
  if (!is.null(mf) && nrow(mf) == n_obs) {
    hfi_plot <- NULL
    
    if ("HFI" %in% colnames(mf)) {
      hfi_plot <- mf[["HFI"]]
    } else if (is.list(model_bundle) &&
               !is.null(model_bundle$x_col) &&
               model_bundle$x_col %in% colnames(mf) &&
               !is.null(model_bundle$standardized_stats$HFI$mean) &&
               !is.null(model_bundle$standardized_stats$HFI$sd)) {
      hfi_plot <- mf[[model_bundle$x_col]] * model_bundle$standardized_stats$HFI$sd +
        model_bundle$standardized_stats$HFI$mean
    }
    
    if (is.null(hfi_plot) && !is.null(df_aligned)) {
      hfi_plot <- df_aligned[["HFI"]]
    }
    if (is.null(hfi_plot) &&
        !is.null(df_fallback) &&
        nrow(df_fallback) == n_obs) {
      hfi_plot <- df_fallback[["HFI"]]
    }
    
    if (!is.null(hfi_plot) && length(hfi_plot) == n_obs) {
      ppc_info <- prepare_ppc_hfi(hfi_plot, y_obs, n_bins = 20)
      ppc_status <- if (is.null(ppc_info)) "failed" else "ready"
    } else {
      ppc_status <- "missing"
    }
  } else {
    ppc_status <- "unaligned"
  }
  
  # 模型模拟
  sim_zero <- numeric(0)
  sim_mean <- numeric(0)
  sim_var <- numeric(0)
  sim_q95 <- numeric(0)
  sim_q99 <- numeric(0)
  sim_max <- numeric(0)
  sim_tail <- numeric(0)
  sim_bin_means_all <- NULL
  sim_error_message <- NULL
  sim_chunks <- split_nsim_into_chunks(nsim, sim_chunk_size)
  
  for (chunk_nsim in sim_chunks) {
    sim_df_chunk <- tryCatch({
      simulate(model_obj, nsim = chunk_nsim)
    }, error = function(e) {
      sim_error_message <<- e$message
      NULL
    })
    
    if (is.null(sim_df_chunk)) break
    
    sim_mat_chunk <- as.matrix(sim_df_chunk)
    sim_stats_list_chunk <- apply(sim_mat_chunk, 2, calc_stats)
    
    sim_zero <- c(sim_zero, sapply(sim_stats_list_chunk, function(s) s$zero_ratio))
    sim_mean <- c(sim_mean, sapply(sim_stats_list_chunk, function(s) s$mean))
    sim_var <- c(sim_var, sapply(sim_stats_list_chunk, function(s) s$variance))
    sim_q95 <- c(sim_q95, sapply(sim_stats_list_chunk, function(s) s$q95))
    sim_q99 <- c(sim_q99, sapply(sim_stats_list_chunk, function(s) s$q99))
    sim_max <- c(sim_max, sapply(sim_stats_list_chunk, function(s) s$max))
    sim_tail <- c(sim_tail, sapply(sim_stats_list_chunk, function(s) s$tail_index))
    
    if (identical(ppc_status, "ready")) {
      sim_bin_chunk <- calc_sim_bin_means(sim_mat_chunk, ppc_info)
      if (is.null(sim_bin_means_all)) {
        sim_bin_means_all <- sim_bin_chunk
      } else {
        sim_bin_means_all <- cbind(sim_bin_means_all, sim_bin_chunk)
      }
      rm(sim_bin_chunk)
    }
    
    rm(sim_df_chunk, sim_mat_chunk, sim_stats_list_chunk)
    invisible(gc(verbose = FALSE))
  }
  
  sim_failed <- !is.null(sim_error_message)
  if (sim_failed) {
    note_vec <- c(note_vec, paste0("simulate_error:", sim_error_message))
  }
  
  if (sim_failed) {
    results <- rbind(results, data.frame(
      分类ID = key,
      rds_file = rds_path,
      csv_file = csv_path,
      n_obs = n_obs,
      nsim = nsim,
      zero_ratio_obs = obs_stats$zero_ratio,
      zero_ratio_sim_mean = NA_real_,
      zero_ratio_sim_q025 = NA_real_,
      zero_ratio_sim_q975 = NA_real_,
      zero_ratio_obs_pct = NA_real_,
      zero_ratio_in95 = NA,
      mean_obs = obs_stats$mean,
      mean_sim_mean = NA_real_,
      mean_sim_q025 = NA_real_,
      mean_sim_q975 = NA_real_,
      mean_obs_pct = NA_real_,
      mean_in95 = NA,
      var_obs = obs_stats$variance,
      var_sim_mean = NA_real_,
      var_sim_q025 = NA_real_,
      var_sim_q975 = NA_real_,
      var_obs_pct = NA_real_,
      var_in95 = NA,
      q95_obs = obs_stats$q95,
      q95_sim_mean = NA_real_,
      q95_sim_q025 = NA_real_,
      q95_sim_q975 = NA_real_,
      q95_obs_pct = NA_real_,
      q95_in95 = NA,
      q99_obs = obs_stats$q99,
      q99_sim_mean = NA_real_,
      q99_sim_q025 = NA_real_,
      q99_sim_q975 = NA_real_,
      q99_obs_pct = NA_real_,
      q99_in95 = NA,
      max_obs = obs_stats$max,
      max_sim_mean = NA_real_,
      max_sim_q025 = NA_real_,
      max_sim_q975 = NA_real_,
      max_obs_pct = NA_real_,
      max_in95 = NA,
      tail_index_obs = obs_stats$tail_index,
      tail_index_sim_mean = NA_real_,
      tail_index_sim_q025 = NA_real_,
      tail_index_sim_q975 = NA_real_,
      tail_index_obs_pct = NA_real_,
      tail_index_in95 = NA,
      note = paste(note_vec, collapse = ";"),
      stringsAsFactors = FALSE
    ))
    cat(paste0("[进度 ", i, "/", total_rds, "] 已处理: ", rds_base, " | note: ", ifelse(length(note_vec) == 0, "ok", paste(note_vec, collapse = ";")), "\n"))
    next
  }
  
  s_zero <- summarize_sim(sim_zero, obs_stats$zero_ratio)
  s_mean <- summarize_sim(sim_mean, obs_stats$mean)
  s_var <- summarize_sim(sim_var, obs_stats$variance)
  s_q95 <- summarize_sim(sim_q95, obs_stats$q95)
  s_q99 <- summarize_sim(sim_q99, obs_stats$q99)
  s_max <- summarize_sim(sim_max, obs_stats$max)
  s_tail <- summarize_sim(sim_tail, obs_stats$tail_index)
  
  results <- rbind(results, data.frame(
    分类ID = key,
    rds_file = rds_path,
    csv_file = csv_path,
    n_obs = n_obs,
    nsim = nsim,
    zero_ratio_obs = obs_stats$zero_ratio,
    zero_ratio_sim_mean = s_zero$mean,
    zero_ratio_sim_q025 = s_zero$q025,
    zero_ratio_sim_q975 = s_zero$q975,
    zero_ratio_obs_pct = s_zero$pct,
    zero_ratio_in95 = s_zero$in95,
    mean_obs = obs_stats$mean,
    mean_sim_mean = s_mean$mean,
    mean_sim_q025 = s_mean$q025,
    mean_sim_q975 = s_mean$q975,
    mean_obs_pct = s_mean$pct,
    mean_in95 = s_mean$in95,
    var_obs = obs_stats$variance,
    var_sim_mean = s_var$mean,
    var_sim_q025 = s_var$q025,
    var_sim_q975 = s_var$q975,
    var_obs_pct = s_var$pct,
    var_in95 = s_var$in95,
    q95_obs = obs_stats$q95,
    q95_sim_mean = s_q95$mean,
    q95_sim_q025 = s_q95$q025,
    q95_sim_q975 = s_q95$q975,
    q95_obs_pct = s_q95$pct,
    q95_in95 = s_q95$in95,
    q99_obs = obs_stats$q99,
    q99_sim_mean = s_q99$mean,
    q99_sim_q025 = s_q99$q025,
    q99_sim_q975 = s_q99$q975,
    q99_obs_pct = s_q99$pct,
    q99_in95 = s_q99$in95,
    max_obs = obs_stats$max,
    max_sim_mean = s_max$mean,
    max_sim_q025 = s_max$q025,
    max_sim_q975 = s_max$q975,
    max_obs_pct = s_max$pct,
    max_in95 = s_max$in95,
    tail_index_obs = obs_stats$tail_index,
    tail_index_sim_mean = s_tail$mean,
    tail_index_sim_q025 = s_tail$q025,
    tail_index_sim_q975 = s_tail$q975,
    tail_index_obs_pct = s_tail$pct,
    tail_index_in95 = s_tail$in95,
    note = ifelse(length(note_vec) == 0, "ok", paste(note_vec, collapse = ";")),
    stringsAsFactors = FALSE
  ))
  row_idx <- nrow(results)
  
  # 生成HFI vs Fire frequency的后验检验图
  if (identical(ppc_status, "ready")) {
    plot_file <- file.path(output_dir, paste0("PPC_HFI_Fire_frequency_", key, ".png"))
    ok_plot <- plot_ppc_hfi(ppc_info, sim_bin_means_all, plot_file)
    if (!ok_plot) note_vec <- c(note_vec, "ppc_plot_failed")
  } else if (identical(ppc_status, "missing")) {
    note_vec <- c(note_vec, "ppc_missing_HFI_or_Fire_frequency")
  } else if (identical(ppc_status, "failed")) {
    note_vec <- c(note_vec, "ppc_plot_failed")
  } else {
    note_vec <- c(note_vec, "ppc_data_not_aligned")
  }
  results$note[row_idx] <- ifelse(length(note_vec) == 0, "ok", paste(note_vec, collapse = ";"))
  cat(paste0("[进度 ", i, "/", total_rds, "] 已处理: ", rds_base, " | note: ", results$note[row_idx], "\n"))
}

write_csv_utf8_bom(results, output_file)
cat(paste0("完成: ", output_file, "\n"))
