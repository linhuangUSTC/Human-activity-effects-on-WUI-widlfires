#!/usr/bin/env Rscript

# ------------------------------------------------------------------------
# 程序: SimCheck_Fire_area_Ordbeta_spline_等样本分箱.R
# 功能: 使用已拟合的GLMM模型进行多次模拟，比较模拟分布与真实数据分布特征，
#       评估模型能否生成与真实 Normalized_fire_area 数据相似的整体分布。
# 模型来源: B:/WUI/Picture and statistic results/model construction/GLMM OrdBeta 自然样条曲线 Fire area/*.rds
# 数据来源: I:/Processing data/渔网分类建模/Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic/*.csv
#       （直接使用 Normalized_fire_area 列，不再计算或截断）
# 匹配方式: 根据CSV文件名中的前三段（如 face_A_WET）匹配RDS文件名
# 输出: B:/WUI/Picture/SimCheck_Fire_area_OrdBeta_spline.csv
#       B:/WUI/Picture/PCC_HFI_Normalized_Fire_BArea_***.png
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

plot_ppc_hfi <- function(hfi_raw, y_obs, sim_mat, out_path, n_bins = 20) {
  valid <- is.finite(hfi_raw) & is.finite(y_obs)
  if (!any(valid)) return(FALSE)
  
  hfi_raw <- hfi_raw[valid]
  y_obs <- y_obs[valid]
  sim_mat <- sim_mat[valid, , drop = FALSE]
  
  bins <- build_hfi_bins(hfi_raw, n_bins = n_bins)
  if (is.null(bins)) return(FALSE)
  
  bin_id <- cut(hfi_raw, breaks = bins$breaks, include.lowest = TRUE, labels = FALSE)
  n_bins_use <- length(bins$centers)
  
  sim_norm <- sim_mat
  sim_bin_means <- matrix(NA_real_, nrow = n_bins_use, ncol = ncol(sim_mat))
  sim_bin_se <- matrix(NA_real_, nrow = n_bins_use, ncol = ncol(sim_mat))
  for (b in seq_len(n_bins_use)) {
    idx <- which(bin_id == b)
    if (length(idx) > 0) {
      sim_bin_means[b, ] <- colMeans(sim_norm[idx, , drop = FALSE], na.rm = TRUE)
      sim_bin_se[b, ] <- apply(sim_norm[idx, , drop = FALSE], 2, stats::sd, na.rm = TRUE) / sqrt(length(idx))
    }
  }

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
  
  obs_norm <- y_obs
  obs_bin_means <- rep(NA_real_, n_bins_use)
  obs_bin_se <- rep(NA_real_, n_bins_use)
  for (b in seq_len(n_bins_use)) {
    idx <- which(bin_id == b)
    if (length(idx) > 0) {
      obs_bin_means[b] <- mean(obs_norm[idx], na.rm = TRUE)
      obs_bin_se[b] <- stats::sd(obs_norm[idx], na.rm = TRUE) / sqrt(length(idx))
    }
  }
  
  sim_keep_means <- sim_bin_means
  sim_keep_se <- sim_bin_se
  smooth_x <- seq(min(bins$centers), max(bins$centers), length.out = 200)
  sim_smooth_list <- lapply(seq_len(ncol(sim_keep_means)), function(j) {
    mean_smooth <- smooth_predict(bins$centers, sim_keep_means[, j], smooth_x)
    se_smooth <- smooth_predict(bins$centers, sim_keep_se[, j], smooth_x)
    se_smooth <- pmax(se_smooth, 0)
    data.frame(
      HFI = smooth_x,
      Normalized_Fire_Area = mean_smooth,
      Lower = mean_smooth - 1.96 * se_smooth,
      Upper = mean_smooth + 1.96 * se_smooth,
      sim = j
    )
  })
  sim_smooth_df <- do.call(rbind, sim_smooth_list)
  obs_df <- data.frame(
    HFI = bins$centers,
    Normalized_Fire_Area = obs_bin_means,
    Lower = obs_bin_means - obs_bin_se,
    Upper = obs_bin_means + obs_bin_se
  )
  
  p <- ggplot(sim_smooth_df, aes(x = HFI, y = Normalized_Fire_Area, group = sim)) +
    geom_ribbon(aes(ymin = Lower, ymax = Upper), fill = "grey80", alpha = 0.05, color = NA) +
    geom_line(color = "grey80", alpha = 0.25, linewidth = 1.0) +
    geom_errorbar(data = obs_df, aes(x = HFI, y = Normalized_Fire_Area, ymin = Lower, ymax = Upper), color = "red3", width = (max(bins$centers) - min(bins$centers)) * 0.05, linewidth = 0.5, inherit.aes = FALSE) +
    geom_point(data = obs_df, aes(x = HFI, y = Normalized_Fire_Area), color = "red3", size = 2, inherit.aes = FALSE) +
    xlab("HFI") +
    ylab("Normalized Fire Area") +
    theme_bw() +
    theme(panel.grid = element_line(colour = "gray90"))
  
  ggsave(out_path, p, width = 6, height = 4, dpi = 600)
  TRUE
}

# 配置
rds_dir <- "B:/WUI/Picture and statistic results/model construction/GLMM ZIOrdbeta 自然样条 Normalized fire area"
csv_dir <- "I:/Processing data/渔网分类建模/Fire_area_with_WUI_area_offset_conclude_0_分类建模数据_Basic"
output_dir <- "B:/WUI/Picture"
output_file <- file.path(output_dir, "SimCheck_Fire_area_Ordbeta_spline.csv")

nsim <- 1000
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
    cat(paste0("[进度 ", i, "/", total_rds, "] RDS: ", basename(rds_path), " | CSV: <no_match>\n"))
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
  
  cat(paste0("[进度 ", i, "/", total_rds, "] RDS: ", basename(rds_path), " | CSV: ", basename(csv_path), "\n"))
  
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
  required_cols <- c("Normalized_fire_area", "HFI", "GRID_ID")
  df_required <- NULL
  if (!all(required_cols %in% colnames(df_raw))) {
    note_vec <- c(note_vec, "csv_missing_cols")
  } else {
    df_required <- df_raw[, required_cols, drop = FALSE]
  }
  
  # 若没有model.frame，才尝试用CSV复现过滤得到观测响应
  df_fallback <- df_required
  if (!is.null(df_fallback) && is.null(mf)) {
    df_fallback <- na.omit(df_fallback)
    df_fallback <- df_fallback[df_fallback[["Normalized_fire_area"]] >= 0, ]
  }
  
  # 真实响应值
  y_obs <- tryCatch({
    if (!is.null(mf)) mf[[1]] else NULL
  }, error = function(e) NULL)
  if (is.null(y_obs)) {
    if (!is.null(df_fallback) && nrow(df_fallback) > 0) {
      y_obs <- df_fallback[["Normalized_fire_area"]]
    } else {
      y_obs <- numeric(0)
    }
  }
  
  obs_stats <- calc_stats(y_obs)
  n_obs <- length(y_obs)
  
  # 模型模拟
  sim_df <- tryCatch({
    simulate(model_obj, nsim = nsim)
  }, error = function(e) {
    note_vec <<- c(note_vec, paste0("simulate_error:", e$message))
    NULL
  })
  
  if (is.null(sim_df)) {
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
  
  sim_mat <- as.matrix(sim_df)
  sim_stats_list <- apply(sim_mat, 2, calc_stats)
  
  # 将模拟统计量拆解成向量
  sim_zero <- sapply(sim_stats_list, function(s) s$zero_ratio)
  sim_mean <- sapply(sim_stats_list, function(s) s$mean)
  sim_var <- sapply(sim_stats_list, function(s) s$variance)
  sim_q95 <- sapply(sim_stats_list, function(s) s$q95)
  sim_q99 <- sapply(sim_stats_list, function(s) s$q99)
  sim_max <- sapply(sim_stats_list, function(s) s$max)
  sim_tail <- sapply(sim_stats_list, function(s) s$tail_index)
  
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
  
  # 生成HFI vs Normalized Fire Area的后验检验图
  if (!is.null(mf) && nrow(mf) == nrow(sim_mat)) {
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
    
    if (is.null(hfi_plot) &&
      !is.null(df_required) &&
      all(rownames(mf) %in% rownames(df_required))) {
      df_mf <- df_required[rownames(mf), , drop = FALSE]
      if ("HFI" %in% colnames(df_mf)) hfi_plot <- df_mf[["HFI"]]
    }
    
    if (!is.null(hfi_plot)) {
      plot_file <- file.path(output_dir, paste0("PPC_HFI_Normalized_Fire_Area_", key, ".png"))
      ok_plot <- plot_ppc_hfi(hfi_plot, y_obs, sim_mat, plot_file, n_bins =20)
      if (!ok_plot) note_vec <- c(note_vec, "ppc_plot_failed")
    } else {
      note_vec <- c(note_vec, "ppc_missing_HFI")
    }
  } else {
    note_vec <- c(note_vec, "ppc_data_not_aligned")
  }
  results$note[row_idx] <- ifelse(length(note_vec) == 0, "ok", paste(note_vec, collapse = ";"))
  cat(paste0("[进度 ", i, "/", total_rds, "] 已处理: ", rds_base, " | note: ", results$note[row_idx], "\n"))
}

write_csv_utf8_bom(results, output_file)
cat(paste0("完成: ", output_file, "\n"))
