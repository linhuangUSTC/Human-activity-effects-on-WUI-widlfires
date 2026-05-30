#!/usr/bin/env Rscript

# ------------------------------------------------------------------------
# 程序: 量化HFI对Fire frequency的影响
# 功能:
#   - 读取 B:/WUI/Picture and statistic results/model construction/GLMM NB 自然样条曲线 Fire num 下的模型产物 (*.rds)
#   - 读取输入目录下的 CSV 数据
#   - 不使用随机效应进行预测（re.form = NA）
#   - 以原始尺度 HFI = 0 作为基线
#   - 计算比值贡献与差值贡献
#   - 将每个 CSV 的结果保存到 B:/WUI/Picture
#   - 输出包含年份字段（从输入 CSV 中自动识别）
# 说明:
#   - 计算时读取的 HFI 为原始值（来自 CSV 的 HFI 列）
#   - 在预测前会按照模型训练时保存的均值/标准差进行标准化
#   - 使用固定效应系数与协方差矩阵，通过 Delta Method 计算 SE 与置信区间
# ------------------------------------------------------------------------

library(glmmTMB)
library(splines)

create_dir_if_not_exists <- function(dir_path) {
  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE)
    cat(paste0("Created directory: ", dir_path, "\n"))
  }
}

safe_predict_no_re <- function(model, newdata) {
  tryCatch({
    predict(model, newdata = newdata, type = "response", re.form = NA, allow.new.levels = TRUE)
  }, error = function(e) {
    # Fallback for older glmmTMB versions without allow.new.levels
    predict(model, newdata = newdata, type = "response", re.form = NA)
  })
}

standardize_hfi <- function(hfi_raw, stats) {
  if (is.null(stats) || is.null(stats$sd) || is.na(stats$sd) || stats$sd == 0) {
    return(rep(0, length(hfi_raw)))
  }
  (hfi_raw - stats$mean) / stats$sd
}

build_spline_basis <- function(x_std, spline_info) {
  if (!is.null(spline_info$knots) && length(spline_info$knots) > 0) {
    if (!is.null(spline_info$boundary_knots) && length(spline_info$boundary_knots) > 0) {
      return(ns(x_std, knots = spline_info$knots, Boundary.knots = spline_info$boundary_knots))
    }
    return(ns(x_std, knots = spline_info$knots))
  }
  # If no knots stored, fall back to df
  if (!is.null(spline_info$df)) {
    if (!is.null(spline_info$boundary_knots) && length(spline_info$boundary_knots) > 0) {
      return(ns(x_std, df = spline_info$df, Boundary.knots = spline_info$boundary_knots))
    }
    return(ns(x_std, df = spline_info$df))
  }
  stop("Spline info is incomplete (no knots or df).")
}

detect_year_column <- function(df) {
  candidates <- c("Year", "YEAR", "year", "YR", "yr", "Fire_Year", "fire_year", "YEAR_ID", "Year_ID", "年份")
  hit <- candidates[candidates %in% colnames(df)]
  if (length(hit) > 0) {
    return(hit[1])
  }
  regex_hit <- grep("year|年份", colnames(df), ignore.case = TRUE, value = TRUE)
  if (length(regex_hit) > 0) {
    return(regex_hit[1])
  }
  return(NULL)
}

add_spline_columns <- function(newdata, x_std, spline_info) {
  basis <- build_spline_basis(x_std, spline_info)
  basis <- as.matrix(basis)
  if (ncol(basis) != length(spline_info$spline_names)) {
    stop("Spline basis column count does not match stored spline names.")
  }
  for (j in seq_len(ncol(basis))) {
    newdata[[spline_info$spline_names[j]]] <- basis[, j]
  }
  newdata
}

get_fixed_effects <- function(model) {
  beta <- fixef(model)$cond
  V <- tryCatch(vcov(model)$cond, error = function(e) NULL)
  if (is.null(V)) {
    vc <- tryCatch(vcov(model), error = function(e) NULL)
    if (is.list(vc) && !is.null(vc$cond)) {
      V <- vc$cond
    } else if (is.matrix(vc)) {
      V <- vc
    }
  }
  if (!is.null(V)) {
    V <- as.matrix(V)
  }
  list(beta = beta, V = V)
}

align_vcov <- function(V, beta_names) {
  if (is.null(V)) {
    return(NULL)
  }
  if (is.null(colnames(V)) || is.null(rownames(V))) {
    if (ncol(V) == length(beta_names)) {
      colnames(V) <- beta_names
      rownames(V) <- beta_names
    }
  }
  if (!all(beta_names %in% colnames(V))) {
    return(NULL)
  }
  V[beta_names, beta_names, drop = FALSE]
}

build_design_matrix <- function(newdata, beta_names, spline_names) {
  n <- nrow(newdata)
  X <- matrix(0, nrow = n, ncol = length(beta_names))
  colnames(X) <- beta_names
  if ("(Intercept)" %in% beta_names) {
    X[, "(Intercept)"] <- 1
  }
  for (sn in spline_names) {
    if (sn %in% beta_names) {
      X[, sn] <- newdata[[sn]]
    }
  }
  X
}

delta_method_effects <- function(beta, V, X, X0, area) {
  eta <- as.vector(X %*% beta)
  eta0 <- as.vector(X0 %*% beta)
  mu_count <- exp(eta + log(area))
  mu0_count <- exp(eta0 + log(area))
  mu_freq <- mu_count / area
  mu0_freq <- mu0_count / area
  delta <- mu_freq - mu0_freq
  logR <- eta - eta0
  R <- exp(logR)
  
  se_delta <- rep(NA_real_, length(mu_freq))
  se_logR <- rep(NA_real_, length(mu_freq))
  if (!is.null(V)) {
    X_mu <- sweep(X, 1, mu_freq, "*")
    X0_mu0 <- sweep(X0, 1, mu0_freq, "*")
    gprime <- X_mu - X0_mu0
    var_delta <- rowSums((gprime %*% V) * gprime)
    se_delta <- sqrt(pmax(var_delta, 0))
    
    diffX <- X - X0
    var_logR <- rowSums((diffX %*% V) * diffX)
    se_logR <- sqrt(pmax(var_logR, 0))
  }
  
  L <- delta - 1.96 * se_delta
  U <- delta + 1.96 * se_delta
  Llog <- logR - 1.96 * se_logR
  Ulog <- logR + 1.96 * se_logR
  
  list(
    mu_freq = mu_freq,
    mu0_freq = mu0_freq,
    mu_count = mu_count,
    mu0_count = mu0_count,
    delta = delta,
    logR = logR,
    R = R,
    se_delta = se_delta,
    se_logR = se_logR,
    L = L,
    U = U,
    Llog = Llog,
    Ulog = Ulog
  )
}

process_single_csv <- function(csv_path, rds_dir, output_dir) {
  file_name <- tools::file_path_sans_ext(basename(csv_path))
  rds_path <- file.path(rds_dir, paste0("GLMM_NaturalSpline_NB_HFI_only_", file_name, "_model.rds"))
  
  if (!file.exists(rds_path)) {
    cat(paste0("Model RDS not found for ", file_name, ": ", rds_path, "\n"))
    return(FALSE)
  }
  
  model_artifact <- readRDS(rds_path)
  model <- model_artifact$model
  standardized_stats <- model_artifact$standardized_stats
  spline_info <- model_artifact$spline_info
  x_col <- if (!is.null(model_artifact$x_col)) model_artifact$x_col else "HFI_std"
  y_col <- if (!is.null(model_artifact$y_col)) model_artifact$y_col else "Fire_num"
  offset_col <- if (!is.null(model_artifact$offset_col)) model_artifact$offset_col else "WUI_area"
  group_var <- if (!is.null(model_artifact$group_var)) model_artifact$group_var else "GRID_ID"
  
  df <- read.csv(csv_path)
  year_col <- detect_year_column(df)
  if (is.null(year_col)) {
    cat(paste0("Warning: Year column not found in ", file_name, ", output will have NA for Year\n"))
  }
  
  # Identify original HFI column name
  orig_x_name <- NULL
  if (!is.null(standardized_stats) && length(standardized_stats) > 0) {
    orig_x_name <- names(standardized_stats)[1]
  }
  if (is.null(orig_x_name) || orig_x_name == "") {
    orig_x_name <- "HFI"
  }
  if (!(orig_x_name %in% colnames(df))) {
    if ("HFI" %in% colnames(df)) {
      orig_x_name <- "HFI"
    } else {
      cat(paste0("Missing HFI column in ", file_name, "\n"))
      return(FALSE)
    }
  }
  
  # Ensure offset column exists
  if (!(offset_col %in% colnames(df))) {
    if ("All_WUI_area" %in% colnames(df)) {
      df[[offset_col]] <- df[["All_WUI_area"]]
    } else {
      cat(paste0("Missing WUI area column in ", file_name, "\n"))
      return(FALSE)
    }
  }
  
  # Ensure group variable exists
  if (!(group_var %in% colnames(df))) {
    if ("GRID_ID" %in% colnames(df)) {
      df[[group_var]] <- df[["GRID_ID"]]
    } else {
      cat(paste0("Missing group column in ", file_name, "\n"))
      return(FALSE)
    }
  }
  
  df[[group_var]] <- as.factor(df[[group_var]])
  
  # Prepare output container
  row_id <- seq_len(nrow(df))
  hfi_raw <- df[[orig_x_name]]
  wui_area <- df[[offset_col]]
  
  valid <- is.finite(hfi_raw) & is.finite(wui_area) & wui_area > 0
  
  pred_freq <- rep(NA_real_, nrow(df))
  base_freq <- rep(NA_real_, nrow(df))
  ratio_contrib <- rep(NA_real_, nrow(df))
  diff_contrib <- rep(NA_real_, nrow(df))
  delta_se <- rep(NA_real_, nrow(df))
  logR_se <- rep(NA_real_, nrow(df))
  delta_ci_l <- rep(NA_real_, nrow(df))
  delta_ci_u <- rep(NA_real_, nrow(df))
  delta_sig <- rep(NA, nrow(df))
  ratio_ci_l <- rep(NA_real_, nrow(df))
  ratio_ci_u <- rep(NA_real_, nrow(df))
  ratio_sig <- rep(NA, nrow(df))
  
  if (any(valid)) {
    hfi_std <- standardize_hfi(hfi_raw[valid], standardized_stats[[orig_x_name]])
    hfi_std_base <- standardize_hfi(rep(0, sum(valid)), standardized_stats[[orig_x_name]])
    
    newdata <- data.frame(
      tmp = hfi_std,
      tmp_wui = wui_area[valid],
      tmp_gid = df[[group_var]][valid]
    )
    names(newdata) <- c(x_col, offset_col, group_var)
    newdata[[x_col]] <- hfi_std
    newdata <- add_spline_columns(newdata, hfi_std, spline_info)
    
    newdata_base <- newdata
    newdata_base[[x_col]] <- hfi_std_base
    newdata_base <- add_spline_columns(newdata_base[, c(x_col, offset_col, group_var)], hfi_std_base, spline_info)
    
    fe <- get_fixed_effects(model)
    beta <- fe$beta
    beta_names <- names(beta)
    V <- align_vcov(fe$V, beta_names)
    if (is.null(V)) {
      cat(paste0("Warning: Var(beta) unavailable for ", file_name, ", SE and CI will be NA\n"))
    }
    
    X <- build_design_matrix(newdata, beta_names, spline_info$spline_names)
    X0 <- build_design_matrix(newdata_base, beta_names, spline_info$spline_names)
    
    dm <- delta_method_effects(beta, V, X, X0, newdata[[offset_col]])
    
    pred_freq[valid] <- dm$mu_count / newdata[[offset_col]]
    base_freq[valid] <- dm$mu0_count / newdata_base[[offset_col]]
    ratio_contrib[valid] <- dm$R
    diff_contrib[valid] <- dm$delta
    delta_se[valid] <- dm$se_delta
    logR_se[valid] <- dm$se_logR
    delta_ci_l[valid] <- dm$L
    delta_ci_u[valid] <- dm$U
    delta_sig[valid] <- (dm$L > 0) | (dm$U < 0)
    ratio_ci_l[valid] <- exp(dm$Llog)
    ratio_ci_u[valid] <- exp(dm$Ulog)
    ratio_sig[valid] <- (dm$Llog > 0) | (dm$Ulog < 0)
  }
  
  out_df <- data.frame(
    row_id = row_id,
    Year = if (!is.null(year_col)) df[[year_col]] else NA,
    model_rds_file = basename(rds_path),
    HFI = hfi_raw,
    WUI_area = wui_area,
    GRID_ID = df[[group_var]],
    pred_fire_frequency = pred_freq,
    baseline_fire_frequency = base_freq,
    ratio_contribution = ratio_contrib,
    diff_contribution = diff_contrib,
    delta_se = delta_se,
    logR_se = logR_se,
    delta_ci_l = delta_ci_l,
    delta_ci_u = delta_ci_u,
    delta_sig = delta_sig,
    ratio_ci_l = ratio_ci_l,
    ratio_ci_u = ratio_ci_u,
    ratio_sig = ratio_sig
  )
  
  out_file <- file.path(output_dir, paste0("HFI_effect_", file_name, ".csv"))
  write.csv(out_df, out_file, row.names = FALSE)
  cat(paste0("Saved: ", out_file, "\n"))
  
  return(TRUE)
}

main <- function() {
  rds_dir <- "B:/WUI/Picture and statistic results/model construction/GLMM NB 自然样条曲线 Fire num"
  input_dir <- "I:/Processing data/渔网分类建模/Fire_num_with_WUI_area_offset_conclude_0_分类建模数据_Basic"
  output_dir <- "B:/WUI/Picture"
  
  create_dir_if_not_exists(output_dir)
  
  csv_files <- list.files(input_dir, pattern = "\\.csv$", ignore.case = TRUE)
  if (length(csv_files) == 0) {
    cat(paste0("No CSV files found in ", input_dir, "\n"))
    return()
  }
  
  cat(paste0("Found ", length(csv_files), " CSV files\n"))
  
  success_count <- 0
  failure_count <- 0
  
  for (csv_file in csv_files) {
    csv_path <- file.path(input_dir, csv_file)
    if (process_single_csv(csv_path, rds_dir, output_dir)) {
      success_count <- success_count + 1
    } else {
      failure_count <- failure_count + 1
    }
  }
  
  cat("\nDone.\n")
  cat(paste0("Success: ", success_count, "\n"))
  cat(paste0("Failure: ", failure_count, "\n"))
}

main()
