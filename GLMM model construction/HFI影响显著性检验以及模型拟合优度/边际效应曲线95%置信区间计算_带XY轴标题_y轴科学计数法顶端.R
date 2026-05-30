#!/usr/bin/env Rscript

# ------------------------------------------------------------------------
# 程序: 边际效应曲线95%置信区间计算_带XY轴标题_y轴科学计数法顶端.R
# 功能: 读取自然样条GLMM模型RDS，计算HFI边际效应曲线及95%置信区间带并绘图。
#       与原程序一致，但显示x/y轴标题，并将y轴科学计数法指数放在y轴顶部。
# 输入: RDS模型
#       - B:/WUI/Picture and statistic results/model construction/GLMM NB 自然样条曲线 Fire num/*.rds
#       - B:/WUI/Picture and statistic results/model construction/GLMM Ordbeta 自然样条 Normalized fire area/*.rds
# 输出: 边际效应图（PNG）
#       - B:/WUI/Picture/*_marginal_effect_with_axis_titles.png
# ------------------------------------------------------------------------

library(glmmTMB)
library(ggplot2)
library(splines)

create_dir_if_not_exists <- function(dir_path) {
  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE)
    cat(paste0("Created directory: ", dir_path, "\n"))
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
  if (is.list(bundle) && !is.null(bundle[[name]])) return(bundle[[name]])
  default
}

get_group_level <- function(model, group_var, n) {
  if (is.null(group_var) || group_var == "") {
    return(list(values = rep("1", n), levels = NULL))
  }
  mf <- tryCatch(model.frame(model), error = function(e) NULL)
  if (!is.null(mf) && group_var %in% colnames(mf)) {
    gv <- mf[[group_var]]
    if (is.factor(gv)) {
      lv <- levels(gv)
      if (length(lv) > 0) {
        return(list(values = factor(rep(lv[1], n), levels = lv), levels = lv))
      }
    }
    return(list(values = rep(as.character(gv[1]), n), levels = NULL))
  }
  list(values = rep("1", n), levels = NULL)
}

build_hfi_seq <- function(spline_info, n_points = 100) {
  if (!is.null(spline_info) && !is.null(spline_info$boundary_knots)) {
    b <- range(spline_info$boundary_knots, na.rm = TRUE)
    if (is.finite(b[1]) && is.finite(b[2]) && b[1] < b[2]) {
      return(seq(b[1], b[2], length.out = n_points))
    }
  }
  if (!is.null(spline_info) && !is.null(spline_info$knots)) {
    b <- range(spline_info$knots, na.rm = TRUE)
    if (is.finite(b[1]) && is.finite(b[2]) && b[1] < b[2]) {
      return(seq(b[1], b[2], length.out = n_points))
    }
  }
  stop("Unable to determine HFI range from spline_info.")
}

build_newdata <- function(hfi_seq, spline_info, x_col, offset_col, offset_val, group_var, group_values) {
  n <- length(hfi_seq)
  new_data <- data.frame(row_id = seq_len(n))
  new_data[[x_col]] <- hfi_seq
  if (!is.null(offset_col) && offset_col != "") {
    new_data[[offset_col]] <- rep(offset_val, n)
  }
  if (!is.null(group_var) && group_var != "") {
    new_data[[group_var]] <- group_values
  }

  if (!is.null(spline_info) && !is.null(spline_info$boundary_knots)) {
    knots <- spline_info$knots
    bknots <- spline_info$boundary_knots
    spline_basis <- ns(hfi_seq, knots = knots, Boundary.knots = bknots)
    spline_names <- spline_info$spline_names
    if (is.null(spline_names) || length(spline_names) != ncol(spline_basis)) {
      spline_names <- paste0("spline_", seq_len(ncol(spline_basis)))
    }
    for (j in seq_len(ncol(spline_basis))) {
      new_data[[spline_names[j]]] <- spline_basis[, j]
    }
  }

  new_data$row_id <- NULL
  new_data
}

predict_marginal_curve <- function(model, new_data, offset_col) {
  pred <- predict(model, newdata = new_data, type = "response", se.fit = TRUE, re.form = NA)
  fit <- as.numeric(pred$fit)
  se <- as.numeric(pred$se.fit)
  if (!is.null(offset_col) && offset_col %in% colnames(new_data)) {
    denom <- new_data[[offset_col]]
    fit <- fit / denom
    se <- se / denom
  }
  list(fit = fit, se = se)
}

choose_y_exponent <- function(values) {
  finite_vals <- values[is.finite(values) & values != 0]
  if (length(finite_vals) == 0) {
    return(0)
  }
  max_abs <- max(abs(finite_vals), na.rm = TRUE)
  if (!is.finite(max_abs) || max_abs < 1e-12) {
    return(0)
  }
  floor(log10(max_abs))
}

make_y_multiplier_label <- function(exponent) {
  if (exponent == 0) {
    return(NULL)
  }
  sprintf("1e%+03d", exponent)
}

make_plot <- function(pred_df, y_label, out_path) {
  x_range <- range(pred_df$hfi_original, na.rm = TRUE)
  y_values <- c(pred_df$pred_freq, pred_df$ci_lower, pred_df$ci_upper)
  y_exponent <- choose_y_exponent(y_values)
  y_scale_factor <- 10.0 ^ y_exponent
  y_multiplier_label <- make_y_multiplier_label(y_exponent)
  scaled_y_label_1dec <- function(x) {
    ifelse(is.finite(x), sprintf("%.1f", x / y_scale_factor), NA_character_)
  }

  p <- ggplot(pred_df, aes(x = hfi_original)) +
    geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper, fill = "95% CI"), alpha = 0.5) +
    geom_line(aes(y = pred_freq, color = "Marginal effect"), linewidth = 1.2) +
    scale_color_manual(NULL, values = c("Marginal effect" = "red")) +
    scale_fill_manual(NULL, values = c("95% CI" = "lightblue")) +
    scale_x_continuous(limits = x_range) +
    scale_y_continuous(labels = scaled_y_label_1dec, expand = expansion(mult = c(0.02, 0.12))) +
    labs(
      x = "Human Footprint",
      y = y_label,
      title = pred_df$class_id[1]
    ) +
    coord_cartesian(clip = "off") +
    theme_bw() +
    theme(
      panel.grid = element_line(colour = "gray90"),
      panel.grid.major.y = element_blank(),
      panel.grid.minor.y = element_blank(),
      text = element_text(size = 32),
      axis.text = element_text(size = 28),
      axis.title.x = element_text(size = 30, margin = margin(t = 12)),
      axis.title.y = element_text(size = 30, margin = margin(r = 12)),
      plot.title = element_text(size = 32, hjust = 0.5),
      plot.margin = margin(t = 20, r = 16, b = 12, l = 16),
      legend.position = "none"
    )

  if (!is.null(y_multiplier_label)) {
    p <- p +
      annotate(
        "text",
        x = -Inf,
        y = Inf,
        label = y_multiplier_label,
        hjust = -0.10,
        vjust = -0.45,
        size = 8.5
      )
  }

  ggsave(out_path, p, width = 7.5, height = 6, units = "in", dpi = 600)
}

extract_class_id <- function(rds_path) {
  base <- tools::file_path_sans_ext(basename(rds_path))
  m <- regexpr("(face|mix)_[^_]+_[^_]+", base, perl = TRUE)
  if (m[1] != -1) {
    return(regmatches(base, m))
  }
  parts <- strsplit(base, "_")[[1]]
  class_idx <- which(parts %in% c("face", "mix"))
  if (length(class_idx) > 0 && length(parts) >= class_idx[1] + 2) {
    return(paste(parts[class_idx[1]:(class_idx[1] + 2)], collapse = "_"))
  }
  if (length(parts) >= 3) {
    return(paste(parts[1:3], collapse = "_"))
  }
  base
}

process_rds <- function(rds_path, output_dir) {
  bundle <- NULL
  model <- NULL
  tryCatch({
    rb <- safe_read_model_bundle(rds_path)
    bundle <- rb$bundle
    model <- rb$model
  }, error = function(e) {
    cat(paste0("Skip (read error): ", basename(rds_path), " | ", e$message, "\n"))
    return(FALSE)
  })

  x_col <- get_meta_value(bundle, "x_col", "HFI_std")
  y_col <- get_meta_value(bundle, "y_col", NA_character_)
  offset_col <- get_meta_value(bundle, "offset_col", "WUI_area")
  group_var <- get_meta_value(bundle, "group_var", "GRID_ID")
  standardized_stats <- get_meta_value(bundle, "standardized_stats", NULL)
  spline_info <- get_meta_value(bundle, "spline_info", NULL)

  if (is.null(spline_info)) {
    cat(paste0("Skip (missing spline_info): ", basename(rds_path), "\n"))
    return(FALSE)
  }

  hfi_seq <- tryCatch(build_hfi_seq(spline_info, n_points = 100), error = function(e) {
    cat(paste0("Skip (HFI range error): ", basename(rds_path), " | ", e$message, "\n"))
    NULL
  })
  if (is.null(hfi_seq)) return(FALSE)

  group_info <- get_group_level(model, group_var, length(hfi_seq))
  new_data <- build_newdata(
    hfi_seq = hfi_seq,
    spline_info = spline_info,
    x_col = x_col,
    offset_col = offset_col,
    offset_val = 1,
    group_var = group_var,
    group_values = group_info$values
  )

  pred <- tryCatch(predict_marginal_curve(model, new_data, offset_col), error = function(e) {
    cat(paste0("Skip (predict error): ", basename(rds_path), " | ", e$message, "\n"))
    NULL
  })
  if (is.null(pred)) return(FALSE)

  orig_var <- gsub("_std$", "", x_col)
  hfi_original <- hfi_seq
  if (!is.null(standardized_stats) &&
      !is.null(standardized_stats[[orig_var]]) &&
      !is.null(standardized_stats[[orig_var]]$mean) &&
      !is.null(standardized_stats[[orig_var]]$sd)) {
    hfi_original <- hfi_seq * standardized_stats[[orig_var]]$sd + standardized_stats[[orig_var]]$mean
  }

  ci_lower <- pmax(0, pred$fit - 1.96 * pred$se)
  ci_upper <- pred$fit + 1.96 * pred$se

  pred_df <- data.frame(
    hfi_original = hfi_original,
    pred_freq = pred$fit,
    ci_lower = ci_lower,
    ci_upper = ci_upper
  )
  pred_df$class_id <- extract_class_id(rds_path)

  y_label <- "Normalized Response"
  if (!is.na(y_col) && grepl("Fire_num", y_col, ignore.case = TRUE)) {
    y_label <- "Fire frequency"
  } else if (!is.na(y_col) && grepl("Fire_area", y_col, ignore.case = TRUE)) {
    y_label <- "Normalized burned area"
  } else if (grepl("fire_num", basename(rds_path), ignore.case = TRUE)) {
    y_label <- "Fire frequency"
  } else if (grepl("fire_area", basename(rds_path), ignore.case = TRUE)) {
    y_label <- "Normalized burned area"
  }

  base <- tools::file_path_sans_ext(basename(rds_path))
  base <- sub("_model$", "", base)
  out_path <- file.path(output_dir, paste0(base, "_marginal_effect_with_axis_titles.png"))

  make_plot(pred_df, y_label, out_path)
  cat(paste0("Saved: ", out_path, "\n"))
  TRUE
}

main <- function() {
  rds_dirs <- c(
    "B:/WUI/Picture and statistic results/model construction/GLMM NB 自然样条曲线 Fire num",
    "B:/WUI/Picture and statistic results/model construction/GLMM Ordbeta 自然样条 Normalized fire area"
  )
  output_dir <- "B:/WUI/Picture"
  create_dir_if_not_exists(output_dir)

  total <- 0
  success <- 0
  failure <- 0

  for (rds_dir in rds_dirs) {
    rds_files <- list.files(rds_dir, pattern = "\\.rds$", full.names = TRUE)
    if (length(rds_files) == 0) {
      cat(paste0("No RDS files found in ", rds_dir, "\n"))
      next
    }
    for (rds_path in rds_files) {
      total <- total + 1
      ok <- process_rds(rds_path, output_dir)
      if (isTRUE(ok)) {
        success <- success + 1
      } else {
        failure <- failure + 1
      }
    }
  }

  cat("\nDone.\n")
  cat(paste0("Total: ", total, "\n"))
  cat(paste0("Success: ", success, "\n"))
  cat(paste0("Failure: ", failure, "\n"))
}

main()
