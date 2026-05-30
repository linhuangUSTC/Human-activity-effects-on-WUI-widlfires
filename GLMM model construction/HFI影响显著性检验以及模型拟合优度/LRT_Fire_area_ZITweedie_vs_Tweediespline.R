#!/usr/bin/env Rscript

# ------------------------------------------------------------------------
# 程序: LRT_Fire_area_Tweedie_noHFI_vs_spline.R
# 功能: 根据分类ID匹配两类模型RDS（ZITweedie线性 vs Tweedie+自然样条），进行LRT检验并输出CSV
#      LRT检验用于判断更复杂模型（Tweedie+自然样条）是否相对简单模型（ZITweedie线性）带来显著拟合改进。
# 输入:
#   - I:/Processing data/渔网分类建模/火灾分类数据汇总/分类建模火灾数据汇总统计_Basic.xlsx
#     sheet: Fire_area_over100, column: 分类ID
#   - B:/WUI/Picture and statistic results/model construction/GLMM ZITweedie 线性 Fire area (ZITweedie线性模型RDS)
#   - B:/WUI/Picture and statistic results/model construction/GLMM Tweedie 自然样条曲线 Fire area (Tweedie+自然样条模型RDS)
# 输出:
#   - B:/WUI/Picture/LRT_Fire_area_Tweedie_noHFI_vs_spline.csv
# ------------------------------------------------------------------------

library(glmmTMB)
library(readxl)

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

# 配置
excel_path <- "I:/Processing data/渔网分类建模/火灾分类数据汇总/分类建模火灾数据汇总统计_Basic.xlsx"
sheet_name <- "Fire_area_over100"
class_id_col <- "分类ID"

rds_dir_zitweedie <- "B:/WUI/Picture and statistic results/model construction/GLMM ZITweedie 线性 Fire area"
rds_dir_tweedie_spline <- "B:/WUI/Picture and statistic results/model construction/GLMM Tweedie 自然样条曲线 Fire area"

output_dir <- "B:/WUI/Picture"
output_file <- file.path(output_dir, "LRT_Fire_area_Tweedie_noHFI_vs_spline.csv")

create_dir_if_not_exists(output_dir)

# 读取分类ID
cat("读取Excel...\n")
excel_df <- readxl::read_excel(excel_path, sheet = sheet_name)
if (!(class_id_col %in% colnames(excel_df))) {
  stop(paste0("Column '", class_id_col, "' not found. Available columns: ", paste(colnames(excel_df), collapse = ", ")))
}

class_ids <- unique(trimws(as.character(excel_df[[class_id_col]])))
class_ids <- class_ids[!is.na(class_ids) & class_ids != ""]

if (length(class_ids) == 0) {
  stop("No valid 分类ID found in the sheet.")
}

# 预加载RDS文件列表
files_zitweedie <- list.files(rds_dir_zitweedie, pattern = "\\.rds$", full.names = TRUE)
files_tweedie_spline <- list.files(rds_dir_tweedie_spline, pattern = "\\.rds$", full.names = TRUE)
base_zitweedie <- basename(files_zitweedie)
base_tweedie_spline <- basename(files_tweedie_spline)

find_rds_for_id <- function(id, files, bases) {
  idx <- which(grepl(id, bases, fixed = TRUE))
  if (length(idx) == 0) {
    return(list(path = NA_character_, note = "no_match"))
  }
  if (length(idx) > 1) {
    return(list(path = files[idx[1]], note = paste0("multiple_matches:", length(idx))))
  }
  list(path = files[idx], note = "ok")
}

results <- data.frame(
  分类ID = character(),
  model_ZITweedie = character(),
  model_TweedieSpline = character(),
  LRT_df = numeric(),
  LRT_Chisq = numeric(),
  LRT_pvalue = numeric(),
  AIC_ZITweedie = numeric(),
  AIC_TweedieSpline = numeric(),
  BIC_ZITweedie = numeric(),
  BIC_TweedieSpline = numeric(),
  HFI_spline_significant = character(),
  Better_model = character(),
  note = character(),
  stringsAsFactors = FALSE
)

cat(paste0("开始处理分类ID数量: ", length(class_ids), "\n"))

for (id in class_ids) {
  note_vec <- character()
  m_zi <- find_rds_for_id(id, files_zitweedie, base_zitweedie)
  m_ts <- find_rds_for_id(id, files_tweedie_spline, base_tweedie_spline)
  if (m_zi$note == "no_match") {
    if (!is.na(m_ts$path)) {
      note_vec <- c(note_vec, "ZITweedie_missing_corresponding_rds")
    } else {
      note_vec <- c(note_vec, "ZITweedie_no_match")
    }
  } else if (m_zi$note != "ok") {
    note_vec <- c(note_vec, paste0("ZITweedie_", m_zi$note))
  }
  if (m_ts$note == "no_match") {
    if (!is.na(m_zi$path)) {
      note_vec <- c(note_vec, "TweedieSpline_missing_corresponding_rds")
    } else {
      note_vec <- c(note_vec, "TweedieSpline_no_match")
    }
  } else if (m_ts$note != "ok") {
    note_vec <- c(note_vec, paste0("TweedieSpline_", m_ts$note))
  }
  
  lrt_df <- NA_real_
  lrt_chisq <- NA_real_
  lrt_p <- NA_real_
  aic_zi <- NA_real_
  aic_ts <- NA_real_
  bic_zi <- NA_real_
  bic_ts <- NA_real_
  hfi_sig <- NA_character_
  better_model <- NA_character_

  # 若某分类ID仅有一个模型，则直接判定该模型更优
  if (!is.na(m_zi$path) && is.na(m_ts$path)) better_model <- "ZITweedie"
  if (is.na(m_zi$path) && !is.na(m_ts$path)) better_model <- "Tweedie+Spline"
  
  if (!is.na(m_zi$path) && !is.na(m_ts$path)) {
    tryCatch({
      model_zi <- readRDS(m_zi$path)
      model_ts <- readRDS(m_ts$path)
      
      # 兼容保存为list的模型
      if (is.list(model_zi) && !inherits(model_zi, "glmmTMB") && !is.null(model_zi$model)) {
        model_zi <- model_zi$model
      }
      if (is.list(model_ts) && !inherits(model_ts, "glmmTMB") && !is.null(model_ts$model)) {
        model_ts <- model_ts$model
      }
      
      if (!inherits(model_zi, "glmmTMB") || !inherits(model_ts, "glmmTMB")) {
        note_vec <- c(note_vec, "model_not_glmmTMB")
      } else {
        aic_zi <- AIC(model_zi)
        aic_ts <- AIC(model_ts)
        bic_zi <- BIC(model_zi)
        bic_ts <- BIC(model_ts)
        
        # LRT: 简单模型在前（ZITweedie线性）
        lrt_tab <- anova(model_zi, model_ts, test = "LRT")
        if (nrow(lrt_tab) >= 2) {
          lrt_df <- lrt_tab$Df[2]
          lrt_chisq <- lrt_tab$Chisq[2]
          lrt_p <- lrt_tab$`Pr(>Chisq)`[2]
        } else {
          note_vec <- c(note_vec, "lrt_failed")
        }

        # 1) HFI样条是否显著：基于LRT p值
        if (is.finite(lrt_p)) {
          hfi_sig <- ifelse(lrt_p < 0.05, "Yes", "No")
        }

        # 2) 综合判断哪个模型更好
        if (is.finite(aic_zi) && is.finite(aic_ts) && is.finite(bic_zi) && is.finite(bic_ts)) {
          aic_better <- ifelse(aic_ts < aic_zi, "Tweedie+Spline", "ZITweedie")
          bic_better <- ifelse(bic_ts < bic_zi, "Tweedie+Spline", "ZITweedie")
          if (!is.na(hfi_sig) && hfi_sig == "Yes" && aic_better == "Tweedie+Spline" && bic_better == "Tweedie+Spline") {
            better_model <- "Tweedie+Spline"
          } else if (!is.na(hfi_sig) && hfi_sig == "No" && aic_better == "ZITweedie" && bic_better == "ZITweedie") {
            better_model <- "ZITweedie"
          } else if (aic_better == bic_better) {
            better_model <- aic_better
          } else {
            better_model <- "mixed"
          }
        }
      }
    }, error = function(e) {
      note_vec <<- c(note_vec, paste0("error:", e$message))
    })
  }
  
  results <- rbind(results, data.frame(
    分类ID = id,
    model_ZITweedie = ifelse(is.na(m_zi$path), "", basename(m_zi$path)),
    model_TweedieSpline = ifelse(is.na(m_ts$path), "", basename(m_ts$path)),
    LRT_df = lrt_df,
    LRT_Chisq = lrt_chisq,
    LRT_pvalue = lrt_p,
    AIC_ZITweedie = aic_zi,
    AIC_TweedieSpline = aic_ts,
    BIC_ZITweedie = bic_zi,
    BIC_TweedieSpline = bic_ts,
    HFI_spline_significant = ifelse(is.na(hfi_sig), "", hfi_sig),
    Better_model = ifelse(is.na(better_model), "", better_model),
    note = ifelse(length(note_vec) == 0, "ok", paste(note_vec, collapse = ";")),
    stringsAsFactors = FALSE
  ))
}

write_csv_utf8_bom(results, output_file)
cat(paste0("完成: ", output_file, "\n"))
