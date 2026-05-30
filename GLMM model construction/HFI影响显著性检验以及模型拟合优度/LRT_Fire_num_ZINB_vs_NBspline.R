#!/usr/bin/env Rscript

# ------------------------------------------------------------------------
# 程序: LRT_Fire_num_NB_noHFI_vs_spline.R
# 功能: 根据分类ID匹配两类模型RDS（ZINB线性 vs NB+自然样条），进行LRT检验并输出CSV
#      LRT检验用于判断更复杂模型（NB+自然样条）是否相对简单模型（ZINB线性）带来显著拟合改进。
# 输入:
#   - I:/Processing data/渔网分类建模/火灾分类数据汇总/分类建模火灾数据汇总统计_Basic.xlsx
#     sheet: Fire_num_over100, column: 分类ID
#   - B:/WUI/Picture and statistic results/model construction/GLMM ZINB 线性 Fire num (ZINB线性模型RDS)
#   - B:/WUI/Picture and statistic results/model construction/GLMM NB 自然样条曲线 Fire num (NB+自然样条模型RDS)
# 输出:
#   - B:/WUI/Picture/LRT_Fire_num_NB_noHFI_vs_spline.csv
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
sheet_name <- "Fire_num_over100"
class_id_col <- "分类ID"

rds_dir_zinb <- "B:/WUI/Picture and statistic results/model construction/GLMM ZINB 线性 Fire num"
rds_dir_nb_spline <- "B:/WUI/Picture and statistic results/model construction/GLMM NB 自然样条曲线 Fire num"

output_dir <- "B:/WUI/Picture"
output_file <- file.path(output_dir, "LRT_Fire_num_NB_noHFI_vs_spline.csv")

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
files_zinb <- list.files(rds_dir_zinb, pattern = "\\.rds$", full.names = TRUE)
files_nb_spline <- list.files(rds_dir_nb_spline, pattern = "\\.rds$", full.names = TRUE)
base_zinb <- basename(files_zinb)
base_nb_spline <- basename(files_nb_spline)

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
  model_ZINB = character(),
  model_NBSpline = character(),
  LRT_df = numeric(),
  LRT_Chisq = numeric(),
  LRT_pvalue = numeric(),
  AIC_ZINB = numeric(),
  AIC_NBSpline = numeric(),
  BIC_ZINB = numeric(),
  BIC_NBSpline = numeric(),
  HFI_spline_significant = character(),
  Better_model = character(),
  note = character(),
  stringsAsFactors = FALSE
)

cat(paste0("开始处理分类ID数量: ", length(class_ids), "\n"))

for (id in class_ids) {
  note_vec <- character()
  m_zi <- find_rds_for_id(id, files_zinb, base_zinb)
  m_ns <- find_rds_for_id(id, files_nb_spline, base_nb_spline)
  if (m_zi$note == "no_match") {
    if (!is.na(m_ns$path)) {
      note_vec <- c(note_vec, "ZINB_missing_corresponding_rds")
    } else {
      note_vec <- c(note_vec, "ZINB_no_match")
    }
  } else if (m_zi$note != "ok") {
    note_vec <- c(note_vec, paste0("ZINB_", m_zi$note))
  }
  if (m_ns$note == "no_match") {
    if (!is.na(m_zi$path)) {
      note_vec <- c(note_vec, "NBSpline_missing_corresponding_rds")
    } else {
      note_vec <- c(note_vec, "NBSpline_no_match")
    }
  } else if (m_ns$note != "ok") {
    note_vec <- c(note_vec, paste0("NBSpline_", m_ns$note))
  }
  
  lrt_df <- NA_real_
  lrt_chisq <- NA_real_
  lrt_p <- NA_real_
  aic_zi <- NA_real_
  aic_ns <- NA_real_
  bic_zi <- NA_real_
  bic_ns <- NA_real_
  hfi_sig <- NA_character_
  better_model <- NA_character_

  # 若某分类ID仅有一个模型，则直接判定该模型更优
  if (!is.na(m_zi$path) && is.na(m_ns$path)) better_model <- "ZINB"
  if (is.na(m_zi$path) && !is.na(m_ns$path)) better_model <- "NB+Spline"
  
  if (!is.na(m_zi$path) && !is.na(m_ns$path)) {
    tryCatch({
      model_zi <- readRDS(m_zi$path)
      model_ns <- readRDS(m_ns$path)
      
      # 兼容保存为list的模型
      if (is.list(model_zi) && !inherits(model_zi, "glmmTMB") && !is.null(model_zi$model)) {
        model_zi <- model_zi$model
      }
      if (is.list(model_ns) && !inherits(model_ns, "glmmTMB") && !is.null(model_ns$model)) {
        model_ns <- model_ns$model
      }
      
      if (!inherits(model_zi, "glmmTMB") || !inherits(model_ns, "glmmTMB")) {
        note_vec <- c(note_vec, "model_not_glmmTMB")
      } else {
        aic_zi <- AIC(model_zi)
        aic_ns <- AIC(model_ns)
        bic_zi <- BIC(model_zi)
        bic_ns <- BIC(model_ns)
        
        # LRT: 简单模型在前（ZINB线性）
        lrt_tab <- anova(model_zi, model_ns, test = "LRT")
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
        if (is.finite(aic_zi) && is.finite(aic_ns) && is.finite(bic_zi) && is.finite(bic_ns)) {
          aic_better <- ifelse(aic_ns < aic_zi, "NB+Spline", "ZINB")
          bic_better <- ifelse(bic_ns < bic_zi, "NB+Spline", "ZINB")
          if (!is.na(hfi_sig) && hfi_sig == "Yes" && aic_better == "NB+Spline" && bic_better == "NB+Spline") {
            better_model <- "NB+Spline"
          } else if (!is.na(hfi_sig) && hfi_sig == "No" && aic_better == "ZINB" && bic_better == "ZINB") {
            better_model <- "ZINB"
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
    model_ZINB = ifelse(is.na(m_zi$path), "", basename(m_zi$path)),
    model_NBSpline = ifelse(is.na(m_ns$path), "", basename(m_ns$path)),
    LRT_df = lrt_df,
    LRT_Chisq = lrt_chisq,
    LRT_pvalue = lrt_p,
    AIC_ZINB = aic_zi,
    AIC_NBSpline = aic_ns,
    BIC_ZINB = bic_zi,
    BIC_NBSpline = bic_ns,
    HFI_spline_significant = ifelse(is.na(hfi_sig), "", hfi_sig),
    Better_model = ifelse(is.na(better_model), "", better_model),
    note = ifelse(length(note_vec) == 0, "ok", paste(note_vec, collapse = ";")),
    stringsAsFactors = FALSE
  ))
}

write_csv_utf8_bom(results, output_file)
cat(paste0("完成: ", output_file, "\n"))
