# ============================================================
# Applied Stats II - Problem Set 2
# Portable & robust script 
# ============================================================

rm(list = ls())

# -----------------------------
# 0) Packages (auto-install)
# -----------------------------
pkgs <- c("dplyr", "broom")
to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_install) > 0) install.packages(to_install, dependencies = TRUE)
invisible(lapply(pkgs, library, character.only = TRUE))

# -----------------------------
# 1) Robust path: locate data
# -----------------------------
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  script_path <- sub(file_arg, "", args[grep(file_arg, args)])
  if (length(script_path) > 0) {
    return(normalizePath(dirname(script_path), winslash = "/", mustWork = FALSE))
  }
  # RStudio / interactive fallback
  normalizePath(getwd(), winslash = "/", mustWork = FALSE)
}

base_dir <- get_script_dir()

candidate_paths <- c(
  file.path(base_dir, "climateSupport.RData"),
  file.path(base_dir, "data", "climateSupport.RData"),
  file.path(getwd(), "climateSupport.RData"),
  file.path(getwd(), "data", "climateSupport.RData")
)

data_path <- candidate_paths[file.exists(candidate_paths)][1]

if (is.na(data_path) || length(data_path) == 0) {
  stop(
    "Cannot find climateSupport.RData.\n",
    "Put it in the same folder as this script OR in a ./data/ subfolder.\n",
    "Searched:\n- ", paste(candidate_paths, collapse = "\n- ")
  )
}
message("Data file found at: ", data_path)

# -----------------------------
# 2) Load .RData safely
# -----------------------------
tmp_env <- new.env()
loaded_names <- load(data_path, envir = tmp_env)
message("Objects loaded: ", paste(loaded_names, collapse = ", "))

# choose the first data.frame-like object
is_df_like <- function(x) inherits(x, c("data.frame", "tbl_df", "tbl"))
df_candidates <- loaded_names[sapply(loaded_names, function(nm) is_df_like(tmp_env[[nm]]))]

if (length(df_candidates) == 0) {
  stop("No data.frame found in the .RData file. Objects: ",
       paste(loaded_names, collapse = ", "))
}

dat_name <- df_candidates[1]
climateSupport <- tmp_env[[dat_name]]
rm(tmp_env)

message("Using data frame object: ", dat_name)
message("N rows: ", nrow(climateSupport), " | N cols: ", ncol(climateSupport))

# -----------------------------
# 3) Clean variables 
# -----------------------------
required_vars <- c("choice", "countries", "sanctions")
missing_vars <- setdiff(required_vars, names(climateSupport))
if (length(missing_vars) > 0) {
  stop("Missing required variables: ", paste(missing_vars, collapse = ", "))
}

# (A) choice: "Supported"/"Not supported" -> 1/0
#  (no silent NA conversion)
choice_chr <- tolower(trimws(as.character(climateSupport$choice)))

climateSupport$choice01 <- dplyr::case_when(
  choice_chr == "supported" ~ 1L,
  choice_chr == "not supported" ~ 0L,
  TRUE ~ NA_integer_
)

if (any(is.na(climateSupport$choice01))) {
  bad <- sort(unique(choice_chr[is.na(climateSupport$choice01)]))
  stop("Unrecognized values in choice: ", paste(bad, collapse = ", "))
}

# (B) countries & sanctions: convert ordered -> unordered factor, set baselines
climateSupport$countries <- factor(as.character(climateSupport$countries), ordered = FALSE)
climateSupport$sanctions <- factor(as.character(climateSupport$sanctions), ordered = FALSE)

# baseline: countries = "20 of 192", sanctions = "None"
if ("20 of 192" %in% levels(climateSupport$countries)) {
  climateSupport$countries <- relevel(climateSupport$countries, ref = "20 of 192")
}
if ("None" %in% levels(climateSupport$sanctions)) {
  climateSupport$sanctions <- relevel(climateSupport$sanctions, ref = "None")
}

message("Countries levels: ", paste(levels(climateSupport$countries), collapse = ", "))
message("Sanctions levels: ", paste(levels(climateSupport$sanctions), collapse = ", "))
message("Choice01 counts: "); print(table(climateSupport$choice01))

# -----------------------------
# 4) Q1: Additive logistic model + global LR test
# -----------------------------
m_null <- glm(choice01 ~ 1, data = climateSupport, family = binomial(link = "logit"))
m_add  <- glm(choice01 ~ countries + sanctions, data = climateSupport, family = binomial(link = "logit"))

cat("\n========================\n")
cat("Q1) Additive model summary\n")
cat("========================\n")
print(summary(m_add))

cat("\n========================\n")
cat("Q1) Global LR test (null vs additive)\n")
cat("========================\n")
lr_global <- anova(m_null, m_add, test = "Chisq")
print(lr_global)

# -----------------------------
# 5) Q2a/Q2b: OR for sanctions 15% vs 5%
# -----------------------------
# OR(15 vs 5) = exp(beta_15 - beta_5)
b <- coef(m_add)
coef_names <- names(b)
print(coef_names)

# coefficient names will look like: sanctions15% and sanctions5%
name_5  <- grep("^sanctions5%$",  coef_names, value = TRUE)
name_15 <- grep("^sanctions15%$", coef_names, value = TRUE)

if (length(name_5) == 0 | length(name_15) == 0) {
  stop("Cannot find sanctions coefficients for 5% and 15%. coef names: ",
       paste(coef_names, collapse = ", "))
}

OR_15_vs_5 <- exp(b[name_15] - b[name_5])

cat("\n========================\n")
cat("Q2a/Q2b) OR for sanctions 15% vs 5%\n")
cat("========================\n")
cat("OR(15% vs 5%) = ", OR_15_vs_5, "\n", sep = "")

# -----------------------------
# 6) Q2c: Predicted probability at countries=80 of 192, sanctions=None
# -----------------------------
newdat <- data.frame(
  countries = factor("80 of 192", levels = levels(climateSupport$countries)),
  sanctions = factor("None", levels = levels(climateSupport$sanctions))
)

p_hat <- predict(m_add, newdata = newdat, type = "response")

cat("\n========================\n")
cat("Q2c) Predicted probability at countries=80 of 192, sanctions=None\n")
cat("========================\n")
cat("Predicted probability = ", p_hat, "\n", sep = "")

# -----------------------------
# 7) Q3: Interaction model + LR test
# -----------------------------
m_int <- glm(choice01 ~ countries * sanctions, data = climateSupport, family = binomial(link = "logit"))

cat("\n========================\n")
cat("Q3) LR test (additive vs interaction)\n")
cat("========================\n")
lr_int <- anova(m_add, m_int, test = "Chisq")
print(lr_int)

# -----------------------------
# 8) Save outputs
# -----------------------------
out_dir <- file.path(base_dir, "results")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

write.csv(broom::tidy(m_add), file.path(out_dir, "model_additive_tidy.csv"), row.names = FALSE)
write.csv(broom::tidy(m_int), file.path(out_dir, "model_interaction_tidy.csv"), row.names = FALSE)

sink(file.path(out_dir, "key_results.txt"))
cat("Data path: ", data_path, "\n", sep = "")
cat("\nAdditive model: choice01 ~ countries + sanctions\n\n")
cat("Global LR test (null vs additive):\n")
print(lr_global)
cat("\nOR(15% vs 5%) = ", OR_15_vs_5, "\n", sep = "")
cat("Predicted P(choice=1) at (80 of 192, None) = ", p_hat, "\n", sep = "")
cat("\nLR test (additive vs interaction):\n")
print(lr_int)
sink()

message("Done. Saved outputs to: ", out_dir)
