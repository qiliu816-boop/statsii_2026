rm(list = ls())
options(stringsAsFactors = FALSE)

# 0) Packages
pkgs <- c("nnet", "MASS", "dplyr", "readr", "broom")
to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_install) > 0) install.packages(to_install, dependencies = TRUE)
invisible(lapply(pkgs, library, character.only = TRUE))

# 1) Robust path
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  script_path <- sub(file_arg, "", args[grep(file_arg, args)])
  
  if (length(script_path) > 0) {
    return(normalizePath(dirname(script_path), winslash = "/", mustWork = FALSE))
  }
  
  return(normalizePath(getwd(), winslash = "/", mustWork = FALSE))
}

script_dir <- get_script_dir()
cat("Current script directory:\n", script_dir, "\n\n")

# 2) File paths

gdp_file <- file.path(script_dir, "gdpChange.csv")
mex_file <- file.path(script_dir, "MexicoMuniData.csv")

if (!file.exists(gdp_file)) stop("gdpChange.csv not found in current folder.")
if (!file.exists(mex_file)) stop("MexicoMuniData.csv not found in current folder.")

# QUESTION 1
cat("QUESTION 1\n")
# 3) Read GDP data
gdp <- read.csv(gdp_file)

cat("Preview of gdp data:\n")
print(head(gdp))
cat("\nVariable names:\n")
print(names(gdp))
cat("\n")

cat("Summary of original numeric GDPWdiff:\n")
print(summary(gdp$GDPWdiff))
cat("\n")

# 4) Create categorical GDPWdiff
gdp$GDPWdiff_cat <- ifelse(
  gdp$GDPWdiff < 0, "negative",
  ifelse(gdp$GDPWdiff == 0, "no change", "positive")
)

cat("Frequency table for constructed GDPWdiff_cat:\n")
print(table(gdp$GDPWdiff_cat, useNA = "ifany"))
cat("\n")

# Keep needed variables
gdp_q1 <- gdp %>%
  dplyr::select(GDPWdiff_cat, REG, OIL) %>%
  na.omit()

cat("Question 1 sample size after NA removal:", nrow(gdp_q1), "\n\n")

# 5) Unordered multinomial logit
gdp_q1$GDPWdiff_cat <- factor(
  gdp_q1$GDPWdiff_cat,
  levels = c("no change", "positive", "negative")
)

gdp_q1$GDPWdiff_cat <- relevel(gdp_q1$GDPWdiff_cat, ref = "no change")

cat("Levels used in multinomial model:\n")
print(levels(gdp_q1$GDPWdiff_cat))
cat("\n")

cat("Counts used in multinomial model:\n")
print(table(gdp_q1$GDPWdiff_cat))
cat("\n")

m_multinom <- nnet::multinom(GDPWdiff_cat ~ REG + OIL, data = gdp_q1, trace = FALSE)

cat("-Unordered Multinomial Logit-\n")
print(summary(m_multinom))
cat("\n")

# Extract coefficients, SE, z, p
multi_coef <- summary(m_multinom)$coefficients
multi_se   <- summary(m_multinom)$standard.errors
multi_z    <- multi_coef / multi_se
multi_p    <- 2 * (1 - pnorm(abs(multi_z)))

multi_results <- data.frame(
  comparison = rep(rownames(multi_coef), each = ncol(multi_coef)),
  term = rep(colnames(multi_coef), times = nrow(multi_coef)),
  estimate = as.vector(t(multi_coef)),
  std.error = as.vector(t(multi_se)),
  z.value = as.vector(t(multi_z)),
  p.value = as.vector(t(multi_p))
)

cat("Tidy multinomial results:\n")
print(multi_results)
cat("\n")

# Predicted probabilities
new_multinom <- expand.grid(
  REG = c(0, 1),
  OIL = c(0, 1)
)

pred_multi <- predict(m_multinom, newdata = new_multinom, type = "probs")
pred_multi_df <- cbind(new_multinom, as.data.frame(pred_multi))

cat("Predicted probabilities from multinomial logit:\n")
print(pred_multi_df)
cat("\n")

# 6) Ordered logit
gdp_q1$GDPWdiff_ord <- ordered(
  gdp_q1$GDPWdiff_cat,
  levels = c("negative", "no change", "positive")
)

cat("Levels used in ordered logit:\n")
print(levels(gdp_q1$GDPWdiff_ord))
cat("\n")

m_ordered <- MASS::polr(GDPWdiff_ord ~ REG + OIL, data = gdp_q1, Hess = TRUE)

cat("-Ordered Logit-\n")
print(summary(m_ordered))
cat("\n")

ord_ctable <- coef(summary(m_ordered))
ord_p <- 2 * (1 - pnorm(abs(ord_ctable[, "t value"])))
ord_results <- cbind(ord_ctable, "p.value" = ord_p)

cat("Ordered logit results (coefficients + cutoffs):\n")
print(ord_results)
cat("\n")

pred_ord <- predict(m_ordered, newdata = new_multinom, type = "probs")
pred_ord_df <- cbind(new_multinom, as.data.frame(pred_ord))

cat("Predicted probabilities from ordered logit:\n")
print(pred_ord_df)
cat("\n")

# QUESTION 2
cat("QUESTION 2\n")

# 7) Read Mexico data
mex <- read.csv(mex_file)

cat("Preview of Mexico data:\n")
print(head(mex))
cat("\nVariable names:\n")
print(names(mex))
cat("\n")

# 8) Keep required variables
mex_q2 <- mex %>%
  dplyr::select(PAN.visits.06, competitive.district, marginality.06, PAN.governor.06) %>%
  na.omit()

cat("Question 2 sample size after NA removal:", nrow(mex_q2), "\n\n")

cat("Summary of variables used in Question 2:\n")
print(summary(mex_q2))
cat("\n")

# 9) Poisson regression
m_pois <- glm(
  PAN.visits.06 ~ competitive.district + marginality.06 + PAN.governor.06,
  family = poisson(link = "log"),
  data = mex_q2
)

cat("Poisson Regression\n")
print(summary(m_pois))
cat("\n")

pois_coef <- summary(m_pois)$coefficients
cat("Poisson coefficient table:\n")
print(pois_coef)
cat("\n")

# 10) Q2(a): test for competitive.district
comp_est <- pois_coef["competitive.district", "Estimate"]
comp_se  <- pois_coef["competitive.district", "Std. Error"]
comp_z   <- pois_coef["competitive.district", "z value"]
comp_p   <- pois_coef["competitive.district", "Pr(>|z|)"]

cat("Q2(a) Test for competitive.district:\n")
cat("Estimate =", comp_est, "\n")
cat("Std. Error =", comp_se, "\n")
cat("z value =", comp_z, "\n")
cat("p value =", comp_p, "\n\n")

# 11) Q2(b): IRR
irr <- exp(coef(m_pois))
cat("Incidence Rate Ratios (IRR = exp(beta)):\n")
print(irr)
cat("\n")

cat("IRR for marginality.06 =", irr["marginality.06"], "\n")
cat("IRR for PAN.governor.06 =", irr["PAN.governor.06"], "\n\n")

# 12) Q2(c): predicted mean count
new_pois <- data.frame(
  competitive.district = 1,
  marginality.06 = 0,
  PAN.governor.06 = 1
)

pred_visits <- predict(m_pois, newdata = new_pois, type = "response")

cat("Q2(c) Predicted mean number of visits:\n")
print(pred_visits)
cat("\n")

# Manual check
b <- coef(m_pois)
lambda_manual <- exp(
  b["(Intercept)"] +
    b["competitive.district"] * 1 +
    b["marginality.06"] * 0 +
    b["PAN.governor.06"] * 1
)

cat("Manual calculation of predicted mean:\n")
print(lambda_manual)
cat("\n")

# 13)overdispersion check
dispersion <- sum(residuals(m_pois, type = "pearson")^2) / m_pois$df.residual
cat("Optional overdispersion statistic (Pearson chi-square / df):\n")
print(dispersion)
cat("\n")
