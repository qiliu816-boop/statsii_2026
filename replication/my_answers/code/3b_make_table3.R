rm(list = ls())

# Packages
library(WeightIt)
library(marginaleffects)
library(cobalt)
library(tidyverse)
library(ivreg)
library(ivDiag)
library(tayloRswift)

# Load processed data
load("./data/processed_data/weighting.RData")

# Equivalence bound
eqb <- .36 * sd(df$chinaecon, na.rm = TRUE)

# Common covariates
covs <- "networks + factor(fanlength) + factor(fanintensity) +
         factor(ideo5_new) + factor(pid3_new) + age + factor(gender)"

# China economic expansion
china_fit1 <- lm(scale(chinaecon) ~ treat1, data = df, weights = wt1)
china_fit1c <- update(china_fit1, paste(". ~ . +", covs))
china_res <- avg_comparisons(
  china_fit1c,
  variables = "treat1",
  vcov = "HC3",
  wts = "wt1",
  equivalence = c(-eqb, eqb)
) |> as.data.frame()

# Pre-sale political activities
preact_fit1 <- lm(scale(engagepre_count) ~ treat1, data = df, weights = wt1)
preact_fit1c <- update(preact_fit1, paste(". ~ . +", covs))
preact_res <- avg_comparisons(
  preact_fit1c,
  variables = "treat1",
  vcov = "HC3",
  wts = "wt1",
  equivalence = c(-eqb, eqb)
) |> as.data.frame()

# Check column names
print(names(china_res))
print(names(preact_res))

# Keep only columns that actually exist
keep_cols <- intersect(
  c("term", "estimate", "std.error", "conf.low", "conf.high", "p.value.equiv"),
  names(china_res)
)


# Extract rows
china_rows <- intersect(c(1, 4), seq_len(nrow(china_res)))
preact_rows <- intersect(c(1, 4), seq_len(nrow(preact_res)))

table3_out <- bind_rows(
  china_res[china_rows, keep_cols],
  preact_res[preact_rows, keep_cols]
)

# Add outcome label if useful
table3_out <- table3_out |> mutate(
  outcome = c(
    rep("chinaecon", length(china_rows)),
    rep("engagepre_count", length(preact_rows))
  ),
  .before = 1
)

# Write table
write.table(
  table3_out,
  file = "./tables/table3.txt",
  row.names = FALSE,
  quote = FALSE,
  sep = "\t"
)

cat("table3.txt has been created in ./tables/\n")