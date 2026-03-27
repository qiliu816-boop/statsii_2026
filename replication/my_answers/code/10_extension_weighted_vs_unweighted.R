rm(list = ls())

library(WeightIt)
library(marginaleffects)
library(cobalt)
library(tidyverse)
library(ivreg)
library(ivDiag)
library(tayloRswift)

load("./data/processed_data/weighting.RData")

pick_cols <- c("estimate", "std.error", "conf.low", "conf.high", "p.value")

extract_res <- function(x, outcome, model){
  out <- as.data.frame(x)[, pick_cols]
  out$outcome <- outcome
  out$model <- model
  out
}

# 1) Ticketmaster monopoly
fit_mono_w <- lm(scale(monopoliestm) ~ treat1, data = df, weights = wt1)
fit_mono_u <- lm(scale(monopoliestm) ~ treat1, data = df)

res_mono_w <- avg_comparisons(fit_mono_w, variables = "treat1", vcov = "HC3", wts = "wt1")
res_mono_u <- avg_comparisons(fit_mono_u, variables = "treat1", vcov = "HC3")

# 2) Ticketmaster inequality
fit_ineq_w <- lm(scale(ineqtm) ~ treat1, data = df, weights = wt1)
fit_ineq_u <- lm(scale(ineqtm) ~ treat1, data = df)

res_ineq_w <- avg_comparisons(fit_ineq_w, variables = "treat1", vcov = "HC3", wts = "wt1")
res_ineq_u <- avg_comparisons(fit_ineq_u, variables = "treat1", vcov = "HC3")

# 3) Total post-sale activities
fit_act_w <- lm(scale(engagepost_count) ~ treat1, data = df, weights = wt1)
fit_act_u <- lm(scale(engagepost_count) ~ treat1, data = df)

res_act_w <- avg_comparisons(fit_act_w, variables = "treat1", vcov = "HC3", wts = "wt1")
res_act_u <- avg_comparisons(fit_act_u, variables = "treat1", vcov = "HC3")

# 4) Filed FTC report
fit_ftc_w <- lm(ftc_completion_strict ~ treat1, data = df, weights = wt1)
fit_ftc_u <- lm(ftc_completion_strict ~ treat1, data = df)

res_ftc_w <- avg_comparisons(fit_ftc_w, variables = "treat1", vcov = "HC3", wts = "wt1")
res_ftc_u <- avg_comparisons(fit_ftc_u, variables = "treat1", vcov = "HC3")

ext_results <- bind_rows(
  extract_res(res_mono_w, "Ticketmaster monopoly", "Weighted"),
  extract_res(res_mono_u, "Ticketmaster monopoly", "Unweighted"),
  extract_res(res_ineq_w, "Ticketmaster inequality", "Weighted"),
  extract_res(res_ineq_u, "Ticketmaster inequality", "Unweighted"),
  extract_res(res_act_w,  "Total post-sale activities", "Weighted"),
  extract_res(res_act_u,  "Total post-sale activities", "Unweighted"),
  extract_res(res_ftc_w,  "Filed FTC report", "Weighted"),
  extract_res(res_ftc_u,  "Filed FTC report", "Unweighted")
) %>%
  select(outcome, model, everything())

print(ext_results)

write.csv(ext_results,
          "./tables/extension_weighted_vs_unweighted.csv",
          row.names = FALSE)

p_ext <- ggplot(ext_results,
                aes(x = model, y = estimate,
                    ymin = conf.low, ymax = conf.high)) +
  geom_pointrange() +
  facet_wrap(~ outcome, scales = "free_y") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  coord_flip() +
  theme_minimal()

ggsave("./graphs/extension_weighted_vs_unweighted.pdf",
       p_ext, width = 10, height = 7)