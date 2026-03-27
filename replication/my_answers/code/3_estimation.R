#################################################################################
# Replication file for: "The Development of an Issue Public: Evidence from      #
# The Eras Tour"                                                                #
#                                                                               #
# Erin L. Rossiter                                                              #
# University of Notre Dame                                                      #
# erossite@nd.edu                                                               #
#                                                                               #
# Jeffrey J. Harden                                                             #
# University of Notre Dame                                                      #
# jeff.harden@nd.edu                                                            #
#                                                                               #
# This file estimates treatment effects.                                        #
#                                                                               #
# Last update: May 20, 2024                                                     #  
#################################################################################

## Packages
library(WeightIt)
library(marginaleffects)
library(cobalt)
library(tidyverse)
library(ivreg)
library(ivDiag)
library(tayloRswift)

load("./data/processed_data/weighting.RData")

## Placebo outcomes----
# eqb = .36\sigma from Hartman and Hidalgo (2018, 1006)
# Economic expansion of China
eqb <- .36 # *sd(df$chinaecon, na.rm = TRUE) 
covs <- "networks + factor(fanlength) + factor(fanintensity) +
         factor(ideo5_new) + factor(pid3_new) + age + factor(gender)"

china_fit1 <- lm(scale(chinaecon) ~ treat1, data = df, weights = wt1)
china_fit1c <- update(china_fit1, paste(". ~ . +", covs))
china_est1 <- avg_comparisons(china_fit1c, variables = "treat1",
                              vcov = "HC3", wts = "wt1",
                              equivalence = c(-eqb, eqb))

china_fit2 <- lm(scale(chinaecon) ~ treat2, data = df, weights = wt2)
china_fit2c <- update(china_fit2, paste(". ~ . +", covs))
china_est2 <- avg_comparisons(china_fit2c, variables = "treat2",
                              vcov = "HC3", wts = "wt2",
                              equivalence = c(-eqb, eqb))

china_fit3 <- lm(scale(chinaecon) ~ treat3, data = df, weights = wt3)
china_fit3c <- update(china_fit3, paste(". ~ . +", covs))
china_est3 <- avg_comparisons(china_fit3c, variables = list(treat3 = "2sd"),
                              vcov = "HC3", wts = "wt3",
                              equivalence = c(-eqb, eqb))

china_fit4 <- ivreg(scale(chinaecon) ~ treat4 | treat1, data = df, weights = wt1)
china_fit4c <- ivreg(scale(chinaecon) ~ treat4 + networks + factor(fanlength) +
                                        factor(fanintensity) + factor(ideo5_new) +
                                        factor(pid3_new) + age + factor(gender) |
                                        treat1 + networks + factor(fanlength) +
                                        factor(fanintensity) + factor(ideo5_new) +
                                        factor(pid3_new) + age + factor(gender),
                     data = df, weights = wt1)
china_est4 <- avg_comparisons(china_fit4c, variables = "treat4",
                              vcov = "HC3", wts = "wt1",
                              equivalence = c(-eqb, eqb))

china_res <- rbind(china_est1, china_est2, china_est3, china_est4)

# 2022 political activities
eqb <- .36 # *sd(df$engagepre_count, na.rm = TRUE)

preact_fit1 <- lm(scale(engagepre_count) ~ treat1, data = subset(df, !is.na(engagepre_1)), weights = wt1)
preact_fit1c <- update(preact_fit1, paste(". ~ . +", covs))
preact_est1 <- avg_comparisons(preact_fit1c, variables = "treat1",
                               vcov = "HC3", wts = "wt1",
                               equivalence = c(-eqb, eqb))

preact_fit2 <- lm(scale(engagepre_count) ~ treat2, data = subset(df, !is.na(engagepre_1)), weights = wt2)
preact_fit2c <- update(preact_fit2, paste(". ~ . +", covs))
preact_est2 <- avg_comparisons(preact_fit2c, variables = "treat2",
                               vcov = "HC3", wts = "wt2",
                               equivalence = c(-eqb, eqb))

preact_fit3 <- lm(scale(engagepre_count) ~ treat3, data = subset(df, !is.na(engagepre_1)), weights = wt3)
preact_fit3c <- update(preact_fit3, paste(". ~ . +", covs))
preact_est3 <- avg_comparisons(preact_fit3c, variables = list(treat3 = "2sd"),
                               vcov = "HC3", wts = "wt3",
                               equivalence = c(-eqb, eqb))

preact_fit4 <- ivreg(scale(engagepre_count) ~ treat4 | treat1, data = subset(df, !is.na(engagepre_1)), weights = wt1)
preact_fit4c <- ivreg(scale(engagepre_count) ~ treat4 + networks + factor(fanlength) +
                                               factor(fanintensity) + factor(ideo5_new) +
                                               factor(pid3_new) + age + factor(gender) |
                                               treat1 + networks + factor(fanlength) +
                                               factor(fanintensity) + factor(ideo5_new) +
                                               factor(pid3_new) + age + factor(gender),
                      data = subset(df, !is.na(engagepre_1)), weights = wt1)
preact_est4 <- avg_comparisons(preact_fit4c, variables = "treat4",
                               vcov = "HC3", wts = "wt1",
                               equivalence = c(-eqb, eqb))

preact_res <- rbind(preact_est1, preact_est2, preact_est3, preact_est4)

## Attitude differences outcomes----
# Ticketmaster monopoly
eqb <- .36 # *sd(df$monopoliestm, na.rm = TRUE)

montmd_fit1 <- lm(scale(monopoliestm) ~ treat1, data = df, weights = wt1)
montmd_fit1c <- update(montmd_fit1, paste(". ~ . +", covs))
montmd_est1 <- avg_comparisons(montmd_fit1c, variables = "treat1",
                               vcov = "HC3", wts = "wt1",
                               equivalence = c(-eqb, eqb))

montmd_fit2 <- lm(scale(monopoliestm) ~ treat2, data = df, weights = wt2)
montmd_fit2c <- update(montmd_fit2, paste(". ~ . +", covs))
montmd_est2 <- avg_comparisons(montmd_fit2c, variables = "treat2",
                               vcov = "HC3", wts = "wt2",
                               equivalence = c(-eqb, eqb))

montmd_fit3 <- lm(scale(monopoliestm) ~ treat3, data = df, weights = wt3)
montmd_fit3c <- update(montmd_fit3, paste(". ~ . +", covs))
montmd_est3 <- avg_comparisons(montmd_fit3c, variables = list(treat3 = "2sd"),
                               vcov = "HC3", wts = "wt3",
                               equivalence = c(-eqb, eqb))

montmd_fit4 <- ivreg(scale(monopoliestm) ~ treat4 | treat1, data = df, weights = wt1)
montmd_fit4c <- ivreg(scale(monopoliestm) ~ treat4 + networks + factor(fanlength) +
                                            factor(fanintensity) + factor(ideo5_new) +
                                            factor(pid3_new) + age + factor(gender) |
                                            treat1 + networks + factor(fanlength) +
                                            factor(fanintensity) + factor(ideo5_new) +
                                            factor(pid3_new) + age + factor(gender),
                      data = df, weights = wt1)
montmd_est4 <- avg_comparisons(montmd_fit4c, variables = "treat4",
                               vcov = "HC3", wts = "wt1",
                               equivalence = c(-eqb, eqb))

montmd_res <- rbind(montmd_est1, montmd_est2, montmd_est3, montmd_est4)

# Ticketmaster distribution inequality
eqb <- .36 # *sd(df$ineqtm, na.rm = TRUE)

ineqtmd_fit1 <- lm(scale(ineqtm) ~ treat1, data = df, weights = wt1)
ineqtmd_fit1c <- update(ineqtmd_fit1, paste(". ~ . +", covs))
ineqtmd_est1 <- avg_comparisons(ineqtmd_fit1c, variables = "treat1",
                                vcov = "HC3", wts = "wt1",
                                equivalence = c(-eqb, eqb))

ineqtmd_fit2 <- lm(scale(ineqtm) ~ treat2, data = df, weights = wt2)
ineqtmd_fit2c <- update(ineqtmd_fit2, paste(". ~ . +", covs))
ineqtmd_est2 <- avg_comparisons(ineqtmd_fit2c, variables = "treat2",
                                vcov = "HC3", wts = "wt2",
                                equivalence = c(-eqb, eqb))

ineqtmd_fit3 <- lm(scale(ineqtm) ~ treat3, data = df, weights = wt3)
ineqtmd_fit3c <- update(ineqtmd_fit3, paste(". ~ . +", covs))
ineqtmd_est3 <- avg_comparisons(ineqtmd_fit3c, variables = list(treat3 = "2sd"),
                                vcov = "HC3", wts = "wt3",
                                equivalence = c(-eqb, eqb))

ineqtmd_fit4 <- ivreg(scale(ineqtm) ~ treat4 | treat1, data = df, weights = wt1)
ineqtmd_fit4c <- ivreg(scale(ineqtm) ~ treat4 + networks + factor(fanlength) +
                                       factor(fanintensity) + factor(ideo5_new) +
                                       factor(pid3_new) + age + factor(gender) |
                                       treat1 + networks + factor(fanlength) +
                                       factor(fanintensity) + factor(ideo5_new) +
                                       factor(pid3_new) + age + factor(gender),
                       data = df, weights = wt1)
ineqtmd_est4 <- avg_comparisons(ineqtmd_fit4c, variables = "treat4",
                                vcov = "HC3", wts = "wt1",
                                equivalence = c(-eqb, eqb))

ineqtmd_res <- rbind(ineqtmd_est1, ineqtmd_est2, ineqtmd_est3, ineqtmd_est4)

# General monopolies
eqb <- .36 # *sd(df$monopoliesgen, na.rm = TRUE)

mongend_fit1 <- lm(scale(monopoliesgen) ~ treat1, data = df, weights = wt1)
mongend_fit1c <- update(mongend_fit1, paste(". ~ . +", covs))
mongend_est1 <- avg_comparisons(mongend_fit1c, variables = "treat1",
                                vcov = "HC3", wts = "wt1",
                                equivalence = c(-eqb, eqb))

mongend_fit2 <- lm(scale(monopoliesgen) ~ treat2, data = df, weights = wt2)
mongend_fit2c <- update(mongend_fit2, paste(". ~ . +", covs))
mongend_est2 <- avg_comparisons(mongend_fit2c, variables = "treat2",
                                vcov = "HC3", wts = "wt2",
                                equivalence = c(-eqb, eqb))

mongend_fit3 <- lm(scale(monopoliesgen) ~ treat3, data = df, weights = wt3)
mongend_fit3c <- update(mongend_fit3, paste(". ~ . +", covs))
mongend_est3 <- avg_comparisons(mongend_fit3c, variables = list(treat3 = "2sd"),
                                vcov = "HC3", wts = "wt3",
                                equivalence = c(-eqb, eqb))

mongend_fit4 <- ivreg(scale(monopoliesgen) ~ treat4 | treat1, data = df, weights = wt1)
mongend_fit4c <- ivreg(scale(monopoliesgen) ~ treat4 + networks + factor(fanlength) +
                                              factor(fanintensity) + factor(ideo5_new) +
                                              factor(pid3_new) + age + factor(gender) |
                                              treat1 + networks + factor(fanlength) +
                                              factor(fanintensity) + factor(ideo5_new) +
                                              factor(pid3_new) + age + factor(gender),
                       data = df, weights = wt1)
mongend_est4 <- avg_comparisons(mongend_fit4c, variables = "treat4",
                                vcov = "HC3", wts = "wt1",
                                equivalence = c(-eqb, eqb))

mongend_res <- rbind(mongend_est1, mongend_est2, mongend_est3, mongend_est4)

# General inequality
eqb <- .36 # *sd(df$ineqgen, na.rm = TRUE)

ineqgend_fit1 <- lm(scale(ineqgen) ~ treat1, data = df, weights = wt1)
ineqgend_fit1c <- update(ineqgend_fit1, paste(". ~ . +", covs))
ineqgend_est1 <- avg_comparisons(ineqgend_fit1c, variables = "treat1",
                                 vcov = "HC3", wts = "wt1",
                                 equivalence = c(-eqb, eqb))

ineqgend_fit2 <- lm(scale(ineqgen) ~ treat2, data = df, weights = wt2)
ineqgend_fit2c <- update(ineqgend_fit2, paste(". ~ . +", covs))
ineqgend_est2 <- avg_comparisons(ineqgend_fit2c, variables = "treat2",
                                 vcov = "HC3", wts = "wt2",
                                 equivalence = c(-eqb, eqb))

ineqgend_fit3 <- lm(scale(ineqgen) ~ treat3, data = df, weights = wt3)
ineqgend_fit3c <- update(ineqgend_fit3, paste(". ~ . +", covs))
ineqgend_est3 <- avg_comparisons(ineqgend_fit3c, variables = list(treat3 = "2sd"),
                                 vcov = "HC3", wts = "wt3",
                                 equivalence = c(-eqb, eqb))

ineqgend_fit4 <- ivreg(scale(ineqgen) ~ treat4 | treat1, data = df, weights = wt1)
ineqgend_fit4c <- ivreg(scale(ineqgen) ~ treat4 + networks + factor(fanlength) +
                                         factor(fanintensity) + factor(ideo5_new) +
                                         factor(pid3_new) + age + factor(gender) |
                                         treat1 + networks + factor(fanlength) +
                                         factor(fanintensity) + factor(ideo5_new) +
                                         factor(pid3_new) + age + factor(gender),
                        data = df, weights = wt1)
ineqgend_est4 <- avg_comparisons(ineqgend_fit4c, variables = "treat4",
                                 vcov = "HC3", wts = "wt1",
                                 equivalence = c(-eqb, eqb))

ineqgend_res <- rbind(ineqgend_est1, ineqgend_est2, ineqgend_est3, ineqgend_est4)

## Behavioral outcomes----
# Contact a government official
eqb <- .36*sd(df$engagepost_3, na.rm = TRUE)

postgov_fit1 <- lm(engagepost_3 ~ treat1, data = df, weights = wt1)
postgov_fit1c <- update(postgov_fit1, paste(". ~ . +", covs))
postgov_est1 <- avg_comparisons(postgov_fit1c, variables = "treat1",
                                vcov = "HC3", wts = "wt1",
                                equivalence = c(-eqb, eqb))

postgov_fit2 <- lm(engagepost_3 ~ treat2, data = df, weights = wt2)
postgov_fit2c <- update(postgov_fit2, paste(". ~ . +", covs))
postgov_est2 <- avg_comparisons(postgov_fit2c, variables = "treat2",
                                vcov = "HC3", wts = "wt2",
                                equivalence = c(-eqb, eqb))

postgov_fit3 <- lm(engagepost_3 ~ treat3, data = df, weights = wt3)
postgov_fit3c <- update(postgov_fit3, paste(". ~ . +", covs))
postgov_est3 <- avg_comparisons(postgov_fit3c, variables = list(treat3 = "2sd"),
                                vcov = "HC3", wts = "wt3",
                                equivalence = c(-eqb, eqb))

postgov_fit4 <- ivreg(engagepost_3 ~ treat4 | treat1, data = df, weights = wt1)
postgov_fit4c <- ivreg(engagepost_3 ~ treat4 + networks + factor(fanlength) +
                                      factor(fanintensity) + factor(ideo5_new) +
                                      factor(pid3_new) + age + factor(gender) |
                                      treat1 + networks + factor(fanlength) +
                                      factor(fanintensity) + factor(ideo5_new) +
                                      factor(pid3_new) + age + factor(gender),
                       data = df, weights = wt1)
postgov_est4 <- avg_comparisons(postgov_fit4c, variables = "treat4",
                                vcov = "HC3", wts = "wt1",
                                equivalence = c(-eqb, eqb))

postgov_res <- rbind(postgov_est1, postgov_est2, postgov_est3, postgov_est4)

# Post or comment online
eqb <- .36*sd(df$engagepost_1, na.rm = TRUE)

postcom_fit1 <- lm(engagepost_1 ~ treat1, data = df, weights = wt1)
postcom_fit1c <- update(postcom_fit1, paste(". ~ . +", covs))
postcom_est1 <- avg_comparisons(postcom_fit1c, variables = "treat1",
                                vcov = "HC3", wts = "wt1",
                                equivalence = c(-eqb, eqb))

postcom_fit2 <- lm(engagepost_1 ~ treat2, data = df, weights = wt2)
postcom_fit2c <- update(postcom_fit2, paste(". ~ . +", covs))
postcom_est2 <- avg_comparisons(postcom_fit2c, variables = "treat2",
                                vcov = "HC3", wts = "wt2",
                                equivalence = c(-eqb, eqb))

postcom_fit3 <- lm(engagepost_1 ~ treat3, data = df, weights = wt3)
postcom_fit3c <- update(postcom_fit3, paste(". ~ . +", covs))
postcom_est3 <- avg_comparisons(postcom_fit3c, variables = list(treat3 = "2sd"),
                                vcov = "HC3", wts = "wt3",
                                equivalence = c(-eqb, eqb))

postcom_fit4 <- ivreg(engagepost_1 ~ treat4 | treat1, data = df, weights = wt1)
postcom_fit4c <- ivreg(engagepost_1 ~ treat4 + networks + factor(fanlength) +
                                      factor(fanintensity) + factor(ideo5_new) +
                                      factor(pid3_new) + age + factor(gender) |
                                      treat1 + networks + factor(fanlength) +
                                      factor(fanintensity) + factor(ideo5_new) +
                                      factor(pid3_new) + age + factor(gender),
                       data = df, weights = wt1)
postcom_est4 <- avg_comparisons(postcom_fit4c, variables = "treat4",
                                vcov = "HC3", wts = "wt1",
                                equivalence = c(-eqb, eqb))

postcom_res <- rbind(postcom_est1, postcom_est2, postcom_est3, postcom_est4)

# Talk to other people
eqb <- .36*sd(df$engagepost_2, na.rm = TRUE)

posttalk_fit1 <- lm(engagepost_2 ~ treat1, data = df, weights = wt1)
posttalk_fit1c <- update(posttalk_fit1, paste(". ~ . +", covs))
posttalk_est1 <- avg_comparisons(posttalk_fit1c, variables = "treat1",
                                 vcov = "HC3", wts = "wt1",
                                 equivalence = c(-eqb, eqb))

posttalk_fit2 <- lm(engagepost_2 ~ treat2, data = df, weights = wt2)
posttalk_fit2c <- update(posttalk_fit2, paste(". ~ . +", covs))
posttalk_est2 <- avg_comparisons(posttalk_fit2c, variables = "treat2",
                                 vcov = "HC3", wts = "wt2",
                                 equivalence = c(-eqb, eqb))

posttalk_fit3 <- lm(engagepost_2 ~ treat3, data = df, weights = wt3)
posttalk_fit3c <- update(posttalk_fit3, paste(". ~ . +", covs))
posttalk_est3 <- avg_comparisons(posttalk_fit3c, variables = list(treat3 = "2sd"),
                                 vcov = "HC3", wts = "wt3",
                                 equivalence = c(-eqb, eqb))

posttalk_fit4 <- ivreg(engagepost_2 ~ treat4 | treat1, data = df, weights = wt1)
posttalk_fit4c <- ivreg(engagepost_2 ~ treat4 + networks + factor(fanlength) +
                                       factor(fanintensity) + factor(ideo5_new) +
                                       factor(pid3_new) + age + factor(gender) |
                                       treat1 + networks + factor(fanlength) +
                                       factor(fanintensity) + factor(ideo5_new) +
                                       factor(pid3_new) + age + factor(gender),
                        data = df, weights = wt1)
posttalk_est4 <- avg_comparisons(posttalk_fit4c, variables = "treat4",
                                 vcov = "HC3", wts = "wt1",
                                 equivalence = c(-eqb, eqb))

posttalk_res <- rbind(posttalk_est1, posttalk_est2, posttalk_est3, posttalk_est4)

# Count of post-sale activities
eqb <- .36 # *sd(df$engagepost_count, na.rm = TRUE)

postact_fit1 <- lm(scale(engagepost_count) ~ treat1, data = subset(df, !is.na(engagepost_1)), weights = wt1)
postact_fit1c <- update(postact_fit1, paste(". ~ . +", covs))
postact_est1 <- avg_comparisons(postact_fit1c, variables = "treat1",
                                 vcov = "HC3", wts = "wt1",
                                 equivalence = c(-eqb, eqb))

postact_fit2 <- lm(scale(engagepost_count) ~ treat2, data = subset(df, !is.na(engagepost_1)), weights = wt2)
postact_fit2c <- update(postact_fit2, paste(". ~ . +", covs))
postact_est2 <- avg_comparisons(postact_fit2c, variables = "treat2",
                                 vcov = "HC3", wts = "wt2",
                                 equivalence = c(-eqb, eqb))

postact_fit3 <- lm(scale(engagepost_count) ~ treat3, data = subset(df, !is.na(engagepost_1)), weights = wt3)
postact_fit3c <- update(postact_fit3, paste(". ~ . +", covs))
postact_est3 <- avg_comparisons(postact_fit3c, variables = list(treat3 = "2sd"),
                                 vcov = "HC3", wts = "wt3",
                                 equivalence = c(-eqb, eqb))

postact_fit4 <- ivreg(scale(engagepost_count) ~ treat4 | treat1, data = subset(df, !is.na(engagepost_1)), weights = wt1)
postact_fit4c <- ivreg(scale(engagepost_count) ~ treat4 + networks + factor(fanlength) +
                                                 factor(fanintensity) + factor(ideo5_new) +
                                                 factor(pid3_new) + age + factor(gender) |
                                                 treat1 + networks + factor(fanlength) +
                                                 factor(fanintensity) + factor(ideo5_new) +
                                                 factor(pid3_new) + age + factor(gender),
                        data = subset(df, !is.na(engagepost_1)), weights = wt1)
postact_est4 <- avg_comparisons(postact_fit4c, variables = "treat4",
                                 vcov = "HC3", wts = "wt1",
                                 equivalence = c(-eqb, eqb))

postact_res <- rbind(postact_est1, postact_est2, postact_est3, postact_est4)

# Completed FTC report
eqb <- .36*sd(df$ftc_completion_strict, na.rm = TRUE)

ftc_fit1 <- lm(ftc_completion_strict ~ treat1, data = df, weights = wt1)
ftc_fit1c <- update(ftc_fit1, paste(". ~ . +", covs))
ftc_est1 <- avg_comparisons(ftc_fit1c, variables = "treat1",
                            vcov = "HC3", wts = "wt1",
                            equivalence = c(-eqb, eqb))

ftc_fit2 <- lm(ftc_completion_strict ~ treat2, data = df, weights = wt2)
ftc_fit2c <- update(ftc_fit2, paste(". ~ . +", covs))
ftc_est2 <- avg_comparisons(ftc_fit2c, variables = "treat2",
                            vcov = "HC3", wts = "wt2",
                            equivalence = c(-eqb, eqb))

ftc_fit3 <- lm(ftc_completion_strict ~ treat3, data = df, weights = wt3)
ftc_fit3c <- update(ftc_fit3, paste(". ~ . +", covs))
ftc_est3 <- avg_comparisons(ftc_fit3c, variables = list(treat3 = "2sd"),
                            vcov = "HC3", wts = "wt3",
                            equivalence = c(-eqb, eqb))

ftc_fit4 <- ivreg(ftc_completion_strict ~ treat4 | treat1, data = df, weights = wt1)
ftc_fit4c <- ivreg(ftc_completion_strict ~ treat4 + networks + factor(fanlength) +
                                           factor(fanintensity) + factor(ideo5_new) +
                                           factor(pid3_new) + age + factor(gender) |
                                           treat1 + networks + factor(fanlength) +
                                           factor(fanintensity) + factor(ideo5_new) +
                                           factor(pid3_new) + age + factor(gender),
                   data = df, weights = wt1)
ftc_est4 <- avg_comparisons(ftc_fit4c, variables = "treat4",
                            vcov = "HC3", wts = "wt1",
                            equivalence = c(-eqb, eqb))

ftc_res <- rbind(ftc_est1, ftc_est2, ftc_est3, ftc_est4)

## IV diagnostics----
z_res <- lm(treat1 ~ networks + factor(fanlength) +
              factor(fanintensity) + factor(ideo5_new) +
              factor(pid3_new) + age + factor(gender), data = df)$residuals
d_res <- lm(treat4 ~ networks + factor(fanlength) +
              factor(fanintensity) + factor(ideo5_new) +
              factor(pid3_new) + age + factor(gender), data = df)$residuals
d_fs <- data.frame(z_res = z_res, d_res = d_res)
summary(lm(d_res ~ z_res, data = d_fs))

theme_set(theme_bw(base_size = 16))

ggplot(d_fs, aes(x = z_res, y = d_res)) +
  geom_point(alpha = .2) +
  geom_smooth(method = "lm") +
  scale_x_continuous(breaks = seq(-1, 1, .25)) +
  scale_y_continuous(breaks = seq(-1, 1, .25)) +
  annotate("text", x = 0, y = .2,
           label = expression(hat(beta)~" = 0.197; SE = 0.01"),
           size = 4, fontface = 2) +
  xlab("Residualized instrument") + ylab("Residualized treatment")
ggsave("./graphs/figA14.pdf", width = 20, height = 10, units = "cm")

# Weak instruments null: "the instrument is weak"; alternative: "the instrument is strong"
# Wu-Hausman null: "OLS and IV are equally consistent"; alternative: "IV is consistent, OLS is not"
iv_diag <- function(iv_model) round(summary(iv_model)$diagnostics[1:2, 3:4], 3)
lapply(list(china = china_fit4c, preact = preact_fit4c, montmd = montmd_fit4c, ineqtmd = ineqtmd_fit4c,
            mongend = mongend_fit4c, ineqgend = ineqgend_fit4c, postgov = postgov_fit4c, postcom = postcom_fit4c,
            posttalk = posttalk_fit4c, postact = postact_fit4c, ftc = ftc_fit4c), iv_diag)

# Effective F statistics
df_iv <- df
df_iv$fanlength <- factor(df_iv$fanlength)
df_iv$fanintensity <- factor(df_iv$fanintensity)
df_iv$ideo5_new <- factor(df_iv$ideo5_new)
df_iv$pid3_new <- factor(df_iv$pid3_new)
df_iv$gender <- factor(df_iv$gender)
outcomes <- c("chinaecon", "engagepre_count", "monopoliestm",
              "ineqtm", "monopoliesgen", "ineqgen", "engagepost_3",
              "engagepost_1", "engagepost_2", "engagepost_count",
              "ftc_completion_strict")
F_eff <- numeric(length(outcomes))
                        
for(i in 1:length(outcomes)){
F_eff[i] <- eff_F(data = df_iv, Y = outcomes[i], D = "treat4", Z = "treat1",
                  controls = c("networks", "fanlength", "fanintensity",
                               "ideo5_new", "pid3_new", "age", "gender"),
                  weights = "wt1")
}
range(round(F_eff, 0))

## Tables----
# Main text
diff_res <- as.data.frame(rbind(montmd_res[c(1, 4), c(1, 3, 4, 8, 9, 6)],
                                ineqtmd_res[c(1, 4), c(1, 3, 4, 8, 9, 6)],
                                mongend_res[c(1, 4), c(1, 3, 4, 8, 9, 6)],
                                ineqgend_res[c(1, 4), c(1, 3, 4, 8, 9, 6)]))
diff_res$outcome <- rep(c("Ticketmaster monopoly", "Ticketmaster inequality",
                          "General monopolies", "General inequality"), each = 2)
diff_res <- diff_res[colnames(diff_res)[c(7, 1:6)]]
diff_res$n <- rep(c(length(resid(montmd_fit1c)), length(resid(ineqtmd_fit1c)),
                    length(resid(mongend_fit1c)), length(resid(ineqgend_fit1c))), each = 2)
diff_res$p.value <- p.adjust(diff_res$p.value, method = "hochberg")

sink("./tables/table1.txt")
stargazer::stargazer(diff_res, type = "latex", summary = FALSE,
                     title = "Estimated Treatment Effects on Attitude Differences")
sink()

beh_res <- as.data.frame(rbind(postgov_res[1, c(1, 3, 4, 8, 9, 6)],
                               postcom_res[1, c(1, 3, 4, 8, 9, 6)],
                               posttalk_res[1, c(1, 3, 4, 8, 9, 6)],
                               postact_res[1, c(1, 3, 4, 8, 9, 6)],
                               ftc_res[c(1, 4), c(1, 3, 4, 8, 9, 6)]))
beh_res$outcome <- rep(c("Contacted gov. official", "Communicated online",
                         "Talked with friends", "Total post-sale activities (standardized)",
                         "Filed FTC report"), times = c(1, 1, 1, 1, 2))
beh_res <- beh_res[colnames(beh_res)[c(7, 1:6)]]
beh_res$n <- rep(c(length(resid(postgov_fit1c)), length(resid(postcom_fit1c)),
                   length(resid(posttalk_fit1c)), length(resid(postact_fit1c)),
                   length(resid(ftc_fit1c))), times = c(1, 1, 1, 1, 2))
beh_res <- beh_res[4:6, ]
beh_res$p.value <- p.adjust(beh_res$p.value, method = "hochberg")

sink("./tables/table2.txt")
stargazer::stargazer(beh_res, type = "latex", summary = FALSE,
                     title = "Estimated Treatment Effects on Behavioral Outcomes")
sink()

plac_res <- as.data.frame(rbind(china_res[c(1, 4), c(1, 3, 4, 8, 9, 17)],
                                preact_res[c(1, 4), c(1, 3, 4, 8, 9, 17)]))
plac_res$outcome <- rep(c("China economic expansion", "Pre-sale political activities"), each = 2)
plac_res <- plac_res[colnames(plac_res)[c(7, 1:6)]]
plac_res$n <- rep(c(length(resid(china_fit1c)), length(resid(preact_fit1c))), each = 2)

sink("./tables/table3.txt")
stargazer::stargazer(plac_res, type = "latex", summary = FALSE,
                     title = "Estimated Treatment Effects on Placebo Outcomes")
sink()

# Appendix
diff_res2 <- as.data.frame(rbind(montmd_res[c(2, 3), c(1, 3, 4, 8, 9, 6)],
                                 ineqtmd_res[c(2, 3), c(1, 3, 4, 8, 9, 6)],
                                 mongend_res[c(2, 3), c(1, 3, 4, 8, 9, 6)],
                                 ineqgend_res[c(2, 3), c(1, 3, 4, 8, 9, 6)]))
diff_res2$outcome <- rep(c("Ticketmaster monopoly", "Ticketmaster inequality",
                          "General monopolies", "General inequality"), each = 2)
diff_res2 <- diff_res2[colnames(diff_res2)[c(7, 1:6)]]
diff_res2$n <- rep(c(length(resid(montmd_fit2c)), length(resid(ineqtmd_fit2c)),
                     length(resid(mongend_fit2c)), length(resid(ineqgend_fit2c))), each = 2)
diff_res2$p.value <- p.adjust(diff_res2$p.value, method = "hochberg")

sink("./tables/tableA3.txt")
stargazer::stargazer(diff_res2, type = "latex", summary = FALSE,
                     title = "Estimated Treatment Effects on Attitude Differences with Alternative Treatments")
sink()

beh_res2 <- as.data.frame(ftc_res[c(2, 3), c(1, 3, 4, 8, 9, 6)])
beh_res2$outcome <- "Filed FTC report"
beh_res2 <- beh_res2[colnames(beh_res2)[c(7, 1:6)]]
beh_res2$n <- rep(length(resid(ftc_fit2c)), times = 2)
beh_res2$p.value <- p.adjust(beh_res2$p.value, method = "hochberg")

sink("./tables/tableA4.txt")
stargazer::stargazer(beh_res2, type = "latex", summary = FALSE,
                     title = "Estimated Treatment Effects on Behavioral Outcomes with Alternative Treatments")
sink()

plac_res2 <- as.data.frame(rbind(china_res[c(2, 3), c(1, 3, 4, 8, 9, 17)],
                                 preact_res[c(2, 3), c(1, 3, 4, 8, 9, 17)]))
plac_res2$outcome <- rep(c("China economic expansion", "Pre-sale political activities"), each = 2)
plac_res2 <- plac_res2[colnames(plac_res2)[c(7, 1:6)]]
plac_res2$n <- rep(c(length(resid(china_fit2c)), length(resid(preact_fit2c))), each = 2)

sink("./tables/tableA5.txt")
stargazer::stargazer(plac_res2, type = "latex", summary = FALSE,
                     title = "Estimated Treatment Effects on Placebo Outcomes with Alternative Treatments")
sink()

### Results with pre-treatment political activities as a covariate----
covs2 <- "networks + factor(fanlength) + factor(fanintensity) +
          factor(ideo5_new) + factor(pid3_new) + age + factor(gender) + engagepre_count"

## Re-estimate effects
# Ticketmaster monopoly
montmd_fit1c_p <- update(montmd_fit1, paste(". ~ . +", covs2))
montmd_est1_p <- avg_comparisons(montmd_fit1c_p, variables = "treat1",
                                 vcov = "HC3", wts = "wt1")
montmd_fit4c_p <- ivreg(scale(monopoliestm) ~ treat4 + networks + factor(fanlength) +
                                              factor(fanintensity) + factor(ideo5_new) +
                                              factor(pid3_new) + age + factor(gender) + engagepre_count |
                                              treat1 + networks + factor(fanlength) +
                                              factor(fanintensity) + factor(ideo5_new) +
                                              factor(pid3_new) + age + factor(gender) + engagepre_count,
                        data = df, weights = wt1)
montmd_est4_p <- avg_comparisons(montmd_fit4c_p, variables = "treat4",
                                 vcov = "HC3", wts = "wt1")

# Ticketmaster distribution inequality
ineqtmd_fit1c_p <- update(ineqtmd_fit1, paste(". ~ . +", covs2))
ineqtmd_est1_p <- avg_comparisons(ineqtmd_fit1c_p, variables = "treat1",
                                  vcov = "HC3", wts = "wt1")
ineqtmd_fit4c_p <- ivreg(scale(ineqtm) ~ treat4 + networks + factor(fanlength) +
                                         factor(fanintensity) + factor(ideo5_new) +
                                         factor(pid3_new) + age + factor(gender) + engagepre_count |
                                         treat1 + networks + factor(fanlength) +
                                         factor(fanintensity) + factor(ideo5_new) +
                                         factor(pid3_new) + age + factor(gender) + engagepre_count,
                         data = df, weights = wt1)
ineqtmd_est4_p <- avg_comparisons(ineqtmd_fit4c_p, variables = "treat4",
                                  vcov = "HC3", wts = "wt1")

# FTC report
ftc_fit1c_p <- update(ftc_fit1, paste(". ~ . +", covs2))
ftc_est1_p <- avg_comparisons(ftc_fit1c_p, variables = "treat1",
                              vcov = "HC3", wts = "wt1")
ftc_fit4c_p <- ivreg(ftc_completion_strict ~ treat4 + networks + factor(fanlength) +
                                             factor(fanintensity) + factor(ideo5_new) +
                                             factor(pid3_new) + age + factor(gender) + engagepre_count |
                                             treat1 + networks + factor(fanlength) +
                                             factor(fanintensity) + factor(ideo5_new) +
                                             factor(pid3_new) + age + factor(gender) + engagepre_count,
                     data = df, weights = wt1)
ftc_est4_p <- avg_comparisons(ftc_fit4c_p, variables = "treat4",
                              vcov = "HC3", wts = "wt1")


w_pol <- data.frame(spec = rep(c("Original", "With pretreatment political activities"), each = 6),
                    outcome = rep(c("Ticketmaster\n monopoly",
                                    "Ticketmaster\n inequality",
                                    "Filed FTC\n report"), each = 2),
                    estimator = rep(c("ITT", "CACE"), times = 3))
w_pol$pe <- c(montmd_res$estimate[c(1, 4)], ineqtmd_res$estimate[c(1, 4)], ftc_res$estimate[c(1, 4)],
              montmd_est1_p$estimate, montmd_est4_p$estimate, ineqtmd_est1_p$estimate, ineqtmd_est4_p$estimate,
              ftc_est1_p$estimate, ftc_est4_p$estimate)
w_pol$lo <- c(montmd_res$conf.low[c(1, 4)], ineqtmd_res$conf.low[c(1, 4)], ftc_res$conf.low[c(1, 4)],
              montmd_est1_p$conf.low, montmd_est4_p$conf.low, ineqtmd_est1_p$conf.low, ineqtmd_est4_p$conf.low,
              ftc_est1_p$conf.low, ftc_est4_p$conf.low)
w_pol$hi <- c(montmd_res$conf.high[c(1, 4)], ineqtmd_res$conf.high[c(1, 4)], ftc_res$conf.high[c(1, 4)],
              montmd_est1_p$conf.high, montmd_est4_p$conf.high, ineqtmd_est1_p$conf.high, ineqtmd_est4_p$conf.high,
              ftc_est1_p$conf.high, ftc_est4_p$conf.high)
w_pol$estimator <- factor(w_pol$estimator, levels = c("ITT", "CACE"))

# Graph results
theme_set(theme_bw(base_size = 18))

ggplot(w_pol, aes(x = outcome, y = pe, color = spec)) +
  geom_errorbar(aes(ymin = lo, ymax = hi),
                linewidth = .35, width = 0,
                position = position_dodge(.5)) +
  geom_point(position = position_dodge(.5), size = 3.5) +
  geom_hline(yintercept = 0, linewidth = .1) +
  scale_color_taylor(palette = "taylor1989", reverse = TRUE) +
  xlab("") + ylab("Estimate") +
  theme(legend.position = "bottom", legend.title = element_blank(), 
        legend.background = element_rect(fill = "transparent"),
        strip.background = element_rect(fill = "white"),
        plot.margin = unit(c(.1, .2, -.5, -1), "line")) +
  coord_flip() + facet_wrap(~ estimator, scales = "free_x")
ggsave("./graphs/figA8.pdf", width = 30, height = 12, units = "cm")

### Save workspace
save.image("./data/processed_data/estimation.RData")
