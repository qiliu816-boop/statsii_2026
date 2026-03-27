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
# This file estimates and assesses weights to improve covariate balance.        #
#                                                                               #
# Last update: May 20, 2024                                                     #  
#################################################################################

## Packages
library(WeightIt)
library(cobalt)
library(tidyverse)
library(tayloRswift)

load("./data/processed_data/balance.RData")

## Estimate weights----
wf1 <- formula(treat1 ~ networks + factor(fanlength) + factor(fanintensity) +
                        factor(ideo5_new) + factor(pid3_new) + age + factor(gender))
wf2 <- update(wf1, treat2 ~ .)
wf3 <- update(wf1, treat3 ~ .)

w.out1 <- weightit(wf1, data = df, estimand = "ATE", method = "cbps",
                   missing = "ind", over = TRUE)
w.out2 <- weightit(wf2, data = df, estimand = "ATE", method = "cbps",
                   missing = "ind", over = TRUE)
w.out3 <- weightit(wf3, data = df, estimand = "ATE", method = "cbps",
                   missing = "ind", over = TRUE)

# Table A8
get_ess <- function(x) round(sum(summary(x)$effective.sample.size[2, ]), 0)

A8 <- data.frame(Treatment = c("November", "November-August", "Continuous"),
                 Mean = c(mean(w.out1$weights), mean(w.out2$weights), mean(w.out3$weights)),
                 SD = c(sd(w.out1$weights), sd(w.out2$weights), sd(w.out3$weights)),
                 `Min.` = c(min(w.out1$weights), min(w.out2$weights), min(w.out3$weights)),
                 `Max.` = c(max(w.out1$weights), max(w.out2$weights), max(w.out3$weights)),
                 ESS = c(get_ess(w.out1), get_ess(w.out2), get_ess(w.out3)))

sink("./tables/tableA8.txt")
stargazer::stargazer(A8, type = "latex", summary = FALSE, digits = 2,
                     title = "CBPS Weight Summaries")
sink()

df$wt1 <- w.out1$weights
df$wt2 <- w.out2$weights
df$wt3 <- w.out3$weights

## Balance
# Standardized mean differences, KS statistics, and complement of overlap
bal_fan1_wt <- bal.tab(covs_fan1, data = df, s.d.denom = "pooled", weights = "wt1",
                       abs = FALSE, int = FALSE, poly = 1, continuous = "std",
                       binary = "std", stats = c("mean.diffs", "ks", "ovl"), un = TRUE)
                       # thresholds = list(m = .1, k = .1, o = .1))
bal_pol1_wt <- bal.tab(covs_pol1, data = df, s.d.denom = "pooled", weights = "wt1",
                       abs = FALSE, int = FALSE, poly = 1, continuous = "std",
                       binary = "std", stats = c("mean.diffs", "ks", "ovl"), un = TRUE)
                       # thresholds = list(m = .1, k = .1, o = .1))
bal_dem1_wt <- bal.tab(covs_dem1, data = df, s.d.denom = "pooled", weights = "wt1",
                       abs = FALSE, int = FALSE, poly = 1, continuous = "std",
                       binary = "std", stats = c("mean.diffs", "ks", "ovl"), un = TRUE)
                       # thresholds = list(m = .1, k = .1, o = .1))

bal_fan2_wt <- bal.tab(covs_fan2, data = df, s.d.denom = "pooled", weights = "wt2",
                       abs = FALSE, int = FALSE, poly = 1, continuous = "std",
                       binary = "std", stats = c("mean.diffs", "ks", "ovl"), un = TRUE)
                       # thresholds = list(m = .1, k = .1, o = .1))
bal_pol2_wt <- bal.tab(covs_pol2, data = df, s.d.denom = "pooled", weights = "wt2",
                       abs = FALSE, int = FALSE, poly = 1, continuous = "std",
                       binary = "std", stats = c("mean.diffs", "ks", "ovl"), un = TRUE)
                       # thresholds = list(m = .1, k = .1, o = .1))
bal_dem2_wt <- bal.tab(covs_dem2, data = df, s.d.denom = "pooled", weights = "wt2",
                       abs = FALSE, int = FALSE, poly = 1, continuous = "std",
                       binary = "std", stats = c("mean.diffs", "ks", "ovl"), un = TRUE)
                       # thresholds = list(m = .1, k = .1, o = .1))

bal_fan3_wt <- bal.tab(covs_fan3, data = df, s.d.denom = "all", weights = "wt3",
                       abs = FALSE, int = FALSE, poly = 1, continuous = "std",
                       binary = "std", stats = c("correlation"), un = TRUE)
                       # thresholds = list(m = .1, k = .1, o = .1))
bal_pol3_wt <- bal.tab(covs_pol3, data = df, s.d.denom = "all", weights = "wt3",
                       abs = FALSE, int = FALSE, poly = 1, continuous = "std",
                       binary = "std", stats = c("correlation"), un = TRUE)
                       # thresholds = list(m = .1, k = .1, o = .1))
bal_dem3_wt <- bal.tab(covs_dem3, data = df, s.d.denom = "all", weights = "wt3",
                       abs = FALSE, int = FALSE, poly = 1, continuous = "std",
                       binary = "std", stats = c("correlation"), un = TRUE)
                       # thresholds = list(m = .1, k = .1, o = .1))
## Love plots----
# Swiftieness
temp1 <- bal_fan1_wt$Balance[ , 2:4]; temp2 <- bal_fan1_wt$Balance[ , 5:7]
colnames(temp1) <- colnames(temp2) <- c("(a) Absolute standardized MD",
                                        "(b) KS statistics",
                                        "(c) Complement of overlap")
temp1$adjust <- "\n(Unweighted)"; temp2$adjust <- "\n(Weighted)"
lp_fan_wt <- rbind(temp1, temp2)
lp_fan_wt$treat <- "Treatment: Nov."
lp_fan_wt$names <- c("Fan length:\n 2020-2022", "Fan length:\n 2014-2019", "Fan length:\n 2010-2013", "Fan length:\n 2009 or earlier", "Fan length NA",
                  "Fan intensity: 1", "Fan intensity: 2", "Fan intensity: 3", "Fan intensity: 4", "Fan intensity: 5", "Fan intensity: 6",
                  "Fan intensity: 7", "Fan intensity: 8", "Fan intensity: 9", "Fan intensity: 10", "Fan intensity NA",
                  "Fan social media", "Fan social media NA", "Stadium time\n zone: Central", "Stadium time\n zone: Eastern",
                  "Stadium time\n zone: Mountain", "Stadium time\n zone: Pacific", "Tour stop capacity")
temp1 <- bal_fan2_wt$Balance[ , 2:4]; temp2 <- bal_fan2_wt$Balance[ , 5:7]
colnames(temp1) <- colnames(temp2) <- c("(a) Absolute standardized MD",
                                        "(b) KS statistics",
                                        "(c) Complement of overlap")
temp1$adjust <- "\n(Unweighted)"; temp2$adjust <- "\n(Weighted)"
temp3 <- rbind(temp1, temp2)
temp3$treat <- "Treatment: Nov.-Aug."
temp3$names <- lp_fan_wt$names
lp_fan_wt <- rbind(lp_fan_wt, temp3)

lp_fan_wt$bias <- if_else(lp_fan_wt$`(a) Absolute standardized MD` < 0, "Bias toward\n control", "Bias toward\n treated")
lp_fan_wt <- gather(lp_fan_wt, stat, value, `(a) Absolute standardized MD`:`(c) Complement of overlap`, factor_key = TRUE)
lp_fan_wt$order <- rep(rank(abs(lp_fan_wt[lp_fan_wt$treat == "Treatment: Nov." & lp_fan_wt$stat == "(a) Absolute standardized MD" &
                                          lp_fan_wt$adjust == "\n(Weighted)", ]$value)), times = 6)
lp_fan_wt[lp_fan_wt$stat != "(a) Absolute standardized MD", ]$bias <- "n/a"

lp_fan_wt$treat_adjust <- paste(lp_fan_wt$treat, lp_fan_wt$adjust, sep = " ")
lp_fan_wt$treat_adjust <- factor(lp_fan_wt$treat_adjust, levels = names(table(lp_fan_wt$treat_adjust))[c(1, 3, 2, 4)])

theme_set(theme_bw(base_size = 18))
xl_fan <- "<-- More balance  Less balance -->           <-- More balance  Less balance -->           <-- More balance  Less balance -->"

ggplot(lp_fan_wt, aes(x = abs(value), y = reorder(names, order),
                   color = bias, alpha = adjust)) + 
  geom_point(aes(shape = treat_adjust), size = 4) +
  geom_vline(xintercept = .1, linetype = 3) +
  geom_vline(xintercept = .2, linetype = 2) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, .1)) +
  xlab(xl_fan) + ylab("") +
  scale_y_discrete() +
  scale_alpha_manual(values = c(.3, 1), guide = "none") +
  scale_shape_manual(values = c(16, 1, 17, 2)) +
  scale_color_taylor(palette = "taylor1989", reverse = TRUE,
                     breaks = c("Bias toward\n control", "Bias toward\n treated")) +
  facet_wrap(~ stat) +
  theme(legend.position = "bottom", legend.title = element_blank(),
        legend.background = element_rect(fill = "transparent"),
        strip.background = element_rect(fill = "white"),
        axis.title = element_text(size = 16, face = "bold"))
ggsave("./graphs/figA10.pdf", width = 40, height = 27, units = "cm")

# Political variables
temp1 <- bal_pol1_wt$Balance[ , 2:4]; temp2 <- bal_pol1_wt$Balance[ , 5:7]
colnames(temp1) <- colnames(temp2) <- c("(a) Absolute standardized MD",
                                        "(b) KS statistics",
                                        "(c) Complement of overlap")
temp1$adjust <- "\n(Unweighted)"; temp2$adjust <- "\n(Weighted)"
lp_pol_wt <- rbind(temp1, temp2)
lp_pol_wt$treat <- "Treatment: Nov."
lp_pol_wt$names <- c("Support TS political statements: 1", "Support TS political statements: 2", "Support TS political statements: 3",
                     "Support TS political statements: 4", "Support TS political statements: 5", "Support TS political statements NA",
                     "Ideology: Very liberal", "Ideology: Liberal", "Ideology: Moderate", "Ideology: Conservative",
                     "Ideology: Very conservative", "Ideology: Not sure", "Party: Democrat", "Party: Republican", "Party: Independent",
                     "Party: Other", "Party: Not sure", "Political interest: Most of the time", "Political interest: Some of the time",
                     "Political interest: Only now and then", "Political interest: Hardly at all", "Political interest: Don't know",
                     "Political interest NA")
temp1 <- bal_pol2_wt$Balance[ , 2:4]; temp2 <- bal_pol2_wt$Balance[ , 5:7]
colnames(temp1) <- colnames(temp2) <- c("(a) Absolute standardized MD",
                                        "(b) KS statistics",
                                        "(c) Complement of overlap")
temp1$adjust <- "\n(Unweighted)"; temp2$adjust <- "\n(Weighted)"
temp3 <- rbind(temp1, temp2)
temp3$treat <- "Treatment: Nov.-Aug."
temp3$names <- lp_pol_wt$names
lp_pol_wt <- rbind(lp_pol_wt, temp3)

lp_pol_wt$bias <- if_else(lp_pol_wt$`(a) Absolute standardized MD` < 0, "Bias toward\n control", "Bias toward\n treated")
lp_pol_wt <- gather(lp_pol_wt, stat, value, `(a) Absolute standardized MD`:`(c) Complement of overlap`, factor_key = TRUE)
lp_pol_wt$order <- rep(rank(abs(lp_pol_wt[lp_pol_wt$treat == "Treatment: Nov." & lp_pol_wt$stat == "(a) Absolute standardized MD" &
                                            lp_pol_wt$adjust == "\n(Weighted)", ]$value)), times = 6)
lp_pol_wt[lp_pol_wt$stat != "(a) Absolute standardized MD", ]$bias <- "n/a"

lp_pol_wt$treat_adjust <- paste(lp_pol_wt$treat, lp_pol_wt$adjust, sep = " ")
lp_pol_wt$treat_adjust <- factor(lp_pol_wt$treat_adjust, levels = names(table(lp_pol_wt$treat_adjust))[c(1, 3, 2, 4)])

theme_set(theme_bw(base_size = 18))
xl_pol <- " <-- More balance  Less balance -->    <-- More balance  Less balance -->    <-- More balance  Less balance -->"

ggplot(lp_pol_wt, aes(x = abs(value), y = reorder(names, order),
                      color = bias, alpha = adjust)) + 
  geom_point(aes(shape = treat_adjust), size = 4) +
  geom_vline(xintercept = .1, linetype = 3) +
  geom_vline(xintercept = .2, linetype = 2) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, .1)) +
  xlab(xl_pol) + ylab("") +
  scale_y_discrete() +
  scale_alpha_manual(values = c(.3, 1), guide = "none") +
  scale_shape_manual(values = c(16, 1, 17, 2)) +
  scale_color_taylor(palette = "taylor1989", reverse = TRUE,
                     breaks = c("Bias toward\n control", "Bias toward\n treated")) +
  facet_wrap(~ stat) +
  theme(legend.position = "bottom", legend.title = element_blank(),
        legend.background = element_rect(fill = "transparent"),
        strip.background = element_rect(fill = "white"),
        axis.title = element_text(size = 16, face = "bold"))
ggsave("./graphs/figA11.pdf", width = 40, height = 27, units = "cm")

# Demographic variables
temp1 <- bal_dem1_wt$Balance[ , 2:4]; temp2 <- bal_dem1_wt$Balance[ , 5:7]
colnames(temp1) <- colnames(temp2) <- c("(a) Absolute standardized MD",
                                        "(b) KS statistics",
                                        "(c) Complement of overlap")
temp1$adjust <- "\n(Unweighted)"; temp2$adjust <- "\n(Weighted)"
lp_dem_wt <- rbind(temp1, temp2)
lp_dem_wt$treat <- "Treatment: Nov."
lp_dem_wt$names <- c("Age: 18-21", "Age: 22-25", "Age: 26-30", "Age: 31-35", "Age: 36-40",
                     "Age: 41-45", "Age: 46-50", "Age: 51-60", "Age: 61+", "Age NA",
                     "Gender: Man", "Gender: Woman", "Gender: Non-binary", "Gender: Self-describe",
                     "Gender NA", "Race: Asian", "Race: Black", "Race: Latino", "Race: Middle Eastern",
                     "Race: Mixed", "Race: Native American", "Race: Other", "Race: White", "Race NA",
                     "Income: < $10k", "Income: $10k-20k", "Income: $20k-30k", "Income: $30k-40k",
                     "Income: $40k-50k", "Income: $50k-60k", "Income: $60k-70k", "Income: $70k-80k",
                     "Income: $80k-100k", "Income: $100k-120k", "Income: $120k-150k", "Income: $150k-200k",
                     "Income: $200k-250k", "Income: $250k-350k", "Income: $350k-500k", "Income: $500k+",
                     "Income NA", "Education: No HS", "Education: HS graduate", "Education: Some college",
                     "Education: 2-year degree", "Education: 4-year degree", "Education: Post-graduate",
                     "Education NA")
temp1 <- bal_dem2_wt$Balance[ , 2:4]; temp2 <- bal_dem2_wt$Balance[ , 5:7]
colnames(temp1) <- colnames(temp2) <- c("(a) Absolute standardized MD",
                                        "(b) KS statistics",
                                        "(c) Complement of overlap")
temp1$adjust <- "\n(Unweighted)"; temp2$adjust <- "\n(Weighted)"
temp3 <- rbind(temp1, temp2)
temp3$treat <- "Treatment: Nov.-Aug."
temp3$names <- lp_dem_wt$names
lp_dem_wt <- rbind(lp_dem_wt, temp3)

lp_dem_wt$bias <- if_else(lp_dem_wt$`(a) Absolute standardized MD` < 0, "Bias toward\n control", "Bias toward\n treated")
lp_dem_wt <- gather(lp_dem_wt, stat, value, `(a) Absolute standardized MD`:`(c) Complement of overlap`, factor_key = TRUE)
lp_dem_wt$order <- rep(rank(abs(lp_dem_wt[lp_dem_wt$treat == "Treatment: Nov." & lp_dem_wt$stat == "(a) Absolute standardized MD" &
                                            lp_dem_wt$adjust == "\n(Weighted)", ]$value)), times = 6)
lp_dem_wt[lp_dem_wt$stat != "(a) Absolute standardized MD", ]$bias <- "n/a"

lp_dem_wt$treat_adjust <- paste(lp_dem_wt$treat, lp_dem_wt$adjust, sep = " ")
lp_dem_wt$treat_adjust <- factor(lp_dem_wt$treat_adjust, levels = names(table(lp_dem_wt$treat_adjust))[c(1, 3, 2, 4)])

theme_set(theme_bw(base_size = 18))
xl_dem <- "<-- More balance  Less balance -->        <-- More balance  Less balance -->        <-- More balance  Less balance -->"

ggplot(lp_dem_wt, aes(x = abs(value), y = reorder(names, order),
                      color = bias, alpha = adjust)) + 
  geom_point(aes(shape = treat_adjust), size = 4) +
  geom_vline(xintercept = .1, linetype = 3) +
  geom_vline(xintercept = .2, linetype = 2) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, .1)) +
  xlab(xl_dem) + ylab("") +
  scale_y_discrete() +
  scale_alpha_manual(values = c(.3, 1), guide = "none") +
  scale_shape_manual(values = c(16, 1, 17, 2)) +
  scale_color_taylor(palette = "taylor1989", reverse = TRUE,
                     breaks = c("Bias toward\n control", "Bias toward\n treated")) +
  facet_wrap(~ stat) +
  theme(legend.position = "bottom", legend.title = element_blank(),
        legend.background = element_rect(fill = "transparent"),
        strip.background = element_rect(fill = "white"),
        axis.title = element_text(size = 16, face = "bold"))
ggsave("./graphs/figA12.pdf", width = 40, height = 30, units = "cm")

# Treatment 3 love plot
lp_fan3_wt <- bal_fan3_wt$Balance[ , 2:3]
lp_fan3_wt$names <- c("Fan length:\n 2020-2022", "Fan length:\n 2014-2019", "Fan length:\n 2010-2013", "Fan length:\n 2009 or earlier",
                      "Fan length NA", "Fan intensity: 1", "Fan intensity: 2", "Fan intensity: 3", "Fan intensity: 4", "Fan intensity: 5",
                      "Fan intensity: 6", "Fan intensity: 7", "Fan intensity: 8", "Fan intensity: 9", "Fan intensity: 10", "Fan intensity NA",
                      "Fan social media", "Fan social media NA", "Stadium time\n zone: Central", "Stadium time\n zone: Eastern",
                      "Stadium time\n zone: Mountain", "Stadium time\n zone: Pacific", "Tour stop capacity")
colnames(lp_fan3_wt)[1:2] <- "value"
lp_fan3_wt <- rbind(lp_fan3_wt[ , c(1, 3)], lp_fan3_wt[ , 2:3])
lp_fan3_wt$adjust <- rep(c("Unweighted", "Weighted"), each = nrow(lp_fan3_wt)/2)
lp_fan3_wt$bias <- if_else(lp_fan3_wt$value < 0, "Bias toward control", "Bias toward treated")
lp_fan3_wt$order <- rank(abs(lp_fan3_wt[lp_fan3_wt$adjust == "Weighted", ]$value))
lp_fan3_wt$category <- "(a) Swiftie fanship variables"

lp_pol3_wt <- bal_pol3_wt$Balance[ , 2:3]
lp_pol3_wt$names <- c("Support TS political\n statements: 1", "Support TS political\n statements: 2", "Support TS political\n statements: 3",
                      "Support TS political\n statements: 4", "Support TS political\n statements: 5", "Support TS political\n statements NA",
                      "Ideology: Very liberal", "Ideology: Liberal", "Ideology: Moderate", "Ideology: Conservative",
                      "Ideology: Very conservative", "Ideology: Not sure", "Party: Democrat", "Party: Republican", "Party: Independent",
                      "Party: Other", "Party: Not sure", "Political interest:\n Most of the time", "Political interest:\n Some of the time",
                      "Political interest:\n Only now and then", "Political interest:\n Hardly at all", "Political interest:\n Don't know",
                      "Political interest NA")
colnames(lp_pol3_wt)[1:2] <- "value"
lp_pol3_wt <- rbind(lp_pol3_wt[ , c(1, 3)], lp_pol3_wt[ , 2:3])
lp_pol3_wt$adjust <- rep(c("Unweighted", "Weighted"), each = nrow(lp_pol3_wt)/2)
lp_pol3_wt$bias <- if_else(lp_pol3_wt$value < 0, "Bias toward control", "Bias toward treated")
lp_pol3_wt$order <- rank(abs(lp_pol3_wt[lp_pol3_wt$adjust == "Weighted", ]$value))
lp_pol3_wt$category <- "(b) Political variables"

lp_dem3_wt <- bal_dem3_wt$Balance[ , 2:3]
lp_dem3_wt$names <- c("Age: 18-21", "Age: 22-25", "Age: 26-30", "Age: 31-35", "Age: 36-40",
                      "Age: 41-45", "Age: 46-50", "Age: 51-60", "Age: 61+", "Age NA",
                      "Gender: Man", "Gender: Woman", "Gender: Non-binary", "Gender: Self-describe",
                      "Gender NA", "Race: Asian", "Race: Black", "Race: Latino", "Race: Middle Eastern",
                      "Race: Mixed", "Race: Native American", "Race: Other", "Race: White", "Race NA",
                      "Income: < $10k", "Income: $10k-20k", "Income: $20k-30k", "Income: $30k-40k",
                      "Income: $40k-50k", "Income: $50k-60k", "Income: $60k-70k", "Income: $70k-80k",
                      "Income: $80k-100k", "Income: $100k-120k", "Income: $120k-150k", "Income: $150k-200k",
                      "Income: $200k-250k", "Income: $250k-350k", "Income: $350k-500k", "Income: $500k+",
                      "Income NA", "Education: No HS", "Education: HS graduate", "Education: Some college",
                      "Education: 2-year degree", "Education: 4-year degree", "Education: Post-graduate",
                      "Education NA")
colnames(lp_dem3_wt)[1:2] <- "value"
lp_dem3_wt <- rbind(lp_dem3_wt[ , c(1, 3)], lp_dem3_wt[ , 2:3])
lp_dem3_wt$adjust <- rep(c("Unweighted", "Weighted"), each = nrow(lp_dem3_wt)/2)
lp_dem3_wt$bias <- if_else(lp_dem3_wt$value < 0, "Bias toward control", "Bias toward treated")
lp_dem3_wt$order <- rank(abs(lp_dem3_wt[lp_dem3_wt$adjust == "Weighted", ]$value))
lp_dem3_wt$category <- "(c) Demographic variables"

lp3_wt <- rbind(lp_fan3_wt, lp_pol3_wt, lp_dem3_wt)

theme_set(theme_bw(base_size = 22))
xl3 <- "<-- More balance  Less balance -->                                                                    <-- More balance  Less balance -->                                                                    <-- More balance  Less balance -->"

ggplot(lp3_wt, aes(x = abs(value), y = reorder(names, order),
                   color = bias, shape = adjust)) + 
  geom_point(size = 4) +
  geom_vline(xintercept = .1, linetype = 3) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, .1)) +
  xlab(xl3) + ylab("") +
  scale_y_discrete() +
  scale_shape_manual(values = c(16, 1)) +
  scale_color_taylor(palette = "taylor1989", reverse = TRUE,
                     breaks = c("Bias toward control", "Bias toward treated")) +
  facet_wrap(~ category, scales = "free_y", nrow = 1) +
  theme(legend.position = "bottom", legend.title = element_blank(),
        legend.background = element_rect(fill = "transparent"),
        strip.background = element_rect(fill = "white"),
        axis.title = element_text(size = 16, face = "bold"))
ggsave("./graphs/figA13.pdf", width = 60, height = 35, units = "cm")

### SMD for Treatment 1 only----
lp_fan_wt$type <- "(a) Swiftie fanship variables"
lp_pol_wt$type <- "(b) Political variables"
lp_dem_wt$type <- "(c) Demographic variables"

smd_res <- rbind(lp_fan_wt[lp_fan_wt$treat == "Treatment: Nov." &
                             lp_fan_wt$stat == "(a) Absolute standardized MD", ],
                 lp_pol_wt[lp_pol_wt$treat == "Treatment: Nov." &
                             lp_pol_wt$stat == "(a) Absolute standardized MD", ],
                 lp_dem_wt[lp_dem_wt$treat == "Treatment: Nov." &
                             lp_dem_wt$stat == "(a) Absolute standardized MD", ])
smd_res <- smd_res %>%
  mutate(adjust2 = case_when(
    adjust == "\n(Unweighted)" ~ "Unweighted",
    adjust == "\n(Weighted)" ~ "Weighted",
  ))

theme_set(theme_bw(base_size = 22))
xl_smd <- "<-- More balance  Less balance -->                                                            <-- More balance  Less balance -->                                                                  <-- More balance  Less balance -->"

ggplot(smd_res, aes(x = abs(value), y = reorder(names, order),
                    color = bias, shape = adjust2)) + 
  geom_point(aes(shape = adjust2), size = 4) +
  geom_vline(xintercept = .1, linetype = 3) +
  geom_vline(xintercept = .2, linetype = 2) +
  scale_x_continuous(limits = c(0, .5), breaks = seq(0, 1, .1)) +
  xlab(xl_smd) + ylab("") +
  scale_y_discrete(labels = function(x) str_wrap(x, width = 25)) +
  scale_shape_manual(values = c(1, 16)) +
  scale_color_taylor(palette = "taylor1989", reverse = TRUE,
                     breaks = c("Bias toward\n control", "Bias toward\n treated")) +
  facet_wrap(~ type, scales = "free", nrow = 1) +
  theme(legend.position = "bottom", legend.title = element_blank(),
        legend.background = element_rect(fill = "transparent"),
        strip.background = element_rect(fill = "white"),
        axis.title = element_text(size = 16, face = "bold")) 
ggsave("./graphs/fig5.pdf", width = 60, height = 35, units = "cm")

### Save workspace for '3_estimation.R'
save.image("./data/processed_data/weighting.RData")
