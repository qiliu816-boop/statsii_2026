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
# This file assesses covariate balance.                                         #
#                                                                               #
# Last update: March 5, 2024                                                    #  
#################################################################################

## Packages
library(cobalt)
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)
library(forcats)
library(Amelia)
library(TOSTER)
library(sandwich)
library(stargazer)
library(fixest)
library(tayloRswift)

## Data---- 
df <- readRDS("./data/raw_data/surveydata.rds")

# Create a subset with only the non-waitlist group
df_sale <- subset(df, screener == 2)

# Count boosts in the non-waitlist group
df_sale$boost_count <- NA
df_sale[df_sale$boost == 1, ]$boost_count <- 0
df_sale[df_sale$boost %in% 2:3, ]$boost_count <- 1
df_sale[df_sale$boost == 4, ]$boost_count <- 2

# Check missingness -- pretty minimal
missmap(subset(df, select = c("fanlength", "fanintensity", "networks", "engagepre_7",
                              "tspolitics", "ideo5", "pid3", "newsint", "age_cat",
                              "gender", "race", "faminc_new", "educ", "timezone",
                              "presaleprep", "friendsfail", "seatgeek", "demand")))

## Equivalence tests----
# eqb = .36\sigma from Hartman and Hidalgo (2018, 1006)
eq_test <- function(dat, x, tr){
  if(tr == "treat1"){
    tost <-
      tsum_TOST(m1 = mean(dat[dat$treat1 == 1, x], na.rm = TRUE),
                sd1 = sd(dat[dat$treat1 == 1, x], na.rm = TRUE),
                n1 = table(dat[ , "treat1"])[2],
                m2 = mean(dat[dat$treat1 == 0, x], na.rm = TRUE),
                sd2 = sd(dat[dat$treat1 == 0, x], na.rm = TRUE),
                n2 = table(dat[ , "treat1"])[1], eqbound_type = "raw",
                hypothesis = "EQU", var.equal = FALSE,
                eqb = .36*sd(dat[ , x], na.rm = TRUE))      
  } else{
    tost <-
      tsum_TOST(m1 = mean(dat[dat$treat2 == 1, x], na.rm = TRUE),
                sd1 = sd(dat[dat$treat2 == 1, x], na.rm = TRUE),
                n1 = table(dat[ , "treat2"])[2],
                m2 = mean(dat[dat$treat2 == 0, x], na.rm = TRUE),
                sd2 = sd(dat[dat$treat2 == 0, x], na.rm = TRUE),
                n2 = table(dat[ , "treat2"])[1], eqbound_type = "raw",
                hypothesis = "EQU", var.equal = FALSE,
                eqb = .36*sd(dat[ , x], na.rm = TRUE))  
  }
  res <- c(as.numeric(tost$effsize[1, c(1, 3:4)]), as.numeric(tost$eqb[1, 2:3]))
  names(res) <- c("diff", "diff_lower", "diff_upper", "eqb_lower", "eqb_upper")
  return(list(tost = tost, res = res))
}

# Swiftieness
eq_fan <- data.frame(variable = rep(c("Fan length", "Fan intensity", "Fan social\n media", "Tour stop\n capacity"), each = 2),
                     treat = rep(c("Treatment: November", "Treatment: November-August"), times = 4),
                     diff = NA, diff_lower = NA, diff_upper = NA, eqb_lower = NA, eqb_upper = NA)
eq_fan[1, 3:7] <- as.numeric(eq_test(df, "fanlength", "treat1")$res)
eq_fan[2, 3:7] <- as.numeric(eq_test(df, "fanlength", "treat2")$res)
eq_fan[3, 3:7] <- as.numeric(eq_test(df, "fanintensity", "treat1")$res)
eq_fan[4, 3:7] <- as.numeric(eq_test(df, "fanintensity", "treat2")$res)
eq_fan[5, 3:7] <- as.numeric(eq_test(df, "networks", "treat1")$res)
eq_fan[6, 3:7] <- as.numeric(eq_test(df, "networks", "treat2")$res)

demand_eq <- data.frame(df$treat1, df$treat2, scale(df$demand))
colnames(demand_eq) <- c("treat1", "treat2", "demand_z")

eq_fan[7, 3:7] <- as.numeric(eq_test(demand_eq, "demand_z", "treat1")$res)
eq_fan[8, 3:7] <- as.numeric(eq_test(demand_eq, "demand_z", "treat2")$res)
eq_fan$type <- "(a) Swiftie fanship variables"

# Political variables
eq_pol <- data.frame(variable = rep(c("Support TS\n political statements",
                                      "Ideology", "Party: Democrat", "Party: Republican", "Party: Independent", "Party: Other",
                                      "Party: Not sure", "Political interest"), each = 2),
                     treat = rep(c("Treatment: November", "Treatment: November-August"), times = 8),
                     diff = NA, diff_lower = NA, diff_upper = NA, eqb_lower = NA, eqb_upper = NA)

ideo_pid_eq <- data.frame(df$treat1, df$treat2, df$ideo5_new, i(df$pid3_new))
colnames(ideo_pid_eq) <- c("treat1", "treat2", "ideo5_new", "dem", "rep", "ind", "other", "notsure")
ideo_pid_eq <- ideo_pid_eq %>%
  mutate(ideo5_new = na_if(ideo5_new, 6))

newsint_eq <- data.frame(df$treat1, df$treat2, df$newsint)
colnames(newsint_eq) <- c("treat1", "treat2", "newsint")
newsint_eq <- newsint_eq %>%
  mutate(newsint = na_if(newsint, 5))

eq_pol[1, 3:7] <- as.numeric(eq_test(df, "tspolitics", "treat1")$res)
eq_pol[2, 3:7] <- as.numeric(eq_test(df, "tspolitics", "treat2")$res)
eq_pol[3, 3:7] <- as.numeric(eq_test(ideo_pid_eq, "ideo5_new", "treat1")$res)
eq_pol[4, 3:7] <- as.numeric(eq_test(ideo_pid_eq, "ideo5_new", "treat2")$res)
eq_pol[5, 3:7] <- as.numeric(eq_test(ideo_pid_eq, "dem", "treat1")$res)
eq_pol[6, 3:7] <- as.numeric(eq_test(ideo_pid_eq, "dem", "treat2")$res)
eq_pol[7, 3:7] <- as.numeric(eq_test(ideo_pid_eq, "rep", "treat1")$res)
eq_pol[8, 3:7] <- as.numeric(eq_test(ideo_pid_eq, "rep", "treat2")$res)
eq_pol[9, 3:7] <- as.numeric(eq_test(ideo_pid_eq, "ind", "treat1")$res)
eq_pol[10, 3:7] <- as.numeric(eq_test(ideo_pid_eq, "ind", "treat2")$res)
eq_pol[11, 3:7] <- as.numeric(eq_test(ideo_pid_eq, "other", "treat1")$res)
eq_pol[12, 3:7] <- as.numeric(eq_test(ideo_pid_eq, "other", "treat2")$res)
eq_pol[13, 3:7] <- as.numeric(eq_test(ideo_pid_eq, "notsure", "treat1")$res)
eq_pol[14, 3:7] <- as.numeric(eq_test(ideo_pid_eq, "notsure", "treat2")$res)
eq_pol[15, 3:7] <- as.numeric(eq_test(newsint_eq, "newsint", "treat1")$res)
eq_pol[16, 3:7] <- as.numeric(eq_test(newsint_eq, "newsint", "treat2")$res)
eq_pol$type <- "(b) Political variables"

# Demographic variables 
eq_dem <- data.frame(variable = rep(c("Age", "Gender: Man", "Gender: Woman", "Gender: Non-binary",
                                      "Gender: Self-describe", "Race: Asian", "Race: Black",
                                      "Race: Latino", "Race: Mid. East.", "Race: Mixed",
                                      "Race: Native Am.", "Race: Other", "Race: White",
                                      "Income", "Education"), each = 2),
                     treat = rep(c("Treatment: November", "Treatment: November-August"), times = 15),
                     diff = NA, diff_lower = NA, diff_upper = NA, eqb_lower = NA, eqb_upper = NA)

age_eq <- data.frame(df$treat1, df$treat2, scale(df$age))
colnames(age_eq) <- c("treat1", "treat2", "age_z")
gen_eq <- data.frame(df$treat1, df$treat2, i(df$gender))
colnames(gen_eq) <- c("treat1", "treat2", "man", "woman", "non-binary", "self-describe")
race_eq <- data.frame(df$treat1, df$treat2, i(df$race))
colnames(race_eq) <- c("treat1", "treat2", "asian", "black", "latino", "middleeastern",
                       "mixed", "nativeamerican", "other", "white")

eq_dem[1, 3:7] <- as.numeric(eq_test(age_eq, "age_z", "treat1")$res)
eq_dem[2, 3:7] <- as.numeric(eq_test(age_eq, "age_z", "treat2")$res)
eq_dem[3, 3:7] <- as.numeric(eq_test(gen_eq, "man", "treat1")$res)
eq_dem[4, 3:7] <- as.numeric(eq_test(gen_eq, "man", "treat2")$res)
eq_dem[5, 3:7] <- as.numeric(eq_test(gen_eq, "woman", "treat1")$res)
eq_dem[6, 3:7] <- as.numeric(eq_test(gen_eq, "woman", "treat2")$res)
eq_dem[7, 3:7] <- as.numeric(eq_test(gen_eq, "non-binary", "treat1")$res)
eq_dem[8, 3:7] <- as.numeric(eq_test(gen_eq, "non-binary", "treat2")$res)
eq_dem[9, 3:7] <- as.numeric(eq_test(gen_eq, "self-describe", "treat1")$res)
eq_dem[10, 3:7] <- as.numeric(eq_test(gen_eq, "self-describe", "treat2")$res)
eq_dem[11, 3:7] <- as.numeric(eq_test(race_eq, "asian", "treat1")$res)
eq_dem[12, 3:7] <- as.numeric(eq_test(race_eq, "asian", "treat2")$res)
eq_dem[13, 3:7] <- as.numeric(eq_test(race_eq, "black", "treat1")$res)
eq_dem[14, 3:7] <- as.numeric(eq_test(race_eq, "black", "treat2")$res)
eq_dem[15, 3:7] <- as.numeric(eq_test(race_eq, "latino", "treat1")$res)
eq_dem[16, 3:7] <- as.numeric(eq_test(race_eq, "latino", "treat2")$res)
eq_dem[17, 3:7] <- as.numeric(eq_test(race_eq, "middleeastern", "treat1")$res)
eq_dem[18, 3:7] <- as.numeric(eq_test(race_eq, "middleeastern", "treat2")$res)
eq_dem[19, 3:7] <- as.numeric(eq_test(race_eq, "mixed", "treat1")$res)
eq_dem[20, 3:7] <- as.numeric(eq_test(race_eq, "mixed", "treat2")$res)
eq_dem[21, 3:7] <- as.numeric(eq_test(race_eq, "nativeamerican", "treat1")$res)
eq_dem[22, 3:7] <- as.numeric(eq_test(race_eq, "nativeamerican", "treat2")$res)
eq_dem[23, 3:7] <- as.numeric(eq_test(race_eq, "other", "treat1")$res)
eq_dem[24, 3:7] <- as.numeric(eq_test(race_eq, "other", "treat2")$res)
eq_dem[25, 3:7] <- as.numeric(eq_test(race_eq, "white", "treat1")$res)
eq_dem[26, 3:7] <- as.numeric(eq_test(race_eq, "white", "treat2")$res)
eq_dem[27, 3:7] <- as.numeric(eq_test(df, "faminc_new", "treat1")$res)
eq_dem[28, 3:7] <- as.numeric(eq_test(df, "faminc_new", "treat2")$res)
eq_dem[29, 3:7] <- as.numeric(eq_test(df, "educ", "treat1")$res)
eq_dem[30, 3:7] <- as.numeric(eq_test(df, "educ", "treat2")$res)
eq_dem$type <- "(c) Demographic variables"

# Stadium pre-sale variables
eq_sale <- data.frame(variable = rep(c("Stadium time\n zone: Central", "Stadium time\n zone: Eastern",
                                       "Stadium time\n zone: Mountain", "Stadium time\n zone: Pacific"), each = 2),
                      treat = rep(c("Treatment: November", "Treatment: November-August"), times = 4),
                      diff = NA, diff_lower = NA, diff_upper = NA, eqb_lower = NA, eqb_upper = NA)

tz_eq <- data.frame(df$treat1, df$treat2, i(df$timezone))
colnames(tz_eq) <- c("treat1", "treat2", "central", "eastern", "mountain", "pacific")

eq_sale[1, 3:7] <- as.numeric(eq_test(tz_eq, "central", "treat1")$res)
eq_sale[2, 3:7] <- as.numeric(eq_test(tz_eq, "central", "treat2")$res)
eq_sale[3, 3:7] <- as.numeric(eq_test(tz_eq, "eastern", "treat1")$res)
eq_sale[4, 3:7] <- as.numeric(eq_test(tz_eq, "eastern", "treat2")$res)
eq_sale[5, 3:7] <- as.numeric(eq_test(tz_eq, "mountain", "treat1")$res)
eq_sale[6, 3:7] <- as.numeric(eq_test(tz_eq, "mountain", "treat2")$res)
eq_sale[7, 3:7] <- as.numeric(eq_test(tz_eq, "pacific", "treat1")$res)
eq_sale[8, 3:7] <- as.numeric(eq_test(tz_eq, "pacific", "treat2")$res)
eq_sale$type <- "(a) Swiftie fanship variables"

# Combine
eq_all <- rbind(eq_fan, eq_pol, eq_dem, eq_sale)
eq_all$pass <- if_else(eq_all$diff_lower < eq_all$eqb_lower |
                       eq_all$diff_upper > eq_all$eqb_upper, "pass", "fail")

theme_set(theme_bw(base_size = 22))

ggplot(eq_all, aes(x = diff, y = reorder(variable, desc(variable)))) +
  geom_point(aes(shape = treat), position = position_dodge(.3), size = 4) +
  geom_errorbarh(aes(group = treat, xmin = diff_lower, xmax = diff_upper),
                 position = position_dodge(.3), height = 0) +
  geom_linerange(aes(xmin = eqb_lower, xmax = eqb_upper),
                 size = 13, alpha = .2) +
  scale_shape_manual(values = c(16, 1)) +
  xlab("Mean differences") + ylab("") +
  geom_vline(xintercept = 0) +
  facet_wrap(~ type, scales = "free", nrow = 1) +
  theme(legend.position = "bottom", legend.title = element_blank(),
        legend.background = element_rect(fill = "transparent"),
        strip.background = element_rect(fill = "white"),
        axis.title = element_text(size = 16, face = "bold"))
ggsave("./graphs/figA9.pdf", width = 50, height = 20, units = "cm")


## Additional balance metrics----
# Treatment 1
covs_fan1 <- formula(treat1 ~ factor(fanlength) + factor(fanintensity) + networks +
                              factor(timezone) + demand) # Swiftieness
covs_pol1 <- formula(treat1 ~ factor(tspolitics) + factor(ideo5_new) +
                              factor(pid3_new) + factor(newsint)) # Politics
covs_dem1 <- formula(treat1 ~ factor(age_cat) + factor(gender) + factor(race) +
                              factor(faminc_new) + factor(educ)) # Demographics

# Treatment 2
covs_fan2 <- update(covs_fan1, treat2 ~ .) # Swiftieness
covs_pol2 <- update(covs_pol1, treat2 ~ .) # Politics
covs_dem2 <- update(covs_dem1, treat2 ~ .) # Demographics

# Standardized mean differences, KS statistics, and complement of overlap
bal_fan1 <- bal.tab(covs_fan1, data = df, s.d.denom = "pooled",
                    abs = FALSE, int = FALSE, poly = 1, continuous = "std",
                    binary = "std", stats = c("mean.diffs", "ks", "ovl"))
bal_pol1 <- bal.tab(covs_pol1, data = df, s.d.denom = "pooled",
                    abs = FALSE, int = FALSE, poly = 1, continuous = "std",
                    binary = "std", stats = c("mean.diffs", "ks", "ovl"))
bal_dem1 <- bal.tab(covs_dem1, data = df, s.d.denom = "pooled",
                    abs = FALSE, int = FALSE, poly = 1, continuous = "std",
                    binary = "std", stats = c("mean.diffs", "ks", "ovl"))

bal_fan2 <- bal.tab(covs_fan2, data = df, s.d.denom = "pooled",
                    abs = FALSE, int = FALSE, poly = 1, continuous = "std",
                    binary = "std", stats = c("mean.diffs", "ks", "ovl"))
bal_pol2 <- bal.tab(covs_pol2, data = df, s.d.denom = "pooled",
                    abs = FALSE, int = FALSE, poly = 1, continuous = "std",
                    binary = "std", stats = c("mean.diffs", "ks", "ovl"))
bal_dem2 <- bal.tab(covs_dem2, data = df, s.d.denom = "pooled",
                    abs = FALSE, int = FALSE, poly = 1, continuous = "std",
                    binary = "std", stats = c("mean.diffs", "ks", "ovl"))

## Love plots
# Swiftieness
lp_fan <- bal_fan1$Balance[ , 2:4]
lp_fan$treat <- "Treatment: November"
lp_fan$names <- c("Fan length:\n 2020-2022", "Fan length:\n 2014-2019", "Fan length:\n 2010-2013", "Fan length:\n 2009 or earlier", "Fan length NA",
                  "Fan intensity: 1", "Fan intensity: 2", "Fan intensity: 3", "Fan intensity: 4", "Fan intensity: 5", "Fan intensity: 6",
                  "Fan intensity: 7", "Fan intensity: 8", "Fan intensity: 9", "Fan intensity: 10", "Fan intensity NA",
                  "Fan social media", "Fan social media NA", "Stadium time\n zone: Central", "Stadium time\n zone: Eastern",
                  "Stadium time\n zone: Mountain", "Stadium time\n zone: Pacific", "Tour stop capacity")
lp_fan <- rbind(lp_fan, data.frame(bal_fan2$Balance[ , 2:4],
                                   treat = "Treatment: November-August",
                                   names = lp_fan$names))
colnames(lp_fan)[1:3] <- c("(a) Absolute standardized MD",
                           "(b) KS statistics",
                           "(c) Complement of overlap")
lp_fan$bias <- if_else(lp_fan$`(a) Absolute standardized MD` < 0, "Bias toward control", "Bias toward treated")
lp_fan <- gather(lp_fan, stat, value, `(a) Absolute standardized MD`:`(c) Complement of overlap`, factor_key = TRUE)
lp_fan$order <- rep(rank(abs(lp_fan[lp_fan$treat == "Treatment: November" & lp_fan$stat == "(a) Absolute standardized MD", ]$value)), times = 6)
lp_fan[lp_fan$stat != "(a) Absolute standardized MD", ]$bias <- "n/a"

theme_set(theme_bw(base_size = 18))
xl_fan <- "<-- More balance  Less balance -->           <-- More balance  Less balance -->           <-- More balance  Less balance -->"

ggplot(lp_fan, aes(x = abs(value), y = reorder(names, order),
                        color = bias)) + 
  geom_point(aes(shape = treat), size = 4) +
  geom_vline(xintercept = .1, linetype = 3) +
  geom_vline(xintercept = .2, linetype = 2) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, .1)) +
  xlab(xl_fan) + ylab("") +
  scale_y_discrete() +
  scale_shape_manual(values = c(16, 1)) +
  scale_color_taylor(palette = "taylor1989", reverse = TRUE,
                     breaks = c("Bias toward control", "Bias toward treated")) +
  facet_wrap(~ stat) +
  theme(legend.position = "bottom", legend.title = element_blank(),
        legend.background = element_rect(fill = "transparent"),
        strip.background = element_rect(fill = "white"),
        axis.title = element_text(size = 16, face = "bold"))
ggsave("./graphs/t1t2_bal_fan-OMITTED.pdf", width = 40, height = 27, units = "cm")

# Politics
lp_pol <- bal_pol1$Balance[ , 2:4]
lp_pol$treat <- "Treatment: November"
lp_pol$names <- c("Support TS political statements: 1", "Support TS political statements: 2", "Support TS political statements: 3",
                  "Support TS political statements: 4", "Support TS political statements: 5", "Support TS political statements NA",
                  "Ideology: Very liberal", "Ideology: Liberal", "Ideology: Moderate", "Ideology: Conservative",
                  "Ideology: Very conservative", "Ideology: Not sure", "Party: Democrat", "Party: Republican", "Party: Independent",
                  "Party: Other", "Party: Not sure", "Political interest: Most of the time", "Political interest: Some of the time",
                  "Political interest: Only now and then", "Political interest: Hardly at all", "Political interest: Don't know",
                  "Political interest NA")
lp_pol <- rbind(lp_pol, data.frame(bal_pol2$Balance[ , 2:4],
                                   treat = "Treatment: November-August",
                                   names = lp_pol$names))
colnames(lp_pol)[1:3] <- c("(a) Absolute standardized MD",
                           "(b) KS statistics",
                           "(c) Complement of overlap")
lp_pol$bias <- if_else(lp_pol$`(a) Absolute standardized MD` < 0, "Bias toward control", "Bias toward treated")
lp_pol <- gather(lp_pol, stat, value, `(a) Absolute standardized MD`:`(c) Complement of overlap`, factor_key = TRUE)
lp_pol$order <- rep(rank(abs(lp_pol[lp_pol$treat == "Treatment: November" & lp_pol$stat == "(a) Absolute standardized MD", ]$value)), times = 6)
lp_pol[lp_pol$stat != "(a) Absolute standardized MD", ]$bias <- "n/a"

xl_pol <- " <-- More balance  Less balance -->    <-- More balance  Less balance -->    <-- More balance  Less balance -->"

ggplot(lp_pol, aes(x = abs(value), y = reorder(names, order),
                   color = bias)) + 
  geom_point(aes(shape = treat), size = 4) +
  geom_vline(xintercept = .1, linetype = 3) +
  geom_vline(xintercept = .2, linetype = 2) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, .1)) +
  xlab(xl_pol) + ylab("") +
  scale_y_discrete() +
  scale_shape_manual(values = c(16, 1)) +
  scale_color_taylor(palette = "taylor1989", reverse = TRUE,
                     breaks = c("Bias toward control", "Bias toward treated")) +
  facet_wrap(~ stat) +
  theme(legend.position = "bottom", legend.title = element_blank(),
        legend.background = element_rect(fill = "transparent"),
        strip.background = element_rect(fill = "white"),
        axis.title = element_text(size = 16, face = "bold"))
ggsave("./graphs/t1t2_bal_pol-OMITTED.pdf", width = 40, height = 27, units = "cm")

# Demographics
lp_dem <- bal_dem1$Balance[ , 2:4]
lp_dem$treat <- "Treatment: November"
lp_dem$names <- c("Age: 18-21", "Age: 22-25", "Age: 26-30", "Age: 31-35", "Age: 36-40",
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

lp_dem <- rbind(lp_dem, data.frame(bal_dem2$Balance[ , 2:4],
                                   treat = "Treatment: November-August",
                                   names = lp_dem$names))
colnames(lp_dem)[1:3] <- c("(a) Absolute standardized MD",
                           "(b) KS statistics",
                           "(c) Complement of overlap")
lp_dem$bias <- if_else(lp_dem$`(a) Absolute standardized MD` < 0, "Bias toward control", "Bias toward treated")
lp_dem <- gather(lp_dem, stat, value, `(a) Absolute standardized MD`:`(c) Complement of overlap`, factor_key = TRUE)
lp_dem$order <- rep(rank(abs(lp_dem[lp_dem$treat == "Treatment: November" & lp_dem$stat == "(a) Absolute standardized MD", ]$value)), times = 6)
lp_dem[lp_dem$stat != "(a) Absolute standardized MD", ]$bias <- "n/a"

xl_dem <- "<-- More balance  Less balance -->        <-- More balance  Less balance -->        <-- More balance  Less balance -->"

ggplot(lp_dem, aes(x = abs(value), y = reorder(names, order),
                   color = bias)) + 
  geom_point(aes(shape = treat), size = 4) +
  geom_vline(xintercept = .1, linetype = 3) +
  geom_vline(xintercept = .2, linetype = 2) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, .1)) +
  xlab(xl_dem) + ylab("") +
  scale_y_discrete() +
  scale_shape_manual(values = c(16, 1)) +
  scale_color_taylor(palette = "taylor1989", reverse = TRUE,
                     breaks = c("Bias toward control", "Bias toward treated")) +
  facet_wrap(~ stat) +
  theme(legend.position = "bottom", legend.title = element_blank(),
        legend.background = element_rect(fill = "transparent"),
        strip.background = element_rect(fill = "white"),
        axis.title = element_text(size = 16, face = "bold"))
ggsave("./graphs/t1t2_bal_dem-OMITTED.pdf", width = 40, height = 30, units = "cm")

## Day of sale treatment validity check----
vc1 <- lm(treat1 ~ tmerrors_count, data = df_sale)
vc2 <- lm(treat1 ~ factor(vqwait), data = df_sale)
vc3 <- lm(treat1 ~ seatgeek, data = df_sale)
vc4 <- lm(treat1 ~ factor(presaleprep), data = df_sale)
vc5 <- lm(treat1 ~ vfcodes, data = df_sale)
vc6 <- lm(treat1 ~ boost_count, data = df_sale)
vc7 <- lm(treat1 ~ friendsfail, data = df_sale)
vc8 <- lm(treat1 ~ tmerrors_count + factor(vqwait) + seatgeek + factor(presaleprep) +
                       vfcodes + boost_count + friendsfail, data = df_sale)

# Robust SEs
vc1_rse <- sqrt(diag(vcovHC(vc1, type = "HC3")))
vc2_rse <- sqrt(diag(vcovHC(vc2, type = "HC3")))
vc3_rse <- sqrt(diag(vcovHC(vc3, type = "HC3")))
vc4_rse <- sqrt(diag(vcovHC(vc4, type = "HC3")))
vc5_rse <- sqrt(diag(vcovHC(vc5, type = "HC3")))
vc6_rse <- sqrt(diag(vcovHC(vc6, type = "HC3")))
vc7_rse <- sqrt(diag(vcovHC(vc7, type = "HC3")))
vc8_rse <- sqrt(diag(vcovHC(vc8, type = "HC3")))

title <- "Ticketmaster Day-of-Sale Correlates of Treatment Status"
dv <- "Outcome: Failed to secure face-value tickets"
covs <- c("Total website errors", "Virtual queue: 1-3 hours", "Virtual queue: 3-4 hours",
          "Virtual queue: 4-7 hours", "Virtual queue: 7+ hours", "Stadium used SeatGeek",
          "Pre-sale prep: 1-2 hours", "Pre-sale prep: 3-4 hours", "Pre-sale prep: 4+ hours",
          "Total verified fan codes", "Total boosts", "Friends with ticket failure", "Intercept")

sink("./tables/tableA9.txt")
stargazer(vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8,
          type = "latex", title = title, dep.var.caption = dv, 
          model.names = FALSE, no.space = TRUE, covariate.labels = covs,
          dep.var.labels.include = FALSE, label = "treat_validity",
          initial.zero = TRUE, keep.stat = c("n", "adj.rsq"), star.cutoffs = 0.05,
          se = list(vc1_rse, vc2_rse, vc3_rse, vc4_rse, vc5_rse, vc6_rse, vc7_rse, vc8_rse))
sink()


## Balance for treat3----
covs_fan3 <- update(covs_fan1, treat3 ~ .)
covs_pol3 <- update(covs_pol1, treat3 ~ .)
covs_dem3 <- update(covs_dem1, treat3 ~ .)

bal_fan3 <- bal.tab(covs_fan3, data = df, stats = c("c", "ks"), continuous = "std",
                    binary = "std", thresholds = c(cor = .1))
bal_pol3 <- bal.tab(covs_pol3, data = df, stats = c("c", "ks"), continuous = "std",
                    binary = "std", thresholds = c(cor = .1))
bal_dem3 <- bal.tab(covs_dem3, data = df, stats = c("c", "ks"), continuous = "std",
                    binary = "std", thresholds = c(cor = .1))

# Treatment 3 love plot
lp_fan3 <- bal_fan3$Balance[ , 2:3]
lp_fan3$names <- c("Fan length:\n 2020-2022", "Fan length:\n 2014-2019", "Fan length:\n 2010-2013", "Fan length:\n 2009 or earlier",
                   "Fan length NA", "Fan intensity: 1", "Fan intensity: 2", "Fan intensity: 3", "Fan intensity: 4", "Fan intensity: 5",
                   "Fan intensity: 6", "Fan intensity: 7", "Fan intensity: 8", "Fan intensity: 9", "Fan intensity: 10", "Fan intensity NA",
                   "Fan social media", "Fan social media NA", "Stadium time\n zone: Central", "Stadium time\n zone: Eastern",
                   "Stadium time\n zone: Mountain", "Stadium time\n zone: Pacific", "Tour stop capacity")
colnames(lp_fan3)[1:2] <- c("correlation", "balanced")
lp_fan3$bias <- if_else(lp_fan3$correlation < 0, "Bias toward control", "Bias toward treated")
lp_fan3$order <- rank(abs(lp_fan3$correlation))
lp_fan3$category <- "(a) Swiftie fanship variables"

lp_pol3 <- bal_pol3$Balance[ , 2:3]
lp_pol3$names <- c("Support TS political\n statements: 1", "Support TS political\n statements: 2", "Support TS political\n statements: 3",
                   "Support TS political\n statements: 4", "Support TS political\n statements: 5", "Support TS political\n statements NA",
                   "Ideology: Very liberal", "Ideology: Liberal", "Ideology: Moderate", "Ideology: Conservative",
                   "Ideology: Very conservative", "Ideology: Not sure", "Party: Democrat", "Party: Republican", "Party: Independent",
                   "Party: Other", "Party: Not sure", "Political interest:\n Most of the time", "Political interest:\n Some of the time",
                   "Political interest:\n Only now and then", "Political interest:\n Hardly at all", "Political interest:\n Don't know",
                   "Political interest NA")
colnames(lp_pol3)[1:2] <- c("correlation", "balanced")
lp_pol3$bias <- if_else(lp_pol3$correlation < 0, "Bias toward control", "Bias toward treated")
lp_pol3$order <- rank(abs(lp_pol3$correlation))
lp_pol3$category <- "(b) Political variables"

lp_dem3 <- bal_dem3$Balance[ , 2:3]
lp_dem3$names <- c("Age: 18-21", "Age: 22-25", "Age: 26-30", "Age: 31-35", "Age: 36-40",
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
colnames(lp_dem3)[1:2] <- c("correlation", "balanced")
lp_dem3$bias <- if_else(lp_dem3$correlation < 0, "Bias toward control", "Bias toward treated")
lp_dem3$order <- rank(abs(lp_dem3$correlation))
lp_dem3$category <- "(c) Demographic variables"

lp3 <- rbind(lp_fan3, lp_pol3, lp_dem3)

theme_set(theme_bw(base_size = 22))
xl3 <- "<-- More balance  Less balance -->                                                                    <-- More balance  Less balance -->                                                                    <-- More balance  Less balance -->"

ggplot(lp3, aes(x = abs(correlation), y = reorder(names, order),
                    color = bias)) + 
  geom_point(size = 4) +
  geom_vline(xintercept = .1, linetype = 3) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, .1)) +
  xlab(xl3) + ylab("") +
  scale_y_discrete() +
  scale_shape_manual(values = 16) +
  scale_color_taylor(palette = "taylor1989", reverse = TRUE,
                     breaks = c("Bias toward control", "Bias toward treated")) +
  facet_wrap(~ category, scales = "free_y", nrow = 1) +
  theme(legend.position = "bottom", legend.title = element_blank(),
        legend.background = element_rect(fill = "transparent"),
        strip.background = element_rect(fill = "white"),
        axis.title = element_text(size = 16, face = "bold"))
ggsave("./graphs/t3_bal-OMITTED.pdf", width = 60, height = 35, units = "cm")


### Save workspace for '2_weighting.R'
save.image("./data/processed_data/balance.RData")
