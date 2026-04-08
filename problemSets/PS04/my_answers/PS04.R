#####################
# load libraries
# set wd
# clear global .envir
#####################

# remove objects
rm(list=ls())
# detach all libraries
detachAllPackages <- function() {
  basic.packages <- c("package:stats", "package:graphics", "package:grDevices", "package:utils", "package:datasets", "package:methods", "package:base")
  package.list <- search()[ifelse(unlist(gregexpr("package:", search()))==1, TRUE, FALSE)]
  package.list <- setdiff(package.list, basic.packages)
  if (length(package.list)>0)  for (package in package.list) detach(package,  character.only=TRUE)
}
detachAllPackages()

# load libraries
pkgTest <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[,  "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg,  dependencies = TRUE)
  sapply(pkg,  require,  character.only = TRUE)
}

# here is where you load any necessary packages
# ex: stringr
# lapply(c("stringr"),  pkgTest)

lapply(c("nnet", "MASS", "eha", "survival", "sampleSelection", "rstudioapi"), pkgTest)

# set wd for current folder
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#####################
# Problem 1
#####################

# load data on child mortality by mother's background and child gender
data("child", package = "eha")

str(child)
names(child)
summary(child)
head(child)

# Cox proportional hazards model
library(survival)

model_q1 <- coxph(Surv(enter, exit, event) ~ m.age + sex, data = child)
summary(model_q1)

# hazard ratios and confidence intervals
exp(coef(model_q1))
exp(confint(model_q1))

# proportional hazards assumption check
cox.zph(model_q1)

#####################
# Problem 2
#####################

# load data
disaster_data <- read.csv("https://raw.githubusercontent.com/ASDS-TCD/StatsII_2026/refs/heads/main/datasets/disaster_response.csv")
str(disaster_data)
names(disaster_data)
summary(disaster_data)
head(disaster_data)

summary(disaster_data[, c("binContribution",
                          "originalContributionMillionUSDLogged",
                          "occurrences",
                          "deathsEM",
                          "normalizedDamageEMLogged")])

library(sampleSelection)

model_q2 <- selection(
  selection = binContribution ~ occurrences + deathsEM + normalizedDamageEMLogged,
  outcome   = originalContributionMillionUSDLogged ~ occurrences + deathsEM + normalizedDamageEMLogged,
  data = disaster_data,
  method = "ml"
)

summary(model_q2)
