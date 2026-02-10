# =========================================
# Problem Set 1 - Applied Stats II
# PS01.R  (Reproducible)
# =========================================

# ----------------------------
# Housekeeping
# ----------------------------
rm(list = ls())
options(stringsAsFactors = FALSE)
set.seed(123)

# ============================================================
# Question 1: Implement a KS test where reference distribution is Normal
# Data: 1000 Cauchy RVs, location=0, scale=1
# ============================================================

# ---- KS statistic D for testing against N(0,1) ----
ks_D_against_norm <- function(x) {
  # x: numeric vector
  x <- sort(as.numeric(x))
  n <- length(x)
  
  # Theoretical CDF at ordered points
  F0 <- pnorm(x)
  
  # i/n and (i-1)/n
  i <- seq_len(n)
  D_plus  <- max(i / n - F0)          # sup (F_n - F0)
  D_minus <- max(F0 - (i - 1) / n)    # sup (F0 - F_n^-)
  
  D <- max(D_plus, D_minus)
  return(D)
}

# ---- Correct series p-value for KS (asymptotic) ----
# Use lambda = sqrt(n) * D
# P(D_n <= d) ≈ 1 - 2 * sum_{k=1}^\infty (-1)^{k-1} exp(-2 k^2 lambda^2)
# so p-value = P(D_n >= d) ≈ 2 * sum_{k=1}^\infty (-1)^{k-1} exp(-2 k^2 lambda^2)
ks_pvalue_asymptotic <- function(D, n, K = 200) {
  if (!is.finite(D) || D <= 0) return(NA_real_)
  lambda <- sqrt(n) * D
  k <- 1:K
  pval <- 2 * sum(((-1)^(k - 1)) * exp(-2 * (k^2) * (lambda^2)))
  
  # Numerical guard
  pval <- max(min(pval, 1), 0)
  return(pval)
}


# ---- Wrapper: run "my KS test" vs Normal ----
my_ks_test_norm <- function(x, K = 200) {
  D <- ks_D_against_norm(x)
  pval <- ks_pvalue_asymptotic(D, n = length(x), K = K)
  list(D = D, p_value_series = pval, K = K, n = length(x))
}


cat("============================================================\n")
cat("Q1: KS test implementation (reference: N(0,1))\n")
cat("============================================================\n")

# Generate data (Cauchy)
set.seed(123)
x_cauchy <- rcauchy(1000, location = 0, scale = 1)

# My implementation
q1_res <- my_ks_test_norm(x_cauchy, K = 500)

cat("My KS D statistic (vs N(0,1)):", q1_res$D, "\n")
cat("My p-value (series approx, K =", q1_res$K, "):", q1_res$p_value_series, "\n")

# Compare with built-in ks.test
# Note: ks.test default uses an asymptotic approximation for the p-value.
ks_builtin <- ks.test(x_cauchy, "pnorm")

cat("\nBuilt-in ks.test results:\n")
cat("ks.test D:", unname(ks_builtin$statistic), "\n")
cat("ks.test p-value:", ks_builtin$p.value, "\n")

# Small diagnostic difference
cat("\nDiagnostics:\n")
cat("Absolute difference in D (my D vs ks.test D):",
    abs(q1_res$D - unname(ks_builtin$statistic)), "\n")


# ============================================================
# Question 2: Estimate OLS regression using BFGS and compare to lm()
# Data generation given in handout
# ============================================================

cat("\n\n============================================================\n")
cat("Q2: OLS via BFGS (quasi-Newton) and compare with lm()\n")
cat("============================================================\n")

# Generate data as required
set.seed(123)
data <- data.frame(x = runif(200, 1, 10))
data$y <- 0 + 2.75 * data$x + rnorm(200, 0, 1.5)

# Fit using lm for reference
fit_lm <- lm(y ~ x, data = data)
coef_lm <- coef(fit_lm)

cat("lm() coefficients:\n")
print(coef_lm)

# Define SSE objective function for OLS
# par = c(beta0, beta1)
sse_ols <- function(par, x, y) {
  b0 <- par[1]
  b1 <- par[2]
  resid <- y - (b0 + b1 * x)
  sum(resid^2)
}

# Run BFGS optimization
# Initial guess: (0,0) is fine for convex quadratic problem
init <- c(0, 0)

opt_bfgs <- optim(
  par = init,
  fn = sse_ols,
  x = data$x,
  y = data$y,
  method = "BFGS",
  control = list(reltol = 1e-12)
)

coef_bfgs <- opt_bfgs$par
names(coef_bfgs) <- c("(Intercept)", "x")

cat("\nBFGS (optim) coefficients:\n")
print(coef_bfgs)

# Compare SSE
sse_lm <- sum(residuals(fit_lm)^2)
sse_bfgs <- opt_bfgs$value

cat("\nSSE comparison:\n")
cat("SSE from lm():  ", sse_lm, "\n")
cat("SSE from BFGS:  ", sse_bfgs, "\n")
cat("Abs diff in SSE:", abs(sse_lm - sse_bfgs), "\n")

# Compare coefficients numerically
cat("\nCoefficient comparison (BFGS - lm):\n")
print(coef_bfgs - coef_lm)

cat("\nall.equal(BFGS, lm) for coefficients:\n")
print(all.equal(as.numeric(coef_bfgs), as.numeric(coef_lm), tolerance = 1e-8))

cat("\nDone.\n")
