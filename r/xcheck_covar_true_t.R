# Cross-check of the Python c(rho, nu) tables against their unchanged CoVaR.true.t (R/ViolSeq_tDist.R).
# Reads data/derived/xcheck/covar_points.csv (columns rho, s1, s2, nu, level) and evaluates
# CoVaR.true.t(alpha = level, beta = level, nu, H = [[s1^2, rho s1 s2], [rho s1 s2, s2^2]]) under seeds 1..5
# (pmvt is randomised QMC and their code sets no seed), writing data/derived/xcheck/covar_R.csv.
# Run from the repo root: Rscript r/xcheck_covar_true_t.R
source("r/utils_io.R")
suppressMessages({library(mvtnorm)})
source("external/dh_replication/R/ViolSeq_tDist.R")

dir <- "data/derived/xcheck"
pts <- read.csv(file.path(dir, "covar_points.csv"))
out <- data.frame()
for (r in seq_len(nrow(pts))) {
  p <- pts[r, ]
  H <- matrix(c(p$s1^2, p$rho * p$s1 * p$s2, p$rho * p$s1 * p$s2, p$s2^2), 2, 2)
  for (seed in 1:5) {
    set.seed(seed)
    out <- rbind(out, data.frame(point = r, seed = seed,
                                 covar = CoVaR.true.t(alpha = p$level, beta = p$level, nu = p$nu, H = H)))
  }
}
write_csv_exact(out, file.path(dir, "covar_R.csv"))
