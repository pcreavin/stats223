# Exact cross-check of the Python detector port against their unchanged R code.
# Reads data/derived/xcheck/indicators.csv (columns I_1..I_J, n rows) and params.csv
# (m, p, mu_UC, sd_UC, mu_IID, sd_IID), runs R/Detectors.R::CoVaR_Detector on every column
# and writes data/derived/xcheck/detectors_R.csv (UC_j, IID_j columns).
# Run from the repo root: Rscript r/xcheck_detectors.R
source("r/utils_io.R")
suppressMessages({library(zoo)})
source("external/dh_replication/R/Detectors.R")

dir <- "data/derived/xcheck"
I_df <- read.csv(file.path(dir, "indicators.csv"))
par <- read.csv(file.path(dir, "params.csv"))
m <- par$m; p <- par$p; n <- nrow(I_df)
ms <- list(mu_UC = par$mu_UC, sd_UC = par$sd_UC, mu_IID = par$mu_IID, sd_IID = par$sd_IID)

out <- list()
for (j in seq_along(I_df)) {
  det <- CoVaR_Detector(I = I_df[[j]], m = m, n = n, p = p, musigma_CoVaR = ms)
  out[[paste0("UC_", j)]] <- det$Det_UC
  out[[paste0("IID_", j)]] <- det$Det_IID
}
write_csv_exact(as.data.frame(out), file.path(dir, "detectors_R.csv"))
