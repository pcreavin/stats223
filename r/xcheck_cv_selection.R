# Adapted copy of Steps 4-5 (i) and (iv) of external/dh_replication/R/CriticalValues.R
# (function Critical_values), used for an exact cross-check of the Python threshold selection.
# Changes w.r.t. the original:
#   1. sup.VaR.MCS / sup.CoVaR.MCS are read from data/derived/xcheck/sups.csv
#      (columns sup_VaR, sup_CoVaR) instead of being simulated (Steps 1-3 removed);
#   2. K and iota are read from data/derived/xcheck/cv_params.csv (columns K, iota; one row per case);
#   3. CoES / MES steps (ii), (iii) removed; moments not computed;
#   4. results are written to data/derived/xcheck/cv_selection_R.csv (full precision, r/utils_io.R);
#   5. wrapped in a loop over the rows of cv_params.csv.
# The quantile / ">=" / which.min code below is copied verbatim.
# Run from the repo root: Rscript r/xcheck_cv_selection.R
source("r/utils_io.R")
dir <- "data/derived/xcheck"
sups <- read.csv(file.path(dir, "sups.csv"))
pars <- read.csv(file.path(dir, "cv_params.csv"))
sup.VaR.MCS <- sups$sup_VaR
sup.CoVaR.MCS <- sups$sup_CoVaR

res <- data.frame()
for (r in seq_len(nrow(pars))) {
  K <- pars$K[r]; iota <- pars$iota[r]

  # Step 4: (1-nu)-quantiles of sup Measure_{MCS}(T)
  nu.grid    <- seq.int(0, 1, by=0.0001)
  c.nu.VaR   <- quantile( sup.VaR.MCS, probs = 1-nu.grid )
  c.nu.CoVaR <- quantile( sup.CoVaR.MCS, probs = 1-nu.grid )

  # Step 5 (i): For CoVaR monitoring
  cv.help <- numeric( length(nu.grid) )
  for(i in 1:length(nu.grid)){
    cv.help[i] <- mean( 1*(sup.VaR.MCS >= c.nu.VaR[i]) ) + K * mean( 1*(sup.CoVaR.MCS >= c.nu.CoVaR[i]) ) -
      K * mean( 1*(sup.CoVaR.MCS >= c.nu.CoVaR[i])*(sup.VaR.MCS >= c.nu.VaR[i]) )
  }
  target.index <- which.min( abs(cv.help - iota) )
  c.iota.VaR.V <- c.nu.VaR[target.index]
  c.iota.CoVaR <- c.nu.CoVaR[target.index]

  # Step 5 (iv): For Reverse CoVaR monitoring
  cv.help <- numeric( length(nu.grid) )
  for(i in 1:length(nu.grid)){
    cv.help[i] <- mean( 1*(sup.VaR.MCS >= c.nu.VaR[i] | sup.CoVaR.MCS >= c.nu.CoVaR[i]) )
  }
  target.index <- which.min( abs(cv.help - iota/K) )
  c.iota.VaR.CRev <- c.nu.VaR[target.index]
  c.iota.CoVaR.Rev <- c.nu.CoVaR[target.index]

  res <- rbind(res, data.frame(K = K, iota = iota,
                               c_VaR = c.iota.VaR.V, c_SysRisk = c.iota.CoVaR,
                               c_VaR_rev = c.iota.VaR.CRev, c_SysRisk_rev = c.iota.CoVaR.Rev))
}
write_csv_exact(res, file.path(dir, "cv_selection_R.csv"))
