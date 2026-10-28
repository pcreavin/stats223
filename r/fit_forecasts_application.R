# Adapted copy of external/dh_replication/applications/apply_SysRisk_Monitoring.R (Milestone D).
# GARCH fitting, forecasts and violation indicators for the application (Figures 5-6), plus their
# UNCHANGED SysRisk_Monitoring on those indicators. Python (experiments/05_application_fig5_fig6.py)
# does our own monitoring on the outputs.
#
# Changes vs their script (everything else, incl. windows, fit specs, solver settings and the nu
# extraction, is copied verbatim):
#  1. Paths relative to the git root via r/utils_io.R (run from the repo root); their R/ files are
#     sourced unchanged from external/dh_replication/R/.
#  2. source("simulations/DGP.R") dropped (file is missing; DGP.R is not needed here).
#  3. library(tidyverse, copula, patchwork) replaced by dplyr/tidyr/tibble/lubridate (the others are
#     not in renv; copula is unused here, patchwork only served the plots).
#  4. prob_level_list reduced to alpha = beta = 0.95.
#  5. set.seed(1) before each ViolSeq_tDist call (their pmvt calls were unseeded).
#  6. The loop over windows runs via parallel::mclapply (one function per window; body unchanged apart
#     from recording fit time and the convergence flag); specs from the command line.
#  7. Plotting, saveRDS(appl_res) and the CoES/MES/long-format reshaping removed.
#  8. Output block added: data/derived/appl/{spec}_{window}.csv (dates, OOS losses, D_t, upper-triangle
#     R_t, indicators), nu_{spec}_{window}.csv, rmon_{spec}_{window}_{covar|rev}.csv (normalised paths)
#     and rmon_..._rl.csv (run lengths), fit_log.csv. `date` comes from the na.omit-ed matrix rows;
#     `date_theirs` is their label tail(data_Assets_wide, n), taken before na.omit.
#
# Run from the repo root: Rscript r/fit_forecasts_application.R [spec ...]   (default CCC_norm DCC_std)
source("r/utils_io.R")
suppressMessages({
  library(dplyr); library(tidyr); library(tibble); library(lubridate)
  library(rmgarch); library(parallel)
})

ext <- "external/dh_replication"
source(file.path(ext, "R/CriticalValues.R"))
source(file.path(ext, "R/mu.R"))
source(file.path(ext, "R/ViolSeq_tDist.R"))
source(file.path(ext, "R/Detectors.R"))
source(file.path(ext, "R/SysRisk_Monitoring.R"))

out_dir <- "data/derived/appl"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

args <- commandArgs(trailingOnly = TRUE)
GARCH_specs <- if (length(args)) args else c("CCC_norm", "DCC_std")


# ===========================================================
# Load data and transform
# ===========================================================
data_Assets_all <- readRDS(file = file.path(ext, "applications/data/data_Assets.rds"))

stock_names <- c("BAC", "C", "GS", "JPM", "WFC")
index_name <- "SPF"

data_Assets <- data_Assets_all %>%
  dplyr::filter(Asset %in% c(index_name, stock_names))
K <- length(stock_names)

iota <- 0.1
n <- 1000
m <- 250

prob_level_list <- list(c(alpha=0.95, beta=0.95))

end_dates <- as_date(c("2009-09-15", "2016-12-30", "2021-03-15", "2025-10-23"))
data_Assets_wide_full <- pivot_wider(data_Assets, names_from = Asset, values_from = NegReturn) %>% na.omit()
end_idx <- match(end_dates, data_Assets_wide_full$Date)
start_dates_raw <- data_Assets_wide_full %>% pull(Date) %>% .[end_idx-2500]
start_dates <- pmax(as_date("1999-05-05"), start_dates_raw) # Because GS starts trading at  "1999-05-05"

settings_tbl <- tibble(name = c("FinCrisis", "CalmTime", "COVID", "TrumpTariff"),
                       start_dates = start_dates,
                       end_dates = end_dates,
                       event_dates =  as_date(c("2008-09-15", NA, "2020-03-13", "2025-04-02")))

df_CritVal <- readRDS(file.path(ext, "simulations/data/df_CritVal_full_20250805.rds")) %>%
  as_tibble() %>%
  mutate(SysRisk = recode(SysRisk, "CoVaR_Exposure" = "CoVaR_Reverse"))

asset_cols <- c(index_name, stock_names)
ut <- which(upper.tri(diag(K+1)), arr.ind = TRUE)
ut <- ut[order(ut[, "row"], ut[, "col"]), , drop = FALSE]


run_setting <- function(i_setting, GARCH_spec_name) {
  t0 <- Sys.time()
  setting_name <- settings_tbl[[i_setting, "name"]]
  start_date <- settings_tbl[[i_setting, "start_dates"]]
  end_date <- settings_tbl[[i_setting, "end_dates"]]

  # Transform data
  data_Assets_wide <- pivot_wider(data_Assets, names_from = Asset, values_from = NegReturn) %>%
    dplyr::filter(Date >= start_date, Date < end_date)

  data_Assets_matrix <- data_Assets_wide %>%
    dplyr::select(all_of(c(index_name, stock_names))) %>%
    na.omit() %>%
    as.matrix()
  dates_matrix <- data_Assets_wide %>%
    dplyr::select(all_of(c("Date", index_name, stock_names))) %>%
    na.omit() %>% pull(Date)

  nE <- dim(data_Assets_matrix)[1]
  E <- nE - n
  conv <- NA

  if (GARCH_spec_name %in% c("DCC_norm", "DCC_std")){
    if (GARCH_spec_name == "DCC_std"){
      garch11.spec.std = ugarchspec(mean.model = list(armaOrder = c(0,0), include.mean=FALSE),
                                    variance.model = list(garchOrder = c(1,1), model = "sGARCH"),
                                    distribution.model = "std")
      DCC.spec = dccspec(uspec = multispec(replicate(K+1, garch11.spec.std)),
                         dccOrder = c(1,1),
                         distribution = "mvt")
    } else {
      garch11.spec.Gaussian = ugarchspec(mean.model = list(armaOrder = c(0,0), include.mean=FALSE),
                                         variance.model = list(garchOrder = c(1,1), model = "sGARCH"),
                                         distribution.model = "norm")
      DCC.spec = dccspec(uspec = multispec(replicate(K+1, garch11.spec.Gaussian)),
                         dccOrder = c(1,1),
                         distribution = "mvnorm")
    }

    # DCC model fitting
    dcc.fit = dccfit(DCC.spec, data=data_Assets_matrix, out.sample=n,
                     solver=c("hybrid", "solnp"), solver.control = list(tol=1e-12),
                     fit.control = list(eval.se = FALSE, stationarity = TRUE, scale = FALSE))
    conv <- dcc.fit@mfit$convergence
    dcc.forecast <- dccforecast(dcc.fit, n.ahead = 1, n.roll = n)

    # Extract conditional correlation matrices and convert to array
    R.t.list          <- rcor(dcc.forecast, type = "R")
    R.t <- array(as.numeric(unlist(R.t.list)), dim=c(K+1, K+1, n+1))
    cov.list <- rcov(dcc.forecast)
    cov.array <- array(as.numeric(unlist(cov.list)), dim=c(K+1, K+1, n+1))
    D.t <- sapply(1:dim(cov.array)[3], function(i) sqrt(diag(cov.array[,,i])))

    # Extract degree of freedoms
    if (DCC.spec@umodel$modeldesc$distribution[1] == "norm" ){nu <- rep.int(10000, times = K+2)}
    if (DCC.spec@umodel$modeldesc$distribution[1] == "std" ){nu <- round(dcc.fit@mfit$coef[grep("shape", names(dcc.fit@mfit$coef))])}

    D.t <- D.t[, -(n+1)]
    R.t <- R.t[,,-(n+1)]

  } else if (GARCH_spec_name %in% c("CCC_norm", "CCC_std")) {
    if (GARCH_spec_name == "CCC_std"){
      resid_name <- "std"
    } else {
      resid_name <- "norm"
    }

    CCC_univGARCH_fits <- lapply(1:(K+1), function(i) {
      ugarchfit(spec = ugarchspec(
        variance.model = list(garchOrder = c(1,1)),
        mean.model = list(armaOrder = c(0,0), include.mean=FALSE),
        distribution.model = resid_name),
        data = data_Assets_matrix[, i],
        out.sample=n)
    })
    conv <- max(sapply(CCC_univGARCH_fits, function(f) f@fit$convergence))

    std_resid <- sapply(CCC_univGARCH_fits, function(f) residuals(f, standardize = TRUE))
    R_CCC <- array(rep(cor(std_resid), n), dim = c(K+1, K+1, n))

    if (GARCH_spec_name == "CCC_std"){
      nu <- round(sapply(CCC_univGARCH_fits, function(f) coef(f)["shape"]))
    } else {
      nu <- rep.int(10000, times = K+1)
    }

    D.t_CCC <- sapply(CCC_univGARCH_fits, function(fit) {
      fc <- ugarchforecast(fit, n.ahead = 1, n.roll = n)
      sigma <- sigma(fc)
      return(sigma)
    })

    R.t <- R_CCC
    D.t <- t(D.t_CCC[-(n+1),])
  }
  t_fit <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  for (index_prob_level in 1:length(prob_level_list)){
    alpha <- prob_level_list[[index_prob_level]][1]
    beta <- prob_level_list[[index_prob_level]][2]

    set.seed(1)
    ViolSeq_values <- ViolSeq_tDist(data_OOS = data_Assets_matrix[(E+1):(E+n),],
                                    R.t = R.t,
                                    D.t = D.t,
                                    nu = nu,
                                    alpha = alpha,
                                    beta = beta)

    I_VaR <- ViolSeq_values$I_VaR
    I_CoVaR <- ViolSeq_values$I_CoVaR
    H_CoES <- ViolSeq_values$H_CoES
    H_MES <- ViolSeq_values$H_MES
    I_VaR_Rev <- ViolSeq_values$I_VaR_Rev
    I_CoVaR_Rev <- ViolSeq_values$I_CoVaR_Rev

    df_CritVal_select <- df_CritVal %>%
      dplyr::filter(alpha %in% c(0,!!alpha), beta==!!beta, m==!!m, K==!!K, n==!!n)

    dates_theirs <- tail(data_Assets_wide, n) %>% pull(Date)
    res_Monitoring <- SysRisk_Monitoring(I_VaR=I_VaR, I_CoVaR=I_CoVaR, H_CoES=H_CoES, H_MES=H_MES,
                                         dates= dates_theirs,
                                         asset_names=c(index_name, stock_names),
                                         iota=iota, m=m, alpha=alpha, beta=beta, risk_measure=c("CoVaR","CoES","MES"), a=0.5,
                                         critical_values=df_CritVal_select)

    res_Monitoring_RevCoVaR <- SysRisk_Monitoring(I_VaR=I_VaR_Rev, I_CoVaR=I_CoVaR_Rev,
                                                  dates= dates_theirs,
                                                  asset_names=c(index_name, stock_names),
                                                  iota=iota, m=m, alpha=alpha, beta=beta, risk_measure=c("CoVaR"), a=0.5,
                                                  critical_values=df_CritVal_select)

    # ---------------------------------------------------------- output block (new)
    tag <- paste0(GARCH_spec_name, "_", setting_name)
    dates_oos <- tail(dates_matrix, n)
    loss <- data_Assets_matrix[(E+1):(E+n), ]
    df <- data.frame(date = as.character(dates_oos), date_theirs = as.character(dates_theirs))
    for (j in 1:(K+1)) df[[paste0("loss_", asset_cols[j])]] <- as.numeric(loss[, j])
    for (j in 1:(K+1)) df[[paste0("sd_", asset_cols[j])]] <- as.numeric(D.t[j, ])
    for (r in seq_len(nrow(ut))) {
      i <- ut[r, "row"]; j <- ut[r, "col"]
      df[[paste0("r_", asset_cols[i], "_", asset_cols[j])]] <- as.numeric(R.t[i, j, ])
    }
    df$I_VaR <- as.integer(I_VaR)
    for (k in 1:K) df[[paste0("I_CoVaR_", stock_names[k])]] <- as.integer(I_CoVaR[k, ])
    for (k in 1:K) df[[paste0("I_VaR_Rev_", stock_names[k])]] <- as.integer(I_VaR_Rev[k, ])
    for (k in 1:K) df[[paste0("I_CoVaR_Rev_", stock_names[k])]] <- as.integer(I_CoVaR_Rev[k, ])
    write_csv_exact(df, file.path(out_dir, paste0(tag, ".csv")))

    nm <- names(nu); if (is.null(nm)) nm <- rep("", length(nu))
    write_csv_exact(data.frame(idx = seq_along(nu), name = nm, nu = as.numeric(nu)),
                    file.path(out_dir, paste0("nu_", tag, ".csv")))

    date_win <- as.character(tail(dates_oos, n-m+1))
    for (mode in c("covar", "rev")) {
      res <- if (mode == "covar") res_Monitoring else res_Monitoring_RevCoVaR
      det <- as.data.frame(res$df_CoVaR_detector)
      det$Date <- as.character(det$Date)
      det <- cbind(date = date_win, date_theirs = det$Date, det[, setdiff(names(det), "Date")])
      write_csv_exact(det, file.path(out_dir, paste0("rmon_", tag, "_", mode, ".csv")))
      write_csv_exact(as.data.frame(res$run_length), file.path(out_dir, paste0("rmon_", tag, "_", mode, "_rl.csv")))
    }
  }
  t_all <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat(sprintf("%s %s: E=%d, fit %.1fs, total %.1fs, convergence %s, nu = %s\n", GARCH_spec_name, setting_name,
              E, t_fit, t_all, conv, paste(nu, collapse = " ")))
  data.frame(spec = GARCH_spec_name, window = setting_name, E = E, fit_sec = t_fit, total_sec = t_all,
             convergence = conv)
}


start.time <- Sys.time()
log_rows <- list()
for (GARCH_spec_name in GARCH_specs) {
  res <- mclapply(1:nrow(settings_tbl), run_setting, GARCH_spec_name = GARCH_spec_name,
                  mc.cores = 4, mc.preschedule = FALSE)
  for (r in res) if (inherits(r, "try-error")) stop(r)
  log_rows <- c(log_rows, res)
}
log_df <- do.call(rbind, log_rows)
log_file <- file.path(out_dir, "fit_log.csv")
if (file.exists(log_file)) log_df <- rbind(read.csv(log_file) %>% filter(!spec %in% GARCH_specs), log_df)
write_csv_exact(log_df, log_file)
cat(sprintf("run time: %.1f min\n", as.numeric(difftime(Sys.time(), start.time, units = "mins"))))
