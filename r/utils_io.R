# Shared helpers for the r/ scripts (new file, not a copy of their code).
# Activate the project's renv library (r/renv) when scripts are run from the repo root.
if (requireNamespace("renv", quietly = TRUE) && dir.exists("r/renv")) {
  suppressMessages(renv::load("r", quiet = TRUE))
}

# write.csv with full double precision (write.csv alone rounds to 15 significant digits)
write_csv_exact <- function(df, file) {
  df[] <- lapply(df, function(x) if (is.double(x)) sprintf("%.17g", x) else x)
  write.csv(df, file, row.names = FALSE, quote = FALSE)
}
