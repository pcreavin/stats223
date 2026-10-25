"""Central file-system locations."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "external" / "dh_replication"
DERIVED = ROOT / "data" / "derived"
SIMS = DERIVED / "sims"
FIGURES = ROOT / "figures"
R_DIR = ROOT / "r"

CRIT_VALUES_RDS = EXT / "simulations" / "data" / "df_CritVal_full_20250805.rds"
ASSETS_RDS = EXT / "applications" / "data" / "data_Assets.rds"
APPL_RES_RDS = EXT / "applications" / "data" / "appl_res_20251024.rds"
SIM_BREAKPOINT_RDS = EXT / "simulations" / "data" / "sim_BreakPoint_20251006.rds"
SIM_PARAMMISSPEC_RDS = EXT / "simulations" / "data" / "sim_ParamMisspec_20251006.rds"


def ensure_dirs() -> None:
    for d in (DERIVED, SIMS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
