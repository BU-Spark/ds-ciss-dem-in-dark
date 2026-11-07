"""
Build analytic panel: drift-robust metrics with explicit below-detection handling.

Inputs: constituency_lights_raw.csv (GEE raw radiance export)
Outputs: constituency_lights_panel.csv (with level, onset, change metrics)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# project layout: scripts/nightlights/ -> project root is two levels up
HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "nightlights"
RAW = DATA / "constituency_lights_raw.csv"
OUT = DATA / "constituency_lights_panel.csv"

# Detection threshold: median_masked == 0 IS the product's own censoring
# ("below detection"). We use that non-arbitrary line -- lit = mean > 0, dark = 0 --
# rather than a hand-picked floor. So a rural area going 0 -> 0.003 counts as
# "became detectable" (accessibility gained), which a floor would wrongly hide.
df = pd.read_csv(RAW)

# auto-detect which years actually made it into the export (mean_<year> columns)
YEARS = sorted(int(c.split("_")[1]) for c in df.columns if c.startswith("mean_"))
print("years detected in raw export:", YEARS, "| threshold: lit = mean > 0")

# --- sanity checks up front (fail loud, not silent) ---
assert len(df) == 275, f"expected 275 rows, got {len(df)}"
empty_id = df[df["con_id"].isna()]
if len(empty_id):
    print(f"WARNING: {len(empty_id)} rows have empty con_id "
          f"(fix upstream): {list(empty_id['cons_name'])}")

# Radiance can be slightly negative after masking; clip to 0 before logs/shares.
for y in YEARS:
    df[f"mean_{y}"] = df[f"mean_{y}"].clip(lower=0)
    df[f"sum_{y}"] = df[f"sum_{y}"].clip(lower=0)

# --- within-year level metrics ---
for y in YEARS:
    df[f"share_{y}"] = df[f"sum_{y}"] / df[f"sum_{y}"].sum()
    df[f"pct_rank_{y}"] = df[f"mean_{y}"].rank(pct=True) * 100
    df[f"log_mean_{y}"] = np.log1p(df[f"mean_{y}"])
    df[f"status_{y}"] = np.where(df[f"mean_{y}"] > 0, "lit", "dark")

# --- onset year: first year with any detectable light (mean > 0) ---
def onset(row):
    for y in YEARS:
        if row[f"mean_{y}"] > 0:
            return y
    return ""
df["onset_year"] = df.apply(onset, axis=1)

# --- change metrics between consecutive election years ---
for a, b in zip(YEARS[:-1], YEARS[1:]):
    df[f"d_rank_{a}_{b}"] = df[f"pct_rank_{b}"] - df[f"pct_rank_{a}"]
    df[f"d_share_{a}_{b}"] = df[f"share_{b}"] - df[f"share_{a}"]
    df[f"abs_{a}_{b}"] = df[f"mean_{b}"] - df[f"mean_{a}"]
    both_pos = (df[f"mean_{a}"] > 0) & (df[f"mean_{b}"] > 0)
    # log-ratio (~% change) needs a non-zero denominator -> only where both > 0;
    # else NaN. (Very small values are still noisy, so prefer d_rank / abs there.)
    df[f"lgr_mean_{a}_{b}"] = np.where(
        both_pos, np.log(df[f"mean_{b}"] / df[f"mean_{a}"].where(both_pos)), np.nan
    )

# --- 2-category split by 2013 detection (data's own zero, non-arbitrary) ---
a0, bN = YEARS[0], YEARS[-1]
df["base2013"] = np.where(df[f"mean_{a0}"] > 0, "lit_2013", "dark_2013")
print("base2013 counts:", df["base2013"].value_counts().to_dict())
# of the dark-in-2013 group, how many are detectable each year (accessibility)
dark0 = df["base2013"] == "dark_2013"
print("dark_2013 group becoming detectable:",
      {y: int((df.loc[dark0, f"mean_{y}"] > 0).sum()) for y in YEARS},
      f"(n={int(dark0.sum())})")

# order columns: keys, per-year blocks, onset/category, change blocks
key_cols = ["con_id", "cons_name"]
year_cols = [c for y in YEARS for c in
             (f"mean_{y}", f"sum_{y}", f"share_{y}", f"pct_rank_{y}",
              f"log_mean_{y}", f"status_{y}")]
cls_cols = ["onset_year", "base2013"]
chg_cols = [c for c in df.columns
            if c.startswith(("d_rank_", "d_share_", "abs_", "lgr_mean_"))]
df = df[key_cols + year_cols + cls_cols + chg_cols]

df.to_csv(OUT, index=False)
print(f"wrote {OUT.name}: {len(df)} rows, {len(df.columns)} cols")
print("level -> pct_rank_<y> (fair 0-1 view when /100) + log_mean_<y> (map); "
      "category -> base2013 (dark/lit by data's own 0); change -> d_rank + abs; "
      "lgr only where both years > 0")
