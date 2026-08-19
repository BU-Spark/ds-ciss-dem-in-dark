"""
Ingest the 13 yearly daily_access_YYYY.csv exports, validate them against the
existing monthly panel, and merge into the analysis panel.

Run after all 13 Earth Engine tasks have finished and the CSVs are sitting in
this directory (or in ./daily/).

    python ingest_daily_access.py
"""

import os
import glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)) or "."

files = sorted(glob.glob(os.path.join(HERE, "daily_access_*.csv")) +
               glob.glob(os.path.join(HERE, "daily", "daily_access_*.csv")))
if not files:
    raise SystemExit(
        "No daily_access_*.csv found.\n"
        "Run gee_daily_blackmarble.js once per year (2012..2024), download the\n"
        "13 CSVs from Google Drive, and put them next to this script.")

print("found %d yearly files:" % len(files))
for f in files:
    print("   ", os.path.basename(f))

D = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print("\nrows: %d   constituencies: %d   months: %d"
      % (len(D), D.cons_name.nunique(), D.groupby(["year", "month"]).ngroups))

# ---------------------------------------------------------------- checks
print("\n" + "=" * 72)
print("CHECK 1  nights_valid should be seasonal: high Dec-Feb, low Jul-Aug")
print("=" * 72)
nv = D.groupby("month").nights_valid.mean().round(1)
print("  " + "  ".join(f"{m}:{v}" for m, v in nv.items()))
if nv.max() / max(nv.min(), 0.01) < 1.3:
    print("  !! nearly flat -- suspect the gap-filled band or a dead quality gate")
else:
    print("  OK: max/min = %.2f" % (nv.max() / nv.min()))

print("\n" + "=" * 72)
print("CHECK 2  lit_share must discriminate, not saturate")
print("=" * 72)
for t in ["0p25", "0p5", "1p0"]:
    c = f"lit_share_{t}"
    if c not in D.columns:
        continue
    v = D[c].dropna()
    print("  %-16s mean %.3f  p10 %.3f  p50 %.3f  p90 %.3f  %s"
          % (c, v.mean(), v.quantile(.1), v.quantile(.5), v.quantile(.9),
             "<-- saturated" if v.quantile(.1) > 0.9 or v.quantile(.9) < 0.1 else ""))

print("\n" + "=" * 72)
print("CHECK 3  Black Marble vs the VCMSLCFG monthly composite")
print("=" * 72)
L = pd.read_csv(os.path.join(HERE, "lights_raw.csv"))
mg = D.merge(L[["cons_name", "year", "month", "avg_rad"]],
             on=["cons_name", "year", "month"], how="inner")
mg = mg[(mg.bm_mean_rad > 0) & (mg.avg_rad > 0)]
r = np.corrcoef(np.log(mg.bm_mean_rad), np.log(mg.avg_rad))[0, 1]
ratio = (mg.bm_mean_rad / mg.avg_rad).median()
print("  matched constituency-months: %d" % len(mg))
print("  corr of logs: %+.3f   %s" % (r, "OK" if r > 0.9 else "!! TOO LOW - investigate"))
print("  median ratio bm_mean_rad / avg_rad: %.3f" % ratio)
print("  (a ratio near 10 or 0.1 means the 0.1 scale factor is wrong somewhere)")

# ---------------------------------------------------------------- merge
PANEL = os.path.join(HERE, "out", "panel_deseasonalized.csv")
if os.path.exists(PANEL):
    P = pd.read_csv(PANEL)
    keep = ["cons_name", "year", "month", "nights_valid", "bm_mean_rad",
            "lit_share_0p25", "lit_share_0p5", "lit_share_1p0"]
    keep = [c for c in keep if c in D.columns]
    P = P.drop(columns=[c for c in keep[3:] if c in P.columns])
    P = P.merge(D[keep], on=["cons_name", "year", "month"], how="left")
    # "average nights of electricity" in the client's language
    if "lit_share_0p5" in P.columns:
        days = pd.to_datetime(
            dict(year=P.year, month=P.month, day=1)).dt.days_in_month
        P["nights_lit_0p5"] = P.lit_share_0p5 * days
    P.to_csv(PANEL, index=False)
    print("\nmerged into %s -> %s" % (os.path.basename(PANEL), P.shape))
    print("match rate: %.1f%%" % (100 * P.lit_share_0p5.notna().mean()))
else:
    print("\npanel not found; wrote nothing. Run seasonal_analysis.py first.")

D.to_csv(os.path.join(HERE, "daily_access_all.csv"), index=False)
print("wrote daily_access_all.csv (%d rows)" % len(D))
