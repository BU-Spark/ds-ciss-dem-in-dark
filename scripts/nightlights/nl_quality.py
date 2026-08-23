"""
Quality-check layer: gate (cf_cvg==0 -> NaN) and missing-rate diagnostics.

Inputs: constituency_lights_monthly_raw.csv (GEE export, long format)
Outputs: Derived fields (is_missing, avg_rad_raw) for downstream use
"""

import pandas as pd
from pathlib import Path


def load_raw(path: Path) -> pd.DataFrame:
    """Load the raw monthly long table and coerce numeric types."""
    df = pd.read_csv(path)
    df["con_id"] = df["con_id"].fillna(df["cons_name"])   # a few rows miss con_id
    df["cf_cvg"] = pd.to_numeric(df["cf_cvg"], errors="coerce")
    df["avg_rad"] = pd.to_numeric(df["avg_rad"], errors="coerce")
    return df


def apply_gate(df: pd.DataFrame) -> pd.DataFrame:
    """THE quality gate (single source of truth). cf_cvg==0 => no cloud-free
    observation that month, so avg_rad carries no light information -> NaN + flag."""
    df = df.copy()
    df["is_missing"] = (df["cf_cvg"].fillna(0) == 0) | df["avg_rad"].isna()
    df["avg_rad_raw"] = df["avg_rad"].where(~df["is_missing"])
    return df


def missing_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Overall missing rate per calendar year (spot thin early years like 2012)."""
    return (df.groupby("year")["is_missing"].mean()
              .rename("missing_rate").reset_index())


def missing_by_constituency(df: pd.DataFrame) -> pd.DataFrame:
    """Missing rate per constituency, sorted worst-first."""
    g = df.groupby(["con_id", "cons_name"])["is_missing"]
    out = pd.concat([g.mean().rename("missing_rate"),
                     g.size().rename("n_months")], axis=1).reset_index()
    return out.sort_values("missing_rate", ascending=False)


def worst_missing(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Top-n constituencies by missing rate (the ones to distrust)."""
    return missing_by_constituency(df).head(n)


def seam_diagnostic(df: pd.DataFrame, seam_year: int = 2014) -> pd.DataFrame:
    """Measure the impact of the VCMCFG -> VCMSLCFG product change at `seam_year`.

    The two products do NOT overlap in time, so we can't compare the same month
    across them. Instead we test for an artificial LEVEL SHIFT: compare the
    year-over-year jump across the seam (2013 -> 2014) with the typical jump in
    the pure VCMSLCFG era (2015 onward). A seam jump far from typical means the
    two sources are not interchangeable for that constituency.
    """
    ann = (df.dropna(subset=["avg_rad_raw"])
             .groupby(["con_id", "year"])["avg_rad_raw"].mean().reset_index())
    piv = ann.pivot(index="con_id", columns="year", values="avg_rad_raw").sort_index(axis=1)
    yoy = piv.pct_change(axis=1)                                   # fractional YoY change
    seam = yoy[seam_year].rename("seam_jump")                      # 2013 -> 2014
    later = [y for y in yoy.columns if y > seam_year]
    typical = yoy[later].median(axis=1).rename("typical_jump")     # normal era jump
    out = pd.concat([seam, typical], axis=1)
    out["excess"] = out["seam_jump"] - out["typical_jump"]         # anomaly at the seam
    return out.reset_index().sort_values(
        "excess", key=lambda s: s.abs(), ascending=False)
