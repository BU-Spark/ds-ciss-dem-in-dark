"""
Seasonal-metrics layer: climatology baseline, deseasonalization, STL strength.

Inputs: constituency_lights_monthly_raw.csv (GEE export)
Outputs: constituency_seasonal_long.csv, constituency_seasonal_metrics.csv, run_manifest.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from nl_quality import load_raw, apply_gate   # single source of truth for the gate

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "nightlights"
IN_CSV = DATA_DIR / "constituency_lights_monthly_raw.csv"
MIN_YEARS = 3                      # min valid years per calendar month for a trusted baseline
DRY_MONTHS = [11, 12, 1, 2, 3]     # Ghana dry season (dam output drops)
WET_MONTHS = [4, 5, 6, 7, 8, 9, 10]


def build_climatology(df: pd.DataFrame) -> pd.DataFrame:
    """Per (con_id, month): median radiance across valid years + a confidence gate."""
    g = df.dropna(subset=["avg_rad_raw"]).groupby(["con_id", "month"])["avg_rad_raw"]
    out = pd.concat([g.median().rename("clim"),
                     g.size().rename("n_years")], axis=1).reset_index()
    out["low_conf"] = out["n_years"] < MIN_YEARS
    return out


def add_deseason(df: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    """Attach the baseline and compute the deseasonalized residual."""
    df = df.merge(clim[["con_id", "month", "clim"]], on=["con_id", "month"], how="left")
    df["deseason"] = df["avg_rad_raw"] - df["clim"]
    return df


def seasonal_strength(ts: pd.Series) -> float:
    """Hyndman FS = max(0, 1 - Var(remainder)/Var(seasonal+remainder)). NaN if too short."""
    s = ts.interpolate(limit_direction="both")    # STL needs a gap-free regular series
    if s.notna().sum() < 24 or s.nunique() < 3:   # need >= 2 cycles and some variation
        return np.nan
    try:
        res = STL(s.to_numpy(), period=12, robust=True).fit()
        denom = np.var(res.seasonal + res.resid)
        return float(max(0.0, 1.0 - np.var(res.resid) / denom)) if denom > 0 else np.nan
    except Exception:
        return np.nan


def build_metrics(df: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    """One row per constituency: strength, amplitude, dry/wet ratio, CV, quality."""
    rows = []
    for con_id, c in clim.groupby("con_id"):
        m = c.set_index("month")["clim"]
        annual = m.mean()
        amplitude = (m.max() - m.min()) / annual if annual else np.nan
        dry, wet = m.reindex(DRY_MONTHS).mean(), m.reindex(WET_MONTHS).mean()
        dry_wet_ratio = dry / wet if wet else np.nan

        sub = df[df["con_id"] == con_id].sort_values(["year", "month"])
        raw = sub["avg_rad_raw"]
        cv = raw.std() / raw.mean() if raw.mean() else np.nan
        ts = (sub.assign(ym=pd.to_datetime(dict(year=sub.year, month=sub.month, day=1)))
                 .set_index("ym")["avg_rad_raw"].asfreq("MS"))

        rows.append({
            "con_id": con_id,
            "cons_name": sub["cons_name"].iloc[0] if len(sub) else None,
            "seasonal_strength": seasonal_strength(ts),   # dimensionless, comparable
            "seasonal_amplitude": amplitude,
            "dry_wet_ratio": dry_wet_ratio,               # < 1 => dry season darker
            "cv": cv,
            "n_months_valid": int(raw.notna().sum()),
            "missing_rate": float(sub["is_missing"].mean()) if len(sub) else np.nan,
            "low_conf_baseline": bool(c["low_conf"].any()),
        })
    return pd.DataFrame(rows).sort_values("seasonal_strength", ascending=False)


def run(path: Path = IN_CSV) -> dict:
    """Convenience end-to-end: returns {'long':..., 'clim':..., 'metrics':...}."""
    df = apply_gate(load_raw(path))
    clim = build_climatology(df)
    long = add_deseason(df, clim)
    metrics = build_metrics(long, clim)
    return {"long": long, "clim": clim, "metrics": metrics}


def main() -> None:
    out = run()
    long, metrics = out["long"], out["metrics"]
    long_cols = ["con_id", "cons_name", "year", "month",
                 "avg_rad_raw", "cf_cvg", "is_missing", "clim", "deseason"]
    long[long_cols].to_csv(DATA_DIR / "constituency_seasonal_long.csv", index=False)
    metrics.to_csv(DATA_DIR / "constituency_seasonal_metrics.csv", index=False)
    manifest = {
        "rows_in": int(len(long)),
        "constituencies": int(long["con_id"].nunique()),
        "year_range": [int(long["year"].min()), int(long["year"].max())],
        "overall_missing_rate": float(long["is_missing"].mean()),
        "low_conf_constituencies": int(metrics["low_conf_baseline"].sum()),
    }
    with open(DATA_DIR / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"OK: {len(metrics)} constituencies | {manifest['year_range']} | "
          f"missing {manifest['overall_missing_rate']:.2%}")


if __name__ == "__main__":
    main()
