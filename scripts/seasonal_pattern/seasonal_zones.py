"""
seasonal_zones.py -- Approach 2: STL seasonal decomposition by 3 climate zones.

Pure-function module (no global execution). Consolidates Ghana's constituencies
into three analytical agroclimatic zones and exposes the building blocks the entry
script (make_zone_seasonal_maps.py) orchestrates:

    1. assign_zones        : con_id -> {Southern bimodal, Transition, Northern unimodal}
    2. build_zone_series   : equal-weight, self-normalized monthly index per zone
    3. decompose_zone      : robust STL split (trend + seasonal + remainder) + FS
    4. zone_seasonal_shape : the folded 12-month seasonal profile (bimodal vs unimodal)
    5. zone_climatology    : cross-year median shape (robustness check vs STL)

Why a *zone* series instead of the whole country: the south is bimodal (two rain
peaks) and the north is unimodal (one). Averaging the nation lets those opposite
shapes cancel (the roadmap's "double trough" atmospheric artifact). Splitting into
homogeneous zones feeds STL a clean series per group -> lower remainder variance,
higher seasonal_strength, and a visible bimodal -> partial -> unimodal gradient.

Why equal-weight + self-normalize: each constituency is divided by its own long-run
mean before averaging, so Accra/Kumasi brightness cannot dominate a zone's shape
(same anti-domination logic as the national curve in the seasonal roadmap).

Prereqs: pandas, numpy, statsmodels
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

# --------------------------------------------------------------------------- #
# 1. Region -> climate-zone lookup
# --------------------------------------------------------------------------- #
# Ghana Meteorological Agency's 5 agroclimatic zones consolidated into 3
# analytical belts. The mapping is at the *administrative region* level because
# that is the finest geography the nightlight panel carries (region_16). Agro-
# climatic boundaries do not align perfectly with regions, so border regions are
# documented below and revisited in the results caveats.
#
#   Southern bimodal (Coastal + Forest): major rains Mar-Jul, Aug break,
#       minor rains Sep-Nov, short dry season Dec-Feb.
#   Transition (middle belt): weakened second peak, one dominant season.
#   Northern unimodal (Guinea + Sudan Savannah): single rains ~May-Oct,
#       long dry season Nov-Apr.

ZONE_SOUTH = "Southern bimodal"
ZONE_TRANS = "Transition"
ZONE_NORTH = "Northern unimodal"

ZONE_ORDER = [ZONE_SOUTH, ZONE_TRANS, ZONE_NORTH]

# Rainfall regime is a *climatological* property that defines each zone (from the
# Ghana Met Agency framework). It is FIXED by the zone, NOT inferred from the
# nightlight shape -- nightlight seasonality is driven mostly by atmosphere
# (Harmattan dust, monsoon cloud) plus supply, so counting humps in the light
# curve does not recover the rainfall modality and must not be relabeled as such.
RAINFALL_REGIME = {
    ZONE_SOUTH: "bimodal",
    ZONE_TRANS: "attenuated bimodal",
    ZONE_NORTH: "unimodal",
}

# Distinct, colour-blind-safe hues used consistently across every figure.
ZONE_COLORS = {
    ZONE_SOUTH: "#1b7837",   # green  - forest/coastal south
    ZONE_TRANS: "#f1a340",   # orange - middle belt
    ZONE_NORTH: "#8073ac",   # purple - savannah north
}

REGION_TO_ZONE: dict[str, str] = {
    # --- Southern bimodal: Coastal + Forest ---
    "Greater Accra": ZONE_SOUTH,   # coastal savanna
    "Central":       ZONE_SOUTH,   # coastal + forest
    "Western":       ZONE_SOUTH,   # rainforest
    "Western North": ZONE_SOUTH,   # rainforest
    "Ashanti":       ZONE_SOUTH,   # forest
    "Eastern":       ZONE_SOUTH,   # forest (SE fringe is drier - see caveats)
    "Ahafo":         ZONE_SOUTH,   # forest (border with transition - see caveats)
    "Volta":         ZONE_SOUTH,   # coastal south + forest (north split off as Oti)
    # --- Transition: middle belt ---
    "Bono":          ZONE_TRANS,   # classic forest-savanna transition
    "Bono East":     ZONE_TRANS,   # transition
    "Oti":           ZONE_TRANS,   # former northern Volta, savanna-transition
    # --- Northern unimodal: Guinea + Sudan Savannah ---
    "Northern":      ZONE_NORTH,   # Guinea Savannah
    "Savannah":      ZONE_NORTH,   # Guinea Savannah
    "North East":    ZONE_NORTH,   # Guinea Savannah
    "Upper East":    ZONE_NORTH,   # Sudan Savannah (shortest rains)
    "Upper West":    ZONE_NORTH,   # Guinea/Sudan Savannah
}

# Regions whose agroclimatic character straddles a zone boundary. Reassigning any
# of these is the main sensitivity check for the zoning (not the STL itself).
BORDER_REGIONS = {"Volta": "S/T", "Ahafo": "S/T", "Eastern": "S/T", "Oti": "T/N"}

# con_id -> region_16 for the handful of polygons that carry a name-as-id and are
# absent from the dictionary/panel lookups (e.g. Zebilla in Bawku West, Upper East).
CON_ID_REGION_OVERRIDE = {"Zebilla": "Upper East"}

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def build_region_lookup(dict_csv, panel_csv=None) -> pd.Series:
    """con_id -> region_16, from constituency_dictionary with a panel fallback.

    Returns a Series indexed by con_id. Only the id-region pair is needed here;
    geometry joins live in the entry script.
    """
    d = pd.read_csv(dict_csv).dropna(subset=["constituency_id", "region_16"])
    lut = (d.drop_duplicates("constituency_id")
             .set_index("constituency_id")["region_16"])
    if panel_csv is not None:
        p = pd.read_csv(panel_csv).dropna(subset=["con_id", "region_16"])
        fallback = p.drop_duplicates("con_id").set_index("con_id")["region_16"]
        lut = lut.combine_first(fallback)     # fill ids the dictionary missed
    override = pd.Series(CON_ID_REGION_OVERRIDE, name="region_16")
    lut = lut.combine_first(override)         # fill known name-as-id polygons
    return lut


def assign_zones(con_ids, region_lut: pd.Series) -> pd.DataFrame:
    """Attach region_16 and climate zone to a set of con_ids.

    Input : iterable of con_id, region lookup Series (con_id -> region_16).
    Output: DataFrame [con_id, region_16, zone]; unmapped rows carry zone=NaN so
            the caller can assert full coverage rather than silently drop them.
    """
    con_ids = pd.Index(pd.unique(pd.Series(list(con_ids))), name="con_id")
    region = con_ids.map(region_lut)
    zone = pd.Series(region, index=con_ids).map(REGION_TO_ZONE)
    return pd.DataFrame({"con_id": con_ids, "region_16": region.values,
                         "zone": zone.values})


# --------------------------------------------------------------------------- #
# 2. Equal-weight, self-normalized zone monthly index
# --------------------------------------------------------------------------- #
def build_zone_series(long_df: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    """Collapse the constituency-month panel into one monthly index per zone.

    Steps (per the anti-domination logic):
      a. drop rows flagged missing (cf_cvg==0) so gaps are honest, not back-filled.
      b. normalize each constituency by its own long-run mean radiance -> unitless
         index centred near 1; brightness level cancels, only shape survives.
      c. equal-weight average the normalized index across constituencies in a zone,
         per (year, month).

    Input : long_df with [con_id, year, month, avg_rad_raw, is_missing],
            zones with [con_id, zone].
    Output: tidy DataFrame [zone, year, month, date, idx, n_cons] on a gap-free
            monthly DatetimeIndex per zone (missing months linearly interpolated
            only at the zone level, where averaging over many units leaves ~no gap).
    """
    df = long_df.copy()
    if "is_missing" in df.columns:
        df.loc[df["is_missing"].astype(bool), "avg_rad_raw"] = np.nan
    df = df.merge(zones[["con_id", "zone"]], on="con_id", how="left")
    df = df.dropna(subset=["zone"])

    # (b) per-constituency self-normalization by long-run mean
    con_mean = df.groupby("con_id")["avg_rad_raw"].transform("mean")
    df["idx"] = df["avg_rad_raw"] / con_mean

    # (c) equal-weight zone average per calendar month
    g = df.groupby(["zone", "year", "month"])
    out = (g["idx"].mean().rename("idx")
           .to_frame()
           .join(g["idx"].count().rename("n_cons"))
           .reset_index())
    out["date"] = pd.to_datetime(dict(year=out["year"], month=out["month"], day=1))

    # regularize each zone onto a gap-free monthly grid for STL
    frames = []
    for z, sub in out.groupby("zone"):
        sub = sub.sort_values("date").set_index("date")
        full = pd.date_range(sub.index.min(), sub.index.max(), freq="MS")
        sub = sub.reindex(full)
        sub["idx"] = sub["idx"].interpolate(limit_direction="both")
        sub["zone"] = z
        sub["year"] = sub.index.year
        sub["month"] = sub.index.month
        frames.append(sub.reset_index(names="date"))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# 3. Robust STL decomposition
# --------------------------------------------------------------------------- #
def decompose_zone(series: pd.Series, period: int = 12) -> dict:
    """Robust STL split of one zone's monthly index series.

    Input : series indexed by monthly DatetimeIndex (gap-free), values = idx.
    Output: dict with observed/trend/seasonal/remainder (all pd.Series on the same
            index) and the dimensionless Hyndman seasonal strength FS in [0, 1]:
                FS = max(0, 1 - Var(remainder) / Var(seasonal + remainder))
    """
    s = series.astype(float).interpolate(limit_direction="both")
    res = STL(s.to_numpy(), period=period, robust=True).fit()
    idx = s.index
    seasonal = pd.Series(res.seasonal, index=idx)
    remainder = pd.Series(res.resid, index=idx)
    denom = float(np.var(seasonal + remainder))
    fs = float(max(0.0, 1.0 - np.var(remainder) / denom)) if denom > 0 else np.nan
    return {
        "observed": pd.Series(res.observed, index=idx),
        "trend": pd.Series(res.trend, index=idx),
        "seasonal": seasonal,
        "remainder": remainder,
        "seasonal_strength": fs,
    }


def zone_seasonal_shape(seasonal: pd.Series) -> pd.Series:
    """Fold STL's year-varying seasonal component into one typical 12-month cycle.

    Input : seasonal component from decompose_zone (monthly DatetimeIndex).
    Output: Series indexed 1..12 (calendar month) = mean seasonal deviation, i.e.
            how much brighter/darker than the annual level that month typically is.
            This is the curve that shows bimodal (two humps) vs unimodal (one).
    """
    s = seasonal.copy()
    prof = s.groupby(s.index.month).mean()
    prof.index.name = "month"
    return prof.reindex(range(1, 13))


# --------------------------------------------------------------------------- #
# 4. Climatology cross-check (independent of STL)
# --------------------------------------------------------------------------- #
def zone_climatology(zone_series: pd.DataFrame) -> pd.DataFrame:
    """Cross-year median shape per zone.

    A second opinion on the STL seasonal shape built straight from the data (median
    over years of the normalized index, by calendar month). Agreement between this
    and zone_seasonal_shape is the robustness check the roadmap asks for.

    Output: DataFrame indexed by zone with columns:
            clim_1..clim_12, peak_month, trough_month.
    """
    rows = {}
    for z, sub in zone_series.groupby("zone"):
        clim = sub.groupby("month")["idx"].median().reindex(range(1, 13))
        rows[z] = {
            **{f"clim_{m}": clim[m] for m in range(1, 13)},
            "peak_month": int(clim.idxmax()),
            "trough_month": int(clim.idxmin()),
        }
    out = pd.DataFrame(rows).T
    return out.reindex([z for z in ZONE_ORDER if z in out.index])


def zone_metrics_table(decomps: dict, clim: pd.DataFrame) -> pd.DataFrame:
    """Assemble the one-row-per-zone summary the entry script writes to CSV.

    rainfall_regime is the fixed climatological label; every other column
    describes the OBSERVED nightlight seasonality that STL/climatology recover.
    """
    rows = []
    for z in ZONE_ORDER:
        if z not in decomps:
            continue
        c = clim.loc[z]
        rows.append({
            "zone": z,
            "rainfall_regime": RAINFALL_REGIME[z],
            "seasonal_strength": round(decomps[z]["seasonal_strength"], 4),
            "brightest_month": MONTH_ABBR[int(c["peak_month"]) - 1],
            "darkest_month": MONTH_ABBR[int(c["trough_month"]) - 1],
        })
    return pd.DataFrame(rows)
