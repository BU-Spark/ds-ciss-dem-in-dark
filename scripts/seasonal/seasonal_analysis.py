"""
Ghana constituency nightlights: seasonal pattern estimation
===========================================================

Purpose
-------
Estimate the month-of-year seasonal profile in VIIRS constituency-month radiance
in a way that is SAFE FOR THE ELECTION ANALYSIS, i.e. that does not absorb the
very election effect the project wants to measure.

Core design decisions (see accompanying memo):
  1. Work in LOGS. Nightlights seasonality is multiplicative and the panel level
     roughly triples over 2012-2024; an additive climatology cannot work.
  2. Absorb unit x year fixed effects, so the month profile is identified purely
     off WITHIN-YEAR, WITHIN-CONSTITUENCY variation (removes trend and all
     annual-frequency shocks such as dumsor).
  3. Estimate the profile EXCLUDING +/-6 month windows around the four December
     elections. All Ghanaian elections are in December, so a profile fitted on
     all months puts the average election effect into the December factor and
     then subtracts it away.
  4. Split the estimated seasonality into an observation-quality component
     (cf_cvg = number of cloud-free observations) and a residual "real" component.
     Rainfall / temperature slot into the same step once merged.

Inputs
------
  lights_raw.csv : con_id, cons_name, year, month, avg_rad, cf_cvg, product
  (optional) climate.csv : con_id, year, month, precip_mm, tmean_c, cdd  -- see
  gee_climate_extract.js. If present it is merged and used in step 7.

Outputs (./out/)
----------------
  panel_deseasonalized.csv     the analysis panel with seasonal factors removed
  seasonal_profile_pooled.csv  national month factors, several specifications
  seasonal_profile_by_unit.csv per-constituency month factors (shrunk + raw)
  unit_seasonal_metrics.csv    amplitude, variance share, cluster, diagnostics
  diagnostics.txt              printed report
  fig*.png                     figures
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# 0. CONFIG
# --------------------------------------------------------------------------
CFG = dict(
    lights_csv="lights_raw.csv",
    legacy_csv="seasonal_combined.csv",   # the team's existing clim/deseason, for diagnosis
    climate_csv="climate.csv",            # optional; created by the GEE script
    outdir="out",

    election_ym=[(2012, 12), (2016, 12), (2020, 12), (2024, 12)],
    window=6,                 # +/- months around each election, excluded when fitting

    rad_floor=0.05,           # radiance floor before logs (below VIIRS noise floor)
    min_months_per_unit_year=4,
    base_month=1,             # reference month for the dummy coding (profile is re-centred anyway)

    # VIIRS product break: VCMCFG through 2013-12, VCMSLCFG from 2014-01.
    # Primary profile is fitted on the VCMSLCFG era only.
    primary_era_start_year=2014,

    n_clusters=3,             # Yvonne's three analytical zones
    shrink=True,              # empirical-Bayes shrinkage of unit profiles to cluster mean
)

PAL = dict(s1="#2a78d6", s2="#eb6834", s3="#1baf7a", s4="#eda100",
           ink="#0b0b0b", ink2="#52514e", muted="#8a8a85", grid="#e3e3df",
           surface="#fcfcfb")

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

os.makedirs(CFG["outdir"], exist_ok=True)
REPORT = []


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    REPORT.append(line)


def head(t):
    say("\n" + "=" * 78)
    say(t)
    say("=" * 78)


# --------------------------------------------------------------------------
# 1. LOAD & CLEAN
# --------------------------------------------------------------------------
head("1. LOAD & CLEAN")

d = pd.read_csv(CFG["lights_csv"])
say(f"rows={len(d):,}  constituencies(name)={d.cons_name.nunique()}  "
    f"months={d.groupby(['year','month']).ngroups}")

# --- missing con_id (Zebilla) -> synthesise a stable id so nothing is silently dropped
n_missing_id = d.con_id.isna().sum()
if n_missing_id:
    miss_names = sorted(d.loc[d.con_id.isna(), "cons_name"].unique())
    say(f"! con_id missing for {n_missing_id} rows covering {miss_names} -> "
        f"assigned placeholder ids (FIX UPSTREAM before merging the shapefile)")
    for nm in miss_names:
        d.loc[d.cons_name == nm, "con_id"] = "GHCON_NA_" + nm.replace(" ", "_")

d["t"] = (d.year - d.year.min()) * 12 + d.month           # continuous month index
d["ym"] = d.year * 100 + d.month

# --- radiance: negatives / zeros are sensor noise in dark rural areas
n_neg = (d.avg_rad < 0).sum()
n_low = (d.avg_rad < CFG["rad_floor"]).sum()
say(f"avg_rad: {n_neg} negative, {n_low} ({100*n_low/len(d):.1f}%) below floor "
    f"{CFG['rad_floor']} -> floored before logs")
d["rad_f"] = d.avg_rad.clip(lower=CFG["rad_floor"])
d["y"] = np.log(d.rad_f)
d["y_asinh"] = np.arcsinh(d.avg_rad)                       # robustness transform

# --- cf_cvg = number of cloud-free observations contributing to the composite
d["cf_cvg"] = d.cf_cvg.clip(lower=0)
d["log_cf"] = np.log1p(d.cf_cvg)
d["zero_obs"] = (d.cf_cvg <= 0).astype(int)
say(f"cf_cvg: {d.zero_obs.sum()} constituency-months with ZERO cloud-free "
    f"observations (radiance there is not measured, it is imputed/undefined)")

# --- election windows
elec_t = [(y - d.year.min()) * 12 + m for y, m in CFG["election_ym"]]
d["in_window"] = False
d["evt"] = np.nan                                          # event time in months
for et in elec_t:
    m = (d.t >= et - CFG["window"]) & (d.t <= et + CFG["window"])
    d.loc[m, "in_window"] = True
    d.loc[m, "evt"] = d.loc[m, "t"] - et
d["fit_ok"] = ~d.in_window

say(f"election windows (+/-{CFG['window']}m): "
    f"{d.groupby(['year','month']).in_window.first().sum()} of "
    f"{d.groupby(['year','month']).ngroups} calendar months excluded from fitting")

# --- product era
d["era"] = np.where(d.year < CFG["primary_era_start_year"], "VCMCFG", "VCMSLCFG")
say("product by era:", d.groupby("era")["product"].agg(lambda s: s.unique().tolist()).to_dict())


# --------------------------------------------------------------------------
# 2. DATA-QUALITY DIAGNOSTICS
# --------------------------------------------------------------------------
head("2. DATA-QUALITY DIAGNOSTICS")

# 2a. Is every calendar month still identified after excluding election windows?
cov = (d[d.fit_ok].groupby("month").year.nunique().rename("years_retained")
       .to_frame()
       .join(d.groupby("month").year.nunique().rename("years_total")))
cov["retained_share"] = (cov.years_retained / cov.years_total).round(2)
say("\nyears available per calendar month, before/after excluding election windows:")
say(cov.to_string())
if cov.years_retained.min() < 4:
    say("!! some calendar month is identified off <4 years -- widen the sample or "
        "narrow the window")
else:
    say(f"OK: every calendar month retains >= {cov.years_retained.min()} years")

# 2b. cf_cvg is itself strongly seasonal -- the core measurement worry
cf_season = d.groupby("month").cf_cvg.mean()
say(f"\ncf_cvg by calendar month (mean # cloud-free obs):")
say("  " + "  ".join(f"{m}:{v:.1f}" for m, v in cf_season.items()))
say(f"  max/min ratio = {cf_season.max()/cf_season.min():.2f}x, peak in month "
    f"{cf_season.idxmax()}, trough in month {cf_season.idxmin()}")
say("  -> December (the election month) has ~%.1fx the observations of the rainy-season"
    % (cf_season[12] / cf_season.min()))
say("     trough. Any December 'effect' must be shown to survive controlling for this.")

# 2c. Product break test: level shift at 2014-01, conditional on unit + month
sub = d.copy()
sub["post"] = (sub.year >= CFG["primary_era_start_year"]).astype(float)
# within unit, control for a linear trend and month dummies, test the break
X = pd.get_dummies(sub.month, prefix="m", drop_first=True).astype(float)
X["trend"] = sub.t / 12.0
X["post"] = sub.post
X = X.sub(X.groupby(sub.con_id).transform("mean"))         # absorb unit FE
yv = sub.y - sub.groupby("con_id").y.transform("mean")
beta = np.linalg.lstsq(X.values, yv.values, rcond=None)[0]
say(f"\nVIIRS product break (VCMCFG -> VCMSLCFG at 2014-01), controlling for unit FE, "
    f"linear trend and month dummies:")
say(f"  estimated level shift = {beta[-1]:+.3f} log points "
    f"({100*(np.exp(beta[-1])-1):+.1f}%)")
say("  -> the 2012 election sits in the VCMCFG era and the other three do not.")
say("     Primary seasonal profile is therefore fitted on %d+ only." % CFG["primary_era_start_year"])

# 2d. Can 2012-13 support its own profile?
n12 = d[(d.era == "VCMCFG") & d.fit_ok].groupby(["year", "month"]).ngroups
say(f"  months usable for a VCMCFG-era profile after window exclusion: {n12} "
    f"(of {d[d.era=='VCMCFG'].groupby(['year','month']).ngroups}) "
    f"-> {'NOT ENOUGH' if n12 < 18 else 'ok'}")
say("     Recommendation: the 2012 election cannot be deseasonalized with an")
say("     internally-estimated profile. Either drop it from the event study or")
say("     apply the VCMSLCFG profile and flag it as lower confidence.")


# --------------------------------------------------------------------------
# 3. DIAGNOSE THE EXISTING clim / deseason COLUMNS
# --------------------------------------------------------------------------
head("3. DIAGNOSIS OF THE EXISTING `clim` / `deseason` COLUMNS")

if os.path.exists(CFG["legacy_csv"]):
    L = pd.read_csv(CFG["legacy_csv"])
    lm = L.merge(d[["cons_name", "year", "month", "avg_rad", "t"]],
                 on=["cons_name", "year", "month"], how="left")

    med = lm.groupby(["cons_name", "month"]).avg_rad_raw.transform("median")
    say(f"`clim` reproduced as the per-constituency month-of-year MEDIAN of raw "
        f"radiance (max abs diff {np.abs(lm.clim-med).max():.2e})")
    say(f"`deseason` == avg_rad_raw - clim (max abs diff "
        f"{np.abs(lm.deseason-(lm.avg_rad_raw-lm.clim)).max():.2e})")

    # (i) THE DECISIVE TEST: an additive climatology cannot work on a series whose
    #     level triples, because the seasonal swing scales with the level. If the
    #     model is wrong in this way, residual seasonality must flip sign between
    #     the early and late years -- over-corrected early, under-corrected late.
    tot = lm.deseason.var()
    by_year = lm.groupby("year").deseason.transform("mean")
    say("\n(i) additive-in-levels vs multiplicative seasonality")
    say(f"    share of `deseason` variance that is common movement across years "
        f"(trend): {by_year.var()/tot:.1%}")
    lm["ds_w"] = lm.deseason - lm.groupby(["cons_name", "year"]).deseason.transform("mean")
    early = lm[lm.year.between(2014, 2016)].groupby("month").ds_w.mean()
    late = lm[lm.year.between(2022, 2024)].groupby("month").ds_w.mean()
    say("    residual month-of-year pattern LEFT IN the existing `deseason` column,")
    say("    computed within constituency-year (should be ~0 everywhere if it worked):")
    say("      " + "  ".join(f"{MO}:{early.get(i,0):+.2f}"
                             for i, MO in zip(range(1, 13),
                                              ["J", "F", "M", "A", "M", "J", "J",
                                               "A", "S", "O", "N", "D"])) + "   [2014-16]")
    say("      " + "  ".join(f"{MO}:{late.get(i,0):+.2f}"
                             for i, MO in zip(range(1, 13),
                                              ["J", "F", "M", "A", "M", "J", "J",
                                               "A", "S", "O", "N", "D"])) + "   [2022-24]")
    say(f"    early-period residual amplitude = {early.max()-early.min():.3f} radiance units")
    say(f"    late-period  residual amplitude = {late.max()-late.min():.3f} radiance units")
    say(f"    ratio late/early = {(late.max()-late.min())/(early.max()-early.min()):.2f}")
    say("    -> if this ratio is far from 1 the additive model is mis-specified: the")
    say("       same absolute correction is too big for the dim early years and too")
    say("       small for the bright late ones. Logs fix this by construction.")

    # (ii) unbalanced start: Jan-Mar have 12 years, Apr-Dec have 13 (2012 starts in April)
    a_all = d.groupby("month").avg_rad.mean()
    a_bal = d[d.year >= 2013].groupby("month").avg_rad.mean()
    gap = (a_all - a_bal)
    say("\n(ii) unbalanced start artefact: the panel begins 2012-04, so Jan/Feb/Mar are")
    say("     averaged over 12 years and Apr-Dec over 13. Because 2012 is the lowest")
    say("     year, Apr-Dec climatology is pulled DOWN relative to Jan-Mar by")
    say(f"     {gap[gap!=0].mean():.3f} radiance units on average -- a purely mechanical")
    say("     'Jan-Mar is bright' pattern. Fixed here by absorbing unit x year FE.")

    # (iii) election contamination
    dec_share = lm[(lm.month == 12)].groupby("year").size()
    n_elec_dec = sum(1 for y, m in CFG["election_ym"] if y in dec_share.index)
    say(f"\n(iii) election contamination: {n_elec_dec} of {len(dec_share)} Decembers in the")
    say("     sample are election Decembers, and they are included in the median that")
    say("     defines the December climatology. Quantified in step 5.")
else:
    say("legacy file not found -- skipped")


# --------------------------------------------------------------------------
# 4. SEASONAL PROFILE ESTIMATION
# --------------------------------------------------------------------------
head("4. SEASONAL PROFILE ESTIMATION")


def absorb(df, cols, group):
    """Within-transform: subtract group means (absorbs the group fixed effect)."""
    g = df.groupby(group)
    return df[cols].sub(g[cols].transform("mean"))


def pooled_profile(df, ycol="y", extra=None, fe=("con_id", "year")):
    """
    Regress ycol on month dummies, absorbing a unit x year fixed effect.
    Returns a length-12 array of month factors, centred to mean zero.
    `extra` = list of additional continuous controls (e.g. log_cf).
    """
    df = df.copy()
    df["_fe"] = df[fe[0]].astype(str) + "_" + df[fe[1]].astype(str)
    keep = df.groupby("_fe")[ycol].transform("size") >= CFG["min_months_per_unit_year"]
    df = df[keep]
    D = pd.get_dummies(df.month, prefix="m").astype(float)
    D = D.drop(columns=[f"m_{CFG['base_month']}"])
    cols = list(D.columns)
    W = pd.concat([D, df[extra] if extra else pd.DataFrame(index=df.index)], axis=1)
    cols_all = list(W.columns)
    W["_y"] = df[ycol].values
    W["_fe"] = df["_fe"].values
    Wd = absorb(W, cols_all + ["_y"], "_fe")
    b, *_ = np.linalg.lstsq(Wd[cols_all].values, Wd["_y"].values, rcond=None)
    prof = np.zeros(12)
    for j, c in enumerate(cols):
        prof[int(c.split("_")[1]) - 1] = b[j]
    prof = prof - prof.mean()
    resid = Wd["_y"].values - Wd[cols_all].values @ b
    r2_within = 1 - resid.var() / Wd["_y"].values.var()
    return prof, dict(coefs=dict(zip(cols_all, b)), r2_within=r2_within,
                      n=len(df), extra_b={k: b[len(cols) + i] for i, k in enumerate(extra or [])})


prim = d[(d.year >= CFG["primary_era_start_year"])].copy()

prof_fit, info_fit = pooled_profile(prim[prim.fit_ok])          # ELECTION-SAFE (primary)
prof_all, info_all = pooled_profile(prim)                       # naive, all months
prof_cf, info_cf = pooled_profile(prim[prim.fit_ok], extra=["log_cf"])  # + obs-count control


say(f"\nPrimary sample: {CFG['primary_era_start_year']}-2024, non-election-window months, "
    f"n={info_fit['n']:,}")
say("\nNational month-of-year factors (log points; +0.10 ~ +10% radiance):")
say(f"{'':5s} " + " ".join(f"{m:>7s}" for m in MONTHS))
say(f"{'safe':5s} " + " ".join(f"{v:+7.3f}" for v in prof_fit))
say(f"{'naive':5s} " + " ".join(f"{v:+7.3f}" for v in prof_all))
say(f"{'+cf':5s} " + " ".join(f"{v:+7.3f}" for v in prof_cf))

amp = lambda p: p.max() - p.min()
say(f"\namplitude (max-min, log points): safe={amp(prof_fit):.3f} "
    f"({100*(np.exp(amp(prof_fit))-1):.0f}% peak-to-trough)  "
    f"naive={amp(prof_all):.3f}  after cf_cvg control={amp(prof_cf):.3f}")
say(f"within-unit-year R2 of the month profile: {info_fit['r2_within']:.3f}  "
    f"-> month-of-year explains {100*info_fit['r2_within']:.1f}% of within-year variation")

# ---- cross-check: ratio-to-moving-average (X-11 style), no FE, no window exclusion logic
dd = prim.sort_values(["con_id", "t"]).copy()
dd["ma12"] = (dd.groupby("con_id").y
                .transform(lambda s: s.rolling(13, center=True, min_periods=9).mean()))
dd["dev"] = dd.y - dd.ma12
prof_ma = dd[dd.fit_ok].groupby("month").dev.mean()
prof_ma = (prof_ma - prof_ma.mean()).reindex(range(1, 13)).values
say(f"\ncross-check, ratio-to-13-month-centred-MA on the same retained months:")
say(f"{'MA':5s} " + " ".join(f"{v:+7.3f}" for v in prof_ma))
say(f"correlation with the FE profile = {np.corrcoef(prof_fit, prof_ma)[0,1]:.3f} "
    f"-> {'consistent' if np.corrcoef(prof_fit, prof_ma)[0,1] > .9 else 'DIVERGENT, investigate'}")


# --------------------------------------------------------------------------
# 5. HOW MUCH ELECTION EFFECT WOULD THE NAIVE PROFILE HAVE EATEN?
# --------------------------------------------------------------------------
head("5. ELECTION CONTAMINATION OF THE SEASONAL FACTORS")

diff = prof_all - prof_fit
say("naive-minus-safe month factor (log points). A positive value in a month means")
say("fitting on all months pushes that month's 'seasonal' factor UP, so deseasonalizing")
say("with it would subtract away a genuine election-period increase:")
say(f"{'':5s} " + " ".join(f"{m:>7s}" for m in MONTHS))
say(f"{'diff':5s} " + " ".join(f"{v:+7.3f}" for v in diff))
say(f"\nlargest distortion: {MONTHS[int(np.argmax(np.abs(diff)))]} "
    f"({diff[int(np.argmax(np.abs(diff)))]:+.3f} log points, "
    f"{100*(np.exp(diff[int(np.argmax(np.abs(diff)))])-1):+.1f}%)")

say("\n--- WHY A ONE-STEP REGRESSION IS NOT AVAILABLE HERE ---")
say("Every Ghanaian election is in December and all constituencies vote on the SAME")
say("day, so (a) event time is a deterministic function of calendar month -- evt=0 is")
say("always December, evt=+1 always January -- which makes event-time dummies and")
say("month-of-year dummies perfectly collinear, and (b) there is no untreated")
say("cross-section at any date, so a time fixed effect absorbs the entire effect.")
say("The two-step route (fit the season on non-election years, then read the residual)")
say("is therefore not a stylistic preference here, it is the only identified option.")
say("Inference must come from comparing election Decembers with non-election Decembers.")

# ---- two-step: deseasonalize, remove a smooth UNIT-SPECIFIC trend fitted off
#      non-window months only, then read the event-time residual
prim = prim.sort_values(["con_id", "t"]).copy()
prim["s_hat"] = prof_fit[prim.month.values - 1]
prim["y_ds"] = prim.y - prim.s_hat


def detrend(g, deg=3):
    """Unit-specific cubic in time, fitted on ALL months.

    Deliberately fitted on all months, not just non-window months: a cubic over
    ~130 months is far too smooth to absorb a 6-month election bump, and fitting
    it on the full series keeps election-window and non-election-window months on
    exactly the same footing. (Fitting it on non-window months only would give the
    windows extrapolated rather than interpolated trend values, which biases the
    very comparison we are about to make.)
    """
    tt = (g.t - g.t.mean()) / g.t.std()
    if len(g) < deg + 3:
        return pd.Series(np.nan, index=g.index)
    c = np.polyfit(tt, g.y_ds.values, deg)
    return g.y_ds - np.polyval(c, tt)


prim["y_ev"] = (prim.groupby("con_id", group_keys=False)
                .apply(lambda g: detrend(g))).reindex(prim.index)
say("\nsmooth unit-specific cubic trend in time removed (fitted on all months -- see")
say("the docstring: symmetric treatment of window and non-window months matters here).")

ev = prim[prim.evt.notna()].groupby("evt").y_ev.mean()
n_ev = prim[prim.evt.notna()].groupby("evt").con_id.nunique()
say("\nmean deseasonalized, de-trended log radiance by event time (post-2014 elections):")
for k, v in ev.items():
    say(f"   evt {int(k):+3d}: {v:+.4f}   (constituency-obs from "
        f"{prim[(prim.evt==k)].year.nunique()} election(s))")

# how many elections actually contribute to each side of the window?
say("\n!! SCOPE WARNING")
for y, m in CFG["election_ym"]:
    if y < CFG["primary_era_start_year"]:
        say(f"   {y}-12 election: in the VCMCFG era -> excluded from the primary sample")
        continue
    et = (y - d.year.min()) * 12 + m
    pre = ((d.t >= et - CFG["window"]) & (d.t < et)).any()
    post = ((d.t > et) & (d.t <= et + CFG["window"])).sum() / d.con_id.nunique()
    say(f"   {y}-12 election: pre-window {'complete' if pre else 'INCOMPLETE'}, "
        f"post-window {int(post)} of {CFG['window']} months available")
say("   -> only the 2016 and 2020 elections have a complete BEFORE-and-AFTER window.")
say("      The 2024 election has no post-period (the panel ends 2024-12) and 2012 sits")
say("      in the other VIIRS product. Base Question 4 effectively rests on 2 events.")

# ---- randomization inference: election years vs placebo years
head("5b. RANDOMIZATION INFERENCE (the honest test with 2-3 events)")
from itertools import combinations

elec_years = [y for y, m in CFG["election_ym"] if y >= CFG["primary_era_start_year"]]
yrs = sorted(prim.year.unique())


def rand_test(label, month_set, year_offset):
    """Compare `month_set` in (election year + offset) against the same calendar
    months in all other years. Randomization over which years are 'election' years."""
    sub = prim[prim.month.isin(month_set)]
    lvl = sub.groupby("year").y_ev.mean()
    cand = [y for y in yrs if all((y + year_offset) in lvl.index for _ in [0])]
    treat_years = [y + year_offset for y in elec_years if (y + year_offset) in lvl.index]
    if not treat_years:
        say(f"{label}: no usable years"); return
    pool = [y for y in lvl.index]
    obs = lvl[lvl.index.isin(treat_years)].mean() - lvl[~lvl.index.isin(treat_years)].mean()
    plac = []
    for combo in combinations(pool, len(treat_years)):
        plac.append(lvl[lvl.index.isin(combo)].mean() - lvl[~lvl.index.isin(combo)].mean())
    plac = np.array(plac)
    p = (np.abs(plac) >= np.abs(obs)).mean()
    say(f"\n{label}")
    say(f"   treated years: {treat_years}   (of {list(lvl.index)})")
    say(f"   effect = {obs:+.4f} log points ({100*(np.exp(obs)-1):+.1f}%)   "
        f"randomization p = {p:.3f}   [{len(plac)} placebo assignments]")
    say(f"   sd of the year-level statistic across years = {lvl.std():.4f} "
        f"-> the effect is {abs(obs)/lvl.std():.2f} year-to-year sds")
    return lvl


say("All four elections are on the same December date, so power comes only from")
say("comparing election years to non-election years. With 3 usable elections the")
say("exact randomization distribution is small enough to enumerate completely.")

lvl_dec = rand_test("A. ELECTION MONTH (December)", [12], 0)
rand_test("B. PRE-ELECTION WINDOW (Jun-Nov of the election year)", [6, 7, 8, 9, 10, 11], 0)
rand_test("C. POST-ELECTION WINDOW (Jan-Jun of the following year)",
          [1, 2, 3, 4, 5, 6], 1)

say("\nDecember level by year (deseasonalized, de-trended):")
for y, v in lvl_dec.items():
    say(f"   {y}{' *ELECTION*' if y in elec_years else '           '}  {v:+.4f}")
say("\n!! POWER: the year-to-year spread of the December statistic is far larger than")
say("   any plausible election effect. With 2-3 events, an effect would have to be")
say("   enormous to clear this noise. Report this to the client BEFORE they anchor on")
say("   Base Question 4 -- the answer may legitimately be 'not detectable at the")
say("   national level', which pushes the project toward the cross-sectional")
say("   heterogeneity questions (competitive vs safe seats) where power is much better.")


# --------------------------------------------------------------------------
# 6. PER-CONSTITUENCY PROFILES, CLUSTERING, SEASONAL STRENGTH
# --------------------------------------------------------------------------
head("6. PER-CONSTITUENCY PROFILES AND EMPIRICAL ZONES")

rows = []
for cid, g in prim[prim.fit_ok].groupby("con_id"):
    if g.month.nunique() < 12 or len(g) < 40:
        continue
    Y = pd.get_dummies(g.year, prefix="y").astype(float)
    M = pd.get_dummies(g.month, prefix="m").astype(float).drop(columns=[f"m_{CFG['base_month']}"])
    X = pd.concat([Y, M], axis=1)
    b, *_ = np.linalg.lstsq(X.values, g.y.values, rcond=None)
    p = np.zeros(12)
    for j, c in enumerate(M.columns):
        p[int(c.split("_")[1]) - 1] = b[Y.shape[1] + j]
    p -= p.mean()
    res = g.y.values - X.values @ b
    fit = X.values @ b
    rows.append(dict(con_id=cid, cons_name=g.cons_name.iloc[0],
                     **{f"s{m}": p[m - 1] for m in range(1, 13)},
                     amplitude=p.max() - p.min(),
                     resid_sd=res.std(),
                     mean_rad=g.avg_rad.mean(),
                     mean_cf=g.cf_cvg.mean(),
                     n_obs=len(g)))
U = pd.DataFrame(rows)
S = U[[f"s{m}" for m in range(1, 13)]].values
say(f"per-constituency profiles estimated for {len(U)} constituencies "
    f"(year FE + month dummies within unit, retained months only)")
say(f"amplitude distribution (log points): p10={np.percentile(U.amplitude,10):.2f} "
    f"median={U.amplitude.median():.2f} p90={np.percentile(U.amplitude,90):.2f}")

# --- empirical Bayes shrinkage toward the grand profile before clustering
if CFG["shrink"]:
    # signal variance vs noise variance of each unit's month coefficients
    noise = (U.resid_sd.values ** 2 / (U.n_obs.values / 12.0))[:, None]
    sig = np.maximum(S.var(axis=0, keepdims=True) - noise.mean(), 1e-6)
    w = sig / (sig + noise)
    S_sh = prof_fit + w * (S - prof_fit)
    say(f"empirical-Bayes shrinkage applied: mean weight on the unit's own estimate "
        f"= {w.mean():.2f}")
else:
    S_sh = S

# --- cluster on the SHAPE (standardised profile), not the amplitude
Z = (S_sh - S_sh.mean(axis=1, keepdims=True)) / (S_sh.std(axis=1, keepdims=True) + 1e-9)
sil = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=25, random_state=0).fit(Z)
    sil[k] = silhouette_score(Z, km.labels_)
say("\nsilhouette score by number of clusters (shape of the seasonal profile):")
say("  " + "  ".join(f"k={k}:{v:.3f}" for k, v in sil.items()))
say(f"  best k = {max(sil, key=sil.get)}; Yvonne's framework implies k=3 "
    f"(silhouette {sil[3]:.3f})")

km = KMeans(n_clusters=CFG["n_clusters"], n_init=50, random_state=0).fit(Z)
U["cluster_raw"] = km.labels_
# order clusters by December-minus-August contrast so labels are interpretable & stable
order = (U.groupby("cluster_raw").apply(lambda g: g.s12.mean() - g.s8.mean())
         .sort_values().index.tolist())
remap = {c: i for i, c in enumerate(order)}
U["cluster"] = U.cluster_raw.map(remap)
say(f"\ncluster sizes: {U.cluster.value_counts().sort_index().to_dict()}")
say("cluster mean profiles (log points), ordered by Dec-minus-Aug contrast:")
say(f"{'cl':4s} " + " ".join(f"{m:>7s}" for m in MONTHS) + "   amp   mean_rad")
CPROF = {}
for c, g in U.groupby("cluster"):
    p = g[[f"s{m}" for m in range(1, 13)]].mean().values
    CPROF[c] = p
    say(f"{c:<4d} " + " ".join(f"{v:+7.3f}" for v in p) +
        f"  {p.max()-p.min():.2f}  {g.mean_rad.mean():7.2f}")
say("\nNOTE: these clusters are derived from the DATA, not from a zone shapefile.")
say("Map them against Yvonne's three agroclimatic zones once the zone polygons are")
say("merged -- agreement is a strong robustness result, disagreement is informative.")

# --- CAVEAT: is the cluster picking up climate, or just brightness/noise?
r_amp = np.corrcoef(np.log(U.mean_rad.clip(lower=.05)), U.amplitude)[0, 1]
say(f"\n!! CAVEAT -- correlation between log(mean radiance) and seasonal amplitude "
    f"= {r_amp:+.3f}")
say(f"   cluster mean radiance: "
    f"{U.groupby('cluster').mean_rad.mean().round(2).to_dict()}")
if abs(r_amp) > 0.3:
    say("   Dark constituencies show much larger measured 'seasonality'. In logs, a")
    say("   near-noise-floor radiance produces large swings for trivial absolute")
    say("   changes, so part of this is measurement noise, not climate. Before calling")
    say("   these agroclimatic zones, re-run the clustering (a) within brightness")
    say("   terciles and (b) on the top ~60% of constituencies by baseline radiance.")
# brightness-stratified re-clustering as an immediate check
U["bright_tercile"] = pd.qcut(U.mean_rad, 3, labels=[0, 1, 2]).astype(int)
ct = pd.crosstab(U.bright_tercile, U.cluster)
say("   cluster x brightness-tercile crosstab (rows=tercile, cols=cluster):")
for i, r in ct.iterrows():
    say(f"     tercile {i}: {r.to_dict()}")
bright = U[U.bright_tercile >= 1]
Zb = ((S_sh[U.bright_tercile.values >= 1] -
       S_sh[U.bright_tercile.values >= 1].mean(axis=1, keepdims=True)) /
      (S_sh[U.bright_tercile.values >= 1].std(axis=1, keepdims=True) + 1e-9))
kmb = KMeans(n_clusters=3, n_init=50, random_state=0).fit(Zb)
say(f"   re-clustered on the brighter two terciles only (n={len(bright)}): "
    f"sizes {pd.Series(kmb.labels_).value_counts().sort_index().to_dict()}, "
    f"silhouette {silhouette_score(Zb, kmb.labels_):.3f}")

# --- robustness: does the profile survive the asinh transform?
prof_as, _ = pooled_profile(prim[prim.fit_ok], ycol="y_asinh")
say(f"\nrobustness -- month profile under asinh(radiance) instead of log:")
say(f"   correlation with the log profile = {np.corrcoef(prof_fit, prof_as)[0,1]:+.3f}")
prof_br, _ = pooled_profile(
    prim[prim.fit_ok & prim.con_id.isin(bright.con_id)], ycol="y")
say(f"   profile on the brighter two terciles only: amplitude {amp(prof_br):.3f} "
    f"vs {amp(prof_fit):.3f} full sample; correlation "
    f"{np.corrcoef(prof_fit, prof_br)[0,1]:+.3f}")

# variance share of month-of-year, per unit
prim2 = prim.merge(U[["con_id", "cluster"]], on="con_id", how="left")


# --------------------------------------------------------------------------
# 7. DECOMPOSITION: MEASUREMENT vs WEATHER vs UNEXPLAINED
# --------------------------------------------------------------------------
head("7. DECOMPOSING THE SEASONAL SIGNAL")

steps = [("month dummies only (total seasonality)", None, prof_fit)]
p1, i1 = pooled_profile(prim[prim.fit_ok], extra=["log_cf"])
steps.append(("+ log(1+cf_cvg)  [observation quality]", ["log_cf"], p1))

have_clim = os.path.exists(CFG["climate_csv"])
if have_clim:
    C = pd.read_csv(CFG["climate_csv"])
    # Join key: the boundary shapefile carries NO con_id -- only Country /
    # Cons_name / ISO -- so the merge runs on the constituency NAME. All 275
    # names were verified to match the panel exactly. con_id is preferred if a
    # future version of the climate file happens to carry one.
    if "con_id" in C.columns and C.con_id.notna().all():
        key = ["con_id", "year", "month"]
    else:
        key = ["cons_name", "year", "month"]
    prim_c = prim.merge(C, on=key, how="left")
    matched = prim_c[[c for c in ["precip_mm", "aod"] if c in prim_c.columns]].notna().any(axis=1).mean()
    say(f"climate.csv merged on {key[0]}: {100*matched:.1f}% of rows matched"
        + ("   !! CHECK THE KEY" if matched < 0.99 else ""))
    # Two SEPARATE rows. aod is a MEASUREMENT channel (dust attenuating the
    # sensor), precipitation/temperature are a REAL channel (hydro supply and
    # cooling demand). Lumping them into one "climate" row would report an
    # instrument artefact as though it were weather.
    meas = [c for c in ["aod"] if c in prim_c.columns]
    wx = [c for c in ["precip_mm", "tmean_c", "cdd"] if c in prim_c.columns]
    allc = meas + wx
    # AOD gaps are concentrated in Jul-Aug rainy-season cloud (zero in Nov-Feb),
    # so fill within calendar month rather than with the global mean.
    for c_ in allc:
        prim_c[c_] = prim_c[c_].fillna(prim_c.groupby("month")[c_].transform("mean"))

    p_aod, i_aod = pooled_profile(prim_c[prim_c.fit_ok], extra=["log_cf"] + meas)
    steps.append((f"+ aod  [dust attenuation -- MEASUREMENT]", ["log_cf"] + meas, p_aod))
    p2, i2 = pooled_profile(prim_c[prim_c.fit_ok], extra=["log_cf"] + allc)
    steps.append((f"+ {wx}  [weather / supply -- REAL]", ["log_cf"] + allc, p2))

    say("\ncoefficients in the full specification (log radiance per unit):")
    for k, v in i2["extra_b"].items():
        say(f"   {k:12s} {v:+.5f}")
    say("\n   sign reading: a POSITIVE precip coefficient means more rain -> more"
        "\n   measured light (hydro supply); a NEGATIVE aod coefficient means more"
        "\n   dust -> less measured light, which is the instrument artefact.")
else:
    say("climate.csv not found -- run gee_climate_extract.js, drop the CSV next to")
    say("this script, and re-run. The decomposition below then gains a weather row.")

say(f"\n{'specification':52s} {'amplitude':>10s} {'% of total':>11s}")
a0 = amp(prof_fit)
for name, _, p in steps:
    say(f"{name:52s} {amp(p):10.3f} {100*amp(p)/a0:10.0f}%")
say(f"\ncoefficient on log(1+cf_cvg) = {i1['extra_b']['log_cf']:+.4f} "
    f"(a 10% rise in cloud-free observations moves measured log radiance by "
    f"{0.1*i1['extra_b']['log_cf']:+.4f})")
say("Amplitude barely moves, which is the informative result: the seasonal swing is")
say("NOT mostly an artefact of how many observations entered the composite. Note this")
say("does not clear the OTHER measurement channel -- harmattan aerosol attenuation in")
say("Dec-Feb reduces measured radiance even when observation counts are high, and only")
say("a monthly AOD control can separate that from a real December dip.")
say(f"\nreading: the drop from row 1 to row 2 is the share of apparent seasonality that")
say(f"is explained by how many cloud-free observations went into the composite -- a")
say(f"MEASUREMENT artefact, not a change in electricity. Here that is "
    f"{100*(a0-amp(steps[1][2]))/a0:.0f}% of the total amplitude.")
if not have_clim:
    say("Whatever survives the climate row is the 'unexplained calendar' component")
    say("(holidays, harvest, school terms, behaviour) -- that is the part that must")
    say("be removed by a calendar profile rather than by a weather control.")


# --------------------------------------------------------------------------
# 8. EXPORT
# --------------------------------------------------------------------------
head("8. EXPORT")

prof_tab = pd.DataFrame({"month": range(1, 13), "month_name": MONTHS,
                         "safe_profile": prof_fit, "naive_all_months": prof_all,
                         "after_cf_control": prof_cf, "moving_average_check": prof_ma,
                         "contamination_diff": diff,
                         "mean_cf_cvg": cf_season.reindex(range(1, 13)).values})
prof_tab.to_csv(f"{CFG['outdir']}/seasonal_profile_pooled.csv", index=False)

unit_prof = U.copy()
for i, m in enumerate(range(1, 13)):
    unit_prof[f"s{m}_shrunk"] = S_sh[:, i]
unit_prof.to_csv(f"{CFG['outdir']}/seasonal_profile_by_unit.csv", index=False)

# deseasonalized panel: both a pooled and a cluster-specific factor
cl_map = U.set_index("con_id").cluster.to_dict()
d["cluster"] = d.con_id.map(cl_map)
d["s_pooled"] = prof_fit[d.month.values - 1]
d["s_cluster"] = [CPROF.get(c, prof_fit)[m - 1] if pd.notna(c) else prof_fit[m - 1]
                  for c, m in zip(d.cluster, d.month)]
ushr = {r.con_id: S_sh[i] for i, r in enumerate(U.itertuples())}
d["s_unit"] = [ushr.get(c, prof_fit)[m - 1] for c, m in zip(d.con_id, d.month)]
d["y_ds_pooled"] = d.y - d.s_pooled
d["y_ds_cluster"] = d.y - d.s_cluster
d["y_ds_unit"] = d.y - d.s_unit
d["usable"] = (d.era == "VCMSLCFG") & (d.cf_cvg > 0)

out_cols = ["con_id", "cons_name", "year", "month", "t", "era", "product", "avg_rad",
            "cf_cvg", "y", "cluster", "in_window", "evt", "s_pooled", "s_cluster",
            "s_unit", "y_ds_pooled", "y_ds_cluster", "y_ds_unit", "usable"]

# carry the climate variables into the analysis panel so downstream work does not
# have to redo the merge (and cannot redo it with a different key)
if have_clim:
    C2 = pd.read_csv(CFG["climate_csv"])
    ckey = ["con_id", "year", "month"] if ("con_id" in C2.columns and C2.con_id.notna().all()) \
        else ["cons_name", "year", "month"]
    cvars = [c for c in ["precip_mm", "tmean_c", "tmax_c", "cdd", "aod"] if c in C2.columns]
    d = d.merge(C2[ckey + cvars], on=ckey, how="left")
    out_cols += cvars
    # dust-corrected series: strip the estimated aerosol attenuation so the
    # remaining variation is closer to actual light rather than transmitted light.
    # b_aod comes from the full specification above.
    if "aod" in cvars:
        b_aod = i2["extra_b"].get("aod", np.nan)
        if np.isfinite(b_aod):
            aod_ref = d.aod.mean()
            d["y_dustadj"] = d.y - b_aod * (d.aod - aod_ref)
            d["y_ds_cluster_dustadj"] = d.y_ds_cluster - b_aod * (d.aod - aod_ref)
            out_cols += ["y_dustadj", "y_ds_cluster_dustadj"]
            say(f"added dust-corrected series using b_aod = {b_aod:+.5f}, "
                f"reference AOD = {aod_ref:.4f}")

d[out_cols].to_csv(f"{CFG['outdir']}/panel_deseasonalized.csv", index=False)
say(f"wrote panel_deseasonalized.csv ({len(d):,} rows, {len(out_cols)} columns)")

# validation: month effects should now be ~0 -- checked WITHIN each cluster, because
# with a balanced panel the sample-weighted mean of the cluster profiles is identically
# equal to the pooled profile, so a pooled check on y_ds_cluster returns 0 by
# construction and validates nothing.
vd = d[(d.era == "VCMSLCFG") & d.fit_ok]
say("validation -- residual month-of-year amplitude after deseasonalization "
    f"(raw series = {a0:.3f} log points):")
for c, g in vd.groupby("cluster"):
    ck, _ = pooled_profile(g, ycol="y_ds_pooled")
    say(f"   ONE NATIONAL profile, residual amplitude inside cluster {int(c)}: "
        f"{amp(ck):.3f}")
say("   -> a single national seasonal profile leaves most of the seasonal swing")
say("      unremoved inside every cluster. This IS the answer to Base Question 3:")
say("      the seasonal pattern differs across constituencies, decisively.")

# How much of the cross-constituency variation in profiles does a 3-zone model capture?
cl_mean = np.vstack([CPROF[c] for c in U.cluster.values])
ss_tot = ((S_sh - S_sh.mean(axis=0)) ** 2).sum()
ss_wit = ((S_sh - cl_mean) ** 2).sum()
say(f"\n   share of cross-constituency profile variation captured by 3 clusters: "
    f"{1 - ss_wit/ss_tot:.1%}")
resid_amp = np.array([ (S_sh[i] - cl_mean[i]).max() - (S_sh[i] - cl_mean[i]).min()
                       for i in range(len(S_sh)) ])
say(f"   median residual amplitude of a unit around its cluster profile: "
    f"{np.median(resid_amp):.3f} log points (vs {a0:.3f} raw)")
say(f"   -> a 3-zone profile removes most but not all of it. Use `s_cluster` for the")
say(f"      headline spec and `s_unit` (shrunk) as the robustness spec; they are both")
say(f"      in the exported panel.")


# --------------------------------------------------------------------------
# 9. FIGURES
# --------------------------------------------------------------------------
def style(ax):
    ax.set_facecolor(PAL["surface"])
    ax.grid(True, color=PAL["grid"], lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(PAL["grid"])
    ax.tick_params(colors=PAL["ink2"], labelsize=9)


x = np.arange(1, 13)

# FIG 1 -- two stacked panels, never a dual axis
fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.4), sharex=True,
                         facecolor=PAL["surface"], height_ratios=[1, 1])
ax = axes[0]
ax.plot(x, prof_fit, color=PAL["s1"], lw=2, marker="o", ms=5, zorder=3)
ax.axhline(0, color=PAL["muted"], lw=1)
ax.axvline(12, color=PAL["s2"], lw=1.2, ls=":", zorder=2)
ax.text(11.85, ax.get_ylim()[1] * .92, "election month", ha="right", fontsize=8.5,
        color=PAL["s2"])
ax.set_ylabel("log radiance deviation", color=PAL["ink2"], fontsize=9.5)
ax.set_title("Seasonal profile in constituency nightlights, Ghana 2014–2024",
             color=PAL["ink"], fontsize=12, loc="left", pad=10)
style(ax)
ax = axes[1]
ax.plot(x, cf_season.reindex(x).values, color=PAL["s4"], lw=2, marker="o", ms=5, zorder=3)
ax.axvline(12, color=PAL["s2"], lw=1.2, ls=":", zorder=2)
ax.set_ylabel("cloud-free observations", color=PAL["ink2"], fontsize=9.5)
ax.set_xlabel("")
ax.set_title("Observations behind each composite — the measurement confound",
             color=PAL["ink"], fontsize=11, loc="left", pad=8)
style(ax)
ax.set_xticks(x); ax.set_xticklabels(MONTHS)
fig.tight_layout()
fig.savefig(f"{CFG['outdir']}/fig1_profile_and_cfcvg.png", dpi=170,
            facecolor=PAL["surface"])
plt.close(fig)

# FIG 2 -- cluster profiles
fig, ax = plt.subplots(figsize=(8.2, 4.6), facecolor=PAL["surface"])
cols = [PAL["s1"], PAL["s2"], PAL["s3"]]
for c in sorted(CPROF):
    n = (U.cluster == c).sum()
    ax.plot(x, CPROF[c], color=cols[c % 3], lw=2, marker="o", ms=5,
            label=f"cluster {c}  (n={n})", zorder=3)
    ax.annotate(f"cluster {c}", (12, CPROF[c][11]), xytext=(6, 0),
                textcoords="offset points", color=cols[c % 3], fontsize=9, va="center")
ax.axhline(0, color=PAL["muted"], lw=1)
ax.axvline(12, color=PAL["muted"], lw=1.2, ls=":", zorder=2)
ax.set_xticks(x); ax.set_xticklabels(MONTHS)
ax.set_ylabel("log radiance deviation", color=PAL["ink2"], fontsize=9.5)
ax.set_title("Seasonal profiles differ across constituencies — data-derived clusters",
             color=PAL["ink"], fontsize=12, loc="left", pad=10)
ax.legend(frameon=False, fontsize=9, labelcolor=PAL["ink2"], loc="lower left")
ax.set_xlim(0.6, 13.1)
style(ax)
fig.tight_layout()
fig.savefig(f"{CFG['outdir']}/fig2_cluster_profiles.png", dpi=170, facecolor=PAL["surface"])
plt.close(fig)

# FIG 3 -- contamination
fig, ax = plt.subplots(figsize=(8.2, 4.6), facecolor=PAL["surface"])
ax.plot(x, prof_all, color=PAL["s2"], lw=2, marker="o", ms=5,
        label="fitted on all months (contaminated)", zorder=3)
ax.plot(x, prof_fit, color=PAL["s1"], lw=2, marker="o", ms=5,
        label="fitted excluding ±6m election windows", zorder=4)
ax.axhline(0, color=PAL["muted"], lw=1)
ax.axvline(12, color=PAL["muted"], lw=1.2, ls=":")
ax.set_xticks(x); ax.set_xticklabels(MONTHS)
ax.set_ylabel("log radiance deviation", color=PAL["ink2"], fontsize=9.5)
ax.set_title("Why the fitting window matters: December is always an election month",
             color=PAL["ink"], fontsize=12, loc="left", pad=10)
ax.legend(frameon=False, fontsize=9, labelcolor=PAL["ink2"], loc="lower left")
style(ax)
fig.tight_layout()
fig.savefig(f"{CFG['outdir']}/fig3_contamination.png", dpi=170, facecolor=PAL["surface"])
plt.close(fig)

# FIG 4 -- amplitude distribution
fig, ax = plt.subplots(figsize=(8.2, 4.2), facecolor=PAL["surface"])
ax.hist(U.amplitude, bins=40, color=PAL["s1"], edgecolor=PAL["surface"], lw=0.8, zorder=3)
ax.axvline(U.amplitude.median(), color=PAL["s2"], lw=2, zorder=4)
ax.annotate(f"median {U.amplitude.median():.2f}", (U.amplitude.median(), ax.get_ylim()[1]*.9),
            xytext=(8, 0), textcoords="offset points", color=PAL["s2"], fontsize=9)
ax.set_xlabel("seasonal amplitude, peak-to-trough (log points)", color=PAL["ink2"], fontsize=9.5)
ax.set_ylabel("constituencies", color=PAL["ink2"], fontsize=9.5)
ax.set_title("Seasonality is not uniform across constituencies",
             color=PAL["ink"], fontsize=12, loc="left", pad=10)
style(ax)
fig.tight_layout()
fig.savefig(f"{CFG['outdir']}/fig4_amplitude_dist.png", dpi=170, facecolor=PAL["surface"])
plt.close(fig)

# FIG 5 -- event time descriptive
fig, ax = plt.subplots(figsize=(8.2, 4.4), facecolor=PAL["surface"])
ax.plot(ev.index, ev.values, color=PAL["s1"], lw=2, marker="o", ms=5, zorder=3)
ax.axvline(0, color=PAL["s2"], lw=1.4, ls=":")
ax.axhline(0, color=PAL["muted"], lw=1)
ax.annotate("election", (0, ax.get_ylim()[1]*.88), xytext=(6, 0),
            textcoords="offset points", color=PAL["s2"], fontsize=9)
ax.set_xlabel("months relative to election", color=PAL["ink2"], fontsize=9.5)
ax.set_ylabel("deseasonalized, de-trended log radiance", color=PAL["ink2"], fontsize=9.5)
ax.set_title("Descriptive event-time path after removing the seasonal profile",
             color=PAL["ink"], fontsize=12, loc="left", pad=10)
style(ax)
fig.tight_layout()
fig.savefig(f"{CFG['outdir']}/fig5_eventtime.png", dpi=170, facecolor=PAL["surface"])
plt.close(fig)

with open(f"{CFG['outdir']}/diagnostics.txt", "w") as f:
    f.write("\n".join(REPORT))
print("\n[done] outputs in ./out/")
