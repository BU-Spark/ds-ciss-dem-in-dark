#!/usr/bin/env python3
"""Shared data preparation for the five political analyses (see specs.md).

Reads panel_monthly_with_elections.csv.gz, constructs all analysis variables,
writes analysis_panel.pkl. Every analysis script imports load() from here.
"""
import numpy as np
import pandas as pd

PANEL = "panel_monthly_with_elections.csv.gz"
ELECTIONS = [2012, 2016, 2020, 2024]

def load(path=PANEL):
    df = pd.read_csv(path)
    assert len(df) == 42075 and df.duplicated(["con_id", "year", "month"]).sum() == 0

    df["ym"] = df.year * 12 + df.month                      # month counter
    df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1)).dt.to_period("M")

    # ---- event time relative to nearest election December (window bookkeeping)
    evt = np.where(df.months_to_elec <= 6, -df.months_to_elec,
          np.where(df.months_since_elec <= 6, df.months_since_elec, np.nan))
    # December of an election year: months_since==0 (and months_to==48 for prior ones)
    evt = np.where(df.months_since_elec == 0, 0, evt)
    df["evt_a"] = evt                                        # NaN outside ±6 window
    df["pre4"] = ((df.months_to_elec <= 3)).astype(int)      # Sep..Dec before an election (evt −3..0 via to==0 handled below)
    df.loc[df.months_since_elec == 0, "pre4"] = 1            # Dec itself
    # relevant election for the window month
    df["elec_rel"] = np.where(df.evt_a < 0, df.next_elec, df.last_elec)

    # ---- margins / swing for the relevant election
    for e in ELECTIONS:
        df[f"absmargin_{e}"] = df[f"margin_{e}"].abs()
    def pick(row_year_col, stub):
        out = pd.Series(np.nan, index=df.index)
        for e in ELECTIONS:
            m = df[row_year_col] == e
            out[m] = df.loc[m, f"{stub}_{e}"]
        return out
    df["margin_rel"] = pick("elec_rel", "margin")
    df["swing_rel"] = (df.margin_rel.abs() < 0.10).astype(float)
    df["swing_rel_05"] = (df.margin_rel.abs() < 0.05).astype(float)
    df["swing_last"] = (df.margin_last.abs() < 0.10).astype(float)
    df["swing_last_05"] = (df.margin_last.abs() < 0.05).astype(float)

    # ---- national ruling party and alignment
    df["ruling_npp"] = ((df.ym >= 2017 * 12 + 1) & (df.ym <= 2024 * 12 + 12)).astype(int)
    winner_npp = (df.margin_last > 0).astype(float).where(df.margin_last.notna())
    df["aligned"] = (winner_npp == df.ruling_npp).astype(float).where(winner_npp.notna())

    # ---- dumsor DID pieces
    df["opp2012"] = (df.margin_2012 > 0).astype(float).where(df.margin_2012.notna())
    df["dumsor"] = ((df.ym >= 2013 * 12 + 1) & (df.ym <= 2015 * 12 + 12)).astype(int)
    df["dumsor_narrow"] = ((df.ym >= 2014 * 12 + 1) & (df.ym <= 2015 * 12 + 12)).astype(int)

    # ---- mechanism outcomes
    df["log_avg_rad"] = np.log(df.avg_rad.clip(lower=0.01))
    df["nights_lit_frac"] = (df.nights_lit_0p5 / df.nights_valid).clip(0, 1)

    # ---- urban flag: top quartile of 2012-13 mean lit_share_0p5
    base = df[df.year <= 2013].groupby("con_id").lit_share_0p5.mean()
    urban_ids = set(base[base >= base.quantile(0.75)].index)
    df["urban"] = df.con_id.isin(urban_ids).astype(int)

    df["usable"] = df.usable.astype(bool)
    return df

def unit_cycle_frame(df):
    """One row per unit x election cycle (2012/2016/2020): Δy outcome, margins, RD vars."""
    rows = []
    y = "y_ds_cluster_dustadj"
    for e in [2012, 2016, 2020]:
        e_ym = e * 12 + 12
        post = df[(df.ym >= e_ym + 7) & (df.ym <= e_ym + 42) & df.usable]
        pre = df[(df.ym >= e_ym - 18) & (df.ym <= e_ym - 1) & df.usable]
        gpost = post.groupby("con_id")[y].agg(["mean", "count"])
        gpre = pre.groupby("con_id")[y].agg(["mean", "count"])
        g = gpost.join(gpre, lsuffix="_post", rsuffix="_pre")
        g = g[(g.count_post >= 12) & (g.count_pre >= 6)]
        base = df[df.ym == e_ym].set_index("con_id")
        for cid, r in g.iterrows():
            b = base.loc[cid]
            margin = b[f"margin_{e}"]
            if pd.isna(margin):
                continue
            ruling_margin = margin if e in (2016, 2020) else -margin  # NPP rules after 2016/2020; NDC after 2012
            rows.append(dict(
                con_id=cid, cycle=e, dy=r["mean_post"] - r["mean_pre"],
                y_pre=r["mean_pre"], margin=margin, absmargin=abs(margin),
                ruling_margin=ruling_margin, ruling_win=float(ruling_margin > 0),
                zone3=b.zone3, urban=b.urban,
                baseline_lit=b.lit_share_0p5,
            ))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = load()
    df.to_pickle("analysis_panel.pkl")
    uc = unit_cycle_frame(df)
    uc.to_pickle("unit_cycle.pkl")
    print("panel:", df.shape, "| usable:", int(df.usable.sum()))
    print("evt window rows:", int(df.evt_a.notna().sum()))
    print("aligned mean:", round(df.aligned.mean(), 3), "| swing_last mean:", round(df.swing_last.mean(), 3))
    print("unit-cycles:", len(uc), "| by cycle:"); print(uc.groupby("cycle").size())
    print("close (<5pp) by cycle:"); print(uc[uc.absmargin < 0.05].groupby("cycle").size())
