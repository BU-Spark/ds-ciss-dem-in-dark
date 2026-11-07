"""
two_line_series.py -- shared panel-loading, event-time, and region-matched
grouping helpers, reused by per_election_gap.py.

Reuses load_alan_panel, nearest_election, and attach_predetermined_close from
election_did.py rather than re-deriving event time / competitiveness logic.

Provides:
    load_panel_with_region  -- con_id x year-month panel (via load_alan_panel)
                                with region_2019 attached.
    attach_real_event_time  -- event time to the nearest December election,
                                predetermined `close`, |k|<=window.
    mean_close_lut          -- one stable competitiveness score per
                                constituency (mean predetermined `close`
                                across election cycles).
    region_lut               -- con_id -> region_2019, one row per
                                constituency.
    region_matched_groups   -- within each region_2019 with >=6
                                constituencies, top/bottom third by close;
                                pooled across regions.

No global execution -- this is a pure library, imported by
per_election_gap.py as `tls`.
"""

from __future__ import annotations

import pandas as pd

import election_did as ed

WINDOW = 18
MIN_REGION_SIZE = 6


# --------------------------------------------------------------------------- #
# Load + event time
# --------------------------------------------------------------------------- #
def load_panel_with_region(path) -> pd.DataFrame:
    """con_id x year-month panel (via load_alan_panel) with region_2019 attached."""
    df = ed.load_alan_panel(path)
    region = pd.read_csv(path, usecols=["con_id", "region_2019"]).drop_duplicates("con_id")
    return df.merge(region, on="con_id", how="left")


def attach_real_event_time(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Event time to the nearest December election, predetermined `close`, |k|<=window."""
    out = df.copy()
    ne = ed.nearest_election(out["year"], out["month"])
    out["k"] = ne["event_k"].to_numpy()
    out["cycle_year"] = ne["cycle_year"].to_numpy()
    out = ed.attach_predetermined_close(out)
    return out[out["k"].abs() <= window].copy()


# --------------------------------------------------------------------------- #
# Competitiveness score + grouping
# --------------------------------------------------------------------------- #
def mean_close_lut(real_df: pd.DataFrame) -> pd.DataFrame:
    """One stable competitiveness score per constituency: mean of `close` across cycles."""
    cyc = real_df.dropna(subset=["close"]).groupby(["con_id", "cycle_year"], as_index=False)["close"].first()
    return cyc.groupby("con_id", as_index=False)["close"].mean().rename(columns={"close": "mean_close"})


def region_lut(df: pd.DataFrame) -> pd.DataFrame:
    """con_id -> region_2019, one row per constituency."""
    return df[["con_id", "region_2019"]].drop_duplicates("con_id")


def region_matched_groups(mean_close: pd.DataFrame, region_map: pd.DataFrame,
                           min_region_size: int = MIN_REGION_SIZE) -> pd.DataFrame:
    """Within each region with >=min_region_size constituencies, top/bottom third by
    close; pooled across qualifying regions."""
    df = mean_close.merge(region_map, on="con_id", how="left")
    rows = []
    for _, sub in df.groupby("region_2019"):
        if len(sub) < min_region_size:
            continue
        sub = sub.sort_values("mean_close", ascending=False)
        third = len(sub) // 3
        rows.append(sub.iloc[:third][["con_id"]].assign(group="competitive"))
        rows.append(sub.iloc[-third:][["con_id"]].assign(group="safe"))
    return pd.concat(rows, ignore_index=True)
