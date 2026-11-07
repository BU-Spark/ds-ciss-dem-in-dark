"""
election_did.py -- shared panel-loading and event-time helpers for the
election / nightlights pipeline.

Provides:
    load_alan_panel             -- read the ALAN monthly panel with election
                                    covariates (margin_last, wide per-cycle
                                    margin columns, weather controls).
    nearest_election             -- signed months to the nearest December
                                    election and which cycle that is.
    attach_predetermined_close   -- predetermined `close` (1 - |margin|) from
                                    the PRIOR election of each row's cycle,
                                    using the wide margin_<year> columns.

No global execution -- functions here are imported by two_line_series.py
and per_election_gap.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ELECTION_YEARS = (2012, 2016, 2020, 2024)
PLACEBO_YEAR = 2018
WINDOW = 6

# Wide-format margin column for each election year, used to pin `close` to the
# PRIOR election of a row's cycle (see attach_predetermined_close). Cycle 2012
# would need a margin_2008 column, which does not exist in the source data.
MARGIN_COL_BY_YEAR = {
    2012: "margin_2012",
    2016: "margin_2016",
    2020: "margin_2020",
    2024: "margin_2024",
}


def load_alan_panel(path) -> pd.DataFrame:
    """Monthly con_id x year-month ALAN (artificial light at night) panel,
    pre-merged with election covariates.

    Keeps con_id, year, month, y, margin_last, the wide per-cycle margin
    columns (margin_2012/2016/2020/2024), aod, precip_mm, tmean_c. Rows with
    a missing outcome are dropped.

    Does NOT build `close` here: `margin_last` is the margin of the
    most-recently-HELD election, which changes value inside the election
    month itself, so deriving `close` from it would make the moderator jump
    discontinuously right at event time k=0. Callers must first build event
    time (nearest_election) to get `cycle_year`, then call
    attach_predetermined_close to pin `close` to the PRIOR election of that
    cycle.
    """
    cols = ["con_id", "year", "month", "y", "margin_last",
            *MARGIN_COL_BY_YEAR.values(), "aod", "precip_mm", "tmean_c"]
    df = pd.read_csv(path, usecols=cols)
    df = df.dropna(subset=["y"]).copy()
    return df


def nearest_election(year: pd.Series, month: pd.Series,
                      election_years=ELECTION_YEARS) -> pd.DataFrame:
    """Signed months to the nearest December election, and which cycle that is.

    event_k > 0 means the row is *after* that election.
    """
    k = e_year = best_abs = None
    for e in election_years:
        cand_k = (year.to_numpy() - e) * 12 + (month.to_numpy() - 12)
        cand_abs = np.abs(cand_k)
        if best_abs is None:
            k, e_year, best_abs = cand_k, np.full(len(year), e), cand_abs
        else:
            better = cand_abs < best_abs
            k = np.where(better, cand_k, k)
            e_year = np.where(better, e, e_year)
            best_abs = np.where(better, cand_abs, best_abs)
    return pd.DataFrame({"event_k": k, "cycle_year": e_year})


def attach_predetermined_close(df: pd.DataFrame,
                                margin_col_by_year: dict[int, str] = MARGIN_COL_BY_YEAR,
                                gap_years: int = 4) -> pd.DataFrame:
    """Predetermined closeness from the wide margin columns, pinned to the
    PRIOR election of each row's cycle.

    Requires a `cycle_year` column (from nearest_election). For each row,
    close = 1 - abs(margin_{cycle_year - gap_years}), picked row-wise from
    margin_col_by_year. Since margin_{cycle_year - gap_years} is fixed for
    the whole cycle, `close` is constant within every (con_id, cycle_year)
    event window and does not move at k=0. Cycle years whose prior election
    year has no margin column (e.g. 2012 -> 2008) get NaN, which the
    estimator drops.
    """
    out = df.copy()
    prior_year = out["cycle_year"] - gap_years
    close = pd.Series(np.nan, index=out.index, dtype=float)
    for year, col in margin_col_by_year.items():
        if col not in out.columns:
            continue
        mask = prior_year == year
        close = close.where(~mask, 1 - out[col].abs())
    out["close"] = close
    return out
