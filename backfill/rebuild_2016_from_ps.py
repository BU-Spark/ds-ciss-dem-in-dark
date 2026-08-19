#!/usr/bin/env python3
"""
Rebuild the 2016 presidential rows from the polling-station file, so that every
column is a real sourced figure rather than an arithmetic plug.

THE DEFECT BEING REPAIRED
-------------------------
The existing 2016 rows took npp_votes and ndc_votes from the polling-station
file's NPP and NDC columns, took total_valid_votes from its TotalValid column,
and then derived

    other_votes = total_valid_votes - npp_votes - ndc_votes

That derivation is not a measurement.  Wherever the file's TotalValid column
disagreed with the sum of its own seven candidate columns - which happens in
240 of 271 constituencies - the disagreement was silently absorbed into
other_votes.  In ten constituencies it pushed other_votes below zero, and in
Atwima Kwanwoma it inflated a genuine minor-party total of 397 votes to 9,295.

It also made the vote accounting identity untestable for 2016: a residual
cannot fail the test it was constructed to satisfy.

WHAT THIS SCRIPT WRITES INSTEAD
-------------------------------
    npp_votes         = sum of the NPP column over the constituency's stations
    ndc_votes         = sum of the NDC column
    other_votes       = sum of CPP, NDP, PPP, PNC and IND
    total_valid_votes = npp_votes + ndc_votes + other_votes

Every component is now something the source actually reports, and the identity
holds because the total is built from the parts rather than the parts from the
total.  Nationally this moves the valid-vote count by +10,709 on 10.4 million,
or 0.10%.

The declared TotalValid figures are not thrown away.  They are written to
presidential_2016_notes.csv alongside a station-by-station verdict on which
side of each disagreement is corrupt, so a reader can see the size and location
of every discrepancy this decision papers over.

WHICH SIDE IS CORRUPT
---------------------
For a station where the candidate columns and TotalValid disagree, TotalIssued
breaks the tie: the surviving figure is the one satisfying valid + rejected =
issued.  Worked example - Atwima Kwanwoma, D/A PRIM KOTWI (B): candidates sum
to 581, TotalValid reads 5822, rejected is 6, issued is 587.  581 + 6 = 587, so
TotalValid lost a digit.  That single mis-keyed cell is most of the 8,898-vote
error in that constituency.

REGISTER REPAIR
---------------
Bibiani Anhwiaso Bekwai and Suaman are not in the polling-station file; their
rows were scraped.  Both carry a registered_voters value identical to their
2020 value, which cannot be right because the register grew between the two
elections.  The scrape evidently picked up the 2020 figure.  There is no 2016
source for these two registers, so they are set to missing rather than left
wrong.  Their vote figures are untouched.

USAGE
    python rebuild_2016_from_ps.py
"""

import shutil
import sys

import numpy as np
import pandas as pd

from canon_lib import canon

PANEL = "presidential_panel.csv"
PS_FILE = "results2016_ps_pres.csv"
NOTES_OUT = "presidential_2016_notes.csv"
CHANGELOG = "presidential_2016_changes.csv"

YEAR = 2016
SOURCE = "ps_aggregate_v2"

PS_ENCODING = "latin-1"

CANDIDATES = ["CPP", "NDP", "NDC", "PPP", "NPP", "PNC", "IND"]
MINOR = ["CPP", "NDP", "PPP", "PNC", "IND"]
NUMERIC = CANDIDATES + ["RegisteredVoters", "TotalValid", "Rejected",
                        "TotalIssued"]

VOTE_COLS = ["npp_votes", "ndc_votes", "other_votes", "total_valid_votes"]

# Register values copied from 2020 by the scrape.  No 2016 source exists.
BLANK_REGISTER = ["BIBIANI ANHWIASO BEKWAI", "SUAMAN"]


def station_verdict(row):
    """Say which of the two valid-vote figures survived transcription."""
    if row.cand_sum == row.TotalValid:
        return "ok"
    cand_ties = (row.cand_sum + row.Rejected) == row.TotalIssued
    decl_ties = (row.TotalValid + row.Rejected) == row.TotalIssued
    if cand_ties and not decl_ties:
        return "declared_valid_is_corrupt"
    if decl_ties and not cand_ties:
        return "candidate_cell_is_corrupt"
    return "unresolved"


def aggregate():
    """Fold the polling-station file up to constituency level."""
    ps = pd.read_csv(PS_FILE, encoding=PS_ENCODING, low_memory=False)
    for col in NUMERIC:
        ps[col] = pd.to_numeric(ps[col], errors="coerce")
    ps["key"] = ps.Constituency.map(canon)
    ps["cand_sum"] = ps[CANDIDATES].sum(axis=1)
    ps["minor_sum"] = ps[MINOR].sum(axis=1)
    ps["verdict"] = ps.apply(station_verdict, axis=1)

    g = ps.groupby("key").agg(
        ps_name=("Constituency", "first"),
        stations=("Psname", "size"),
        npp_votes=("NPP", "sum"),
        ndc_votes=("NDC", "sum"),
        other_votes=("minor_sum", "sum"),
        declared_valid=("TotalValid", "sum"),
        rejected_ballots=("Rejected", "sum"),
        registered_voters=("RegisteredVoters", "sum"),
    )
    g["total_valid_votes"] = g.npp_votes + g.ndc_votes + g.other_votes
    g["total_votes_cast"] = g.total_valid_votes + g.rejected_ballots
    g["declared_valid_delta"] = g.total_valid_votes - g.declared_valid

    # Per-constituency counts of each station verdict, so the notes file says
    # not just how big a discrepancy is but how many cells caused it.
    verdicts = ps.pivot_table(index="key", columns="verdict",
                              values="Psname", aggfunc="size").fillna(0)
    for col in ("declared_valid_is_corrupt", "candidate_cell_is_corrupt",
                "unresolved"):
        g["stations_" + col] = verdicts[col].astype(int) \
            if col in verdicts.columns else 0
    return g


def main():
    g = aggregate()

    panel = pd.read_csv(PANEL)
    panel["key"] = panel.constituency.map(canon)
    shutil.copy(PANEL, "presidential_panel_prev.csv")

    is_year = ((panel.election_year == YEAR)
               & (panel.race == "presidential")
               & (panel["round"] == 1))

    changes = []
    missing = []

    for key, r in g.iterrows():
        hit = is_year & (panel.key == key)
        if not hit.any():
            missing.append(r.ps_name)
            continue
        idx = panel.index[hit][0]
        before = {c: panel.at[idx, c] for c in VOTE_COLS}
        after = {
            "npp_votes": float(r.npp_votes),
            "ndc_votes": float(r.ndc_votes),
            "other_votes": float(r.other_votes),
            "total_valid_votes": float(r.total_valid_votes),
        }
        if all(before[c] == after[c] for c in VOTE_COLS):
            panel.at[idx, "source"] = SOURCE
            continue
        for c in VOTE_COLS:
            panel.at[idx, c] = after[c]
        panel.at[idx, "registered_voters"] = float(r.registered_voters)
        panel.at[idx, "source"] = SOURCE
        changes.append({
            "constituency": panel.at[idx, "constituency"],
            "old_other": before["other_votes"], "new_other": after["other_votes"],
            "old_valid": before["total_valid_votes"],
            "new_valid": after["total_valid_votes"],
            "valid_delta": after["total_valid_votes"] - before["total_valid_votes"],
        })

    if missing:
        sys.exit("polling-station constituencies absent from the panel: "
                 + ", ".join(sorted(missing)))

    # --- register repair --------------------------------------------------
    for name in BLANK_REGISTER:
        hit = is_year & (panel.key == canon(name))
        if not hit.any():
            continue
        idx = panel.index[hit][0]
        old = panel.at[idx, "registered_voters"]
        if pd.isna(old):
            continue
        panel.at[idx, "registered_voters"] = np.nan
        changes.append({
            "constituency": panel.at[idx, "constituency"],
            "old_other": np.nan, "new_other": np.nan,
            "old_valid": np.nan, "new_valid": np.nan,
            "valid_delta": np.nan,
            "note": f"registered_voters {old:.0f} dropped: copied from 2020",
        })

    # --- gates ------------------------------------------------------------
    # Nothing is written unless the rebuilt rows are internally sound and no
    # other year has been disturbed.
    y = panel[is_year & panel.total_valid_votes.notna()]

    broken = y[y.npp_votes + y.ndc_votes + y.other_votes
               != y.total_valid_votes]
    if len(broken):
        print(broken[["constituency"] + VOTE_COLS].to_string(index=False))
        sys.exit("ABORTED: the rebuilt 2016 rows fail the vote identity.")

    neg = y[(y.npp_votes < 0) | (y.ndc_votes < 0) | (y.other_votes < 0)]
    if len(neg):
        print(neg[["constituency"] + VOTE_COLS].to_string(index=False))
        sys.exit("ABORTED: negative vote components remain in 2016.")

    prev = pd.read_csv("presidential_panel_prev.csv")
    other_years = ~((prev.election_year == YEAR) & (prev.race == "presidential")
                    & (prev["round"] == 1))
    now_other = ~is_year
    if not prev[other_years].reset_index(drop=True).equals(
            panel.drop(columns=["key"])[now_other].reset_index(drop=True)):
        sys.exit("ABORTED: rows outside 2016 round 1 were modified.")

    dupes = y[y.duplicated("key", keep=False)]
    if len(dupes):
        print(dupes[["constituency"]].to_string(index=False))
        sys.exit("ABORTED: duplicate 2016 constituency rows.")

    # --- write ------------------------------------------------------------
    panel.drop(columns=["key"]).to_csv(PANEL, index=False)

    notes = g.reset_index().rename(columns={"ps_name": "constituency"})
    notes[["constituency", "key", "stations", "registered_voters",
           "npp_votes", "ndc_votes", "other_votes", "total_valid_votes",
           "declared_valid", "declared_valid_delta", "rejected_ballots",
           "total_votes_cast", "stations_declared_valid_is_corrupt",
           "stations_candidate_cell_is_corrupt",
           "stations_unresolved"]].to_csv(NOTES_OUT, index=False)
    pd.DataFrame(changes).to_csv(CHANGELOG, index=False)

    print(f"2016 rows rebuilt from stations : {len(g)}")
    print(f"rows whose figures changed      : "
          f"{len([c for c in changes if pd.notna(c.get('valid_delta'))])}")
    print(f"national valid, candidate basis : {int(g.total_valid_votes.sum()):,}")
    print(f"national valid, declared basis  : {int(g.declared_valid.sum()):,}")
    print(f"net change                      : "
          f"{int(g.declared_valid_delta.sum()):+,}")
    print(f"national rejected               : {int(g.rejected_ballots.sum()):,}")
    print(f"national registered             : {int(g.registered_voters.sum()):,}")
    print(f"negative other_votes remaining  : {int((y.other_votes < 0).sum())}")
    print(f"wrote {PANEL}, {NOTES_OUT}, {CHANGELOG}")
    print("backup: presidential_panel_prev.csv")


if __name__ == "__main__":
    main()
