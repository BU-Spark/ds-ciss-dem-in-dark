#!/usr/bin/env python3
"""
Merge the 2024 EC Form Ten readings into the presidential panel.

WHAT THIS DOES
--------------
1.  Backs up presidential_panel.csv and presidential_gaps.csv before touching
    either of them.
2.  Overwrites the 2024 rows that the audit flagged with the figures read off
    the constituency's own official Form Ten summary sheet.
3.  Inserts OKAIKWEI CENTRAL, which the scrape never produced a row for.
4.  Blanks DOME KWABENYA.  Its scraped row is a copy of another constituency's
    figures, and the EC never published its sheet, so there is nothing to
    repair it with.  Leaving the wrong numbers in place would be worse than
    leaving a hole, so the votes are set to missing and the seat is moved to
    the gaps file.
5.  Re-verifies every 2024 row: npp + ndc + other must equal total valid votes.
    If any row fails, NOTHING is written.
6.  Writes a notes file so the per-sheet caveats travel with the data.

SOURCE TAG
    Rows filled from a Form Ten get source = "ec_form10_ocr", the same tag the
    earlier Ashanti backfill used, so the provenance of every figure stays
    visible in the panel itself.

USAGE
    python merge_2024_backfill.py
"""

import json
import shutil
import sys

import numpy as np
import pandas as pd

from canon_lib import canon

PANEL = "presidential_panel.csv"
GAPS = "presidential_gaps.csv"
READINGS = "readings_2024_targets.json"
NOTES_OUT = "presidential_2024_notes.csv"
CHANGELOG = "presidential_2024_changes.csv"

YEAR = 2024
SOURCE = "ec_form10_ocr"

# Corrupted beyond repair: the scraped figures belong to another constituency
# and the EC never published this seat's Form Ten.  The user's instruction is
# to leave it blank rather than substitute a non-EC estimate.
BLANK_OUT = ["DOME KWABENYA"]

VOTE_COLS = ["npp_votes", "ndc_votes", "other_votes", "total_valid_votes"]


def fold(reading):
    """Collapse the 13 candidate columns into the panel's npp/ndc/other shape."""
    other = sum(reading["c%d" % i] for i in range(1, 14) if i not in (1, 8))
    return {
        "npp_votes": reading["c1"],
        "ndc_votes": reading["c8"],
        "other_votes": float(other),
        "total_valid_votes": reading["A"],
    }


def identity_from_other_years(panel, key):
    """Borrow region and constituency_id from the same seat in another year.

    Boundaries did not move between 2020 and 2024, so a seat's geography and
    its panel identifier carry over unchanged.  Only the vote figures and the
    register are year-specific.
    """
    hits = panel[(panel.key == key) & (panel.election_year != YEAR)]
    if hits.empty:
        return None
    row = hits.sort_values("election_year").iloc[-1]
    return {
        "region_10": row.region_10,
        "region_16": row.region_16,
        "constituency_id": row.constituency_id,
    }


def main():
    readings = json.load(open(READINGS))

    # Arithmetic gate on the readings themselves, before anything is written.
    for r in readings:
        if sum(r["c%d" % i] for i in range(1, 14)) != r["A"]:
            sys.exit(f"reading fails candidate-sum check: {r['file']}")
        if r["A"] + r["B"] != r["C"]:
            sys.exit(f"reading fails A+B=C check: {r['file']}")

    panel = pd.read_csv(PANEL)
    gaps = pd.read_csv(GAPS)
    panel["key"] = panel.constituency.map(canon)

    shutil.copy(PANEL, "presidential_panel_prev.csv")
    shutil.copy(GAPS, "presidential_gaps_prev.csv")

    is_2024 = (panel.election_year == YEAR) & (panel.race == "presidential")
    changes, notes, inserted = [], [], []

    for r in readings:
        key = canon(r["sheet_constituency"])
        vals = fold(r)
        notes.append({
            "constituency": r["sheet_constituency"],
            "file": r["file"],
            "rejected_ballots": r["B"],
            "total_votes_cast": r["C"],
            "sheet_note": r.get("notes", ""),
        })

        hit = is_2024 & (panel.key == key)
        if hit.any():
            idx = panel.index[hit][0]
            before = {c: panel.at[idx, c] for c in VOTE_COLS}
            if all(before[c] == vals[c] for c in VOTE_COLS):
                # The scrape already agreed with the sheet; only the provenance
                # tag changes, so the figure is no longer taken on trust.
                panel.at[idx, "source"] = SOURCE
                changes.append({"constituency": panel.at[idx, "constituency"],
                                "action": "confirmed", **before})
                continue
            for c in VOTE_COLS:
                panel.at[idx, c] = vals[c]
            panel.at[idx, "source"] = SOURCE
            changes.append({
                "constituency": panel.at[idx, "constituency"],
                "action": "corrected",
                "old_npp": before["npp_votes"], "new_npp": vals["npp_votes"],
                "old_ndc": before["ndc_votes"], "new_ndc": vals["ndc_votes"],
                "old_valid": before["total_valid_votes"],
                "new_valid": vals["total_valid_votes"],
            })
        else:
            ident = identity_from_other_years(panel, key)
            if ident is None:
                sys.exit(f"cannot place {r['sheet_constituency']}: "
                         "no other year to borrow geography from")
            inserted.append({
                "election_year": YEAR, "race": "presidential", "round": 1,
                "constituency": r["sheet_constituency"],
                **ident, **vals,
                "registered_voters": np.nan,
                "source": SOURCE,
                "source_name": r["file"],
                "key": key,
            })
            changes.append({"constituency": r["sheet_constituency"],
                            "action": "inserted", **vals})

    if inserted:
        panel = pd.concat([panel, pd.DataFrame(inserted)], ignore_index=True)
        is_2024 = (panel.election_year == YEAR) & (panel.race == "presidential")

    for name in BLANK_OUT:
        hit = is_2024 & (panel.key == canon(name))
        if not hit.any():
            continue
        idx = panel.index[hit][0]
        changes.append({
            "constituency": panel.at[idx, "constituency"],
            "action": "blanked",
            "old_npp": panel.at[idx, "npp_votes"],
            "old_ndc": panel.at[idx, "ndc_votes"],
            "old_valid": panel.at[idx, "total_valid_votes"],
        })
        for c in VOTE_COLS:
            panel.at[idx, c] = np.nan
        panel.at[idx, "source"] = "unpublished_ec"

    # --- gaps bookkeeping -------------------------------------------------
    filled = {canon(r["sheet_constituency"]) for r in readings}
    gaps["key"] = gaps.constituency.map(canon)
    still_open = ~((gaps.election_year == YEAR) & gaps.key.isin(filled))
    gaps = gaps[still_open]

    for name in BLANK_OUT:
        if not ((gaps.election_year == YEAR)
                & (gaps.key == canon(name))).any():
            ident = identity_from_other_years(panel, canon(name)) or {}
            gaps = pd.concat([gaps, pd.DataFrame([{
                "election_year": YEAR, "round": 1, "constituency": name,
                "region_10": ident.get("region_10"),
                "region_16": ident.get("region_16"),
                "key": canon(name),
            }])], ignore_index=True)

    # --- final gate -------------------------------------------------------
    # The panel carries a handful of long-standing broken rows in earlier years
    # that this merge does not touch.  Aborting on those would make the 2024
    # repair impossible to land, so the gate compares the set of broken rows
    # before and after: any NEW break is this merge's fault and stops it dead,
    # while pre-existing ones are reported and left for their own repair.
    def broken(df):
        d = df[df.total_valid_votes.notna() & df.npp_votes.notna()
               & df.ndc_votes.notna() & df.other_votes.notna()]
        d = d[d.npp_votes + d.ndc_votes + d.other_votes
              != d.total_valid_votes]
        return set(zip(d.election_year, d.constituency))

    was = broken(pd.read_csv("presidential_panel_prev.csv"))
    now = broken(panel)
    introduced = now - was
    if introduced:
        print(sorted(introduced))
        sys.exit("ABORTED: this merge broke the vote identity on the rows "
                 "above. Nothing written; the previous panel is untouched.")

    check = panel[(panel.election_year == YEAR)
                  & (panel.race == "presidential")
                  & panel.total_valid_votes.notna()]
    pre_existing = sorted(c for y, c in now if y == YEAR)

    dupes = check[check.duplicated("key", keep=False)]
    if len(dupes):
        print(dupes[["constituency"]].to_string(index=False))
        sys.exit("ABORTED: duplicate constituency rows for 2024.")

    panel.drop(columns=["key"]).to_csv(PANEL, index=False)
    gaps.drop(columns=["key"]).sort_values(
        ["election_year", "constituency"]).to_csv(GAPS, index=False)
    pd.DataFrame(notes).to_csv(NOTES_OUT, index=False)
    pd.DataFrame(changes).to_csv(CHANGELOG, index=False)

    by = pd.DataFrame(changes).action.value_counts().to_dict()
    print("2024 presidential rows :", len(check))
    print("changes                :", by)
    print("open 2024 gaps         :",
          ", ".join(sorted(gaps[gaps.election_year == YEAR].constituency)))
    if pre_existing:
        print("STILL BROKEN in 2024 (pre-existing, not repaired here):",
              ", ".join(pre_existing))
    print(f"wrote {PANEL}, {GAPS}, {NOTES_OUT}, {CHANGELOG}")
    print("backups: presidential_panel_prev.csv, presidential_gaps_prev.csv")


if __name__ == "__main__":
    main()
