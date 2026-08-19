#!/usr/bin/env python3
"""
Compare the 44 Ashanti-plus-Zebilla control sheets against the scraped panel.

WHY THIS EXISTS
---------------
The 35 rows the audit flagged are known-bad: they failed a duplicate, turnout
or 2024/2020-ratio test.  But a copied vote quadruple only shows up as a
duplicate when BOTH the source row and the corrupted row happen to sit in the
file.  A row that quietly inherited figures from a constituency outside the
panel would pass every test.  So the flagged set is a floor, not necessarily a
total.

Reading a whole region's official sheets and diffing them against their scraped
counterparts measures the true error rate on rows the audit did NOT flag.  If
the unflagged Ashanti rows all match, the audit caught everything.  If they do
not, the repair scope has to grow and the client has to be told.

USAGE
    python compare_ashanti_control.py
"""

import json

import pandas as pd

from canon_lib import canon

PANEL = "presidential_panel.csv"
READINGS = "readings_2024_ashanti.json"
TARGETS = "ec_2024_targets.csv"
OUT = "ashanti_control_diff.csv"

YEAR = 2024


def sheet_rows():
    """Fold each Form Ten reading into the panel's npp/ndc/other/valid shape."""
    out = []
    for r in json.load(open(READINGS)):
        other = sum(r["c%d" % i] for i in range(1, 14) if i not in (1, 8))
        out.append({
            "constituency": r["sheet_constituency"],
            "key": canon(r["sheet_constituency"]),
            "sheet_npp": r["c1"],
            "sheet_ndc": r["c8"],
            "sheet_other": other,
            "sheet_valid": r["A"],
            "sheet_rejected": r["B"],
            "sheet_cast": r["C"],
        })
    return pd.DataFrame(out)


def main():
    sheets = sheet_rows()

    panel = pd.read_csv(PANEL)
    y = panel[(panel.election_year == YEAR) & (panel.race == "presidential")].copy()
    y["key"] = y.constituency.map(canon)

    # Rows the audit already flagged are expected to differ; the interesting
    # signal is a mismatch on a row nobody suspected.
    flagged = {canon(c) for c in pd.read_csv(TARGETS).constituency}

    m = sheets.merge(
        y[["key", "constituency", "npp_votes", "ndc_votes",
           "other_votes", "total_valid_votes"]],
        on="key", how="left", suffixes=("", "_panel"))

    m["in_panel"] = m.npp_votes.notna()
    m["flagged"] = m.key.isin(flagged)
    m["npp_diff"] = m.npp_votes - m.sheet_npp
    m["ndc_diff"] = m.ndc_votes - m.sheet_ndc
    m["valid_diff"] = m.total_valid_votes - m.sheet_valid
    m["match"] = (m.npp_diff == 0) & (m.ndc_diff == 0) & (m.valid_diff == 0)

    m.sort_values(["match", "constituency"]).to_csv(OUT, index=False)

    absent = m[~m.in_panel]
    present = m[m.in_panel]
    ok = present[present.match]
    bad = present[~present.match]
    surprise = bad[~bad.flagged]

    print(f"sheets read              : {len(m)}")
    print(f"not present in panel     : {len(absent)}"
          + (" -> " + ", ".join(absent.constituency) if len(absent) else ""))
    print(f"matched the panel exactly: {len(ok)}")
    print(f"disagreed with the panel : {len(bad)}")
    print(f"  of which already flagged: {len(bad[bad.flagged])}")
    print(f"  NOT flagged by the audit: {len(surprise)}")
    if len(surprise):
        print("\nUNFLAGGED MISMATCHES - the audit missed these:")
        for _, r in surprise.iterrows():
            print(f"  {r.constituency:28s} "
                  f"panel {int(r.npp_votes):>7}/{int(r.ndc_votes):>7}/"
                  f"{int(r.total_valid_votes):>7}   "
                  f"sheet {r.sheet_npp:>7}/{r.sheet_ndc:>7}/{r.sheet_valid:>7}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
