#!/usr/bin/env python3
"""
Find panel rows where npp + ndc + other does not equal total valid votes.

WHY THIS IS A SEPARATE TEST
---------------------------
The 2024 audit looked for whole vote quadruples duplicated across rows, plus
impossible turnout and implausible year-on-year ratios.  Those catch a row that
inherited ALL FOUR figures from another constituency.  They do not catch a row
that inherited just ONE column - the quadruple is then unique, turnout still
looks sane, and the ratio barely moves.

The accounting identity catches exactly that case, because a single borrowed
column almost never happens to preserve the sum.  It found BAWKU CENTRAL in
2024, whose other_votes is a copy of BAWKU WEST's, plus twelve long-standing
breaks in earlier years that predate the 2024 work.

USAGE
    python audit_vote_identity.py [panel.csv]
"""

import sys

import pandas as pd

PANEL = sys.argv[1] if len(sys.argv) > 1 else "presidential_panel.csv"
OUT = "identity_failures.csv"

COLS = ["npp_votes", "ndc_votes", "other_votes", "total_valid_votes"]


def main():
    p = pd.read_csv(PANEL)
    d = p.dropna(subset=COLS).copy()
    d["gap"] = (d.npp_votes + d.ndc_votes + d.other_votes
                - d.total_valid_votes)
    bad = d[d.gap != 0].sort_values(["election_year", "constituency"])

    cols = ["election_year", "constituency"] + COLS + ["gap", "source"]
    bad[cols].to_csv(OUT, index=False)

    print(f"rows checked        : {len(d)}")
    print(f"identity failures   : {len(bad)}")
    if len(bad):
        print()
        print(bad[cols].to_string(index=False))
        by_year = bad.election_year.value_counts().sort_index().to_dict()
        print(f"\nby year: {by_year}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
