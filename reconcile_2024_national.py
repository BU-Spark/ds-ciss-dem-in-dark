#!/usr/bin/env python3
"""
Reconcile the repaired 2024 panel against the EC's declared national totals.

WHY
---
Every check up to this point has been local: does this sheet balance, does this
row match its sheet.  A national reconciliation is the only global test.  If a
constituency somewhere still carries figures copied from a neighbour, or a
column has been silently truncated, the aggregate will drift away from the
published national count even though every individual row looks clean.

THE BENCHMARK
    The EC's updated declaration covers 275 of the 276 constituencies.
    Ablekuma North (registered 121,269) was excluded from the declaration
    because its collation was not completed.

USAGE
    python reconcile_2024_national.py
"""

import pandas as pd

PANEL = "presidential_panel.csv"
YEAR = 2024

# EC updated declaration, 275 constituencies (Ablekuma North excluded).
EC = {
    "ndc": 6_591_790,
    "npp": 4_877_611,
    "valid": 11_683_483,
}


def main():
    p = pd.read_csv(PANEL)
    y = p[(p.election_year == YEAR) & (p.race == "presidential")]
    have = y[y.total_valid_votes.notna()]
    missing = y[y.total_valid_votes.isna()]

    npp = int(have.npp_votes.sum())
    ndc = int(have.ndc_votes.sum())
    valid = int(have.total_valid_votes.sum())

    rows = [
        ("NPP (Bawumia)", npp, EC["npp"]),
        ("NDC (Mahama)", ndc, EC["ndc"]),
        ("Total valid", valid, EC["valid"]),
    ]

    print(f"constituencies with figures : {len(have)} of 276")
    absent = sorted(set(missing.constituency))
    print(f"blank in the panel          : {len(absent) + 276 - len(y)}"
          + (f" ({', '.join(absent)} + rows never created)" if absent else ""))
    print()
    print(f"{'':16s}{'panel':>13s}{'EC declared':>14s}"
          f"{'difference':>13s}{'':>9s}")
    for label, mine, ec in rows:
        d = mine - ec
        print(f"{label:16s}{mine:>13,d}{ec:>14,d}{d:>13,d}"
              f"{d / ec:>9.2%}")

    print()
    print("The panel is expected to run BELOW the EC total, because the seats "
          "left blank\ncontributed votes to the national count that the panel "
          "does not carry.")
    print()
    print(f"NDC share of valid, panel : {ndc / valid:.2%}   "
          f"EC: {EC['ndc'] / EC['valid']:.2%}")
    print(f"NPP share of valid, panel : {npp / valid:.2%}   "
          f"EC: {EC['npp'] / EC['valid']:.2%}")
    print("\nShares are the sharper test: they are insensitive to how many "
          "seats are\nmissing, so a drift here means the surviving rows "
          "themselves are wrong.")


if __name__ == "__main__":
    main()
