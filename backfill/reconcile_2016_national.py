#!/usr/bin/env python3
"""
Reconcile the rebuilt 2016 panel against the published national figures, and
test whether the national benchmark can be used to infer the two missing seats.

THE SHORT ANSWER IS THAT IT CANNOT, AND THIS SCRIPT SHOWS WHY
-------------------------------------------------------------
The tempting shortcut for Tema Central and Trobu is subtraction: take the
national total, subtract the 273 constituencies already in the panel, and call
the remainder the two missing seats.  That is only valid if the national
figures and the constituency figures come from the same accounting.  They do
not, and the arithmetic says so out loud in three places:

  1.  The published national figures do not balance with each other.  Valid
      plus rejected comes to 10,781,609, but the published votes-cast figure is
      10,781,917 - a gap of 308 ballots that exists inside the benchmark
      itself, before any constituency data is involved.

  2.  The 271 constituencies in the polling-station file already report MORE
      rejected ballots than the published national rejected total for all 275.
      Four constituencies are missing, so this is impossible.  Most of it is
      one station: SPECIAL_NEW JUABEN SOUTH records 1,224 rejected against
      1,223 valid, its valid and rejected cells evidently transposed.

  3.  The same is true of minor-party votes: the 271 constituencies already
      exceed the published national minor-party total.

Because of (2) and (3), subtraction produces a NEGATIVE minor-party count for
Tema Central and Trobu combined, and an NDC figure less than half what those
two seats returned in both 2012 and 2020.  A residual that implausible is the
benchmark failing, not the seats being strange.

The conclusion is a practical one: the two missing seats have to be sourced
from a constituency-level record, which is what scrape_2016_peacefm.py is for.
The national comparison below is still worth running as a sanity band - it
confirms the party shares are right to within a tenth of a point - but it must
not be used as a gate.

USAGE
    python reconcile_2016_national.py
"""

import pandas as pd

from canon_lib import canon

PANEL = "presidential_panel.csv"
NOTES = "presidential_2016_notes.csv"

YEAR = 2016

# Published national figures for the 2016 presidential election.
# Sources: IFES Election Guide (election id 2579) and contemporaneous
# reporting of the Electoral Commission's declaration.
EC = {
    "registered": 15_712_499,
    "valid": 10_615_361,
    "rejected": 166_248,
    "cast_published": 10_781_917,
    "npp": 5_716_026,
    "ndc": 4_713_277,
}
EC["minor"] = EC["valid"] - EC["npp"] - EC["ndc"]
EC["cast_implied"] = EC["valid"] + EC["rejected"]

# The two seats the polling-station file never covered.
MISSING = ["TEMA CENTRAL", "TROBU"]


def money(x):
    return f"{x:>12,.0f}"


def main():
    panel = pd.read_csv(PANEL)
    panel["key"] = panel.constituency.map(canon)
    p = panel[(panel.race == "presidential") & (panel["round"] == 1)]
    y = p[(p.election_year == YEAR) & p.total_valid_votes.notna()]

    have = {
        "npp": y.npp_votes.sum(),
        "ndc": y.ndc_votes.sum(),
        "minor": y.other_votes.sum(),
        "valid": y.total_valid_votes.sum(),
        "registered": y.registered_voters.sum(),
    }

    print("BENCHMARK SELF-CONSISTENCY")
    print(f"  valid + rejected            {money(EC['cast_implied'])}")
    print(f"  published votes cast        {money(EC['cast_published'])}")
    print(f"  gap inside the benchmark    {money(EC['cast_published'] - EC['cast_implied'])}")

    print(f"\nPANEL COVERAGE, {YEAR}")
    print(f"  constituencies with figures : {len(y)} of 275")
    print(f"  absent                      : {', '.join(MISSING)}")

    print("\nPANEL vs PUBLISHED NATIONAL")
    print(f"  {'':<16}{'panel':>13}{'published':>14}{'residual':>13}")
    for label, k in (("NPP", "npp"), ("NDC", "ndc"),
                     ("minor parties", "minor"), ("total valid", "valid"),
                     ("registered", "registered")):
        print(f"  {label:<16}{money(have[k])}{money(EC[k])}"
              f"{money(EC[k] - have[k])}")

    print("\nVOTE SHARES (the part that does hold up)")
    for label, k in (("NPP", "npp"), ("NDC", "ndc")):
        print(f"  {label:<16}panel {100 * have[k] / have['valid']:6.2f}%"
              f"   published {100 * EC[k] / EC['valid']:6.2f}%"
              f"   diff {100 * have[k] / have['valid'] - 100 * EC[k] / EC['valid']:+.2f}pp")

    # --- the impossibility checks ----------------------------------------
    notes = pd.read_csv(NOTES)
    ps_rejected = notes.rejected_ballots.sum()

    print("\nIMPOSSIBILITY CHECKS")
    print(f"  rejected, 271 constituencies: {money(ps_rejected)}")
    print(f"  rejected, published, all 275: {money(EC['rejected'])}")
    print(f"  -> 271 seats already exceed the national total by "
          f"{ps_rejected - EC['rejected']:,.0f}")
    print(f"  minor votes, {len(y)} constituencies: {money(have['minor'])}")
    print(f"  minor votes, published      : {money(EC['minor'])}")
    if have["minor"] > EC["minor"]:
        print(f"  -> the panel already exceeds the national total by "
              f"{have['minor'] - EC['minor']:,.0f}")

    # --- what subtraction would produce for the two missing seats --------
    resid = {k: EC[k] - have[k] for k in ("npp", "ndc", "minor", "valid")}
    print(f"\nWHAT SUBTRACTION WOULD ASSIGN TO {' + '.join(MISSING)}")
    for label, k in (("NPP", "npp"), ("NDC", "ndc"),
                     ("minor parties", "minor"), ("total valid", "valid")):
        print(f"  {label:<16}{money(resid[k])}")

    # Compare against the same two seats in the elections either side.
    keys = [canon(m) for m in MISSING]
    for other_year in (2012, 2020):
        o = p[(p.election_year == other_year) & p.key.isin(keys)]
        if len(o) != len(MISSING):
            continue
        print(f"  same two seats in {other_year}: "
              f"NPP {o.npp_votes.sum():,.0f}   NDC {o.ndc_votes.sum():,.0f}   "
              f"valid {o.total_valid_votes.sum():,.0f}")

    verdict = []
    if resid["minor"] < 0:
        verdict.append("minor-party votes come out negative")
    if resid["ndc"] < 0:
        verdict.append("NDC votes come out negative")
    print("\nVERDICT")
    if verdict:
        print("  Subtraction is unusable: " + "; ".join(verdict) + ".")
    else:
        print("  Subtraction produces non-negative figures, but check them "
              "against the neighbouring elections above before using them.")
    print("  Source Tema Central and Trobu from a constituency-level record "
          "instead - see scrape_2016_peacefm.py.")


if __name__ == "__main__":
    main()
