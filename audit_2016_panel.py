#!/usr/bin/env python3
"""
Audit the 2016 presidential rows in the panel, and audit the polling-station
file they were built from.

WHY THIS GOES DEEPER THAN THE 2024 AUDIT
----------------------------------------
The 2024 rows came from a web scrape, so the failure mode was one row
inheriting another row's figures.  The 2016 rows came from a polling-station
file (results2016_ps_pres.csv, 28,418 stations), so the failure mode is
different: individual cells were mis-keyed during transcription, and the
aggregation step then papered over the damage.

Specifically, the panel's other_votes for 2016 was computed as a RESIDUAL,

    other_votes = total_valid_votes - npp_votes - ndc_votes

which makes the accounting identity hold by construction.  Running
audit_vote_identity.py on 2016 therefore returns a clean bill of health that
means nothing - the test cannot fail no matter how bad the inputs are.  Ten
constituencies ended up with NEGATIVE other_votes, which is the residual
admitting that the declared valid total is smaller than the two major parties'
votes alone.

This script tests the things that can actually fail:

  1.  Panel-level checks that still apply: duplicate vote quadruples,
      turnout bounds, 2012 -> 2016 ratio outliers, negative components.
  2.  registered_voters copied from a different election year.
  3.  The polling-station file's own internal consistency, at station level:
        a.  do the seven candidate columns sum to the declared TotalValid?
        b.  does TotalValid + Rejected equal TotalIssued?
        c.  is TotalIssued at most RegisteredVoters?
  4.  For every station where (a) fails, which side is corrupt.  The
      tie-breaker is TotalIssued: whichever of the two candidate-sum or
      declared-valid figures satisfies "valid + rejected = issued" is the one
      that survived transcription.  This resolves the large majority of
      disagreements to a specific corrupt cell.

USAGE
    python audit_2016_panel.py
"""

import pandas as pd

from canon_lib import canon

PANEL = "presidential_panel.csv"
PS_FILE = "results2016_ps_pres.csv"

YEAR = 2016
PREV_YEAR = 2012

# The polling-station file is not UTF-8; it carries Windows-1252 punctuation
# in some station names and fails to decode as UTF-8 at byte 0x83.
PS_ENCODING = "latin-1"

CANDIDATES = ["CPP", "NDP", "NDC", "PPP", "NPP", "PNC", "IND"]
MAJOR = ["NPP", "NDC"]
MINOR = [c for c in CANDIDATES if c not in MAJOR]

NUMERIC = CANDIDATES + [
    "RegisteredVoters", "TotalValid", "Rejected", "TotalIssued",
]

OUT_PANEL = "audit_2016_panel_rows.csv"
OUT_STATIONS = "audit_2016_station_defects.csv"
OUT_CONSTIT = "audit_2016_constituency_deltas.csv"


def load_stations():
    """Read the polling-station file and add the derived columns."""
    ps = pd.read_csv(PS_FILE, encoding=PS_ENCODING, low_memory=False)
    for col in NUMERIC:
        ps[col] = pd.to_numeric(ps[col], errors="coerce")
    ps["key"] = ps.Constituency.map(canon)
    ps["cand_sum"] = ps[CANDIDATES].sum(axis=1)
    ps["minor_sum"] = ps[MINOR].sum(axis=1)
    # Which side of the disagreement is consistent with the issued total.
    ps["cand_ties"] = (ps.cand_sum + ps.Rejected) == ps.TotalIssued
    ps["decl_ties"] = (ps.TotalValid + ps.Rejected) == ps.TotalIssued
    return ps


def audit_stations(ps):
    """Station-level transcription defects, classified by which cell is bad."""
    disagree = ps.cand_sum != ps.TotalValid

    def verdict(row):
        if row.cand_sum == row.TotalValid:
            return "ok"
        if row.cand_ties and not row.decl_ties:
            return "declared_valid_is_corrupt"
        if row.decl_ties and not row.cand_ties:
            return "candidate_cell_is_corrupt"
        return "unresolved"

    ps = ps.copy()
    ps["verdict"] = ps.apply(verdict, axis=1)
    ps["valid_gap"] = ps.cand_sum - ps.TotalValid

    print("POLLING-STATION FILE")
    print(f"  stations                       : {len(ps):,}")
    print(f"  constituencies                 : {ps.key.nunique()}")
    print(f"  candidate sum == declared valid: "
          f"{int((~disagree).sum()):,}")
    print(f"  disagree                       : {int(disagree.sum()):,}")
    counts = ps[disagree].verdict.value_counts().to_dict()
    for k in ("declared_valid_is_corrupt", "candidate_cell_is_corrupt",
              "unresolved"):
        n = counts.get(k, 0)
        votes = int(ps.loc[disagree & (ps.verdict == k), "valid_gap"].sum())
        print(f"    {k:<28}: {n:>5,} stations, net {votes:+,} votes")

    print(f"  valid + rejected != issued     : "
          f"{int(((ps.TotalValid + ps.Rejected) != ps.TotalIssued).sum()):,}")
    over = ps[(ps.cand_sum + ps.Rejected) > ps.RegisteredVoters]
    print(f"  votes cast exceed the register : {len(over):,} stations, "
          f"{int((over.cand_sum + over.Rejected - over.RegisteredVoters).sum()):,}"
          " excess votes")
    print("    (special-voting stations legitimately draw on a separate "
          "register, so some of these are structural, not errors)")

    bad = ps[disagree].copy()
    cols = ["Region", "Constituency", "Psname", "RegisteredVoters"] \
        + CANDIDATES + ["cand_sum", "TotalValid", "Rejected", "TotalIssued",
                        "valid_gap", "verdict"]
    bad.reindex(bad.valid_gap.abs().sort_values(ascending=False).index)[
        cols].to_csv(OUT_STATIONS, index=False)
    print(f"  wrote {OUT_STATIONS}")
    return ps


def audit_constituency_deltas(ps):
    """How much each constituency's total moves under each basis."""
    g = ps.groupby("key").agg(
        constituency=("Constituency", "first"),
        stations=("Psname", "size"),
        npp=("NPP", "sum"),
        ndc=("NDC", "sum"),
        minor=("minor_sum", "sum"),
        declared_valid=("TotalValid", "sum"),
        rejected=("Rejected", "sum"),
        issued=("TotalIssued", "sum"),
        registered=("RegisteredVoters", "sum"),
    )
    g["candidate_valid"] = g.npp + g.ndc + g.minor
    g["delta"] = g.candidate_valid - g.declared_valid
    g["delta_pct"] = 100 * g.delta / g.declared_valid

    g.sort_values("delta_pct").to_csv(OUT_CONSTIT)

    moved = g[g.delta != 0]
    print("\nCONSTITUENCY TOTALS: candidate-sum basis vs declared-valid basis")
    print(f"  constituencies that move       : {len(moved)} of {len(g)}")
    print(f"  national candidate-sum valid   : {int(g.candidate_valid.sum()):,}")
    print(f"  national declared valid        : {int(g.declared_valid.sum()):,}")
    print(f"  net difference                 : {int(g.delta.sum()):+,} "
          f"({100 * g.delta.sum() / g.declared_valid.sum():+.3f}%)")
    big = g[g.delta_pct.abs() > 0.5]
    if len(big):
        print(f"\n  constituencies moving more than 0.5%:")
        print(big.sort_values("delta_pct")[
            ["constituency", "npp", "ndc", "minor",
             "candidate_valid", "declared_valid", "delta", "delta_pct"]
        ].to_string(index=False))
    print(f"  wrote {OUT_CONSTIT}")
    return g


def audit_panel(panel, stations_by_key):
    """The checks that apply to the panel rows as they stand today."""
    p = panel.copy()
    p["key"] = p.constituency.map(canon)
    p = p[(p.race == "presidential") & (p["round"] == 1)]
    y = p[p.election_year == YEAR].copy()
    prev = p[p.election_year == PREV_YEAR].set_index("key")

    print("\nPANEL ROWS, 2016")
    print(f"  rows with figures              : {int(y.total_valid_votes.notna().sum())}")

    flags = []

    # 1. Negative components.  Only possible if a column was plugged.
    neg = y[(y.other_votes < 0) | (y.npp_votes < 0) | (y.ndc_votes < 0)]
    for _, r in neg.iterrows():
        flags.append({"constituency": r.constituency, "check": "negative_component",
                      "detail": f"other_votes = {r.other_votes:.0f}"})
    print(f"  negative vote components       : {len(neg)}")

    # 2. other_votes is a residual rather than a real minor-party sum.
    y["true_minor"] = y.key.map(stations_by_key.minor)
    plugged = y[y.true_minor.notna() & (y.true_minor != y.other_votes)]
    for _, r in plugged.iterrows():
        flags.append({
            "constituency": r.constituency, "check": "other_votes_is_a_plug",
            "detail": f"panel {r.other_votes:.0f} vs actual minor-party sum "
                      f"{r.true_minor:.0f}"})
    print(f"  other_votes != minor-party sum : {len(plugged)}")

    # 3. Duplicated vote quadruples - the 2024 failure mode.
    quad = ["npp_votes", "ndc_votes", "other_votes", "total_valid_votes"]
    dup = y[y.duplicated(quad, keep=False) & y.total_valid_votes.notna()]
    for _, r in dup.iterrows():
        flags.append({"constituency": r.constituency, "check": "duplicate_quadruple",
                      "detail": ""})
    print(f"  duplicate vote quadruples      : {len(dup)}")

    # 4. Turnout.  Uses the panel's own register column.
    y["turnout"] = y.total_valid_votes / y.registered_voters
    odd = y[y.turnout.notna() & ((y.turnout < 0.40) | (y.turnout > 0.95))]
    for _, r in odd.iterrows():
        flags.append({"constituency": r.constituency, "check": "turnout_out_of_band",
                      "detail": f"{100 * r.turnout:.1f}%"})
    print(f"  turnout outside 40-95%         : {len(odd)}")

    # 5. Year-on-year movement against 2012 on the same boundaries.
    y["prev_valid"] = y.key.map(prev.total_valid_votes)
    y["ratio"] = y.total_valid_votes / y.prev_valid
    swing = y[y.ratio.notna() & ((y.ratio < 0.80) | (y.ratio > 1.30))]
    for _, r in swing.iterrows():
        flags.append({"constituency": r.constituency, "check": "ratio_vs_2012",
                      "detail": f"{r.ratio:.3f}"})
    print(f"  2016/2012 ratio outside 0.8-1.3: {len(swing)}")

    # 6. Register values copied from another election year.  A constituency's
    #    register grows every cycle, so an exact repeat is a copy, not a fact.
    reg = p.pivot_table(index="key", columns="election_year",
                        values="registered_voters", aggfunc="first")
    copied = []
    for other in [c for c in reg.columns if c != YEAR]:
        if YEAR not in reg.columns:
            break
        both = reg[[YEAR, other]].dropna()
        hit = both[both[YEAR] == both[other]]
        for k in hit.index:
            name = y.loc[y.key == k, "constituency"]
            copied.append({"constituency": name.iloc[0] if len(name) else k,
                           "check": "register_copied_from_another_year",
                           "detail": f"{YEAR} register {both.loc[k, YEAR]:.0f} "
                                     f"is identical to {other}"})
    flags.extend(copied)
    print(f"  register copied from other year: {len(copied)}")
    for c in copied:
        print(f"    {c['constituency']:<26} {c['detail']}")

    pd.DataFrame(flags).to_csv(OUT_PANEL, index=False)
    print(f"\n  {len(flags)} flags written to {OUT_PANEL}")


def main():
    ps = load_stations()
    ps = audit_stations(ps)
    g = audit_constituency_deltas(ps)
    audit_panel(pd.read_csv(PANEL), g)


if __name__ == "__main__":
    main()
