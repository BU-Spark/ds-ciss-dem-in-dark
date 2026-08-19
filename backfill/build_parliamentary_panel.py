#!/usr/bin/env python3
"""
Build the parliamentary constituency-level panel (1996-2016) from CLEA.

Input:
    clea_ghana.csv        - Ghana rows extracted from the CLEA Lower Chamber
                            Elections Archive (candidate-level, one row per
                            candidate per constituency per election).
    canon_lib.py          - project-wide constituency name canonicalizer
                            (the confirmed crosswalk).
    Scraped_Constituency_elections_Panel_19962024_LABELED.xlsx
                          - used only as a lookup for region names and
                            GHCON constituency ids.
    manual_presidential_clean_LABELED.csv
                          - used only to define the official seat list of
                            each boundary vintage (its per-year constituency
                            lists are complete: 200 / 230 / 275 seats).

Output:
    parliamentary_panel_1996_2016.csv - one row per constituency per election,
        same schema as presidential_panel.csv (race='parliamentary', round=1;
        Ghana parliamentary elections have no run-off).

Notes:
  * CLEA candidate votes (cv1) use negative sentinel codes for missing
    (-990 etc.); they are treated as NaN.
  * npp/ndc votes = sum of that party's candidate votes in the constituency
    (a party normally fields one candidate, but the sum is robust to
    data-entry duplicates).
  * total_valid_votes = CLEA's constituency-level vv1 when present,
    otherwise the sum of candidate votes.
  * CLEA contains a handful of corrupted constituency names in 1996
    ('M', 'WA', 'WUOGON', 'DENKYIRA'); rows that cannot be matched to the
    official seat list are written to parliamentary_unmatched.csv instead
    of the panel.
"""

import pandas as pd
from canon_lib import canon

CLEA = "clea_ghana.csv"
SCRAPED = "Scraped_Constituency_elections_Panel_19962024_LABELED.xlsx"
MANUAL = "manual_presidential_clean_LABELED.csv"
OUT = "parliamentary_panel_1996_2016.csv"
OUT_UNMATCHED = "parliamentary_unmatched.csv"

NPP = {"new patriotic party", "npp"}
NDC = {"national democratic congress", "ndc"}


def main():
    d = pd.read_csv(CLEA, low_memory=False)
    d = d[d.yr >= 1996].copy()          # 1992 is too sparse in CLEA (boycott year)
    d["cname"] = d.cst_n.map(canon)
    d["party"] = d.pty_n.astype(str).str.lower().str.strip()
    d["cv"] = d.cv1.where(d.cv1 >= 0)   # negative sentinels -> NaN

    # --- official seat list per boundary vintage (from the manual file) ---
    mn = pd.read_csv(MANUAL)
    mn["cname"] = mn.constituency_name.map(canon)
    ref96 = set(mn[mn.election_year == 1996].cname)   # 200-seat vintage (1992-2000)
    ref08 = set(mn[mn.election_year == 2008].cname)   # 230-seat vintage (2004-2008)
    ref12 = set(mn[mn.election_year == 2012].cname)   # 275-seat vintage (2012-2016)
    REF = {1996: ref96, 2000: ref96, 2004: ref08, 2008: ref08, 2012: ref12, 2016: ref12}

    # --- region / constituency-id lookups from the scraped panel ---
    lk = pd.read_excel(SCRAPED)
    lk["cname"] = lk.constituency_name.map(canon)
    lk = lk.sort_values("election_year")
    region16 = lk.groupby("cname").region_name_16regions.last().to_dict()
    region10 = lk.groupby("cname").region_name_10regions.last().to_dict()
    conid = lk.groupby("cname").constituency_id.last().to_dict()
    region10.update({k: v for k, v in
                     mn.groupby("cname").region_10.last().to_dict().items()
                     if k not in region10})

    rows, unmatched = [], []
    for (y, cname), g in d.groupby(["yr", "cname"]):
        if cname not in REF[y]:
            unmatched.append({"election_year": y, "clea_name": g.cst_n.iloc[0],
                              "canonical": cname,
                              "reason": "not in official seat list (corrupt CLEA name)"})
            continue
        npp = g.loc[g.party.isin(NPP), "cv"].sum(min_count=1)
        ndc = g.loc[g.party.isin(NDC), "cv"].sum(min_count=1)
        vv = g.vv1.max()
        valid = vv if pd.notna(vv) and vv > 0 else g.cv.sum(min_count=1)
        other = (valid - (npp or 0) - (ndc or 0)) if pd.notna(valid) else None
        winner = g.loc[g.seat == 1, "pty_n"].str.lower().iloc[0] if (g.seat == 1).any() else None
        rows.append({
            "election_year": y, "race": "parliamentary", "round": 1,
            "constituency": cname,
            "region_10": region10.get(cname), "region_16": region16.get(cname),
            "constituency_id": conid.get(cname),
            "npp_votes": npp, "ndc_votes": ndc, "other_votes": other,
            "total_valid_votes": valid,
            "winner_party": winner,
            "n_candidates": len(g),
            "source": "clea_r18",
        })

    panel = pd.DataFrame(rows).sort_values(["election_year", "constituency"])
    panel.to_csv(OUT, index=False)
    pd.DataFrame(unmatched).to_csv(OUT_UNMATCHED, index=False)

    print(f"wrote {OUT}: {len(panel)} rows")
    print("coverage per year (vs official seat count):")
    for y, ref in REF.items():
        n = (panel.election_year == y).sum()
        print(f"  {y}: {n}/{len(ref)}")
    print(f"unmatched CLEA rows: {len(unmatched)} -> {OUT_UNMATCHED}")


if __name__ == "__main__":
    main()
