#!/usr/bin/env python3
"""
Fetch the 2016 presidential results for Tema Central and Trobu from Peace FM's
constituency pages.

WHY THIS RUNS ON YOUR MACHINE AND NOT IN THE SESSION
----------------------------------------------------
ghanaelections.peacefmonline.com does not resolve from the analysis sandbox,
and the sandbox is not permitted to fall back to curl or a raw HTTP client, so
the fetch has to happen somewhere with ordinary internet access.  Everything
after the fetch - parsing, checking, merging - runs back in the session on the
files this script writes.

WHY THESE TWO SEATS NEED A SOURCE OF THEIR OWN
----------------------------------------------
The 2016 polling-station file covers 271 of the 275 constituencies.  Bibiani
Anhwiaso Bekwai and Suaman were filled from an earlier scrape; Tema Central and
Trobu were never filled at all.

Two shortcuts were tried and rejected.  CLEA has both seats, but its Ghana 2016
series is PARLIAMENTARY - it reproduces the parliamentary panel exactly across
all 275 constituencies while disagreeing with the presidential panel on all 273
overlapping ones, so its Tema Central 41,480 and Trobu 76,084 are the wrong
race.  Subtracting the 273 known constituencies from the published national
total was also tried; it assigns the two seats a NEGATIVE minor-party count and
an NDC vote less than half what they returned in 2012 and 2020, because the
published national aggregates do not balance against the polling-station file.
See reconcile_2016_national.py for that arithmetic in full.

WHAT THIS WRITES
----------------
    raw_html_2016/<slug>.html   a snapshot of every page fetched, so parsing
                                can be redone or corrected later without
                                touching the site again
    backfill_2016_pres.csv      the parsed figures
    backfill_2016_misses.csv    anything that could not be found or parsed

If a row comes out PARSE-INCOMPLETE, do not re-run the scrape.  Zip
raw_html_2016/ and send it back; the parser can be fixed against the saved
pages.

THE 2016 BALLOT
---------------
Seven candidates stood.  The panel only needs NPP, NDC and the sum of the rest,
but all seven are parsed so the figures can be checked against each other.

    NPP   Nana Addo Dankwa Akufo-Addo
    NDC   John Dramani Mahama
    CPP   Ivor Kobina Greenstreet
    PPP   Papa Kwesi Nduom
    NDP   Nana Konadu Agyeman-Rawlings
    PNC   Edward Nasigri Mahama
    IND   Jacob Osei Yeboah

USAGE
    pip install requests beautifulsoup4 pandas
    python scrape_2016_peacefm.py

    # if the automatic URL guesses all miss, open the region page in a browser,
    # copy the two constituency links, and pass them directly:
    python scrape_2016_peacefm.py --url "TEMA CENTRAL=https://..." \
                                  --url "TROBU=https://..."

Etiquette: one request per second, and well under a hundred requests in total.
"""

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://ghanaelections.peacefmonline.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (academic research; constituency election panel)"
}
POLITE_DELAY = 1.0
RAW_DIR = Path("raw_html_2016")

YEAR = 2016

# The two seats to fetch, with the region slug their pages sit under.
TARGETS = [
    {"constituency": "TEMA CENTRAL", "region": "greateraccra"},
    {"constituency": "TROBU", "region": "greateraccra"},
]

# Party labels as they appear on the page, mapped to the panel's columns.
PARTIES = ["NPP", "NDC", "CPP", "PPP", "NDP", "PNC", "IND"]
MAJOR = ["NPP", "NDC"]


def slug_variants(name):
    """Peace FM's slugs are inconsistent, so try the plausible spellings.

    Observed in the wild: '/greateraccra/trobu/'.  Multi-word constituencies
    appear variously joined, hyphenated or underscored, so all three are tried
    before giving up and falling back to the region index.
    """
    low = name.lower()
    words = re.split(r"[^a-z0-9]+", low)
    words = [w for w in words if w]
    return list(dict.fromkeys([
        "".join(words),
        "_".join(words),
        "-".join(words),
    ]))


def get(url):
    """GET a URL, snapshot the HTML to disk, return the text."""
    RAW_DIR.mkdir(exist_ok=True)
    resp = requests.get(url, headers=HEADERS, timeout=60)
    fname = re.sub(r"[^A-Za-z0-9]+", "_", url.replace(BASE, "")).strip("_")
    (RAW_DIR / (fname[:120] + ".html")).write_text(
        resp.text, encoding="utf-8", errors="replace")
    time.sleep(POLITE_DELAY)
    resp.raise_for_status()
    return resp.text


def candidate_urls(target):
    """Every URL worth trying for one constituency, best guess first."""
    return [f"{BASE}/pages/{YEAR}/president/{target['region']}/{s}/"
            for s in slug_variants(target["constituency"])]


def find_via_region_index(target):
    """Fall back to reading the region page and matching the link text."""
    index = f"{BASE}/pages/{YEAR}/president/{target['region']}/"
    try:
        html = get(index)
    except Exception as exc:
        print(f"    region index unreachable: {exc}")
        return None
    soup = BeautifulSoup(html, "html.parser")
    want = re.sub(r"[^a-z0-9]", "", target["constituency"].lower())
    for a in soup.find_all("a", href=True):
        text = re.sub(r"[^a-z0-9]", "", a.get_text(" ").lower())
        href = a["href"]
        if not text or want not in text:
            continue
        return href if href.startswith("http") else BASE + href
    return None


def parse_results(html, url):
    """Pull the seven party vote counts out of a constituency page.

    The pages render a results table with a party label and a vote figure on
    each row.  Rather than depend on the exact table markup, which changes
    between seasons of the site, this walks the text and takes the first vote
    figure that follows each party label.  Percentages are ignored: a figure is
    only accepted if it has no decimal point and is not immediately followed by
    a percent sign.
    """
    text = BeautifulSoup(html, "html.parser").get_text("\n")

    votes = {}
    for party in PARTIES:
        pattern = rf"\b{party}\b[^\d%]{{0,120}}?([\d][\d,]{{2,}})(?!\s*%)"
        match = re.search(pattern, text, flags=re.I)
        if match:
            votes[party] = int(match.group(1).replace(",", ""))

    declared = None
    m = re.search(r"(?:TOTAL\s+)?VALID\s+VOTES?[^\d]{0,60}([\d][\d,]{2,})",
                  text, flags=re.I)
    if m:
        declared = int(m.group(1).replace(",", ""))

    rejected = None
    m = re.search(r"(?:TOTAL\s+)?REJECT(?:ED)?\s+(?:BALLOTS?|VOTES?)"
                  r"[^\d]{0,60}([\d][\d,]{0,})", text, flags=re.I)
    if m:
        rejected = int(m.group(1).replace(",", ""))

    registered = None
    m = re.search(r"REGISTERED\s+VOTERS?[^\d]{0,60}([\d][\d,]{2,})",
                  text, flags=re.I)
    if m:
        registered = int(m.group(1).replace(",", ""))

    have_all = all(p in votes for p in PARTIES)
    npp, ndc = votes.get("NPP"), votes.get("NDC")
    other = (sum(v for p, v in votes.items() if p not in MAJOR)
             if have_all else None)

    row = {
        "election_year": YEAR,
        "race": "presidential",
        "round": 1,
        "npp_votes": npp,
        "ndc_votes": ndc,
        "other_votes": other,
        "total_valid_votes": (npp + ndc + other)
        if None not in (npp, ndc, other) else None,
        "declared_valid_on_page": declared,
        "rejected_ballots": rejected,
        "registered_voters": registered,
        "parties_parsed": len(votes),
        "source": "peacefm_2016",
        "source_url": url,
    }
    for party in PARTIES:
        row["raw_" + party] = votes.get(party)
    return row


def fetch_one(target, override=None):
    """Try the guessed URLs, then the region index, and parse whatever loads."""
    urls = [override] if override else candidate_urls(target)
    for url in urls:
        try:
            html = get(url)
        except Exception as exc:
            print(f"    {url} -> {exc}")
            continue
        row = parse_results(html, url)
        if row["npp_votes"] and row["ndc_votes"]:
            return row, None
        print(f"    {url} loaded but parsed only "
              f"{row['parties_parsed']} of {len(PARTIES)} parties")
        return row, "parse incomplete"

    if override:
        return None, "supplied URL did not load"

    print("    falling back to the region index")
    found = find_via_region_index(target)
    if not found:
        return None, "no URL guess worked and the region index had no match"
    try:
        html = get(found)
    except Exception as exc:
        return None, f"index match {found} failed to load: {exc}"
    row = parse_results(html, found)
    return row, (None if row["npp_votes"] and row["ndc_votes"]
                 else "parse incomplete")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", action="append", default=[],
                    help='override a lookup, as "CONSTITUENCY=https://..."')
    args = ap.parse_args()

    overrides = {}
    for item in args.url:
        name, _, url = item.partition("=")
        overrides[name.strip().upper()] = url.strip()

    rows, misses = [], []
    for target in TARGETS:
        name = target["constituency"]
        print(f"{name}")
        row, problem = fetch_one(target, overrides.get(name))
        if row is None:
            misses.append({"constituency": name, "reason": problem})
            print(f"  MISS: {problem}")
            continue
        row["constituency"] = name
        rows.append(row)
        if problem:
            misses.append({"constituency": name, "reason": problem})
            print(f"  PARSE-INCOMPLETE")
        else:
            check = ""
            if row["declared_valid_on_page"] not in (None,
                                                     row["total_valid_votes"]):
                check = (f"  [page's own valid total says "
                         f"{row['declared_valid_on_page']:,}, candidates sum to "
                         f"{row['total_valid_votes']:,}]")
            print(f"  OK  NPP {row['npp_votes']:,}  NDC {row['ndc_votes']:,}  "
                  f"valid {row['total_valid_votes']:,}{check}")

    pd.DataFrame(rows).to_csv("backfill_2016_pres.csv", index=False)
    pd.DataFrame(misses).to_csv("backfill_2016_misses.csv", index=False)

    ok = sum(1 for r in rows if r["npp_votes"] and r["ndc_votes"])
    print(f"\nwrote backfill_2016_pres.csv ({len(rows)} rows, {ok} fully parsed)")
    print(f"wrote backfill_2016_misses.csv ({len(misses)} entries)")
    print(f"raw pages saved under {RAW_DIR}/")
    if ok < len(TARGETS):
        print("\nSend back backfill_2016_pres.csv AND a zip of "
              f"{RAW_DIR}/ - the parser can be corrected against the saved "
              "pages without hitting the site again.")


if __name__ == "__main__":
    main()
