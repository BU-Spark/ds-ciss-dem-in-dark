#!/usr/bin/env python3
"""
Download the still-missing 2024 EC presidential summary sheets (Form Ten).

WHY THIS SCRIPT REPLACES download_ec_pdfs.py
--------------------------------------------
The first version of the downloader missed constituencies for two reasons,
both of which are fixed here.

1. FILENAME PATTERN.  The old script looked for links matching `_PRES.pdf`
   (underscore).  That is the Ashanti / Bono East / Eastern convention, but
   the Greater Accra page uses a HYPHEN instead:

       Ashanti        F2901_AFIGYA-KWABRE-NORTH_PRES.pdf   <- underscore
       Greater Accra  C1501_OKAIKWEI-CENTRAL-PRES.pdf      <- hyphen

   So every Greater Accra sheet was invisible to the old regex, and
   OKAIKWEI CENTRAL - which really is published - was never fetched.
   A handful of files carry no PRES marker at all (for example
   C1101_AYAWASO-WEST-WUOGON.pdf); those are reported separately for a
   human to confirm rather than silently downloaded.

2. FUZZY NAME MATCHING.  The old script used difflib to pick the "closest"
   filename for each wanted constituency.  When the wanted sheet did not
   exist, difflib happily returned a NEIGHBOURING constituency instead, so
   the download set contained TECHIMAN NORTH in place of TECHIMAN SOUTH,
   ABUAKWA NORTH in place of a missing Eastern seat, and so on.  Those four
   files were downloaded, never matched to a panel row, and quietly wasted.
   This version does EXACT canonical-name matching only.  A wanted
   constituency with no exact match is reported as NOT FOUND - never
   substituted.

WHAT IT DOES
------------
* Reads ec_2024_targets.csv (falling back to the 2024 rows of
  presidential_gaps.csv) and takes its `constituency` column as the wanted
  list. The target file holds both the seats missing from the panel and the
  seats flagged by the 2024 audit as having corrupted scraped figures.
* Scrapes the EC regional result pages for every PDF link.
* Canonicalises each filename's constituency part with canon_lib.canon and
  matches it exactly against the wanted list.
* Downloads only exact matches into ec_pdfs_2024_v2/.
* Prints a three-way report: downloaded / not published / ambiguous.

WHERE TO RUN IT
---------------
Run this on your own machine, not in the chat sandbox: ec.gov.gh refuses
connections from the sandbox's proxy (403).  Then send the downloaded PDFs
back and they will be read and merged into the panel.

    pip install requests beautifulsoup4 pandas
    python download_ec_pdfs_2024_v2.py

EXPECTED RESULT AS OF THIS WRITING
----------------------------------
ec_2024_targets.csv holds 39 constituencies (35 flagged by audit_2024_scrape.py
plus 4 with no 2024 row at all), so expect roughly:

    downloaded      ~36 sheets, including OKAIKWEI CENTRAL
    not published   ABLEKUMA NORTH, AHAFO ANO NORTH, TECHIMAN SOUTH

Those three show a placeholder image on the EC site instead of a PDF; all
three are among the nine constituencies whose December 2024 collation was
disputed and re-run.  The script is written so that if the EC uploads them
later, re-running it picks them up with no code change.
"""

import os
import re
import sys
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canon_lib import canon  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
GAPS = os.path.join(BASE, "presidential_gaps.csv")
TARGETS = os.path.join(BASE, "ec_2024_targets.csv")
OUTDIR = os.path.join(BASE, "ec_pdfs_2024_v2")
YEAR = 2024

# All sixteen regional result pages. Pages that 404 are skipped with a
# warning rather than aborting the run, so a slug change on the EC site
# degrades gracefully instead of losing the whole download.
REGION_PAGES = [
    "https://ec.gov.gh/ashanti-region/",
    "https://ec.gov.gh/greater-accra-region/",
    "https://ec.gov.gh/eastern-region/",
    "https://ec.gov.gh/central-region/",
    "https://ec.gov.gh/western-region/",
    "https://ec.gov.gh/western-north-region/",
    "https://ec.gov.gh/volta-region/",
    "https://ec.gov.gh/oti-region/",
    "https://ec.gov.gh/northern-region/",
    "https://ec.gov.gh/north-east-region/",
    "https://ec.gov.gh/savannah-region/",
    "https://ec.gov.gh/upper-east-region/",
    "https://ec.gov.gh/upper-west-region/",
    "https://ec.gov.gh/bono-region/",
    "https://ec.gov.gh/bono-east-region/",
    "https://ec.gov.gh/ahafo-region/",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (academic research; election data backfill)"}

# Filenames look like  <CODE>_<CONSTITUENCY>[-_]PRES.pdf  where CODE is a
# region letter plus four or five digits. Both separators are accepted.
PRES_RE = re.compile(
    r"/(?P<code>[A-Z]\d{4,5})[_-](?P<name>.+?)[_-]PRES\.pdf$", re.IGNORECASE)
# Same shape but with no PRES / PARL marker - cannot be classified safely.
BARE_RE = re.compile(
    r"/(?P<code>[A-Z]\d{4,5})[_-](?P<name>.+?)\.pdf$", re.IGNORECASE)


def wanted_constituencies():
    """Canonical names of every 2024 seat whose EC sheet is needed.

    Two kinds of seat are in scope: the ones missing from the panel entirely,
    and the ones the 2024 audit flagged because their scraped figures are a
    verbatim copy of another constituency's. Both need the same thing - the
    official Form Ten - so they share one download list.
    """
    if os.path.exists(TARGETS):
        rows = pd.read_csv(TARGETS)
    else:
        gaps = pd.read_csv(GAPS)
        rows = gaps[gaps.election_year == YEAR]
    return {canon(c): c for c in rows.constituency}


def harvest(url):
    """Return (pres_links, bare_links) found on one regional page."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
    except Exception as exc:                       # noqa: BLE001
        print(f"  ! could not read {url}: {exc}")
        return [], []

    soup = BeautifulSoup(r.text, "html.parser")
    pres, bare = [], []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.lower().endswith(".pdf"):
            continue
        m = PRES_RE.search(href)
        if m:
            pres.append((canon(m.group("name").replace("-", " ")), href))
            continue
        # Anything ending in PARL is a parliamentary sheet, not our target.
        if re.search(r"[_-]PARL\.pdf$", href, re.IGNORECASE):
            continue
        m = BARE_RE.search(href)
        if m:
            bare.append((canon(m.group("name").replace("-", " ")), href))
    return pres, bare


def main():
    want = wanted_constituencies()
    if not want:
        print("no 2024 gaps left in presidential_gaps.csv - nothing to do")
        return
    print(f"looking for {len(want)} missing 2024 sheets: "
          f"{', '.join(sorted(want.values()))}\n")

    found, ambiguous = {}, {}
    for url in REGION_PAGES:
        print(f"scanning {url}")
        pres, bare = harvest(url)
        print(f"  {len(pres)} presidential links, {len(bare)} unlabelled links")
        for name, href in pres:
            # Exact canonical match only. No nearest-neighbour fallback:
            # a near miss is how the previous run collected the wrong seats.
            if name in want and name not in found:
                found[name] = href
        for name, href in bare:
            if name in want and name not in found:
                ambiguous[name] = href
        time.sleep(1)          # be gentle with the EC's server

    os.makedirs(OUTDIR, exist_ok=True)
    print()
    for name, href in sorted(found.items()):
        dest = os.path.join(OUTDIR, os.path.basename(href))
        if os.path.exists(dest):
            print(f"already have  {name:<20} {os.path.basename(href)}")
            continue
        try:
            r = requests.get(href, headers=HEADERS, timeout=120)
            r.raise_for_status()
            with open(dest, "wb") as f:
                f.write(r.content)
            print(f"downloaded    {name:<20} {os.path.basename(href)} "
                  f"({len(r.content)/1024:,.0f} KB)")
        except Exception as exc:                   # noqa: BLE001
            print(f"FAILED        {name:<20} {href}: {exc}")
        time.sleep(1)

    missing = sorted(set(want) - set(found) - set(ambiguous))
    if ambiguous:
        print("\nambiguous - a PDF exists under this constituency's code but "
              "the filename carries no PRES/PARL marker, so it must be opened "
              "and checked by hand before use:")
        for name, href in sorted(ambiguous.items()):
            print(f"  {name:<20} {href}")
    if missing:
        print("\nnot published - the EC region page shows a placeholder image "
              "instead of a presidential PDF for these seats. They need a "
              "non-EC source; do NOT substitute a similarly named sheet:")
        for name in missing:
            print(f"  {name}")

    print(f"\nsummary: {len(found)} downloaded, {len(ambiguous)} ambiguous, "
          f"{len(missing)} not published")
    print(f"files are in {OUTDIR}")


if __name__ == "__main__":
    main()
