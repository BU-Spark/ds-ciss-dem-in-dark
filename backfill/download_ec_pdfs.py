#!/usr/bin/env python3
"""
Download the official EC result PDFs for the 53 missing 2024 presidential
constituencies.

Source: Electoral Commission of Ghana, "2024 Results by Constituencies"
    https://ec.gov.gh/2024-presidential-election-results/
Each region page links one summary PDF per constituency, named like:
    https://ec.gov.gh/wp-content/uploads/2025/01/F0401_ADANSI-ASOKWA_PRES.pdf

What this script does
---------------------
1. Reads presidential_gaps.csv and takes the 2024 targets (53 seats).
2. Fetches the region pages that contain them (Ashanti, Greater Accra,
   Eastern, Bono East, Upper East, North East).
3. Collects every *_PRES.pdf link, fuzzy-matches the constituency part of
   the filename against the targets.
4. Downloads the matched PDFs into ec_pdfs_2024/ and writes a match report.

Afterwards: zip the ec_pdfs_2024/ folder and upload it — the PDFs are
scanned images, so the number extraction (OCR) happens on the other side.

Usage
-----
    pip install requests beautifulsoup4 pandas
    python download_ec_pdfs.py
"""

import difflib
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

REGION_PAGES = [
    "https://ec.gov.gh/ashanti-region/",
    "https://ec.gov.gh/greater-accra-region/",
    "https://ec.gov.gh/eastern-region/",
    "https://ec.gov.gh/bono-east-region/",
    "https://ec.gov.gh/upper-east-region/",
    "https://ec.gov.gh/north-east-region/",
]
HEADERS = {"User-Agent": "Mozilla/5.0 (academic research; election data backfill)"}
OUT_DIR = Path("ec_pdfs_2024")
OUT_DIR.mkdir(exist_ok=True)
POLITE_DELAY = 1.0


def normalize(s):
    s = str(s).upper().strip()
    s = re.sub(r"[\/\-\.\,\(\)\'_]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    gaps = pd.read_csv("presidential_gaps.csv")
    targets = {normalize(c): c for c in gaps[gaps.election_year == 2024].constituency}
    print(f"targets: {len(targets)} constituencies")

    # 1) collect all *_PRES.pdf links from the region pages
    pdf_links = {}  # normalized constituency name (from filename) -> url
    for page in REGION_PAGES:
        try:
            r = requests.get(page, headers=HEADERS, timeout=60)
            r.raise_for_status()
        except Exception as e:
            print(f"  region page failed: {page} -> {e}")
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        n = 0
        for a in soup.find_all("a", href=True):
            href = a["href"]
            m = re.search(r"/([A-Z]\d+)_([^/]+?)_PRES\.pdf$", href, flags=re.I)
            if m:
                name = normalize(m.group(2))
                pdf_links[name] = href if href.startswith("http") else "https://ec.gov.gh" + href
                n += 1
        print(f"  {page} -> {n} PRES pdf links")
        time.sleep(POLITE_DELAY)
    print(f"total unique PRES pdfs found: {len(pdf_links)}")

    # 2) fuzzy-match targets to pdf filenames and download
    report, missing = [], []
    names = list(pdf_links)
    for tnorm, toriginal in targets.items():
        match = difflib.get_close_matches(tnorm, names, n=1, cutoff=0.6)
        if not match:
            missing.append({"constituency": toriginal, "reason": "no PDF link matched"})
            print(f"  MISS  {toriginal}")
            continue
        url = pdf_links[match[0]]
        fname = OUT_DIR / url.rsplit("/", 1)[-1]
        try:
            r = requests.get(url, headers=HEADERS, timeout=120)
            r.raise_for_status()
            fname.write_bytes(r.content)
            report.append({"constituency": toriginal, "matched_pdf_name": match[0],
                           "source_url": url, "file": fname.name,
                           "bytes": len(r.content)})
            print(f"  OK    {toriginal}  <- {fname.name}")
        except Exception as e:
            missing.append({"constituency": toriginal, "reason": f"download failed: {e}"})
            print(f"  FAIL  {toriginal} -> {e}")
        time.sleep(POLITE_DELAY)

    pd.DataFrame(report).to_csv("ec_pdf_download_report.csv", index=False)
    pd.DataFrame(missing).to_csv("ec_pdf_missing.csv", index=False)
    print(f"\ndownloaded {len(report)} PDFs into {OUT_DIR}/")
    print(f"missing/failed: {len(missing)} (see ec_pdf_missing.csv)")
    print("Next: zip the ec_pdfs_2024 folder together with ec_pdf_download_report.csv "
          "and upload it for OCR extraction.")


if __name__ == "__main__":
    main()
