#!/usr/bin/env python3
"""
Download the 2020 presidential Form Ten summary sheets from the EC website.
v3 - handles two server quirks discovered on previous runs:
  1. The page embeds asset URLs pointing at a retired server IP
     (http://34.248.68.255/...). Any raw-IP host is rewritten to ec.gov.gh.
  2. The full-size originals were not migrated to the current server (404),
     but the WordPress *scaled* variants (-1152x1536 etc.) still exist.
     For each sheet we therefore try every known variant, LARGEST FIRST,
     and keep the first one that downloads.

Usage:
    pip install requests beautifulsoup4
    python download_ec_2020_sheets.py
Then zip the ec_sheets_2020/ folder and upload it.
"""

import re
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

PAGE = "https://ec.gov.gh/constituency-summary-sheet/"
HEADERS = {"User-Agent": "Mozilla/5.0 (academic research; election data backfill)"}
OUT = Path("ec_sheets_2020")
OUT.mkdir(exist_ok=True)
POLITE_DELAY = 0.4

ASSET_RE = re.compile(r"wp-content/uploads/.+\.(?:jpe?g|png|gif|webp|pdf)$", re.I)
SIZE_RE = re.compile(r"-(\d+)x(\d+)(\.\w+)$")


def collect_variants(html, base):
    """Return {base_filename: [urls sorted largest-first]}."""
    soup = BeautifulSoup(html, "html.parser")
    seen = set()

    def norm(u):
        if not u:
            return None
        u = u.split("?")[0].strip()
        if not ASSET_RE.search(u):
            return None
        full = urljoin(base, u)
        full = re.sub(r"^https?://\d{1,3}(?:\.\d{1,3}){3}", "https://ec.gov.gh", full)
        return full

    for img in soup.find_all("img"):
        for attr in ("src", "data-src", "data-lazy-src", "data-full-url", "data-orig-file"):
            u = norm(img.get(attr))
            if u:
                seen.add(u)
        for part in (img.get("srcset") or img.get("data-srcset") or "").split(","):
            u = norm(part.strip().split(" ")[0])
            if u:
                seen.add(u)
    for a in soup.find_all("a", href=True):
        u = norm(a["href"])
        if u:
            seen.add(u)

    groups = defaultdict(list)
    for u in seen:
        basename = SIZE_RE.sub(r"\3", u.rsplit("/", 1)[-1])
        groups[basename].append(u)

    def area(u):
        m = SIZE_RE.search(u)
        return int(m.group(1)) * int(m.group(2)) if m else 10**9  # unsized first
    for k in groups:
        groups[k].sort(key=area, reverse=True)
    return groups


def main():
    r = requests.get(PAGE, headers=HEADERS, timeout=90)
    r.raise_for_status()
    groups = collect_variants(r.text, PAGE)
    print(f"unique sheets found: {len(groups)}")
    if not groups:
        Path("page_snapshot.html").write_text(r.text, encoding="utf-8")
        raise SystemExit("No assets found - upload page_snapshot.html for inspection.")

    ok = fail = 0
    failures = []
    for i, (name, variants) in enumerate(sorted(groups.items()), 1):
        dest = OUT / name
        if dest.exists() and dest.stat().st_size > 0:
            ok += 1
            continue
        got = False
        for u in variants:  # largest first; skip 404s and take what exists
            try:
                resp = requests.get(u, headers=HEADERS, timeout=120)
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
                dest.write_bytes(resp.content)
                got = True
                break
            except requests.RequestException:
                continue
        if got:
            ok += 1
        else:
            fail += 1
            failures.append(name)
            print(f"  FAIL all variants: {name}")
        if i % 20 == 0:
            print(f"  {i}/{len(groups)} processed (ok={ok})")
        time.sleep(POLITE_DELAY)

    print(f"\ndownloaded {ok}/{len(groups)} sheets into {OUT}/ ({fail} with no working variant)")
    if failures:
        Path("failed_sheets.txt").write_text("\n".join(failures), encoding="utf-8")
        print("failed list written to failed_sheets.txt")
    print("Next: zip the ec_sheets_2020 folder and upload it.")


if __name__ == "__main__":
    main()
