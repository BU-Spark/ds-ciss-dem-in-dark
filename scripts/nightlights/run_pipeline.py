"""
Pipeline orchestrator: validate raw GEE export, build panel, generate Q1 visuals.

Inputs: constituency_lights_raw.csv (GEE export)
Outputs: panel CSV, Q1 figures, run_manifest.json with validation metadata
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# project layout: scripts/nightlights/ -> project root is two levels up
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data" / "nightlights"        # csv inputs/outputs + manifest
IMG = ROOT / "images" / "nightlights"       # figure outputs
RAW_DEFAULT = DATA / "constituency_lights_raw.csv"
PANEL = DATA / "constituency_lights_panel.csv"
MANIFEST = DATA / "run_manifest.json"


def log(stage, msg):
    print(f"[{stage:^9}] {msg}", flush=True)


def fail(stage, msg):
    print(f"[{stage:^9}] ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def sha1(path):
    h = hashlib.sha1()
    h.update(Path(path).read_bytes())
    return h.hexdigest()[:12]


def validate_raw(raw):
    import pandas as pd  # local import so --help works without pandas

    if not raw.exists():
        fail("validate", f"{raw.name} not found. Run the GEE export first "
                         "(gee_nightlights_ghana.js) and download the CSV here.")
    df = pd.read_csv(raw)
    if len(df) != 275:
        fail("validate", f"expected 275 rows, got {len(df)}")
    for col in ("con_id", "cons_name"):
        if col not in df.columns:
            fail("validate", f"missing required column '{col}'")
    years = sorted(int(c.split("_")[1]) for c in df.columns if c.startswith("mean_"))
    if not years:
        fail("validate", "no mean_<year> columns found")
    empty_id = int(df["con_id"].isna().sum())
    log("validate", f"{len(df)} rows, years={years}, empty con_id={empty_id} "
                    f"(name-matched downstream)")
    return years, empty_id


def run_step(stage, script):
    log(stage, f"running {script}")
    r = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE)
    if r.returncode != 0:
        fail(stage, f"{script} exited with code {r.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW_DEFAULT), help="raw GEE export CSV")
    ap.add_argument("--skip-maps", action="store_true", help="build panel only")
    args = ap.parse_args()

    raw = Path(args.raw)
    started = datetime.now(timezone.utc)
    log("start", f"pipeline @ {started.isoformat(timespec='seconds')}")

    years, empty_id = validate_raw(raw)
    run_step("build", "build_panel.py")
    if not PANEL.exists():
        fail("build", "panel not produced")

    if not args.skip_maps:
        run_step("visualize", "make_q1_maps.py")

    figures = sorted(f"images/nightlights/{p.name}" for p in IMG.glob("*.png"))
    csvs = [f"data/nightlights/{PANEL.name}",
            "data/nightlights/constituency_lights_panel_with_geo.csv"]
    outputs = [o for o in (figures + csvs) if (ROOT / o).exists()]
    manifest = {
        "run_utc": started.isoformat(timespec="seconds"),
        "raw_input": raw.name,
        "raw_sha1": sha1(raw),
        "years": years,
        "empty_con_id": empty_id,
        "outputs": outputs,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    log("manifest", f"wrote {MANIFEST.name}")
    log("done", f"panel + {len([o for o in outputs if o.endswith('.png')])} figures ready")


if __name__ == "__main__":
    main()
