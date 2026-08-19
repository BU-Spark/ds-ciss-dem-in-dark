#!/usr/bin/env python3
"""
Render a 2024 EC Form Ten PDF and cut it into two legible crops.

WHY CROPS RATHER THAN THE WHOLE PAGE
------------------------------------
These sheets are handwritten scans with no OCR text layer, so the figures have
to be read visually.  A full A4 page downscaled to fit a single view loses the
stroke detail that separates 3 from 5 and 6 from 0, which is exactly where a
transcription error would do the most damage.  Cutting the page into the two
bands that actually carry data keeps every digit at readable size.

BAND 1  candidate rows 1-13 plus the A / B / C totals, figures and words
        columns only.  The words column is the arithmetic safety net: where a
        figure is ambiguous the written-out amount settles it.
BAND 2  the constituency and region header, so the sheet can be tied to a row
        without trusting the filename.

USAGE
    python crop_form10.py <pdf> <outdir>
"""

import os
import subprocess
import sys

from PIL import Image

DPI = 200

# Fractions of page height for each band, measured off a reference sheet.
# Every Form Ten is the same printed template, so these hold for all of them.
HEADER = (0.06, 0.11, 0.99, 0.16)      # left, top, right, bottom
TABLE = (0.06, 0.155, 0.99, 0.56)


def render(pdf, outdir):
    """Rasterise page 1 and write the two crops. Returns their paths."""
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(pdf))[0]
    base = os.path.join(outdir, stem)
    subprocess.run(
        ["pdftoppm", "-r", str(DPI), "-png", "-singlefile", pdf, base],
        check=True)
    im = Image.open(base + ".png")
    w, h = im.size
    out = []
    for tag, (l, t, r, b) in (("hdr", HEADER), ("tbl", TABLE)):
        box = (int(l * w), int(t * h), int(r * w), int(b * h))
        p = f"{base}_{tag}.png"
        im.crop(box).save(p)
        out.append(p)
    os.remove(base + ".png")
    return out


if __name__ == "__main__":
    for p in render(sys.argv[1], sys.argv[2]):
        print(p)
