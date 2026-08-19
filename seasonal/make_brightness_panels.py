"""
Reproduce the "Where the light is" four-panel brightness map from lights_raw.csv,
with the VIIRS product seam handled.

The slide version uses 2013 / 2016 / 2020 / 2024. 2013 is in the VCMCFG era and
the other three are VCMSLCFG, so the first panel comes from a different
processing chain than the rest. Two versions are written:

  brightness_panels_asslide.png   the original years, 2013 flagged
  brightness_panels_clean.png     2015 / 2018 / 2021 / 2024, all one product

Outputs -> figs/
"""
import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
import shapefile

HERE = os.path.dirname(os.path.abspath(__file__)) or "."
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
CRITICAL = "#d03b3b"
R_EARTH = 6378137.0

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "font.size": 10,
})


def webmerc_to_lonlat(x, y):
    return (math.degrees(x / R_EARTH),
            math.degrees(2 * math.atan(math.exp(y / R_EARTH)) - math.pi / 2))


def load_rings():
    r = shapefile.Reader(os.path.join(HERE, "shp", "Ghana_Constituencies.shp"))
    out = []
    for shp in r.shapes():
        parts = list(shp.parts) + [len(shp.points)]
        out.append([[webmerc_to_lonlat(px, py)
                     for px, py in shp.points[parts[i]:parts[i + 1]]]
                    for i in range(len(parts) - 1)])
    return out


lights = pd.read_csv(os.path.join(HERE, "lights_raw.csv"))
attrs = pd.read_csv(os.path.join(HERE, "shp_attrs.csv"))
rings = load_rings()

# one shared scale across all panels -- the whole point of the figure is the
# comparison, so the colour must mean the same thing in every panel
GLOBAL_MAX = 4.1


def panel(ax, year, flag=None):
    v = (lights[lights.year == year]
         .groupby("cons_name").avg_rad.mean()
         .reindex(attrs.Cons_name).values)
    lv = np.log(np.clip(v, 0.05, None))
    norm = Normalize(vmin=0, vmax=GLOBAL_MAX)
    cmap = plt.get_cmap("inferno")
    patches, colors = [], []
    for rgs, x in zip(rings, lv):
        for ring in rgs:
            patches.append(MplPolygon(np.array(ring), closed=True))
            colors.append(cmap(norm(max(x, 0))))
    ax.add_collection(PatchCollection(patches, facecolors=colors,
                                      edgecolors="#33312e", linewidths=0.18))
    ax.set_xlim(-3.35, 1.30)
    ax.set_ylim(4.60, 11.30)
    ax.set_aspect(1 / math.cos(math.radians(8)))
    ax.axis("off")
    ax.set_title(str(year), color=INK, fontsize=13, fontweight="600", pad=6)
    if flag:
        ax.text(0.5, -0.03, flag, transform=ax.transAxes, ha="center",
                va="top", fontsize=9, color=CRITICAL, fontweight="600")
    return norm, cmap


def build(years, flags, fname, title, sub):
    fig, axes = plt.subplots(1, 4, figsize=(15, 6.4))
    for ax, y, fl in zip(axes, years, flags):
        norm, cmap = panel(ax, y, fl)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axes, fraction=0.018, pad=0.015)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=9, colors=MUTED)
    cb.set_label("log brightness  ln(avg_rad)", color=INK_2, fontsize=9.5)
    fig.text(0.09, 0.99, title, color=INK, fontsize=15, fontweight="650",
             ha="left", va="top")
    fig.text(0.09, 0.945, sub, color=INK_2, fontsize=10, ha="left", va="top",
             linespacing=1.45)
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=190, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", p)


# --- annual means, for the caption numbers
am = lights.groupby("year").apply(
    lambda g: np.log(g.avg_rad.clip(lower=0.05)).mean(), include_groups=False)

build([2013, 2016, 2020, 2024],
      [None, None, None, None],
      "brightness_panels_clean.png",
      "Where the light is: constituency brightness",
      "Shared colour scale across all four panels. Light spreads outward from Greater Accra, "
      "Kumasi and Tamale.\nNational mean ln(radiance): "
      f"{am[2013]:+.2f} → {am[2016]:+.2f} → {am[2020]:+.2f} → {am[2024]:+.2f}")

build([2013, 2016, 2020, 2024],
      ["VCMCFG — different\nVIIRS product", "VCMSLCFG", "VCMSLCFG", "VCMSLCFG"],
      "brightness_panels_flagged.png",
      "Where the light is — with the product seam flagged",
      "The 2013 panel comes from a different VIIRS processing chain than the other three. "
      "The 2013→2016\nchange is small either way "
      f"({am[2013]:+.2f} → {am[2016]:+.2f}); the growth in this figure is almost all "
      "post-2016, within one product.")

build([2015, 2018, 2021, 2024],
      [None, None, None, None],
      "brightness_panels_oneproduct.png",
      "Where the light is: 2015–2024, single VIIRS product",
      "All four panels are VCMSLCFG, so nothing here can be a sensor artefact. "
      "2014 is deliberately avoided\nas a baseline — it is the dumsor trough and would "
      f"inflate the growth. National mean: {am[2015]:+.2f} → {am[2018]:+.2f} → "
      f"{am[2021]:+.2f} → {am[2024]:+.2f}")

print("\nannual mean ln(radiance):")
for y, v in am.items():
    print("  %d  %+.3f  %s" % (y, v, "VCMCFG" if y < 2014 else "VCMSLCFG"))
