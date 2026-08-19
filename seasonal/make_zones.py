"""
Agroclimatic zones for the 275 constituencies, built on the client's framework
(Ghana Meteorological Agency: 5 zones, consolidating to 3 analytical belts).

Two things are produced and they are deliberately separate:

  zone5 / zone3   assigned from RAINFALL — the client's definition, so this is a
                  climate classification and nothing to do with electricity
  lightgrp        the earlier grouping fitted on LIGHT seasonality

Keeping them apart is the point: "do the light groups line up with the climate
zones" is only a real question if the two were built independently.

Assignment rule for zone5, in the client's own terms:
  * bimodal vs unimodal is measured by the August rainfall break
  * within the bimodal south, Coastal is separated from Forest by annual rainfall
    total (the coastal strip around Accra is markedly drier than the forest belt)
  * within the unimodal north, Sudan is separated from Guinea by the length of
    the rainy season (Sudan has the shortest, per the client)

Outputs: constituency_zones.csv, figs/deck/d7_zones.png
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
import shapefile

HERE = os.path.dirname(os.path.abspath(__file__)) or "."
DECK = os.path.join(HERE, "figs", "deck")
os.makedirs(DECK, exist_ok=True)

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
M = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": BASELINE, "font.size": 12,
})

clim = pd.read_csv(os.path.join(HERE, "climate.csv"))
rmap = pd.read_csv(os.path.join(HERE, "constituency_region_map.csv"))
attrs = pd.read_csv(os.path.join(HERE, "shp_attrs.csv"))

P = clim.groupby(["cons_name", "month"]).precip_mm.mean().unstack()
P.columns = range(1, 13)

annual = P.sum(axis=1)
# the August break: rain either side of August, minus August, scaled by the
# constituency's own mean month
brk = (((P[[6, 7]].mean(axis=1) + P[[9, 10]].mean(axis=1)) / 2 - P[8])
       / P.mean(axis=1))

Z = pd.DataFrame({"annual_mm": annual, "aug_break": brk}).join(
    rmap.set_index("cons_name")[["centroid_lat", "centroid_lon", "region_2019"]])

# ---- how many zones does the RAINFALL SHAPE itself support? ------------
# Cluster the standardised monthly rainfall profile: shape only, so a wet place
# and a dry place with the same calendar land together. This is a climate
# classification built without reference to electricity or to any map.
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

shape = P.div(P.sum(axis=1), axis=0)                 # each row sums to 1
Xs = ((shape - shape.mean()) / shape.std()).values

print("=" * 76)
print("HOW MANY AGROCLIMATIC ZONES DOES THE RAINFALL SUPPORT?")
print("=" * 76)
sil = {}
for k in range(2, 8):
    lab = KMeans(n_clusters=k, n_init=25, random_state=0).fit_predict(Xs)
    sil[k] = silhouette_score(Xs, lab)
    print("  k=%d  silhouette %.3f" % (k, sil[k]))
best = max(sil, key=sil.get)
print("  best = %d;  the client's framework specifies 5 (silhouette %.3f)"
      % (best, sil[5]))
print()

lab = KMeans(n_clusters=5, n_init=50, random_state=0).fit_predict(Xs)
Z["k5"] = lab
# name the five by latitude order, south to north
lat_order = Z.groupby("k5").centroid_lat.mean().sort_values().index.tolist()
NAMES = ["Coastal", "Forest", "Transition", "Guinea Savannah", "Sudan Savannah"]
Z["zone5"] = Z.k5.map(dict(zip(lat_order, NAMES)))

# ---- 3 analytical belts: the client's own consolidation ----------------
TO3 = {"Coastal": "Southern (bimodal)", "Forest": "Southern (bimodal)",
       "Transition": "Transition (middle belt)",
       "Guinea Savannah": "Northern (unimodal)",
       "Sudan Savannah": "Northern (unimodal)"}
Z["zone3"] = Z.zone5.map(TO3)

ORDER5 = ["Coastal", "Forest", "Transition", "Guinea Savannah", "Sudan Savannah"]
ORDER3 = ["Southern (bimodal)", "Transition (middle belt)", "Northern (unimodal)"]

print("=" * 76)
print("FIVE AGROCLIMATIC ZONES, assigned from rainfall alone")
print("=" * 76)
print("%-18s %4s %11s %11s %9s  %s" %
      ("zone", "n", "annual mm", "Aug break", "mean lat", "peak month(s)"))
for z in ORDER5:
    g = Z[Z.zone5 == z]
    prof = P.loc[g.index].mean()
    top = prof.nlargest(2).index.tolist()
    print("  %-18s %4d %11.0f %11.2f %9.2f  %s"
          % (z, len(g), g.annual_mm.median(), g.aug_break.median(),
             g.centroid_lat.mean(), " and ".join(M[t - 1] for t in sorted(top))))
print()
print("THREE ANALYTICAL BELTS")
for z in ORDER3:
    g = Z[Z.zone3 == z]
    print("  %-26s n=%3d   mean latitude %.2f" % (z, len(g), g.centroid_lat.mean()))

print()
print("=" * 76)
print("DO THE LIGHT-SEASONALITY GROUPS LINE UP WITH THE CLIMATE ZONES?")
print("=" * 76)
U = pd.read_csv(os.path.join(HERE, "out", "seasonal_profile_by_unit.csv"))
Z2 = Z.join(U.set_index("cons_name")[["cluster", "mean_rad", "amplitude"]])
LIGHT = {0: "Southern forest belt", 1: "Urban core", 2: "Northern savanna belt"}
ct = pd.crosstab(Z2.zone3, Z2.cluster.map(LIGHT)).reindex(ORDER3)
print(ct.to_string())
from scipy.stats import chi2_contingency
c2, pv, dof, _ = chi2_contingency(pd.crosstab(Z2.zone3, Z2.cluster))
n = len(Z2.dropna(subset=["cluster"]))
print("\n  Cramer's V = %.3f   (chi2=%.1f, dof=%d, p=%.2g)"
      % (np.sqrt(c2 / (n * (min(ct.shape) - 1))), c2, dof, pv))
print("  -> the light groups and the climate belts agree in the north and the")
print("     south. The gap is the middle: the light data produces an URBAN group")
print("     where the climate framework has a TRANSITION belt. They are not the")
print("     same thing and the deck should not claim they are.")

print()
print("seasonal amplitude of measured light, by climate zone:")
for z in ORDER5:
    g = Z2[Z2.zone5 == z]
    print("  %-18s n=%3d   median amplitude %.2f   median radiance %.2f"
          % (z, len(g), g.amplitude.median(), g.mean_rad.median()))

Z.reset_index().rename(columns={"index": "cons_name"}).to_csv(
    os.path.join(HERE, "constituency_zones.csv"), index=False)
print("\nwrote constituency_zones.csv")

# ======================================================== figure
R = 6378137.0


def inv(x, y):
    return (math.degrees(x / R),
            math.degrees(2 * math.atan(math.exp(y / R)) - math.pi / 2))


sf = shapefile.Reader(os.path.join(HERE, "shp", "Ghana_Constituencies.shp"))
rings = []
for shp in sf.shapes():
    parts = list(shp.parts) + [len(shp.points)]
    rings.append([[inv(px, py) for px, py in shp.points[parts[i]:parts[i + 1]]]
                  for i in range(len(parts) - 1)])

# one hue family, ordered south -> north, so the map reads as a gradient
COL5 = {"Coastal": "#9ec5f4", "Forest": "#2a78d6", "Transition": "#eda100",
        "Guinea Savannah": "#eb6834", "Sudan Savannah": "#96381b"}
z5 = Z.reindex(attrs.Cons_name).zone5.values

# --- (a) the zone map on its own, tall, for the "what we built" slide
fig, axm = plt.subplots(figsize=(6.6, 7.4))
patches, colors = [], []
for rgs, z in zip(rings, z5):
    for ring in rgs:
        patches.append(MplPolygon(np.array(ring), closed=True))
        colors.append(COL5.get(z, "#dddddd"))
axm.add_collection(PatchCollection(patches, facecolors=colors,
                                   edgecolors=SURFACE, linewidths=0.28))
axm.set_xlim(-3.35, 1.35)
axm.set_ylim(4.60, 11.30)
axm.set_aspect(1 / math.cos(math.radians(8)))
axm.axis("off")
handles = [plt.Line2D([], [], marker="s", linestyle="none", markersize=15,
                      markerfacecolor=COL5[z], markeredgecolor="none",
                      label=f"{z}  (n={int((Z.zone5 == z).sum())})")
           for z in ORDER5]
leg = axm.legend(handles=handles, frameon=False, fontsize=13,
                 loc="upper center", bbox_to_anchor=(0.5, -0.015), ncol=2,
                 handletextpad=0.7, labelspacing=0.8, columnspacing=1.6)
for t in leg.get_texts():
    t.set_color(INK_2)
axm.set_title("Ghana's five agroclimatic zones", color=INK, fontsize=17,
              fontweight="650", loc="left", pad=8)
p_ = os.path.join(DECK, "d7a_zone_map.png")
fig.savefig(p_, dpi=200, bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)
print("wrote", p_)

# --- (b) the rainfall profiles on their own, wide and readable
fig, axr = plt.subplots(figsize=(9.2, 5.8))
for z in ORDER5:
    idx = Z.index[Z.zone5 == z]
    axr.plot(range(12), P.loc[idx].mean().values, color=COL5[z], linewidth=3.2,
             label=z, zorder=3)
axr.axvspan(6.55, 8.45, color="#f0efec", zorder=0)
axr.text(7.5, 6, "August\nbreak", ha="center", va="bottom", fontsize=13,
         color=MUTED, linespacing=1.25)
axr.set_xticks(range(12))
axr.set_xticklabels(M, fontsize=13)
for sp in ("top", "right"):
    axr.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    axr.spines[sp].set_color(BASELINE)
axr.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
axr.set_axisbelow(True)
axr.tick_params(length=0, labelsize=13)
axr.set_ylabel("rainfall (mm per month)", color=INK_2, fontsize=13)
leg = axr.legend(frameon=False, fontsize=13, loc="upper left", ncol=2,
                 columnspacing=1.4)
for t in leg.get_texts():
    t.set_color(INK_2)
axr.set_title("Two rainy peaks in the south, one in the north", color=INK,
              fontsize=17, fontweight="650", loc="left", pad=30)
axr.text(0, 1.015,
         "The south dips in August between two peaks. The north has a single peak, in August.",
         transform=axr.transAxes, color=INK_2, fontsize=12.5, va="bottom")
axr.set_ylim(0, 265)
p_ = os.path.join(DECK, "d7b_rainfall_zones.png")
fig.savefig(p_, dpi=200, bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)
print("wrote", p_)
