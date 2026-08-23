"""
Q1 visuals: choropleths (brightness, rank) and rank time-series (biggest movers).

Inputs: constituency_lights_panel.csv, constituencies_4326.gpkg, constituency_dictionary.csv
Outputs: PNG maps and CSV with geographic data
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
from pathlib import Path

# project layout: scripts/nightlights/ -> project root is two levels up
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data" / "nightlights"        # panel + with_geo outputs
BOUND = ROOT / "data" / "boundaries"        # constituency polygons
IMG = ROOT / "result" / "nightlights"       # figure outputs
IMG.mkdir(parents=True, exist_ok=True)
GPKG = BOUND / "constituencies_4326.gpkg"
PANEL = DATA / "constituency_lights_panel.csv"

g = gpd.read_file(GPKG)
df = pd.read_csv(PANEL)

# auto-detect years present in the panel (log_mean_<year> columns)
YEARS = sorted(int(c.split("_")[2]) for c in df.columns if c.startswith("log_mean_"))
print("years detected in panel:", YEARS)

# join on constituency id (gpkg: constituency_id, panel: con_id);
# fall back to name for rows with a missing id (e.g. Zebilla).
have_id = df[df["con_id"].notna()]
no_id = df[df["con_id"].isna()]
g = g.merge(have_id, left_on="constituency_id", right_on="con_id", how="left")
if len(no_id):
    name_map = no_id.set_index("cons_name")
    for name in name_map.index:
        mask = g["Cons_name"] == name
        for col in no_id.columns:
            if col not in ("con_id", "cons_name"):
                g.loc[mask, col] = name_map.loc[name, col]
    print(f"note: name-matched {len(no_id)} row(s) without con_id: "
          f"{list(no_id['cons_name'])}")
missing = g[f"log_mean_{YEARS[-1]}"].isna().sum()
if missing:
    print(f"WARNING: {missing} constituencies had no light value (shown grey)")

# --- 16-region overlay: dissolve constituencies to regions for red boundaries + labels
DICT = BOUND / "constituency_dictionary.csv"
d16 = pd.read_csv(DICT)
# IMPORTANT: only join on REAL ids. The dictionary has ~80 rows with a blank
# constituency_id; pandas would match NaN==NaN and cross-join them onto any
# id-less polygon (e.g. Zebilla), exploding the row count. Drop null keys + dedupe.
d16_id = d16.dropna(subset=["constituency_id"]).drop_duplicates("constituency_id")
g = g.merge(d16_id[["constituency_id", "region_16"]], on="constituency_id", how="left")
# fill region for id-less polygons (e.g. Zebilla) by name, using the dictionary
# and the EC official reference as a second source.
ec = pd.read_csv(BOUND / "EC_official_constituencies_reference.csv")
name2reg = pd.concat([
    d16.dropna(subset=["region_16"]).set_index("constituency_name")["region_16"],
    ec.dropna(subset=["region"]).set_index("constituency")["region"],
])
name2reg = name2reg[~name2reg.index.duplicated()]
miss_reg = g["region_16"].isna()
g.loc[miss_reg, "region_16"] = g.loc[miss_reg, "Cons_name"].map(name2reg)
region_gdf = g.dropna(subset=["region_16"]).dissolve(by="region_16")
assert len(g) == 275, f"row count exploded to {len(g)} — check id merge"
print(f"regions dissolved (16-region): {len(region_gdf)}")

# --- 10-region overlay: Ghana used 10 regions through 2018, split into 16 at
# end of 2018. Panels for years <= REGION_SPLIT_YEAR use the 10-region layer.
REGION_SPLIT_YEAR = 2018
# normalize spelling: the dictionary has both "Brong Ahafo" and "Brong-Ahafo"
# for the same region; without normalizing, dissolve yields 11 polygons, not 10.
d16["region_10"] = d16["region_10"].replace("Brong Ahafo", "Brong-Ahafo")
d10_id = d16.dropna(subset=["constituency_id"]).drop_duplicates("constituency_id")
g = g.merge(d10_id[["constituency_id", "region_10"]], on="constituency_id", how="left")
name2reg10 = d16.dropna(subset=["region_10"]).set_index("constituency_name")["region_10"]
name2reg10 = name2reg10[~name2reg10.index.duplicated()]
miss_reg10 = g["region_10"].isna()
g.loc[miss_reg10, "region_10"] = g.loc[miss_reg10, "Cons_name"].map(name2reg10)
print(f"polygons with no region_10: {g['region_10'].isna().sum()}")
assert len(g) == 275, f"row count exploded to {len(g)} — check id merge"
region_gdf_10 = g.dropna(subset=["region_10"]).dissolve(by="region_10")
assert len(region_gdf_10) == 10, f"expected 10 regions, got {len(region_gdf_10)}"
print(f"regions dissolved (10-region): {len(region_gdf_10)}")


def add_regions(ax):
    """Draw the 16 administrative regions as red outlines with white-haloed labels."""
    region_gdf.boundary.plot(ax=ax, edgecolor="red", linewidth=1.1)
    for name, geom in zip(region_gdf.index, region_gdf.geometry):
        pt = geom.representative_point()
        ax.annotate(name, (pt.x, pt.y), ha="center", va="center",
                    fontsize=7, color="red", fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])


# 1) brightness (log) small-multiples: ONE figure, shared scale -> comparable across years
vmax = max(g[f"log_mean_{y}"].max() for y in YEARS)
fig, axes = plt.subplots(1, len(YEARS), figsize=(4.2 * len(YEARS), 8))
for ax_i, y in zip(axes, YEARS):
    g.plot(column=f"log_mean_{y}", cmap="inferno", vmin=0, vmax=vmax, ax=ax_i,
           missing_kwds={"color": "lightgrey"})
    if y <= REGION_SPLIT_YEAR:
        region_gdf_10.boundary.plot(ax=ax_i, edgecolor="red", linewidth=0.5)
        basis = "10 regions"
    else:
        region_gdf.boundary.plot(ax=ax_i, edgecolor="red", linewidth=0.5)
        basis = "16 regions"
    ax_i.set_title(f"{y} ({basis})", fontsize=12); ax_i.axis("off")
sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmin=0, vmax=vmax))
fig.colorbar(sm, ax=axes, shrink=0.5, label="log brightness")
fig.suptitle("Ghana constituency brightness (log radiance) by year — shared scale",
             fontsize=14)
plt.savefig(IMG / "logmean_by_year.png", dpi=150, bbox_inches="tight")
plt.close()

# 2) percentile-rank map for the latest year (relative position) -------------
ax = g.plot(
    column=f"pct_rank_{YEARS[-1]}", cmap="viridis", legend=True, figsize=(8, 9),
    missing_kwds={"color": "lightgrey", "label": "no data"},
)
add_regions(ax)
ax.set_title(f"Constituency brightness percentile rank - {YEARS[-1]}", fontsize=13)
ax.axis("off")
plt.tight_layout()
plt.savefig(IMG / f"choropleth_rank_{YEARS[-1]}.png", dpi=150, bbox_inches="tight")
plt.close()

# 3) RELATIVE growth map = log-ratio, first -> last year, among lit-throughout.
# log-ratio ln(mean_b / mean_a) keeps an interpretable fold-change, is symmetric
# (0 = no change), and dampens the near-zero explosion that raw % suffers from.
# Still mildly favors low starting points -> pair with the absolute top-5 figure.
a, b = YEARS[0], YEARS[-1]
lit_all = (g[[f"mean_{y}" for y in YEARS]] > 0).all(axis=1)
g["log_ratio"] = np.where(lit_all, np.log(g[f"mean_{b}"] / g[f"mean_{a}"]), np.nan)
ax = g.plot(
    column="log_ratio", cmap="YlGn", legend=True, figsize=(8, 9),
    missing_kwds={"color": "lightgrey", "label": "not lit throughout"},
)
add_regions(ax)
# label the top-5 fastest relative growers (same 5 as loggrowth_top5.png)
top5g = g.nlargest(5, "log_ratio")
for _, r in top5g.iterrows():
    pt = r.geometry.representative_point()
    ax.annotate(r["Cons_name"], (pt.x, pt.y), ha="center", va="center",
                fontsize=7, fontweight="bold", color="black",
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
ax.set_title(f"Relative brightness growth {a}->{b}: log-ratio ln(mean_{b}/mean_{a})\n"
             f"lit throughout; higher = faster relative growth (top 5 labeled)",
             fontsize=11)
ax.axis("off")
plt.tight_layout()
plt.savefig(IMG / f"choropleth_loggrowth_{a}_{b}.png", dpi=150, bbox_inches="tight")
plt.close()

# 3b) 2-category map by 2013 detection: dark (mean==0) vs lit (mean>0)
if "base2013" in g.columns:
    bcolors = {"lit_2013": "#4575b4", "dark_2013": "#d73027"}
    g["_bc"] = g["base2013"].map(bcolors).fillna("lightgrey")
    ax = g.plot(color=g["_bc"], figsize=(8, 9))
    add_regions(ax)
    handles = [mpatches.Patch(color=c, label=l) for l, c in bcolors.items()]
    ax.legend(handles=handles, loc="lower left", fontsize=9, title=f"{a} detection")
    ax.set_title(f"Detected light in {a}: dark (=0) vs lit (>0)", fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(IMG / f"choropleth_base2013_{a}.png", dpi=150, bbox_inches="tight")
    plt.close()

def _clean_line_ax(ax, title, ylabel):
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("election year"); ax.set_ylabel(ylabel)
    ax.set_xticks(YEARS); ax.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=9, frameon=False)

# 4A) share GAINERS: top 5 constituencies by share change vs 2013 baseline (positive only)
dsh = df.copy()
dsh["dshare"] = dsh[f"share_{YEARS[-1]}"] - dsh[f"share_{YEARS[0]}"]
gain5 = dsh[dsh["dshare"] > 0].nlargest(5, "dshare")
plt.figure(figsize=(8, 5)); ax = plt.gca()
for _, row in gain5.sort_values(f"share_{YEARS[-1]}", ascending=False).iterrows():
    ax.plot(YEARS, [row[f"share_{y}"] * 100 for y in YEARS],
            marker="o", linewidth=2, label=row["cons_name"])
_clean_line_ax(ax, f"Biggest gains in national share vs {YEARS[0]} baseline: top 5, "
                   f"{YEARS[0]}–{YEARS[-1]}", "share of national brightness (%)")
plt.tight_layout(); plt.savefig(IMG / "share_movers.png", dpi=150, bbox_inches="tight")
plt.close()

# 4B) fastest RELATIVE growth: log-ratio vs 2013 baseline, top 5 (lit throughout)
lit_all2 = (df[[f"mean_{y}" for y in YEARS]] > 0).all(axis=1)
dl = df[lit_all2].copy()
for y in YEARS:
    dl[f"lr_{y}"] = np.log(dl[f"mean_{y}"] / dl[f"mean_{YEARS[0]}"])
top5lr = dl.nlargest(5, f"lr_{YEARS[-1]}")
plt.figure(figsize=(8, 5)); ax = plt.gca()
for _, row in top5lr.sort_values(f"lr_{YEARS[-1]}", ascending=False).iterrows():
    ax.plot(YEARS, [row[f"lr_{y}"] for y in YEARS],
            marker="o", linewidth=2, label=row["cons_name"])
_clean_line_ax(ax, f"Fastest relative growth: log-ratio vs {YEARS[0]} baseline, top 5",
               f"log-ratio  ln(mean_y / mean_{YEARS[0]})")
plt.tight_layout(); plt.savefig(IMG / "loggrowth_top5.png", dpi=150, bbox_inches="tight")
plt.close()

# 5) accessibility: the constituencies dark (0) in 2013 -> brightness onset
if "base2013" in df.columns:
    dark0 = df[df["base2013"] == "dark_2013"]
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    for _, row in dark0.iterrows():
        ax.plot(YEARS, [row[f"mean_{y}"] for y in YEARS],
                marker="o", linewidth=2, label=row["cons_name"])
    ax.set_title(f"From dark to lit: {len(dark0)} rural constituencies gained\n"
                 f"detectable electricity, {YEARS[0]}–{YEARS[-1]}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("election year"); ax.set_ylabel("mean nighttime brightness")
    ax.set_xticks(YEARS)
    ax.grid(False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(IMG / "accessibility_dark2013_onset.png", dpi=150, bbox_inches="tight")
    plt.close()

# save merged table for downstream joins with the election panel
g.drop(columns="geometry").to_csv(
    DATA / "constituency_lights_panel_with_geo.csv", index=False
)
print("done: logmean_by_year, choropleth_rank_2024, choropleth_base2013_*, "
      "choropleth_loggrowth_*, loggrowth_top5, share_movers, "
      "accessibility_dark2013_onset, constituency_lights_panel_with_geo.csv")
