"""
make_zone_seasonal_maps.py -- Approach 2 entry point (orchestration + visualization).

Reference script. Consolidates the 275 constituencies into 3 agroclimatic zones,
runs a robust STL decomposition on each zone's equal-weight monthly nightlight
index, and renders:

    result/seasonal_zones/zone_stl_decomposition.png   STL grid: 4 components x 3 zones
    result/seasonal_zones/zone_stl_components.csv      long-format trend/seasonal/remainder
    result/seasonal_zones/zone_map_and_shapes.png      combined map + curves (headline)

All numerical logic lives in seasonal_zones.py; this file only loads data, calls
those pure functions, and draws. Run:  python make_zone_seasonal_maps.py

Prereqs: pip install geopandas matplotlib pandas statsmodels
"""

from pathlib import Path

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch

import seasonal_zones as sz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data" / "nightlights"
BOUND = ROOT / "data" / "boundaries"
OUT = ROOT / "result" / "seasonal_zones"
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_zone_assignments() -> pd.DataFrame:
    """con_id -> region_16 -> zone for every constituency in the panel."""
    long_df = pd.read_csv(DATA / "constituency_seasonal_long.csv")
    lut = sz.build_region_lookup(
        BOUND / "constituency_dictionary.csv",
        DATA / "constituency_lights_panel_with_geo.csv",
    )
    zones = sz.assign_zones(long_df["con_id"].unique(), lut)
    missing = zones["zone"].isna().sum()
    assert missing == 0, f"{missing} constituencies unmapped to a zone"
    return long_df, zones


def load_zone_geometry(zones: pd.DataFrame):
    """Constituency polygons coloured by zone + the 16 regions for red outlines."""
    g = gpd.read_file(BOUND / "constituencies_4326.gpkg")
    g = g.merge(zones[["con_id", "region_16", "zone"]],
                left_on="constituency_id", right_on="con_id", how="left")
    # name fallback for the rare id-less polygon (e.g. Zebilla)
    if g["zone"].isna().any():
        by_name = zones.merge(
            pd.read_csv(DATA / "constituency_seasonal_long.csv")[["con_id", "cons_name"]]
            .drop_duplicates("con_id"), on="con_id", how="left")
        name2zone = by_name.dropna(subset=["cons_name"]).set_index("cons_name")["zone"]
        miss = g["zone"].isna()
        g.loc[miss, "zone"] = g.loc[miss, "Cons_name"].map(name2zone)
    region_gdf = g.dropna(subset=["region_16"]).dissolve(by="region_16")
    return g, region_gdf


def add_regions(ax, region_gdf) -> None:
    """16 regions as red outlines with white-haloed labels (reused convention)."""
    region_gdf.boundary.plot(ax=ax, edgecolor="red", linewidth=1.0)
    for name, geom in zip(region_gdf.index, region_gdf.geometry):
        pt = geom.representative_point()
        ax.annotate(name, (pt.x, pt.y), ha="center", va="center",
                    fontsize=6.5, color="red", fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def draw_zone_map(ax, g, region_gdf) -> None:
    """Categorical choropleth: each constituency filled by its climate zone."""
    for zone in sz.ZONE_ORDER:
        sub = g[g["zone"] == zone]
        sub.plot(ax=ax, color=sz.ZONE_COLORS[zone], edgecolor="white", linewidth=0.15)
    unmapped = g[g["zone"].isna()]
    if not unmapped.empty:
        unmapped.plot(ax=ax, color="lightgrey", edgecolor="white", linewidth=0.15)
    add_regions(ax, region_gdf)
    ax.set_title("Ghana constituencies by agroclimatic zone (3-belt consolidation)",
                 fontsize=12)
    ax.axis("off")
    counts = g["zone"].value_counts()
    handles = [Patch(facecolor=sz.ZONE_COLORS[z], edgecolor="white",
                     label=f"{z}  (n={int(counts.get(z, 0))})") for z in sz.ZONE_ORDER]
    ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=True,
              title="Climate zone")


def draw_seasonal_shapes(ax, shapes: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """Three folded 12-month STL seasonal curves: bimodal vs unimodal at a glance."""
    m = metrics.set_index("zone")
    for zone in sz.ZONE_ORDER:
        y = shapes[f"stl_{zone}"].to_numpy()
        lbl = (f"{zone} - rain {m.loc[zone, 'rainfall_regime']} "
               f"(FS={m.loc[zone, 'seasonal_strength']:.2f})")
        ax.plot(range(1, 13), y, marker="o", ms=4, lw=2,
                color=sz.ZONE_COLORS[zone], label=lbl)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.axvspan(10.5, 12.9, color="grey", alpha=0.08)
    ax.axvspan(0.1, 3.5, color="grey", alpha=0.08)   # Nov-Mar dry window (shaded)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(sz.MONTH_ABBR, fontsize=8)
    ax.set_ylabel("STL seasonal deviation\n(index units, 0 = annual level)", fontsize=9)
    ax.set_title("Typical 12-month seasonal shape by zone\n"
                 "(grey band = Nov-Mar dry season)", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)


def fig_stl_grid(decomps: dict) -> None:
    """4 rows (observed/trend/seasonal/remainder) x 3 columns (zones)."""
    comps = ["observed", "trend", "seasonal", "remainder"]
    fig, axes = plt.subplots(len(comps), len(sz.ZONE_ORDER),
                             figsize=(13, 9), sharex="col")
    for j, zone in enumerate(sz.ZONE_ORDER):
        d = decomps[zone]
        for i, comp in enumerate(comps):
            ax = axes[i, j]
            ax.plot(d[comp].index, d[comp].values,
                    color=sz.ZONE_COLORS[zone], lw=0.9)
            if i == 0:
                ax.set_title(f"{zone}\nFS = {d['seasonal_strength']:.2f}", fontsize=10)
            if j == 0:
                ax.set_ylabel(comp, fontsize=9)
            ax.grid(alpha=0.2)
    fig.suptitle("Robust STL decomposition of the zone nightlight index "
                 "(2012-04 to 2024-12)", fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "zone_stl_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_map_and_shapes(g, region_gdf, shapes, metrics) -> None:
    """Headline deliverable: shapefile map beside the three seasonal curves."""
    fig, (axm, axs) = plt.subplots(1, 2, figsize=(16, 8.5),
                                   gridspec_kw={"width_ratios": [1, 1.05]})
    draw_zone_map(axm, g, region_gdf)
    draw_seasonal_shapes(axs, shapes, metrics)
    fig.suptitle("Ghana nightlight seasonality across three agroclimatic zones",
                 fontsize=15, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "zone_map_and_shapes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main() -> None:
    long_df, zones = load_zone_assignments()
    print("zone assignment:\n", zones["zone"].value_counts().to_string())

    zone_series = sz.build_zone_series(long_df, zones)
    clim = sz.zone_climatology(zone_series)

    decomps, shape_cols = {}, {}
    for zone in sz.ZONE_ORDER:
        s = (zone_series[zone_series["zone"] == zone]
             .set_index("date")["idx"].sort_index())
        d = sz.decompose_zone(s)
        decomps[zone] = d
        shape_cols[f"stl_{zone}"] = sz.zone_seasonal_shape(d["seasonal"])
        shape_cols[f"clim_{zone}"] = (clim.loc[zone,
                                      [f"clim_{m}" for m in range(1, 13)]]
                                      .to_numpy() - 1.0)  # centre on annual level

    shapes = pd.DataFrame(shape_cols)
    shapes.index = range(1, 13)
    metrics = sz.zone_metrics_table(decomps, clim)

    stl_parts = []
    for zone in sz.ZONE_ORDER:
        d = decomps[zone]
        stl_parts.append(pd.DataFrame({
            "date": d["trend"].index, "zone": zone,
            "trend": d["trend"].to_numpy(), "seasonal": d["seasonal"].to_numpy(),
            "remainder": d["remainder"].to_numpy(),
        }))
    pd.concat(stl_parts, ignore_index=True).to_csv(
        OUT / "zone_stl_components.csv", index=False)

    print("\nzone metrics:\n", metrics.to_string(index=False))

    # geometry + figures
    g, region_gdf = load_zone_geometry(zones)
    fig_stl_grid(decomps)
    fig_map_and_shapes(g, region_gdf, shapes, metrics)
    print(f"\nwrote 2 figures + 1 CSV to {OUT}")


if __name__ == "__main__":
    main()
