"""
per_election_gap.py -- per-election "gap" series and figure: the
competitive-minus-safe nightlights gap, in log points, around each December
election, plus a client-facing chart of that gap expressed as a deviation
from each window's own average.

Reuses load_alan_panel, nearest_election, and attach_predetermined_close
from election_did.py, and the region_matched grouping pipeline
(load_panel_with_region, attach_real_event_time, mean_close_lut, region_lut,
region_matched_groups) from two_line_series.py, rather than re-deriving any
of that logic.

Four windows, each +/-6 months around a December:
    2012, 2016, 2020   -- real elections
    2018_placebo        -- December 2018, no election (validity check)

Each window is built independently: for cycle year Y, a row is kept if
abs(months from December Y) <= 6. In practice this is computed via
nearest_election() with a 6-month window rather than a bespoke per-row
subtraction: Ghana's real elections are 48 months apart, so at a 6-month
radius "nearest election" and "months from December Y" pick out exactly the
same rows -- there is no ambiguity to resolve. The 2018 placebo has no real
election to be "nearest" to, so its window is computed directly as months
from a fake December 2018.

Grouping is the single, fixed region_matched scheme, reused unchanged from
two_line_series.py: within each region_2019 with >=6 constituencies, the
top third of constituencies by mean predetermined `close` are "competitive"
and the bottom third are "safe", pooled across regions. `close` itself is
attach_predetermined_close's predetermined closeness of a constituency's
PRIOR election (e.g. cycle 2016's close comes from margin_2012, cycle
2020's from margin_2016, cycle 2024's from margin_2020 -- see
MARGIN_COL_BY_YEAR in election_did.py). mean_close_lut averages that close
across whichever cycles have it defined for a given constituency; cycle
2012 has no prior-election margin column (would need margin_2008, which
does not exist in the source data) and so contributes nothing to the mean,
but constituencies still get a score from their 2016/2020/2024 values.
Because the resulting competitiveness score, and hence group membership, is
a single fixed number per constituency, the SAME groups apply to every
window built here -- including the 2018 placebo. There is no 2018 election
to derive a margin from, so the placebo does not need a special case: it
simply reuses the membership every real cycle already uses.

No re-basing to any anchor month: unlike two_line_series.py's index (fixed
to 100 at k=-1), gap and national_dev are left un-anchored here. Earlier
drafts normalised to k=-1 and that created a symmetric fanning artefact
around whatever k=-1 happened to look like; the raw log-point gap, and a
national series centred on its own window mean, avoid that.

The figure draws four columns, one per cycle, each with two rows:
    top    -- national_dev (all seats), plain grey line, its own y-scale
    bottom -- gap_dev (competitive minus safe), coloured line with a
              shaded 95% band, y-scale shared across all four cycles

The top strip is deliberately not tied to the bottom panel's y-scale:
national_dev's range runs several times larger than gap_dev's, so sharing
an axis would flatten the gap panel into a nearly invisible sliver. Keeping
them on separate scales lets a nationwide lift still register, without
distorting the (flat) competitive-vs-safe comparison below it.

Outputs (both under result/election_impact/):
    per_election_gap.csv -- the gap table, one row per (cycle, k)
    per_election_gap.png -- the four-cycle gap_dev figure

No global execution -- run via `python3 per_election_gap.py`.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

import election_did as ed
import two_line_series as tls

WINDOW = 6
REAL_CYCLES = (2012, 2016, 2020)
PLACEBO_YEAR = 2018
CI_MULT = 1.96

CYCLE_ORDER = ["2012", "2016", "2018_placebo", "2020"]
CYCLE_TITLES = {
    "2012": "2012",
    "2016": "2016",
    "2018_placebo": "2018 - no election",
    "2020": "2020",
}
PLACEBO_CYCLE = "2018_placebo"

GAP_COLOR = "#c44e52"
NATIONAL_COLOR = "#8c8c8c"
PLACEBO_TITLE_COLOR = "#b35806"
PLACEBO_BG_COLOR = "#fbe9d8"

Y_LABEL_MAIN = "Competitive minus safe (log points, deviation from window average)"
Y_LABEL_TOP = "All seats"
X_LABEL = "Months relative to December"

DEFAULT_SUPTITLE = (
    "The gap moves no more around a real election than around a December "
    "with no election"
)
DEFAULT_FOOTNOTE = (
    "Each panel is one December. The coloured line is the difference "
    "between competitive and safe constituencies, measured against its own "
    "average for that window; the shaded band is a 95% confidence "
    "interval. The grey strip above shows how all seats moved together, on "
    "its own scale. Months whose band clears zero: 4 in 2012, 2 in 2016 "
    "and 1 in 2020 - against 2 in 2018, a year with no election. The "
    "wobble is the same size either way, and it does not repeat from one "
    "election to the next."
)


# --------------------------------------------------------------------------- #
# Event windows, one per cycle, built independently
# --------------------------------------------------------------------------- #
def attach_cycle_window(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Months to the nearest December election and which cycle, |k| <= window.
    At a 6-month radius this is equivalent to computing months-from-
    December-Y independently for each real election year Y."""
    out = df.copy()
    ne = ed.nearest_election(out["year"], out["month"])
    out["k"] = ne["event_k"].to_numpy()
    out["cycle_year"] = ne["cycle_year"].to_numpy()
    return out[out["k"].abs() <= window].copy()


def attach_placebo_window(df: pd.DataFrame, fake_year: int = PLACEBO_YEAR,
                           window: int = WINDOW) -> pd.DataFrame:
    """Months to a fake December election, |k| <= window."""
    out = df.copy()
    out["k"] = (out["year"] - fake_year) * 12 + (out["month"] - 12)
    out["cycle_year"] = fake_year
    return out[out["k"].abs() <= window].copy()


def build_windows(panel: pd.DataFrame, real_cycles=REAL_CYCLES,
                   placebo_year: int = PLACEBO_YEAR,
                   window: int = WINDOW) -> dict[str, pd.DataFrame]:
    """cycle label -> independently-windowed panel slice, real cycles + placebo."""
    real = attach_cycle_window(panel, window=window)
    windows = {str(y): real[real["cycle_year"] == y].copy() for y in real_cycles}
    windows[f"{placebo_year}_placebo"] = attach_placebo_window(panel, placebo_year, window)
    return windows


# --------------------------------------------------------------------------- #
# Fixed region_matched group membership (reused from two_line_series.py)
# --------------------------------------------------------------------------- #
def group_lookup(panel_with_region: pd.DataFrame) -> pd.DataFrame:
    """Fixed competitive/safe membership shared by every cycle window."""
    real_df = tls.attach_real_event_time(panel_with_region)
    mean_close = tls.mean_close_lut(real_df)
    regions = tls.region_lut(real_df)
    return tls.region_matched_groups(mean_close, regions)


# --------------------------------------------------------------------------- #
# Per-window aggregation: gap, gap_se, national_dev
# --------------------------------------------------------------------------- #
def group_stats(window_df: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Per (group, k): constituency count, mean y, and SE of the mean."""
    merged = window_df.merge(groups, on="con_id", how="inner")
    g = merged.groupby(["group", "k"])["y"]
    agg = g.agg(n="count", mean="mean", sd="std").reset_index()
    agg["se"] = agg["sd"] / np.sqrt(agg["n"])
    return agg


def national_deviation(window_df: pd.DataFrame) -> pd.DataFrame:
    """Mean y across ALL constituencies per k, minus that same national
    series' mean over the whole window -- centred on zero, no anchor month."""
    by_k = window_df.groupby("k", as_index=False)["y"].mean().rename(columns={"y": "national_mean"})
    by_k["national_dev"] = by_k["national_mean"] - by_k["national_mean"].mean()
    return by_k[["k", "national_dev"]]


def constituency_deviation(window_df: pd.DataFrame, window: int) -> tuple[pd.DataFrame, int]:
    """Within-constituency deviation dev_ck = y_ck - mean_k(y_ck) for each
    constituency's own window.

    A constituency's permanent brightness level appears identically in
    every month of its own window, so it cancels exactly in this
    subtraction -- unlike the cross-sectional spread of raw y used by
    group_stats(), which is dominated by that permanent level and is not a
    valid SE input once the series has been demeaned.

    Constituencies without the full set of 2*window+1 months in this
    window are dropped: a partial-window mean is not a fair stand-in for
    the true window mean, and keeping them would bias dev toward whichever
    months happen to be present. Returns (dev_df with con_id/k/dev,
    n_dropped).
    """
    expected_months = 2 * window + 1
    month_counts = window_df.groupby("con_id")["k"].size()
    keep_ids = month_counts.index[month_counts == expected_months]
    n_dropped = int((month_counts != expected_months).sum())

    sub = window_df[window_df["con_id"].isin(keep_ids)].copy()
    sub["dev"] = sub["y"] - sub.groupby("con_id")["y"].transform("mean")
    return sub[["con_id", "k", "dev"]], n_dropped


def dev_group_stats(dev_df: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Per (group, k): constituency count, mean within-constituency
    deviation, and SE of that mean across constituencies -- the
    within-constituency analogue of group_stats(), used for gap_dev
    instead of the level SE."""
    merged = dev_df.merge(groups, on="con_id", how="inner")
    g = merged.groupby(["group", "k"])["dev"]
    agg = g.agg(n="count", mean="mean", sd="std").reset_index()
    agg["se"] = agg["sd"] / np.sqrt(agg["n"])
    return agg


def build_cycle_gap(window_df: pd.DataFrame, groups: pd.DataFrame, cycle_label: str,
                     window: int = WINDOW, ci_mult: float = CI_MULT) -> pd.DataFrame:
    """gap / gap_se / gap_lo / gap_hi / gap_dev / gap_dev_se / gap_dev_lo /
    gap_dev_hi / national_dev for one cycle's window.

    gap is an absolute level: competitive constituencies are systematically
    dimmer than safe ones regardless of the election, so gap sits at a
    roughly constant offset in every window. gap_dev removes that constant
    cross-sectional offset by subtracting the window's own mean gap, the
    same treatment national_dev already gets -- so a chart of gap_dev shows
    only within-window movement, not the permanent level difference.

    gap_se and gap_dev_se are DIFFERENT quantities, deliberately: gap_se is
    the SE of a difference of group mean LEVELS, driven by how much
    constituencies differ in permanent brightness (huge, since that's a
    cross-sectional comparison). gap_dev_se is the SE of a difference of
    group mean WITHIN-constituency deviations, where permanent brightness
    has already cancelled out per constituency -- so it reflects only
    genuine month-to-month movement. Applying gap_se's level-driven spread
    to gap_dev would overstate its uncertainty by an order of magnitude.
    """
    stats = group_stats(window_df, groups)
    comp = stats[stats["group"] == "competitive"].set_index("k")
    safe = stats[stats["group"] == "safe"].set_index("k")

    all_k = sorted(set(comp.index) | set(safe.index))
    out = pd.DataFrame({"k": all_k}).set_index("k")
    out["gap"] = comp["mean"] - safe["mean"]
    out["gap_se"] = np.sqrt(comp["se"] ** 2 + safe["se"] ** 2)
    out["gap_lo"] = out["gap"] - ci_mult * out["gap_se"]
    out["gap_hi"] = out["gap"] + ci_mult * out["gap_se"]

    dev_df, n_dropped = constituency_deviation(window_df, window)
    n_total = window_df["con_id"].nunique()
    print(f"{cycle_label}: dropped {n_dropped} / {n_total} constituencies "
          f"with incomplete windows before computing gap_dev SE")
    dev_stats = dev_group_stats(dev_df, groups)
    comp_dev = dev_stats[dev_stats["group"] == "competitive"].set_index("k")
    safe_dev = dev_stats[dev_stats["group"] == "safe"].set_index("k")
    out["gap_dev"] = comp_dev["mean"] - safe_dev["mean"]
    out["gap_dev_se"] = np.sqrt(comp_dev["se"] ** 2 + safe_dev["se"] ** 2)
    out["gap_dev_lo"] = out["gap_dev"] - ci_mult * out["gap_dev_se"]
    out["gap_dev_hi"] = out["gap_dev"] + ci_mult * out["gap_dev_se"]
    out["n_competitive"] = comp["n"]
    out["n_safe"] = safe["n"]
    out = out.reset_index()

    nat = national_deviation(window_df)
    out = out.merge(nat, on="k", how="left")

    out.insert(0, "cycle", cycle_label)
    out["n_competitive"] = out["n_competitive"].astype(int)
    out["n_safe"] = out["n_safe"].astype(int)
    return out[["cycle", "k", "gap", "gap_se", "gap_lo", "gap_hi",
                "gap_dev", "gap_dev_se", "gap_dev_lo", "gap_dev_hi", "national_dev",
                "n_competitive", "n_safe"]]


def build_per_election_gap(panel_with_region: pd.DataFrame, groups: pd.DataFrame,
                            real_cycles=REAL_CYCLES, placebo_year: int = PLACEBO_YEAR,
                            window: int = WINDOW) -> pd.DataFrame:
    """Full output table: one row per (cycle, k), sorted by cycle then k."""
    windows = build_windows(panel_with_region, real_cycles, placebo_year, window)
    rows = [build_cycle_gap(df, groups, label, window=window) for label, df in windows.items()]
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["cycle", "k"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Sanity / summary report
# --------------------------------------------------------------------------- #
def sanity_report(series: pd.DataFrame, groups: pd.DataFrame) -> None:
    print("=== row count ===")
    print(f"{len(series)} rows total")

    print("\n=== n_competitive / n_safe per cycle ===")
    for cycle, sub in series.groupby("cycle", sort=False):
        n_comp = sub["n_competitive"].unique()
        n_safe = sub["n_safe"].unique()
        print(f"{cycle}: n_competitive={n_comp.tolist()}, n_safe={n_safe.tolist()}")

    print("\n=== group sizes (fixed, from region_matched groups) ===")
    print(groups.groupby("group")["con_id"].nunique().to_dict())

    print("\n=== mean gap: pre (k=-6..-1) vs post (k=0..6), and their difference ===")
    for cycle, sub in series.groupby("cycle", sort=False):
        pre = sub.loc[sub["k"].between(-6, -1), "gap"].mean()
        post = sub.loc[sub["k"].between(0, 6), "gap"].mean()
        print(f"{cycle}: pre={pre:.4f}, post={post:.4f}, diff={post - pre:.4f}")

    print("\n=== months (of 13) where the gap CI excludes zero ===")
    for cycle, sub in series.groupby("cycle", sort=False):
        excludes_zero = ((sub["gap_lo"] > 0) | (sub["gap_hi"] < 0)).sum()
        print(f"{cycle}: {excludes_zero} / {len(sub)}")

    print("\n=== gap_dev range, and months (of 13) where the gap_dev CI excludes zero ===")
    for cycle, sub in series.groupby("cycle", sort=False):
        rng = sub["gap_dev"].max() - sub["gap_dev"].min()
        excludes_zero = ((sub["gap_dev_lo"] > 0) | (sub["gap_dev_hi"] < 0)).sum()
        print(f"{cycle}: range={rng:.4f}, excludes_zero={excludes_zero} / {len(sub)}")

    print("\n=== gap_dev vs the old level-demeaning formula gap - mean_k(gap) ===")
    print("(demeaning is linear, so these should agree to floating-point tolerance,")
    print("modulo any constituencies dropped for an incomplete window)")
    for cycle, sub in series.groupby("cycle", sort=False):
        old_style_gap_dev = sub["gap"] - sub["gap"].mean()
        max_abs_diff = (sub["gap_dev"] - old_style_gap_dev).abs().max()
        print(f"{cycle}: max_abs_diff={max_abs_diff:.8f}")

    print("\n=== mean gap_se (old, level) vs mean gap_dev_se (new, within-constituency) ===")
    for cycle, sub in series.groupby("cycle", sort=False):
        print(f"{cycle}: gap_se={sub['gap_se'].mean():.4f}, "
              f"gap_dev_se={sub['gap_dev_se'].mean():.4f}")

    print("\n=== range (max - min) of national_dev per cycle ===")
    for cycle, sub in series.groupby("cycle", sort=False):
        rng = sub["national_dev"].max() - sub["national_dev"].min()
        print(f"{cycle}: {rng:.4f}")

    print("\n=== 2012 window population check (panel starts 2012-04) ===")
    sub_2012 = series[series["cycle"] == "2012"]
    print(f"2012: {len(sub_2012)} rows (expect 13), k range "
          f"{sub_2012['k'].min()}..{sub_2012['k'].max()}, "
          f"any NaNs: {sub_2012.isna().any().any()}")


# --------------------------------------------------------------------------- #
# Figure: per-election gap_dev, four cycles, national_dev strip above each
# --------------------------------------------------------------------------- #
def select_cycle(df: pd.DataFrame, cycle: str) -> pd.DataFrame:
    """Filter to one cycle, sorted by event time k."""
    return df[df["cycle"] == cycle].sort_values("k").copy()


def plot_per_election(df: pd.DataFrame, out_path: Path,
                       suptitle: str | None = None,
                       footnote: str | None = None) -> None:
    """Draw the four-cycle gap_dev figure (with a national_dev strip above
    each panel) and save it to out_path.

    df must already contain the gap-table columns built by
    build_per_election_gap(). One column per cycle in CYCLE_ORDER; two rows
    per column via gridspec (a thin national_dev strip on its own y-scale,
    and a taller gap_dev panel sharing one y-scale across all four cycles).
    """
    if suptitle is None:
        suptitle = DEFAULT_SUPTITLE
    if footnote is None:
        footnote = DEFAULT_FOOTNOTE

    n_cols = len(CYCLE_ORDER)
    fig = plt.figure(figsize=(14, 6.5))
    gs = GridSpec(2, n_cols, figure=fig, height_ratios=[1, 4],
                  hspace=0.15, wspace=0.08)

    top_axes: list = []
    bottom_axes: list = []

    for i, cycle in enumerate(CYCLE_ORDER):
        sub = select_cycle(df, cycle)
        is_placebo = cycle == PLACEBO_CYCLE

        top_ax = fig.add_subplot(gs[0, i], sharey=top_axes[0] if top_axes else None)
        bottom_ax = fig.add_subplot(gs[1, i], sharex=top_ax,
                                     sharey=bottom_axes[0] if bottom_axes else None)
        top_axes.append(top_ax)
        bottom_axes.append(bottom_ax)

        # --- top strip: national_dev, own y-scale, shared across columns ---
        top_ax.plot(sub["k"], sub["national_dev"], color=NATIONAL_COLOR,
                    linewidth=1.3, zorder=3)
        top_ax.spines["top"].set_visible(False)
        top_ax.spines["right"].set_visible(False)
        top_ax.grid(False)
        top_ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=2))
        top_ax.xaxis.set_major_locator(mticker.MultipleLocator(3))
        top_ax.tick_params(axis="both", labelsize=7)
        plt.setp(top_ax.get_xticklabels(), visible=False)

        title_color = PLACEBO_TITLE_COLOR if is_placebo else "black"
        top_ax.set_title(CYCLE_TITLES[cycle], fontsize=12, color=title_color,
                         fontweight="bold" if is_placebo else "normal")
        if is_placebo:
            top_ax.set_facecolor(PLACEBO_BG_COLOR)
            bottom_ax.set_facecolor(PLACEBO_BG_COLOR)

        # --- bottom panel: gap_dev, coloured line + 95% band ---
        bottom_ax.axhline(0, color="black", linestyle="-", linewidth=1.6, zorder=2)
        bottom_ax.axvline(0, color="grey", linestyle="--", linewidth=1.1, zorder=2)
        bottom_ax.plot(sub["k"], sub["gap_dev"], color=GAP_COLOR, linewidth=2, zorder=4)
        bottom_ax.fill_between(sub["k"], sub["gap_dev_lo"], sub["gap_dev_hi"],
                               color=GAP_COLOR, alpha=0.2, linewidth=0, zorder=3)
        bottom_ax.spines["top"].set_visible(False)
        bottom_ax.spines["right"].set_visible(False)
        bottom_ax.grid(False)
        bottom_ax.xaxis.set_major_locator(mticker.MultipleLocator(3))
        bottom_ax.tick_params(axis="both", labelsize=8)

        if i == 0:
            top_ax.set_ylabel(Y_LABEL_TOP, fontsize=8, labelpad=8)
            # Wrap: a single long rotated label is centred on the axes and,
            # at full length, overflows past the panel's top edge into the
            # strip above it. Wrapping to a few short lines keeps its
            # rotated extent within the panel's own height instead.
            wrapped_ylabel = "\n".join(textwrap.wrap(Y_LABEL_MAIN, width=30))
            bottom_ax.set_ylabel(wrapped_ylabel, fontsize=9, linespacing=1.4)
        else:
            plt.setp(top_ax.get_yticklabels(), visible=False)
            plt.setp(bottom_ax.get_yticklabels(), visible=False)

    # Wrap the suptitle so an unexpectedly long conclusion sentence breaks
    # onto extra lines instead of overflowing past the canvas edge.
    wrapped_suptitle = textwrap.fill(suptitle, width=100)
    suptitle_lines = wrapped_suptitle.count("\n") + 1
    axes_top = 0.86 - 0.04 * (suptitle_lines - 1)

    fig.suptitle(wrapped_suptitle, fontsize=13, y=0.97)

    # Stack the footnote and the shared x-axis label from the bottom up, so
    # a long footnote pushes the label clear instead of colliding with it
    # (this has bitten this repo's other figures twice already).
    wrapped_footnote = textwrap.fill(footnote, width=150)
    footnote_lines = wrapped_footnote.count("\n") + 1
    footnote_y = 0.012
    footnote_top = footnote_y + footnote_lines * 0.024
    x_label_y = footnote_top + 0.02

    fig.text(0.5, x_label_y, X_LABEL, ha="center", va="bottom", fontsize=10)
    fig.text(0.5, footnote_y, wrapped_footnote,
              ha="center", va="bottom", fontsize=7.5, style="italic",
              color="grey", linespacing=1.5)

    bottom_margin = x_label_y + 0.055
    fig.subplots_adjust(left=0.08, right=0.98, top=axes_top,
                         bottom=bottom_margin, hspace=0.15, wspace=0.08)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main() -> None:
    root = Path(__file__).resolve().parents[2]
    data_path = root / "data" / "elections" / "panel_monthly_with_elections.csv"
    out_dir = root / "result" / "election_impact"
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = tls.load_panel_with_region(data_path)
    groups = group_lookup(panel)
    series = build_per_election_gap(panel, groups)

    csv_path = out_dir / "per_election_gap.csv"
    series.to_csv(csv_path, index=False)
    print(f"wrote {len(series)} rows to {csv_path}")

    sanity_report(series, groups)

    png_path = out_dir / "per_election_gap.png"
    plot_per_election(series, png_path)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
