"""
The three electricity-access measures the client's brief lists, put on one page.

Same specification for all three: month-of-year dummies absorbing constituency x
year fixed effects, fitted outside +/-6 month election windows, VCMSLCFG era.

The point of the figure is that they disagree.
"""
import os
import numpy as np
import pandas as pd
import numpy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)) or "."
OUT = os.path.join(HERE, "figs")

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
CRITICAL = "#d03b3b"
M = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": BASELINE, "font.size": 10,
})


def style(ax, ylabel=None, title=None, sub=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
        ax.spines[s].set_linewidth(0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9.5)
    n = (sub.count("\n") + 1) if sub else 0
    if title:
        ax.set_title(title, color=INK, fontsize=12, fontweight="600",
                     loc="left", pad=8 + 15 * n)
    if sub:
        ax.text(0, 1.012, sub, transform=ax.transAxes, color=INK_2,
                fontsize=9, va="bottom", linespacing=1.35)


d = pd.read_csv(os.path.join(HERE, "out", "panel_deseasonalized.csv"))
d = d[d.usable].copy()
d = d[d.bm_mean_rad.notna() & (d.bm_mean_rad > 0)]
d["uy"] = d.cons_name + "_" + d.year.astype(str)
d["lbm"] = np.log(d.bm_mean_rad.clip(lower=0.01))
ELEC = [(2012, 12), (2016, 12), (2020, 12), (2024, 12)]
d["win"] = [any(abs(y * 12 + m - (a * 12 + b)) <= 6 for a, b in ELEC)
            for y, m in zip(d.year, d.month)]


def prof(df, yv):
    df = df[df[yv].notna()]
    X = pd.get_dummies(df.month, prefix="m").astype(float)
    X = X.sub(X.groupby(df.uy.values).transform("mean"))
    y = (df[yv] - df.groupby("uy")[yv].transform("mean")).values
    b = la.lstsq(X.values, y, rcond=None)[0]
    return b - b.mean()


f = d[~d.win]
A = prof(f, "y") * 100
B = prof(f, "lbm") * 100
C = prof(f, "lit_share_0p5") * 100
N = prof(f, "nights_valid")

fig, axes = plt.subplots(3, 1, figsize=(8.6, 10.2), sharex=True,
                         gridspec_kw={"hspace": 0.42})
fig.subplots_adjust(top=0.865)

specs = [
    (A, S1, "Intensity — VCMSLCFG monthly composite",
     "log mean radiance, the series used so far", "deviation (%)"),
    (B, S2, "Intensity — Black Marble, high-quality nights only",
     "same quantity, but only nights with a genuine clean retrieval", "deviation (%)"),
    (C, S3, "Access — share of pixel-nights lit (threshold 0.5)",
     "the client brief's \"on/off frequency\"", "deviation (pp)"),
]
for ax, (v, col, title, sub, ylab) in zip(axes, specs):
    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.plot(range(12), v, color=col, linewidth=2, zorder=3)
    ax.scatter(range(12), v, s=30, color=col, zorder=4,
               edgecolor=SURFACE, linewidth=1.5)
    style(ax, ylabel=ylab, title=title, sub=sub)
    amp = v.max() - v.min()
    ax.text(0.995, 0.06, f"amplitude {amp:.1f}", transform=ax.transAxes,
            ha="right", fontsize=9.5, color=INK_2, fontweight="600")

# call out July on the top two panels
axes[0].annotate(f"{A[6]:+.0f}%", (6, A[6]), textcoords="offset points",
                 xytext=(0, -20), ha="center", fontsize=10, color=CRITICAL,
                 fontweight="700")
axes[0].annotate("July", (6, A[6]), textcoords="offset points",
                 xytext=(0, -34), ha="center", fontsize=8.5, color=MUTED)
axes[1].annotate(f"{B[6]:+.0f}% — the trough is gone", (6, B[6]),
                 textcoords="offset points", xytext=(0, -30), ha="center",
                 fontsize=9.5, color=CRITICAL, fontweight="700")
axes[1].set_ylim(min(B) - 12, max(B) + 8)
axes[0].set_ylim(min(A) - 22, max(A) + 12)

axes[2].set_xticks(range(12))
axes[2].set_xticklabels(M)

fig.text(0.09, 0.985,
         "Three measures of the same thing, three different seasonal patterns",
         color=INK, fontsize=14, fontweight="650", ha="left", va="top")
fig.text(0.09, 0.945,
         "Identical specification throughout: month dummies absorbing constituency × year "
         "fixed effects,\nfitted outside ±6-month election windows, 2014–2024. "
         f"Profile correlations: composite↔Black Marble {np.corrcoef(A, B)[0, 1]:+.2f}, "
         f"composite↔access {np.corrcoef(A, C)[0, 1]:+.2f}.",
         color=INK_2, fontsize=9.5, ha="left", va="top", linespacing=1.45)

p = os.path.join(OUT, "fig10_three_measures.png")
fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)
print("wrote", p)

# ---- second figure: valid nights, the mechanism candidate
fig, ax = plt.subplots(figsize=(8.6, 4.2))
ax.axhline(0, color=BASELINE, linewidth=1)
ax.bar(range(12), N, color=[CRITICAL if i in (6, 7, 8) else "#c3c2b7"
                            for i in range(12)], width=0.62, zorder=3)
ax.set_xticks(range(12))
ax.set_xticklabels(M)
style(ax, ylabel="deviation from the constituency's own year (nights)",
      title="Why the two intensity measures disagree: how many nights survive quality control",
      sub="Black Marble accepts 7.6 nights in July against 21.6 in December. The monthly composite\n"
          "keeps marginal nights that Black Marble rejects — and July has the most of them.")
p = os.path.join(OUT, "fig11_valid_nights.png")
fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)
print("wrote", p)
