#!/usr/bin/env python3
"""A5: Mechanism decomposition — extensive vs intensive vs frequency. Spec: specs.md A5."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis_utils import fe_ols, style_ax, OI

df = pd.read_pickle("analysis_panel.pkl")
d = df[df.usable].copy()

OUTCOMES = [("lit_share_0p5", "Extensive: share of pixels lit"),
            ("log_avg_rad", "Intensive: log mean radiance"),
            ("nights_lit_frac", "Frequency: lit nights / valid nights")]

rows = []
for var, lab in OUTCOMES:
    sd = d[var].std()
    for sample, slab in [(d, "all"), (d[d.urban == 1], "urban"), (d[d.urban == 0], "non-urban")]:
        m = fe_ols(f"{var} ~ pre4 + C(con_id) + C(month) + C(year)", sample)
        rows.append(dict(outcome=lab, sample=slab, b=m.params["pre4"], se=m.bse["pre4"],
                         b_sd=m.params["pre4"]/sd, se_sd=m.bse["pre4"]/sd,
                         p=m.pvalues["pre4"], n=int(m.nobs)))
tab = pd.DataFrame(rows)
tab.to_csv("tables/a5_mechanism.csv", index=False)

fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=200)
ax.axvline(0, color="#999999", lw=0.8)
ylabels, ypos = [], []
colors = {"all": OI["blue"], "urban": OI["verm"], "non-urban": OI["green"]}
yy = 0
for var, lab in OUTCOMES:
    for slab in ["all", "urban", "non-urban"]:
        r = tab[(tab.outcome == lab) & (tab["sample"] == slab)].iloc[0]
        ax.errorbar(r.b_sd, yy, xerr=1.96*r.se_sd, fmt="o", color=colors[slab],
                    ms=6, mec="white", mew=1, capsize=3, lw=1.5)
        ylabels.append(f"{lab.split(':')[0]} — {slab}"); ypos.append(yy); yy -= 1
    yy -= 0.6
ax.set_yticks(ypos); ax.set_yticklabels(ylabels, fontsize=8.5)
ax.set_xlabel("Pre-election (Sep-Dec) effect, SD units, 95% CI", fontsize=9)
ax.set_title("What moves before elections: breadth vs intensity vs frequency", fontsize=10.5, loc="left")
style_ax(ax); ax.grid(axis="x", color="#d9d9d9", lw=0.6, alpha=0.7); ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig("figs/a5_mechanism.png")
print(tab.round(4).to_string(index=False))
