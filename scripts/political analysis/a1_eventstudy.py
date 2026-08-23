#!/usr/bin/env python3
"""A1: Election event-study on deseasonalized, dust-adjusted night lights.
Spec: specs.md A1 + addendum. Windows: 2016, 2020 full; 2024 pre-side."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis_utils import fe_ols, coef_frame, style_ax, OI

df = pd.read_pickle("analysis_panel.pkl")
d = df[df.usable].copy()

# evt dummies, reference = -6
for e in range(-6, 7):
    d[f"evt_m{abs(e)}" if e < 0 else f"evt_p{e}"] = (d.evt_a == e).astype(float)
evt_terms = [f"evt_m{k}" for k in range(5, 0, -1)] + ["evt_p0"] + [f"evt_p{k}" for k in range(1, 7)]

Y = "y_ds_cluster_dustadj"
f_main = f"{Y} ~ " + " + ".join(evt_terms) + " + C(con_id) + C(month) + C(year)"
m_main = fe_ols(f_main, d)

rows = []
order = list(range(-6, 7))
labels = [f"evt_m{abs(e)}" if e < 0 else f"evt_p{e}" for e in order]
for e, t in zip(order, labels):
    if e == -6:
        rows.append(dict(evt=e, b=0.0, se=0.0)); continue
    rows.append(dict(evt=e, b=m_main.params[t], se=m_main.bse[t]))
res = pd.DataFrame(rows)
res["lo"] = res.b - 1.96 * res.se; res["hi"] = res.b + 1.96 * res.se
res.to_csv("tables/a1_eventstudy_coefs.csv", index=False)

# pooled pre4 + heterogeneity
d["pre4_swing"] = d.pre4 * d.swing_rel
d["pre4_aligned"] = d.pre4 * d.aligned
m_pre  = fe_ols(f"{Y} ~ pre4 + C(con_id) + C(month) + C(year)", d)
m_het  = fe_ols(f"{Y} ~ pre4 + pre4_swing + swing_rel + C(con_id) + C(month) + C(year)", d)
m_het2 = fe_ols(f"{Y} ~ pre4 + pre4_aligned + aligned + C(con_id) + C(month) + C(year)", d.dropna(subset=["aligned"]))
# robustness: y_ds_unit outcome; weighted all-months
m_rob  = fe_ols("y_ds_unit ~ pre4 + C(con_id) + C(month) + C(year)", d)

def line(m, term, label, n):
    return dict(model=label, term=term, b=m.params[term], se=m.bse[term], p=m.pvalues[term], n=n)
tab = pd.DataFrame([
    line(m_pre, "pre4", "main: pre-election Sep-Dec", int(m_pre.nobs)),
    line(m_het, "pre4", "het: pre4 (safe seats)", int(m_het.nobs)),
    line(m_het, "pre4_swing", "het: + swing (<10pp)", int(m_het.nobs)),
    line(m_het2, "pre4", "het: pre4 (misaligned)", int(m_het2.nobs)),
    line(m_het2, "pre4_aligned", "het: + aligned w/ ruling party", int(m_het2.nobs)),
    line(m_rob, "pre4", "robust: y_ds_unit outcome", int(m_rob.nobs)),
])
tab.to_csv("tables/a1_pre4_heterogeneity.csv", index=False)

# ---- figure
fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
ax.axhline(0, color="#999999", lw=0.8)
ax.axvline(0, color=OI["gray"], lw=0.8, ls=":")
ax.fill_between(res.evt, res.lo, res.hi, color=OI["blue"], alpha=0.15, lw=0)
ax.plot(res.evt, res.b, color=OI["blue"], lw=2, marker="o", ms=5,
        mfc=OI["blue"], mec="white", mew=1)
ax.annotate("election\n(December)", xy=(0, ax.get_ylim()[1]*0.02), fontsize=8,
            color="#666666", ha="left", xytext=(0.15, res.hi.max()*0.9))
ax.set_xlabel("Months relative to election (Dec = 0)", fontsize=10)
ax.set_ylabel("Deseasonalized log radiance (dust-adj.)", fontsize=10)
ax.set_title("Night lights around elections: 2016, 2020 (full) + 2024 (pre only)",
             fontsize=11, loc="left", color=OI["ink"])
ax.set_xticks(range(-6, 7))
style_ax(ax)
fig.text(0.01, 0.01, "Unit, calendar-month and year FE; 95% CI, SE clustered by constituency (275). Ref. month = -6.",
         fontsize=7.5, color="#777777")
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig("figs/a1_eventstudy.png")

print(res.round(4).to_string(index=False))
print(); print(tab.round(4).to_string(index=False))
print(); print("windows contributing: evt<=0:", sorted(d.loc[d.evt_a.le(0) & d.evt_a.notna(), "elec_rel"].unique()),
      "| evt>0:", sorted(d.loc[d.evt_a.gt(0), "elec_rel"].unique()))

# ---- election-specific pre-election effects (added after leave-one-out verification
#      showed the pooled pre4 is unstable across elections; see MEMO) ----
for e in [2016, 2020, 2024]:
    d[f"pre4_{e}"] = d.pre4 * (d.next_elec == e) * (d.year == e)
m_by = fe_ols(f"{Y} ~ pre4_2016 + pre4_2020 + pre4_2024 + C(con_id) + C(month) + C(year)", d)
by = pd.DataFrame([dict(election=e, b=m_by.params[f"pre4_{e}"], se=m_by.bse[f"pre4_{e}"],
                        p=m_by.pvalues[f"pre4_{e}"]) for e in [2016, 2020, 2024]])
by.to_csv("tables/a1_pre4_by_election.csv", index=False)
print(); print("per-election pre4:"); print(by.round(4).to_string(index=False))

# amend figure: add per-election panel
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200, gridspec_kw={"width_ratios": [3, 1.6]})
ax = axes[0]
ax.axhline(0, color="#999999", lw=0.8); ax.axvline(0, color=OI["gray"], lw=0.8, ls=":")
ax.fill_between(res.evt, res.lo, res.hi, color=OI["blue"], alpha=0.15, lw=0)
ax.plot(res.evt, res.b, color=OI["blue"], lw=2, marker="o", ms=5, mfc=OI["blue"], mec="white", mew=1)
ax.set_xlabel("Months relative to election (Dec = 0)", fontsize=10)
ax.set_ylabel("Deseasonalized log radiance (dust-adj.)", fontsize=10)
ax.set_title("Pooled event-study: 2016, 2020 (full) + 2024 (pre only)", fontsize=10.5, loc="left")
ax.set_xticks(range(-6, 7)); style_ax(ax)
ax = axes[1]
ax.axhline(0, color="#999999", lw=0.8)
ax.errorbar([2016, 2020, 2024], by.b, yerr=1.96*by.se, fmt="o", color=OI["verm"],
            ms=7, mec="white", mew=1, capsize=3, lw=1.5)
ax.set_xticks([2016, 2020, 2024])
ax.set_title("Pre-election (Sep-Dec) effect,\nby election", fontsize=10.5, loc="left")
style_ax(ax)
fig.text(0.01, 0.01, "Unit, calendar-month, year FE; 95% CI, SE clustered by constituency. Pooled ref. month = -6.",
         fontsize=7.5, color="#777777")
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig("figs/a1_eventstudy.png")
