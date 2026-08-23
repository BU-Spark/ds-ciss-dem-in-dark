#!/usr/bin/env python3
"""A2: Close-election alignment RD. Spec: specs.md A2 + addendum (cycles 2016, 2020)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from analysis_utils import style_ax, OI

uc = pd.read_pickle("unit_cycle.pkl")
uc = uc.dropna(subset=["dy", "ruling_margin"]).copy()
uc["x"] = uc.ruling_margin
uc["D"] = (uc.x > 0).astype(float)
uc["Dx"] = uc.D * uc.x

def rd(data, h):
    s = data[data.x.abs() <= h].copy()
    s["w"] = 1 - (s.x.abs() / h)                      # triangular kernel
    m = smf.wls("dy ~ D + x + Dx", data=s, weights=s.w).fit(
        cov_type="cluster", cov_kwds={"groups": s.con_id})
    return dict(h=h, b=m.params["D"], se=m.bse["D"], p=m.pvalues["D"],
                n=int(m.nobs), n_left=int((s.x < 0).sum()), n_right=int((s.x > 0).sum()))

res = pd.DataFrame([rd(uc, h) for h in [0.05, 0.10, 0.15]])
# balance: baseline lights at cutoff (h=0.10)
s = uc[uc.x.abs() <= 0.10].copy(); s["w"] = 1 - s.x.abs()/0.10
bal = smf.wls("y_pre ~ D + x + Dx", data=s, weights=s.w).fit(
    cov_type="cluster", cov_kwds={"groups": s.con_id})
res = pd.concat([res, pd.DataFrame([dict(h=0.10, b=bal.params["D"], se=bal.bse["D"],
      p=bal.pvalues["D"], n=int(bal.nobs), n_left=np.nan, n_right=np.nan)])], ignore_index=True)
res["spec"] = ["dy, h=5pp", "dy, h=10pp", "dy, h=15pp", "balance: y_pre, h=10pp"]
res.to_csv("tables/a2_rd.csv", index=False)

# density counts near cutoff
dens = uc[uc.x.abs() <= 0.04].groupby(uc.x > 0).size()

# figure: binned scatter + fits (h=0.15)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200, gridspec_kw={"width_ratios":[3,2]})
ax = axes[0]
sub = uc[uc.x.abs() <= 0.30]
bins = np.arange(-0.30, 0.301, 0.025)
sub = sub.assign(bin=pd.cut(sub.x, bins))
bs = sub.groupby("bin", observed=True).agg(x=("x","mean"), y=("dy","mean"), n=("dy","size"))
ax.axvline(0, color="#999999", lw=0.9, ls=":")
ax.scatter(uc.x, uc.dy, s=7, color=OI["gray"], alpha=0.25, lw=0)
ax.scatter(bs.x, bs.y, s=np.sqrt(bs.n)*9, color=OI["blue"], zorder=3, edgecolor="white", lw=1)
h = 0.15
for side, mcol in [(uc.x.between(-h, 0, inclusive="left"), OI["verm"]), (uc.x.between(0, h), OI["verm"])]:
    ss = uc[side & (uc.x.abs() <= h)].copy()
    if len(ss) < 5: continue
    ss["w"] = 1 - ss.x.abs()/h
    f = np.polyfit(ss.x, ss.dy, 1, w=ss.w)
    xs = np.linspace(ss.x.min(), ss.x.max(), 20)
    ax.plot(xs, np.polyval(f, xs), color=mcol, lw=2)
ax.set_xlabel("Ruling-party margin at election (x, pp)", fontsize=9)
ax.set_ylabel("Change in deseason. log radiance\n(post +7..+42 minus pre -18..-1)", fontsize=9)
ax.set_title("Alignment RD, cycles 2016 + 2020 pooled", fontsize=10, loc="left")
style_ax(ax)
ax = axes[1]
ax.axhline(0, color="#999999", lw=0.8)
main3 = res.iloc[:3]
ax.errorbar([5,10,15], main3.b, yerr=1.96*main3.se, fmt="o", color=OI["blue"], ms=6,
            mec="white", mew=1, capsize=3, lw=1.5)
for xx, (b, n) in zip([5,10,15], zip(main3.b, main3.n)):
    ax.annotate(f"n={n}", (xx, b), textcoords="offset points", xytext=(8, -3), fontsize=8, color="#666")
ax.set_xticks([5,10,15]); ax.set_xlim(2, 18)
ax.set_xlabel("Bandwidth (pp)", fontsize=9)
ax.set_title("RD estimate of ruling-party win", fontsize=10, loc="left")
style_ax(ax)
fig.tight_layout()
fig.savefig("figs/a2_rd.png")

print(res.round(4).to_string(index=False))
print("\ndensity near cutoff (|x|<=4pp): left(False)/right(True):"); print(dens)
