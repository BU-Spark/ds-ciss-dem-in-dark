#!/usr/bin/env python3
"""A4: Political incidence of the dry-season burden + dumsor DID. Spec: specs.md A4 + addendum."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis_utils import fe_ols, style_ax, OI

df = pd.read_pickle("analysis_panel.pkl")

# ---------- (a) cross-section: seasonal amplitude vs politics ----------
prof = df[df.usable].groupby(["con_id", "month"]).s_unit.mean().unstack()
amp = (prof.max(axis=1) - prof.min(axis=1)).rename("amplitude")
dry = prof[[11, 12, 1, 2, 3]].mean(axis=1) - prof[[5, 6, 7, 8, 9]].mean(axis=1)
dry = dry.rename("dry_minus_wet")
pol = df[df.usable].groupby("con_id").agg(
    mean_npp=("npp_share_last", "mean"),
    mean_absmargin=("margin_last", lambda s: s.abs().mean()),
    logbright=("y", "mean"), zone3=("zone3", "first"))
cs = pol.join(amp).join(dry).dropna()
cs["con_id"] = cs.index

m_amp = fe_ols("amplitude ~ mean_npp + mean_absmargin + logbright + C(zone3)", cs)
m_dry = fe_ols("dry_minus_wet ~ mean_npp + mean_absmargin + logbright + C(zone3)", cs)
rows = []
for m, dv in [(m_amp, "amplitude"), (m_dry, "dry_minus_wet")]:
    for t in ["mean_npp", "mean_absmargin", "logbright"]:
        rows.append(dict(outcome=dv, term=t, b=m.params[t], se=m.bse[t], p=m.pvalues[t], n=int(m.nobs)))
pd.DataFrame(rows).to_csv("tables/a4_amplitude_crosssection.csv", index=False)

# ---------- (b) dumsor DID ----------
d = df[df.usable & (df.ym >= 2014*12+1) & (df.ym <= 2016*12+9)].copy()
d["opp_dumsor"] = d.opp2012 * d.dumsor
m_did = fe_ols("y_ds_cluster_dustadj ~ opp_dumsor + C(con_id) + C(ym)", d.dropna(subset=["opp2012"]))

# dynamic: opp2012 x year, 2014-2019, base 2018, excluding Oct-Dec of election years
dd = df[df.usable & df.year.between(2014, 2019)].copy()
dd = dd[~((dd.year.isin([2016])) & (dd.month >= 10))]
dd = dd.dropna(subset=["opp2012"])
for yy in [2014, 2015, 2016, 2017, 2019]:
    dd[f"oppXy{yy}"] = dd.opp2012 * (dd.year == yy)
terms = [f"oppXy{y}" for y in [2014, 2015, 2016, 2017, 2019]]
m_dyn = fe_ols("y_ds_cluster_dustadj ~ " + " + ".join(terms) + " + C(con_id) + C(ym)", dd)
dyn = pd.DataFrame([dict(year=int(t[-4:]), b=m_dyn.params[t], se=m_dyn.bse[t]) for t in terms] +
                   [dict(year=2018, b=0.0, se=0.0)]).sort_values("year")
dyn["lo"] = dyn.b - 1.96*dyn.se; dyn["hi"] = dyn.b + 1.96*dyn.se
pd.concat([
    pd.DataFrame([dict(model="static DID 2014m1-2016m9", term="opp2012 x dumsor(2014-15)",
                       b=m_did.params["opp_dumsor"], se=m_did.bse["opp_dumsor"],
                       p=m_did.pvalues["opp_dumsor"], n=int(m_did.nobs))]),
    dyn.assign(model="dynamic, base 2018", term=lambda x: "opp2012 x " + x.year.astype(str),
               p=np.nan, n=int(m_dyn.nobs))[["model","term","b","se","p","n"]],
]).to_csv("tables/a4_dumsor_did.csv", index=False)

# ---------- figure: two panels ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)
ax = axes[0]
q = pd.qcut(cs.mean_npp, 12)
bs = cs.groupby(q, observed=True).agg(x=("mean_npp","mean"), y=("amplitude","mean"))
ax.scatter(cs.mean_npp, cs.amplitude, s=8, color=OI["gray"], alpha=0.35, lw=0)
ax.scatter(bs.x, bs.y, s=42, color=OI["blue"], zorder=3, edgecolor="white", lw=1)
b, se = m_amp.params["mean_npp"], m_amp.bse["mean_npp"]
ax.set_title(f"Seasonal amplitude vs NPP vote share\nconditional slope {b:+.3f} (se {se:.3f}), zone+brightness controls",
             fontsize=9.5, loc="left")
ax.set_xlabel("Mean NPP share (last election, 2014-24 avg)", fontsize=9)
ax.set_ylabel("Unit seasonal amplitude (log points)", fontsize=9)
style_ax(ax)

ax = axes[1]
ax.axhline(0, color="#999999", lw=0.8)
ax.axvspan(2013.7, 2015.3, color=OI["verm"], alpha=0.08, lw=0)
ax.text(2014.5, 0.9, "dumsor\npeak", ha="center", fontsize=8, color=OI["verm"])
ax.errorbar(dyn.year, dyn.b, yerr=1.96*dyn.se, fmt="o-", color=OI["verm"], lw=1.8,
            ms=5, mec="white", mew=1, capsize=2)
ax.set_title("NPP-won (opposition) constituencies x year\nvs 2018 baseline; NPP took power Jan 2017",
             fontsize=9.5, loc="left")
ax.set_xlabel("Year", fontsize=9)
ax.set_ylabel("Deseason. log radiance gap", fontsize=9)
ax.set_ylim(min(dyn.lo.min()*1.4, -0.05), max(dyn.hi.max()*1.4, 0.05))
style_ax(ax)
fig.tight_layout()
fig.savefig("figs/a4_seasonal_burden.png")

print("cross-section:"); print(pd.DataFrame(rows).round(4).to_string(index=False))
print(); print(f"static DID opp2012 x dumsor: b={m_did.params['opp_dumsor']:.4f} se={m_did.bse['opp_dumsor']:.4f} p={m_did.pvalues['opp_dumsor']:.4f} n={int(m_did.nobs)}")
print(); print("dynamic:"); print(dyn.round(4).to_string(index=False))
