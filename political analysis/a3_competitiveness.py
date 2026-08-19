#!/usr/bin/env python3
"""A3: Competitiveness and electricity — monthly FE panel + cycle-growth. Spec: specs.md A3 + addendum."""
import numpy as np, pandas as pd
from analysis_utils import fe_ols

df = pd.read_pickle("analysis_panel.pkl")
d = df[df.usable].dropna(subset=["swing_last", "aligned"]).copy()
d["swingXaligned"] = d.swing_last * d.aligned
Y = "y_ds_cluster_dustadj"

m1 = fe_ols(f"{Y} ~ swing_last + C(con_id) + C(ym)", d)
m2 = fe_ols(f"{Y} ~ swing_last + aligned + swingXaligned + C(con_id) + C(ym)", d)
m3 = fe_ols(f"{Y} ~ swing_last_05 + aligned + C(con_id) + C(ym)", d)   # robustness: 5pp

rows = []
for m, lab, ts in [(m1, "monthly: swing only", ["swing_last"]),
                   (m2, "monthly: full", ["swing_last", "aligned", "swingXaligned"]),
                   (m3, "monthly: swing<5pp", ["swing_last_05", "aligned"])]:
    for t in ts:
        rows.append(dict(model=lab, term=t, b=m.params[t], se=m.bse[t], p=m.pvalues[t], n=int(m.nobs)))

# cycle growth
uc = pd.read_pickle("unit_cycle.pkl")
uc["absmXruling"] = uc.absmargin * uc.ruling_win
g1 = fe_ols("dy ~ absmargin + C(cycle) + C(zone3)", uc)
g2 = fe_ols("dy ~ absmargin + ruling_win + absmXruling + C(cycle) + C(zone3)", uc)
g3 = fe_ols("dy ~ absmargin + ruling_win + y_pre + C(cycle) + C(zone3)", uc)
for m, lab, ts in [(g1, "cycle dy: competitiveness", ["absmargin"]),
                   (g2, "cycle dy: + ruling win", ["absmargin", "ruling_win", "absmXruling"]),
                   (g3, "cycle dy: + baseline y", ["absmargin", "ruling_win"])]:
    for t in ts:
        rows.append(dict(model=lab, term=t, b=m.params[t], se=m.bse[t], p=m.pvalues[t], n=int(m.nobs)))
tab = pd.DataFrame(rows)
tab.to_csv("tables/a3_competitiveness.csv", index=False)
print(tab.round(4).to_string(index=False))
