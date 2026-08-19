"""Shared regression helpers: OLS with dummy FE and constituency-clustered SE."""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

OI = dict(blue="#0072B2", verm="#D55E00", green="#009E73", gray="#808080",
          ink="#1a1a1a", grid="#d9d9d9")

def fe_ols(formula, data, cluster="con_id"):
    m0 = smf.ols(formula, data=data)
    used = m0.data.row_labels                     # rows kept after NaN handling
    groups = data.loc[used, cluster]
    return m0.fit(cov_type="cluster", cov_kwds={"groups": groups})

def coef_frame(model, pattern):
    idx = [i for i in model.params.index if pattern in i]
    return pd.DataFrame({"term": idx, "b": model.params[idx].values,
                         "se": model.bse[idx].values, "p": model.pvalues[idx].values})

def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#999999")
    ax.grid(axis="y", color=OI["grid"], lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#444444", labelsize=9)
