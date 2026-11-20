"""Sensitivity analyses of the primate offset (Supplementary Table S4, sections C-G; Supplementary Fig. S1a) and of
the caliber - 50% attenuation distance correlation to the membrane parameters (section H; Supplementary Fig. S1b).

Writes results/sensitivity.csv and results/membrane_sweep.csv.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import optimize, stats

import pgls
from pgls import X, Y

neurons = pd.read_csv(pgls.RESULTS / "neurons.csv")
prim, other = neurons[neurons.primary], neurons[neurons.without_imaging_criterion]
sp = pgls.species_means(prim)
rows = []


def add(section, analysis, g, group="primate", fit=None):
    fit = pgls.fit_group(g, group) if fit is None else fit
    rows.append({"section": section, "analysis": analysis, "n_species": len(g), "n_neurons": int(g.n_neurons.sum()),
                 "n_in_group": int(g[group].sum()), **{k: fit.get(k) for k in
                 ("offset", "offset_ci_lo", "offset_ci_hi", "offset_p", "slope")}})


# ---- dataset and grouping ------------------------------------------------------------------------------------------
S = "dataset and grouping"
sp_other = pgls.species_means(other)
add(S, "primary dataset, primate grouping", sp)
add(S, "primary dataset, anthropoid grouping", sp, "anthropoid")
add(S, "without the imaging criterion, primate grouping", sp_other)
add(S, "without the imaging criterion, anthropoid grouping", sp_other, "anthropoid")
add(S, "primary dataset, human removed", sp[sp.species != "Human"])
add(S, "primary dataset plus elephant", pgls.species_means(pd.concat([prim, other[other.species == "Elephant"]])))

# ---- unequal precision of the species means -------------------------------------------------------------------------
S = "precision of the species means"
C = pgls.phylo_corr(sp.species.tolist())
Xm, names = pgls.design(sp, [X, "primate"])
y, se2, i = sp[Y].to_numpy(), sp.se_log10.to_numpy() ** 2, names.index("primate")


def with_sampling_variance(R):
    """GLS with V = sigma2 * R + diag(se2); sigma2 by maximum likelihood."""
    def nll(log_s2):
        V = np.exp(log_s2) * R + np.diag(se2)
        Vi = np.linalg.inv(V)
        b = np.linalg.solve(Xm.T @ Vi @ Xm, Xm.T @ Vi @ y)
        r = y - Xm @ b
        return 0.5 * (np.linalg.slogdet(V)[1] + r @ Vi @ r)
    s2 = float(np.exp(optimize.minimize_scalar(nll, bounds=(-20, 2), method="bounded").x))
    Vi = np.linalg.inv(s2 * R + np.diag(se2))
    cov = np.linalg.inv(Xm.T @ Vi @ Xm)
    b, se, df = cov @ Xm.T @ Vi @ y, np.sqrt(np.diag(cov)), len(y) - Xm.shape[1]
    tc = stats.t.ppf(0.975, df)
    return {"offset": b[i], "offset_ci_lo": b[i] - tc * se[i], "offset_ci_hi": b[i] + tc * se[i],
            "offset_p": float(2 * stats.t.sf(abs(b[i] / se[i]), df))}


for lam in pgls.fixed_lambdas():
    add(S, f"sampling variance of the species means, lambda = {lam:g}", sp, fit=with_sampling_variance(pgls.pagel(C, lam)))
d = prim.merge(sp[["species", X, "primate"]], on="species")
d["ly"] = np.log10(d.caliber_um)
mm = smf.mixedlm(f"ly ~ {X} + primate", d, groups=d.species).fit(reml=True)
ci = mm.conf_int().loc["primate"]
add(S, "reconstruction-level mixed model (species as random intercept)", sp,
    fit={"offset": float(mm.params["primate"]), "offset_ci_lo": float(ci[0]), "offset_ci_hi": float(ci[1]),
         "offset_p": float(mm.pvalues["primate"])})

# ---- cortical region and depth ----------------------------------------------------------------------------------------
S = "cortical region and depth"
for r in pd.read_csv(pgls.DATA / "region_restrictions.csv").fillna("").itertuples():
    if r.column == "":
        continue
    column = {"lobe": "region"}.get(r.column, r.column)
    g = pgls.species_means(prim[prim[column].isin(r.values.split("|"))])
    if g.primate.sum() >= 1 and (g.primate == 0).sum() >= 5:
        add(S, r.restriction.replace("_", " "), g)

# ---- caliber definition ---------------------------------------------------------------------------------------------
S = "caliber definition"
add(S, "length-weighted caliber (primary analysis)", sp)
for column, name in (("point_radius_mean_um", "sample-point mean radius"), ("point_radius_median_um", "sample-point median radius")):
    add(S, name, pgls.species_means(prim.assign(caliber_um=2 * prim[column])))

# ---- leave-out ------------------------------------------------------------------------------------------------------
S = "leave-out"
for pub in sorted(prim.publication.unique()):
    add(S, f"without publication: {pub}", pgls.species_means(prim[prim.publication != pub]))
for s in sp[sp.primate == 1].species:
    add(S, f"without primate species: {s}", sp[sp.species != s])
for o in sorted(sp.order.unique()):
    if o != "Primates":
        add(S, f"without order: {o}", sp[sp.order != o])
pd.DataFrame(rows).to_csv(pgls.RESULTS / "sensitivity.csv", index=False)

# ---- membrane parameters ----------------------------------------------------------------------------------------------
att = pd.read_csv(pgls.RESULTS / "attenuation.csv").merge(prim[["neuron_name", "species", "caliber_um"]], on="neuron_name")
labels = {"baseline": "Baseline", **pd.read_csv(pgls.DATA / "membrane_sweep.csv").set_index("setting").label.to_dict()}
out = []
for setting, label in labels.items():
    d = att.dropna(subset=[setting])
    g = d.groupby("species").agg(half=(setting, "mean"), caliber=("caliber_um", "mean"))
    r, p = stats.pearsonr(np.log10(g.caliber), np.log10(g.half))
    out.append({"setting": label, "r": r, "p": p, "n_species": len(g), "n_neurons": len(d),
                "n_without_50pct_crossing": len(att) - len(d)})
pd.DataFrame(out).to_csv(pgls.RESULTS / "membrane_sweep.csv", index=False)
print(pd.DataFrame(rows)[["analysis", "n_species", "offset"]].to_string(index=False))
