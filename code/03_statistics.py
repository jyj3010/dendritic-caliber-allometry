"""Species means and the main analyses of the primary dataset.

Writes to results/: species_means.csv, composition.csv, allometry.csv, model_comparison.csv,
phylogenetic_signal.csv, primate_residuals.csv, electrotonic.csv.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

import pgls
from pgls import X, Y

neurons = pd.read_csv(pgls.RESULTS / "neurons.csv")
prim = neurons[neurons.primary]
LAMBDAS = pgls.fixed_lambdas()

# ---- species means -------------------------------------------------------------------------------------------
sp = pgls.species_means(prim)
half = prim.dropna(subset=["half_attenuation_um"])
sp = sp.merge(half.groupby("species").agg(
    half_attenuation_um=("half_attenuation_um", "mean"), half_attenuation_sd=("half_attenuation_um", "std"),
    n_half=("half_attenuation_um", "size"), caliber_half_um=("caliber_um", "mean")).reset_index(), on="species")
sp = sp.merge(prim.groupby("species").electrotonic_extent.agg(
    electrotonic_extent="mean", electrotonic_extent_sd="std").reset_index(), on="species")
sp["log10_half"], sp["log10_caliber_half"] = np.log10(sp.half_attenuation_um), np.log10(sp.caliber_half_um)
other = pgls.species_means(neurons[neurons.without_imaging_criterion])
pd.concat([sp.assign(dataset="primary"), other.assign(dataset="without imaging criterion")]).drop(
    columns=["log10_half", "log10_caliber_half"]).sort_values(["dataset", "brain_g"]).to_csv(
    pgls.RESULTS / "species_means.csv", index=False)

# ---- composition of the datasets per species ------------------------------------------------------------------
rows = []
for dataset, d in [("primary", prim), ("without imaging criterion", neurons[neurons.without_imaging_criterion])]:
    d = d.assign(objective=d.magnification.astype(str) + "x " + d.objective)
    for var in ("objective", "publication", "region", "depth"):
        c = d.groupby(["species", var]).size().reset_index(name="n_neurons")
        rows.append(c.rename(columns={var: "level"}).assign(dataset=dataset, variable=var))
pd.concat(rows)[["dataset", "species", "variable", "level", "n_neurons"]].to_csv(pgls.RESULTS / "composition.csv", index=False)

# ---- caliber ~ brain mass --------------------------------------------------------------------------------------
y, C = sp[Y].to_numpy(), pgls.phylo_corr(sp.species.tolist())
ref = sp[sp.primate == 0]
out = []


def add(analysis, term, c, lam="", n=len(sp)):
    out.append({"analysis": analysis, "term": term, "lambda": lam, "n_species": n, **c})


for name, d in [("correlation, all species", sp), ("correlation, non-primate species", ref)]:
    r, p = stats.pearsonr(d[X], d[Y])
    add(name, "r", {"estimate": r, "p": p}, n=len(d))
Xr, nr = pgls.design(ref, [X])
fit_ref = pgls.gls(Xr, ref[Y].to_numpy(), names=nr)
add("log10 caliber ~ log10 brain mass, non-primate species", "slope", pgls.coef(fit_ref, X), 0.0, len(ref))
for lam in LAMBDAS:
    Xm, nm = pgls.design(sp, [X, "primate"])
    f = pgls.gls(Xm, y, pgls.pagel(C, lam), nm)
    add("log10 caliber ~ log10 brain mass + primate", "slope", pgls.coef(f, X), lam)
    add("log10 caliber ~ log10 brain mass + primate", "primate offset", pgls.coef(f, "primate"), lam)
for lam in (0.0, 1.0):
    Xm, nm = pgls.design(sp, [X, "primate", f"{X}:primate"])
    f = pgls.gls(Xm, y, pgls.pagel(C, lam), nm)
    add("slope homogeneity", "brain mass x primate", pgls.coef(f, f"{X}:primate"), lam)
pd.DataFrame(out).to_csv(pgls.RESULTS / "allometry.csv", index=False)

# ---- which grouping: none, anthropoid, primate, or any other order with at least two species ---------------------
groupings = {"none": None, "anthropoid": sp.anthropoid, "primate": sp.primate}
for order, n in sp.order.value_counts().items():
    if n >= 2 and order != "Primates":
        groupings[order] = (sp.order == order).astype(float)
rows = []
for name, member in groupings.items():
    d = sp.assign(group=0.0 if member is None else member.to_numpy())
    Xm, nm = pgls.design(d, [X] + ([] if member is None else ["group"]))
    lam_ml = pgls.lambda_ml(Xm, y, C)["lambda"]
    for method, lam, extra in [("lambda = 0", 0.0, 0), ("lambda estimated", lam_ml, 1), ("lambda = 1", 1.0, 0)]:
        f = pgls.gls(Xm, y, pgls.pagel(C, lam), nm)
        rows.append({"grouping": name, "n_in_group": int(d.group.sum()), "method": method, "lambda": lam,
                     "loglik": f["loglik"], "AICc": pgls.aicc(f, extra)})
mc = pd.DataFrame(rows)
mc["dAICc"] = mc.AICc - mc.groupby("method").AICc.transform("min")
mc.sort_values(["method", "AICc"]).to_csv(pgls.RESULTS / "model_comparison.csv", index=False)

# ---- phylogenetic signal ------------------------------------------------------------------------------------------
ones = np.ones((len(sp), 1))
models = [("trait: log10 caliber", ones, y), ("trait: log10 brain mass", ones, sp[X].to_numpy()),
          ("log10 caliber ~ log10 brain mass", pgls.design(sp, [X])[0], y),
          ("log10 caliber ~ log10 brain mass + primate", pgls.design(sp, [X, "primate"])[0], y)]
pd.DataFrame([{"model": m, **pgls.lambda_ml(Xm, yy, C)} for m, Xm, yy in models]).to_csv(
    pgls.RESULTS / "phylogenetic_signal.csv", index=False)

# ---- primates against the 95% prediction interval of the non-primate fit -----------------------------------------
rows = []
for relation, xcol, ycol in [("log10 caliber ~ log10 brain mass", X, Y),
                             ("log10 50% attenuation distance ~ log10 caliber", "log10_caliber_half", "log10_half"),
                             ("electrotonic extent ~ log10 brain mass", X, "electrotonic_extent")]:
    Xn, nn = pgls.design(ref, [xcol])
    f = pgls.gls(Xn, ref[ycol].to_numpy(), names=nn)
    for a in sp[sp.primate == 1].itertuples():
        x0 = np.array([1.0, getattr(a, xcol)])
        resid, hw = float(getattr(a, ycol) - x0 @ f["beta"]), pgls.prediction_half_width(f, Xn, x0)
        t = resid / (hw / stats.t.ppf(0.975, f["df"]))
        rows.append({"relation": relation, "species": a.species, "residual": resid, "interval_half_width": hw,
                     "position": "below" if resid < -hw else "above" if resid > hw else "inside",
                     "p": float(2 * stats.t.sf(abs(t), f["df"]))})
pd.DataFrame(rows).to_csv(pgls.RESULTS / "primate_residuals.csv", index=False)

# ---- simulated electrotonic measures ------------------------------------------------------------------------------
out = []
for name, d in [("correlation log10 50% attenuation distance - log10 caliber, all species", sp),
                ("correlation log10 50% attenuation distance - log10 caliber, non-primate species", ref)]:
    r, p = stats.pearsonr(d.log10_caliber_half, d.log10_half)
    add(name, "r", {"estimate": r, "p": p}, n=len(d))
Xm, nm = pgls.design(sp, ["log10_caliber_half", "primate"])
add("log10 50% attenuation distance ~ log10 caliber + primate", "primate offset",
    pgls.coef(pgls.gls(Xm, sp.log10_half.to_numpy(), names=nm), "primate"), 0.0)
d = half.assign(lh=np.log10(half.half_attenuation_um), lc=np.log10(half.caliber_um))
mm = smf.mixedlm("lh ~ lc", d, groups=d["species"]).fit()
ci = mm.conf_int().loc["lc"]
out.append({"analysis": "reconstruction level: log10 50% attenuation distance ~ log10 caliber + (1 | species)",
            "term": "slope", "lambda": "", "n_species": len(sp), "n_neurons": len(d), "estimate": float(mm.params["lc"]),
            "ci_lo": float(ci[0]), "ci_hi": float(ci[1]), "p": float(mm.pvalues["lc"])})
Xm, nm = pgls.design(sp, [X, "primate"])
for lam in LAMBDAS:
    f = pgls.gls(Xm, sp.electrotonic_extent.to_numpy(), pgls.pagel(C, lam), nm)
    add("electrotonic extent ~ log10 brain mass + primate", "primate offset", pgls.coef(f, "primate"), lam)
Xm, nm = pgls.design(sp, [X, Y, "primate"])
f = pgls.gls(Xm, sp.electrotonic_extent.to_numpy(), names=nm)
for term in (X, Y, "primate"):
    add("electrotonic extent ~ log10 brain mass + log10 caliber + primate", term, pgls.coef(f, term), 0.0)
pd.DataFrame(out).to_csv(pgls.RESULTS / "electrotonic.csv", index=False)

o = pd.read_csv(pgls.RESULTS / "allometry.csv")
o = o[(o.term == "primate offset") & (o["lambda"] == 0)].iloc[0]
print(f"{len(sp)} species, {len(prim)} reconstructions; primate offset {o.estimate:.3f} ({o.ci_lo:.3f} to {o.ci_hi:.3f})")
