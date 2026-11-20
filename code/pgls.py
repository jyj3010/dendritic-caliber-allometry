"""Tables shared by the scripts and generalised least squares with Pagel's lambda."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS, SWC = ROOT / "data", ROOT / "results", ROOT / "swc"
RESULTS.mkdir(exist_ok=True)
Y, X = "log10_caliber", "log10_brain"


def parameters():
    return pd.read_csv(DATA / "parameters.csv").set_index("parameter")["value"]


def fixed_lambdas():
    return [float(v) for v in str(parameters()["lambda_fixed"]).split("|")]


def species_table():
    """Species with order, group membership and brain mass (mean of the animals with a reported value)."""
    sp = pd.read_csv(DATA / "species.csv")
    animals = pd.read_csv(DATA / "specimens.csv").dropna(subset=["brain_mass_g"])
    brain = animals.groupby("species").agg(
        brain_g=("brain_mass_g", "mean"), n_animals=("brain_mass_g", "size"),
        brain_source=("source", lambda s: "; ".join(dict.fromkeys(s)))).reset_index()
    sp = sp.merge(brain, on="species", validate="one_to_one")
    sp[X] = np.log10(sp.brain_g)
    return sp


def species_means(neurons, min_neurons=None):
    """Species mean caliber (arithmetic mean, then log10) with the standard error of the log10 mean."""
    if min_neurons is None:
        min_neurons = int(parameters()["min_neurons"])
    g = neurons.groupby("species").caliber_um.agg(n_neurons="size", caliber_um="mean", sd="std").reset_index()
    g = g[g.n_neurons >= min_neurons].copy()
    g[Y] = np.log10(g.caliber_um)
    g["se_log10"] = (g.sd / g.caliber_um) / np.sqrt(g.n_neurons) / math.log(10)
    return g.merge(species_table(), on="species", validate="one_to_one")


def phylo_corr(species):
    """Phylogenetic correlation matrix (root height scaled to 1) in the order of `species`."""
    d = pd.read_csv(DATA / "phylogeny_distance.csv", index_col=0)
    height = d.to_numpy(float).max() / 2.0
    return (height - 0.5 * d.loc[species, species].to_numpy(float)) / height


def pagel(C, lam):
    return lam * C + (1.0 - lam) * np.diag(np.diag(C))


def design(df, terms):
    """Intercept plus the listed columns; 'a:b' is a product term."""
    cols = [np.ones(len(df))]
    for t in terms:
        cols.append(np.prod([df[c].to_numpy(float) for c in t.split(":")], axis=0))
    return np.column_stack(cols), ["intercept"] + list(terms)


def gls(Xm, y, V=None, names=None):
    """Generalised least squares (V = None: ordinary least squares). Standard errors use the unbiased
    residual variance; the log-likelihood is the maximum-likelihood one."""
    Xm, y = np.asarray(Xm, float), np.asarray(y, float)
    n, p = Xm.shape
    V = np.eye(n) if V is None else V
    Vi = np.linalg.inv(V)
    cov = np.linalg.inv(Xm.T @ Vi @ Xm)
    beta = cov @ Xm.T @ Vi @ y
    res = y - Xm @ beta
    rss = float(res @ Vi @ res)
    df = n - p
    se = np.sqrt(np.diag(cov) * rss / df)
    tcrit = stats.t.ppf(0.975, df)
    loglik = -0.5 * (n * math.log(2 * math.pi * rss / n) + np.linalg.slogdet(V)[1] + n)
    return {"n": n, "k": p + 1, "df": df, "names": names or [f"b{i}" for i in range(p)], "beta": beta, "se": se,
            "ci_lo": beta - tcrit * se, "ci_hi": beta + tcrit * se, "p": 2 * stats.t.sf(np.abs(beta / se), df),
            "s2": rss / df, "loglik": loglik}


def coef(fit, name):
    i = fit["names"].index(name)
    return {"estimate": float(fit["beta"][i]), "ci_lo": float(fit["ci_lo"][i]), "ci_hi": float(fit["ci_hi"][i]),
            "p": float(fit["p"][i])}


def aicc(fit, extra_parameters=0):
    k, n = fit["k"] + extra_parameters, fit["n"]
    return 2 * k - 2 * fit["loglik"] + (2 * k * (k + 1)) / (n - k - 1)


def lambda_ml(Xm, y, C):
    """Maximum-likelihood lambda on a grid of 0.005, profile-likelihood 95% interval, and likelihood-ratio
    tests against 0 and 1."""
    grid = np.round(np.linspace(0.0, 1.0, 201), 6)
    ll = np.array([gls(Xm, y, pagel(C, float(v)))["loglik"] for v in grid])
    best = int(ll.argmax())
    inside = grid[ll >= ll[best] - stats.chi2.ppf(0.95, 1) / 2.0]
    lrt = lambda ref: float(stats.chi2.sf(max(2 * (ll[best] - ref), 0.0), 1))
    return {"lambda": float(grid[best]), "ci_lo": float(inside.min()), "ci_hi": float(inside.max()),
            "p_lambda_0": lrt(ll[0]), "p_lambda_1": lrt(ll[-1])}


def fit_group(g, group, V=None):
    """log10 caliber ~ log10 brain mass + group: common slope and group offset."""
    Xm, names = design(g, [X, group])
    f = gls(Xm, g[Y].to_numpy(), V, names)
    o, s = coef(f, group), coef(f, X)
    return {"n_species": len(g), "n_group": int(g[group].sum()), "offset": o["estimate"], "offset_ci_lo": o["ci_lo"],
            "offset_ci_hi": o["ci_hi"], "offset_p": o["p"], "slope": s["estimate"], "slope_ci_lo": s["ci_lo"],
            "slope_ci_hi": s["ci_hi"], "slope_p": s["p"]}


def prediction_half_width(fit, Xm, x0):
    """Half-width of the 95% prediction interval of an ordinary least-squares fit at the row x0."""
    return stats.t.ppf(0.975, fit["df"]) * math.sqrt(fit["s2"] * (1 + x0 @ np.linalg.inv(Xm.T @ Xm) @ x0))
