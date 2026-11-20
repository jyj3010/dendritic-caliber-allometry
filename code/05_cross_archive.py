"""Within-archive contrasts between a primate and a non-primate species in external archives (Fig. 3).

Each contrast is the ratio of the observed primate/non-primate caliber ratio to the ratio expected from the
non-primate slope of the primary dataset. Its uncertainty is species-level: twice the residual variance of
the non-primate fit plus the uncertainty of the slope. The contrasts are reported one by one and not pooled.

data/external_neurons.csv holds the mean dendritic radius of the reconstructions of each archive (computed
from the CNG SWC files; for the Allen archive, half the diameter given by the NeuroMorpho.Org morphometry).

Writes results/cross_archive.csv and results/external_summary.csv.
"""
import math

import pandas as pd
from scipy import stats

import pgls
from pgls import X, Y

neurons = pd.read_csv(pgls.RESULTS / "neurons.csv")
sp = pgls.species_means(neurons[neurons.primary])
ref = sp[sp.primate == 0]
Xr, names = pgls.design(ref, [X])
f = pgls.gls(Xr, ref[Y].to_numpy(), names=names)
slope, se_slope, s_res = f["beta"][1], f["se"][1], math.sqrt(f["s2"])
tc = stats.t.ppf(0.975, f["df"])

own = pgls.fit_group(sp, "primate")
rows = [{"archive": "Jacobs (this study)", "contrast": f"{own['n_group']} primate vs {len(ref)} non-primate species",
         "n_primate": own["n_group"], "n_nonprimate": len(ref), "ratio": 10 ** own["offset"],
         "ci_lo": 10 ** own["offset_ci_lo"], "ci_hi": 10 ** own["offset_ci_hi"], "p": own["offset_p"]}]
ext = pd.read_csv(pgls.DATA / "external_neurons.csv")
ext["diameter_um"] = 2 * ext.mean_radius_um
archives = pd.read_csv(pgls.DATA / "external_archives.csv")
for a in archives[archives.used == "yes"].itertuples():
    d = ext[ext.archive == a.archive]
    p, n = d[d.species == a.primate].diameter_um, d[d.species == a.nonprimate].diameter_um
    dx = math.log10(a.primate_brain_g) - math.log10(a.nonprimate_brain_g)
    est = math.log10(p.mean()) - math.log10(n.mean()) - slope * dx
    se = math.sqrt(2 * s_res ** 2 + (dx * se_slope) ** 2)
    rows.append({"archive": a.archive, "contrast": f"{a.primate_name} vs {a.nonprimate_name}", "n_primate": len(p),
                 "n_nonprimate": len(n), "primate_mean_um": p.mean(), "nonprimate_mean_um": n.mean(),
                 "ratio": 10 ** est, "ci_lo": 10 ** (est - tc * se), "ci_hi": 10 ** (est + tc * se),
                 "p": float(2 * stats.t.sf(abs(est / se), f["df"]))})
out = pd.DataFrame(rows)
out.to_csv(pgls.RESULTS / "cross_archive.csv", index=False)

summary = ext.groupby(["archive", "species"]).diameter_um.agg(
    n_neurons="size", mean_um="mean", median_um="median", sd_um="std").reset_index()
summary.merge(archives[["archive", "used", "reason_not_used"]], on="archive").to_csv(
    pgls.RESULTS / "external_summary.csv", index=False)
print(out[["archive", "contrast", "ratio", "ci_lo", "ci_hi"]].to_string(index=False))
