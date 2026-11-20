"""Per-reconstruction table of the two datasets.

Dendritic caliber (and the mean and median radius of the dendritic sample points, the two alternative caliber
definitions of the sensitivity analysis) is measured from the SWC file; the electrotonic extent (primary dataset) is computed from
the SWC file and the membrane parameters; the 50% attenuation distance is read from results/attenuation.csv.
Cortical region and depth are classified from the brain_region field with data/region_keywords.csv.

Writes results/neurons.csv.
"""
import pandas as pd

import morphology
import pgls

P = pgls.parameters()
Rm, Ra, W = float(P["Rm_ohm_cm2"]), float(P["Ra_ohm_cm"]), float(P["window_um"])
keywords = pd.read_csv(pgls.DATA / "region_keywords.csv").sort_values("priority")
label = pd.read_csv(pgls.DATA / "publications.csv").set_index("pmid").label


def classify(text, kind):
    for r in keywords[keywords.kind == kind].itertuples():
        if r.keyword in str(text).lower():
            return r.value
    return "unspecified"


sel = pd.read_csv(pgls.RESULTS / "selection.csv")
cand = pd.read_csv(pgls.DATA / "candidates.csv")
d = cand.merge(sel[["neuron_id", "primary", "without_imaging_criterion"]], on="neuron_id")
d = d[d.primary | d.without_imaging_criterion].copy()
d["region"] = d.brain_region.map(lambda s: classify(s, "lobe"))
d["depth"] = d.brain_region.map(lambda s: classify(s, "depth"))
d["publication"] = d.pmid.map(label)

caliber, extent, point = [], [], []
for r in d.itertuples():
    pts = morphology.read_swc(pgls.SWC / f"{r.neuron_name}.CNG.swc")
    caliber.append(morphology.caliber(pts))
    point.append(morphology.point_radius(pts))
    extent.append(morphology.electrotonic_extent(pts, Rm, Ra, W) if r.primary else None)
d["caliber_um"] = caliber
d["point_radius_mean_um"], d["point_radius_median_um"] = zip(*point)
d["half_attenuation_um"] = d.neuron_name.map(pd.read_csv(pgls.RESULTS / "attenuation.csv").set_index("neuron_name").baseline)
d["electrotonic_extent"] = extent

d[["neuron_id", "neuron_name", "species", "animal", "brain_region", "region", "depth", "magnification", "objective",
   "pmid", "publication", "caliber_um", "point_radius_mean_um", "point_radius_median_um", "half_attenuation_um",
   "electrotonic_extent", "primary",
   "without_imaging_criterion"]].to_csv(pgls.RESULTS / "neurons.csv", index=False)
print(f"{len(d)} reconstructions: primary {int(d.primary.sum())}, without the imaging criterion "
      f"{int(d.without_imaging_criterion.sum())}")
