"""Select the reconstructions analysed in the article from the candidates of the Jacobs archive.

Writes results/selection.csv: one row per candidate with the criterion that removed it, if any, and its
membership of the primary dataset and of the dataset without the 60x oil-immersion criterion (all eligible
reconstructions; it contains the primary dataset).
"""
import pandas as pd

import pgls

P = pgls.parameters()
cand = pd.read_csv(pgls.DATA / "candidates.csv")
words = lambda key: [w.lower() for w in str(P[key]).split("|")]


def removed_by(r):
    if any(w in str(r.brain_region).lower() for w in words("region_excluded")):
        return "brain region"
    if any(w in str(r.cell_type).lower() for w in words("cell_type_excluded")):
        return "cell type"
    if str(r.experiment_condition).lower() not in words("condition_allowed"):
        return "experimental condition"
    if str(r.age_classification).lower() in words("age_excluded"):
        return "age"
    return ""


def keep_species(d):
    """Reconstructions of the species with at least the minimum number of reconstructions."""
    n = d.groupby("species").neuron_id.transform("size")
    return set(d.neuron_id[n >= int(P["min_neurons"])])


cand["removed_by"] = cand.apply(removed_by, axis=1)
eligible = cand[cand.removed_by == ""]
criterion = (eligible.magnification == int(P["magnification"])) & (eligible.objective == P["objective"])
cand["primary"] = cand.neuron_id.isin(keep_species(eligible[criterion]))
# secondary check: all eligible reconstructions, without the imaging criterion
cand["without_imaging_criterion"] = cand.neuron_id.isin(keep_species(eligible))
assert set(cand.neuron_id[cand.primary]) <= set(cand.neuron_id[cand.without_imaging_criterion])
cand[["neuron_id", "neuron_name", "species", "removed_by", "primary", "without_imaging_criterion"]].to_csv(
    pgls.RESULTS / "selection.csv", index=False)

print(cand.removed_by.replace("", "eligible").value_counts().to_string())
for col in ("primary", "without_imaging_criterion"):
    d = cand[cand[col]]
    print(f"{col}: {len(d)} reconstructions, {d.species.nunique()} species")
