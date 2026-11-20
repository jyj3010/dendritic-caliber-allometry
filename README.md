# dendritic-caliber-allometry

Analysis code and derived data for

Jang Y-J. Dendritic caliber scales with brain mass across mammals. Manuscript submitted to
Brain Structure and Function.

## Running the analysis

Python 3.12 or later.

    pip install -r requirements.txt
    python code/00_select.py
    python code/fetch_swc.py
    python code/01_simulate.py 8
    python code/02_measure.py
    python code/03_statistics.py
    python code/04_sensitivity.py
    python code/05_cross_archive.py

Neuron reconstructions are not included. `fetch_swc.py` downloads the standardised (CNG) SWC
files of the selected reconstructions from NeuroMorpho.Org into `swc/`. `01_simulate.py` needs
NEURON (9.0, Python module; the argument is the number of processes) and can be skipped: the
later scripts then read the simulated distances from `results/attenuation.csv`. Each script
writes to `results/`; the files in `results/` are the outputs used in the article.

| script | does |
|---|---|
| `00_select.py` | inclusion criteria, imaging criterion (60x oil immersion), minimum of 10 per species |
| `fetch_swc.py` | downloads the SWC files of the 3,041 selected reconstructions |
| `01_simulate.py` | passive cable simulation of the primary dataset: 50% attenuation distance per reconstruction, with the standard membrane parameters and with four altered settings |
| `02_measure.py` | dendritic caliber, electrotonic extent, cortical region and depth per reconstruction |
| `03_statistics.py` | species means, caliber-brain mass regression with Pagel's lambda, grouping comparison, phylogenetic signal, prediction intervals, electrotonic measures |
| `04_sensitivity.py` | dataset and grouping, precision of the species means, region and depth, leave-out analyses, membrane parameters |
| `05_cross_archive.py` | within-archive primate versus non-primate contrasts in external archives |
| `morphology.py`, `pgls.py` | SWC measurements; shared tables and generalised least squares |

## Files

| file | contents | article |
|---|---|---|
| `data/candidates.csv` | 4,606 reconstructions of the Jacobs archive (NeuroMorpho.Org) with species, animal, region, cell type, condition, age class, objective and source publication | Methods |
| `data/parameters.csv` | inclusion criteria, membrane parameters, fixed values of lambda | Methods |
| `data/species.csv` | species, order, group membership | Table S1 |
| `data/specimens.csv` | brain mass of the animals that provided the reconstructions, with source | Table S1 |
| `data/phylogeny_distance.csv` | pairwise patristic distances, pruned from the mammal tree of Upham et al. (2019) | Methods |
| `data/publications.csv` | source publications of the reconstructions | Table S3 |
| `data/region_keywords.csv`, `data/region_restrictions.csv` | classification of region and depth; restrictions of the sensitivity analysis | Table S4 |
| `data/membrane_sweep.csv` | altered membrane parameters | Fig. S1b |
| `data/external_archives.csv`, `data/external_neurons.csv` | external archives and the mean dendritic radius of their reconstructions | Fig. 3, Results |
| `results/selection.csv` | outcome of the criteria for every candidate | Methods |
| `results/neurons.csv` | per-reconstruction table: NeuroMorpho.Org identifier, species, region, imaging objective, source publication, caliber, mean and median radius of the sample points, 50% attenuation distance, electrotonic extent, dataset membership (the per-neuron measurement table; the reconstructions of the primary dataset are listed in Table S2) | Table S2, Table S4 |
| `results/attenuation.csv` | 50% attenuation distance per reconstruction and membrane setting | Fig. 2, Fig. S1b |
| `results/species_means.csv` | species means of both datasets | Figs. 1-2, Table S1 |
| `results/composition.csv` | objective, publication, region and depth per species | Table S3 |
| `results/allometry.csv` | correlations, slopes and primate offset with lambda fixed, slope homogeneity | Fig. 1, Results, Table S4 |
| `results/model_comparison.csv` | AICc of the candidate groupings | Results |
| `results/phylogenetic_signal.csv` | Pagel's lambda with interval and likelihood-ratio tests | Results, Table S4 |
| `results/primate_residuals.csv` | primates against the 95% prediction interval of the non-primate fit | Figs. 1-2 |
| `results/electrotonic.csv` | 50% attenuation distance and electrotonic extent analyses | Fig. 2, Results |
| `results/sensitivity.csv` | sensitivity analyses of the primate offset | Table S4, Fig. S1a |
| `results/membrane_sweep.csv` | caliber-50% attenuation distance correlation under altered membrane parameters | Fig. S1b |
| `results/cross_archive.csv`, `results/external_summary.csv` | within-archive contrasts; summary of the external archives | Fig. 3, Results |

In the tables, `primary` marks the primary dataset (reconstructions imaged with a 60x
oil-immersion objective) and `without_imaging_criterion` the dataset without that criterion
(all eligible reconstructions; it contains the primary dataset).

Running the scripts from the downloaded SWC files reproduces the values reported in the
article.

## Licence

Code: MIT. Data tables in `data/` and `results/`: CC BY 4.0. See `LICENSE`.
