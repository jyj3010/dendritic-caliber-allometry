"""Passive cable simulation of the primary dataset (NEURON).

For every reconstruction, a current-free passive model is built with the same membrane parameters, the
steady-state voltage transfer from the soma is averaged over the dendritic segments in path-distance bins,
and the distance at which the transfer falls to 50% is taken by linear interpolation. The simulation is run
with the membrane parameters of data/parameters.csv and with each setting of data/membrane_sweep.csv.

Writes results/attenuation.csv: 50% attenuation distance (µm) per reconstruction and setting; empty where
the transfer stays above 50% within the simulated distance.

Usage: python 01_simulate.py [number of processes]
"""
import os
import sys
import tempfile
from multiprocessing import get_context

import numpy as np
import pandas as pd

import pgls

P = pgls.parameters()
BIN, DMAX = float(P["bin_um"]), float(P["dmax_um"])
h = None


def half_distance(mid, transfer):
    x, y = np.concatenate([[0.0], mid]), np.concatenate([[1.0], transfer])
    for i in range(1, len(x)):
        if y[i] < 0.5 <= y[i - 1]:
            return x[i - 1] + (0.5 - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])
    return np.nan


def simulate(name, Rm, Ra, Cm):
    global h
    if h is None:
        from neuron import h as _h
        h = _h
        h.load_file("stdlib.hoc")
        h.load_file("import3d.hoc")
    for s in list(h.allsec()):
        h.delete_section(sec=s)
    lines = [l for l in (pgls.SWC / f"{name}.CNG.swc").read_text(errors="ignore").replace("\r", "").split("\n") if l.strip()]
    tmp = os.path.join(tempfile.gettempdir(), f"swc_{os.getpid()}.swc")
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    cell = h.Import3d_SWC_read()
    cell.input(tmp)
    h.Import3d_GUI(cell, 0).instantiate(None)
    secs = list(h.allsec())
    for s in secs:
        s.Ra, s.cm = Ra, Cm
        s.insert("pas")
        s.nseg = max(1, int((s.L / (0.1 * h.lambda_f(100, sec=s)) + 0.9) / 2) * 2 + 1)      # d_lambda rule
        for seg in s:
            seg.pas.g, seg.pas.e = 1.0 / Rm, 0
    soma = next((s for s in secs if "soma" in s.name()), secs[0])
    h.distance(0, soma(0.5))
    imp = h.Impedance()
    imp.loc(0.5, sec=soma)
    imp.compute(0)
    n = int(DMAX / BIN)
    total, count = np.zeros(n), np.zeros(n)
    for s in secs:
        if not ("dend" in s.name() or "apic" in s.name()):
            continue
        for seg in s:
            d = h.distance(seg)
            if 0 < d < DMAX:
                total[int(d / BIN)] += imp.ratio(seg.x, sec=s)
                count[int(d / BIN)] += 1
    ok = count > 0
    return half_distance((np.arange(n) * BIN + BIN / 2)[ok], total[ok] / count[ok])


def run(job):
    setting, Rm, Ra, Cm, names = job
    return [(name, setting, simulate(name, Rm, Ra, Cm)) for name in names]


if __name__ == "__main__":
    jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sel = pd.read_csv(pgls.RESULTS / "selection.csv")
    names = list(sel[sel.primary].neuron_name)
    Rm, Ra, Cm = float(P["Rm_ohm_cm2"]), float(P["Ra_ohm_cm"]), float(P["Cm_uF_cm2"])
    settings = [("baseline", Rm, Ra)] + [(s.setting, Rm * s.rm_scale, Ra * s.ra_scale)
                                         for s in pd.read_csv(pgls.DATA / "membrane_sweep.csv").itertuples()]
    chunks = [names[i::jobs * 4] for i in range(jobs * 4)]
    work = [(tag, rm, ra, Cm, c) for tag, rm, ra in settings for c in chunks if c]
    with get_context("spawn").Pool(jobs) as pool:
        rows = [r for part in pool.map(run, work) for r in part]
    out = pd.DataFrame(rows, columns=["neuron_name", "setting", "half"]).pivot(
        index="neuron_name", columns="setting", values="half")
    out = out.loc[names, [s[0] for s in settings]]
    out.to_csv(pgls.RESULTS / "attenuation.csv")
    print(f"{len(out)} reconstructions; without a 50% crossing at baseline: {int(out.baseline.isna().sum())}")
