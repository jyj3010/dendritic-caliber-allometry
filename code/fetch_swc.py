"""Download the standardised (CNG) SWC files of the selected reconstructions from NeuroMorpho.Org into swc/."""
import time
import urllib.request

import pandas as pd

import pgls

URL = "https://neuromorpho.org/dableFiles/jacobs/CNG%20version/{name}.CNG.swc"

sel = pd.read_csv(pgls.RESULTS / "selection.csv")
names = sel[sel.primary | sel.without_imaging_criterion].neuron_name
pgls.SWC.mkdir(exist_ok=True)
failed = []
for i, name in enumerate(names, 1):
    out = pgls.SWC / f"{name}.CNG.swc"
    if out.exists() and out.stat().st_size > 0:
        continue
    try:
        with urllib.request.urlopen(URL.format(name=urllib.request.quote(name)), timeout=60) as r:
            out.write_bytes(r.read())
    except Exception as e:
        failed.append(name)
        print(f"failed: {name} ({e})")
    time.sleep(0.2)
    if i % 200 == 0:
        print(f"{i}/{len(names)}")
print(f"done: {len(names) - len(failed)} files in {pgls.SWC}, {len(failed)} failed")
