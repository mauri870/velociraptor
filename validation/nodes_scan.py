"""The error of the quadrature (the cross-check of the closed form) against the number of nodes, as a function of the distance from the two tangencies -> the key nodes_near_tangency of validation/results.json

usage: OMP_NUM_THREADS=1 PYTHONPATH=src python validation/nodes_scan.py
Reference: 768 nodes. The separations are |1 - k| + d and 1 + k - d with d from 1e-9 to 0.3."""
import json
from pathlib import Path

import numpy as np

from velociraptor import quadrature_flux as quadratic_flux

RESULTS = Path(__file__).parent / "results.json"
NODES = (16, 24, 32, 48, 96, 192)


def scan():
    """For each number of quadrature nodes, the largest difference from the quadrature of 768 nodes over radius ratios, limb-darkening pairs and separations near the internal and external contacts."""
    d = np.logspace(-9, np.log10(0.3), 40)
    rows = {}
    for n in NODES:
        worst = {"internal": [0.0, None], "external": [0.0, None]}
        for k in (0.05, 0.1, 0.3, 0.5, 0.9, 1.2):
            for u in ((0.0, 0.0), (0.4, 0.25), (0.3, 0.7)):
                for name, z in (("internal", abs(1 - k) + d), ("external", 1 + k - d)):
                    difference = np.abs(quadratic_flux(z, k, *u, nodes=n) - quadratic_flux(z, k, *u, nodes=768))
                    if difference.max() > worst[name][0]:
                        worst[name] = [float(difference.max()), [k, list(u), float(d[difference.argmax()])]]
        rows[str(n)] = worst
        print(n, worst, flush=True)
    return rows


if __name__ == "__main__":
    results = json.loads(RESULTS.read_text())
    results["nodes_near_tangency"] = scan()
    RESULTS.write_text(json.dumps(results, indent=1) + "\n")
