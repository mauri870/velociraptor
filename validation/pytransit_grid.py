"""PyTransit 2.2.0 against the model on the grid of README.md (rule V5) -> the keys pytransit_flux and pytransit_orbits of validation/results.json

usage: PYTHONPATH=src python validation/pytransit_grid.py        in the environment of validation/pytransit_env.md
"""
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
for name, alias in (("int", int), ("float", float), ("bool", bool)):  # PyTransit 2.2.0 uses these aliases, which numpy removed
    if not hasattr(np, name):
        setattr(np, name, alias)
import pytransit
from grid import K, U, separations

from velociraptor import quadratic_flux, transit_flux

RESULTS = Path(__file__).parent / "results.json"


def flux_grid():
    """The largest difference of the closed form from PyTransit's QuadraticModel for each radius ratio and limb-darkening pair, at the separations of the grid on a circular orbit of a/R* 20."""
    period, a = 1.0, 20.0
    rows = {}
    for k in K:
        z = separations(k)
        z = z[z < 19.0]
        model = pytransit.QuadraticModel(interpolate=False)
        model.set_data(np.arcsin(z / a) * period / (2 * np.pi))
        for u1, u2 in U:
            theirs = model.evaluate(k, [u1, u2], 0.0, period, a, np.pi / 2)
            difference = np.abs(quadratic_flux(z, k, u1, u2) - theirs)
            rows[f"k={k} u=({u1},{u2})"] = {"largest_difference": float(np.nanmax(difference)), "at_separation": float(z[np.nanargmax(difference)])}
    return rows


def orbits(count=300):
    """The largest difference of `transit_flux` from PyTransit's light curve on `count` seeded random orbits, grouped by cells of a/R* and eccentricity, with their median."""
    rng = np.random.default_rng(1)
    cells = {}
    for _ in range(count):
        period, a = rng.uniform(0.5, 30), rng.uniform(2, 40)
        inc = np.arccos(min(rng.uniform(0, 1.1) / a, 1.0))
        ecc, omega, k = rng.uniform(0, 0.9), rng.uniform(0, 2 * np.pi), rng.uniform(0.01, 0.5)
        exposure, samples = float(rng.choice([0.0, 30 / 1440])), int(rng.choice([1, 5, 20]))
        time = np.linspace(-0.3, 0.3, 121) * period / a * 2
        model = pytransit.QuadraticModel(interpolate=False)
        model.set_data(time, exptimes=exposure, nsamples=samples if exposure else 1)
        theirs = model.evaluate_ps(k, [0.4, 0.25], 0.0, period, a, inc, ecc, omega)
        ours = transit_flux(time, k, 0.4, 0.25, period, a, inc, ecc, omega, exposure=exposure, samples=samples if exposure else 1)
        cell = f"a/R* {'<3' if a < 3 else '3-5' if a < 5 else '5-10' if a < 10 else '10-20' if a < 20 else '20-40'}, e {'<0.3' if ecc < 0.3 else '0.3-0.6' if ecc < 0.6 else '>0.6'}"
        cells.setdefault(cell, []).append(float(np.nanmax(np.abs(ours - theirs))))
    return {cell: {"orbits": len(values), "median_largest_difference": float(np.median(values)), "largest_difference": float(np.max(values))} for cell, values in sorted(cells.items())}


if __name__ == "__main__":
    results = json.loads(RESULTS.read_text())
    results["pytransit_flux"], results["pytransit_orbits"] = flux_grid(), orbits()
    RESULTS.write_text(json.dumps(results, indent=1) + "\n")
    worst = max(results["pytransit_flux"].items(), key=lambda item: item[1]["largest_difference"])
    print("flux grid, largest difference:", worst)
    for cell, row in results["pytransit_orbits"].items():
        print(cell, row)
