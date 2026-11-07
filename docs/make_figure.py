"""Writes docs/figure.png: the folded, binned TESS light curve of TOI-1411 b with the light curves of two scenarios that TRICERATOPS weighs, computed by velociraptor's transit model at the draws that fit the data best, and prints every number that the caption of the README states.

The star is TOI-1411 (TIC 116483514), a confirmed planet host, observed at 2-minute cadence by TESS in sectors 16, 23 and 24.
Sources of what is not computed here:
  planet and ephemeris: NASA Exoplanet Archive, table pscomppars/ps, planet "TOI-1411 b" (period 1.4520527 d, mid-transit BJD 2458739.47404, from Polanski et al. 2024),
    https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+pl_name,tic_id,pl_orbper,pl_tranmid+from+pscomppars+where+pl_name='TOI-1411+b'&format=csv
  published TRICERATOPS result: Giacalone et al. 2022, AJ 163, 99, arXiv:2201.12661, doi:10.3847/1538-3881/ac4334, Section V ("TOI-1411.01"):
    FPP = (1.18 +- 0.68) x 10^-4 and NFPP = 0.0, from the 3 sectors 16, 23 and 24, with the neighbouring stars cleared by ground-based photometry and no companion in Keck/NIRC2 imaging
  light curve: the SPOC PDCSAP light curve of the sectors, from MAST through lightkurve (https://docs.lightkurve.org/reference/api/lightkurve.search_lightcurve.html)
  scenarios and FPP: TRICERATOPS 1.1.0, FPP = 1 - (TP + PTP + DTP) of its table of scenarios (triceratops.TARGET.calc_probs, https://github.com/stevengiacalone/triceratops)

usage: python docs/make_figure.py [OUTPUT.png] [--cache DIRECTORY] [--draws N] [--seed S] [--no-stock]
needs matplotlib, lightkurve, TRICERATOPS 1.1.0 and the network (MAST, the TIC and Gaia catalogs, the TESS cutouts); the calculation runs on the device of install("cuda") when torch has one and on the CPU otherwise.
"""
import argparse
import contextlib
import os
import sys
import tempfile
import time as clock
import types
import warnings
from pathlib import Path

import numpy as np

TIC = 116483514
SECTORS = (16, 23, 24)
PERIOD = 1.4520527  # days
MID_TRANSIT = 2458739.47404 - 2457000.0  # BTJD
PUBLISHED = {"FPP": 1.18e-4, "FPP_error": 0.68e-4, "NFPP": 0.0}
WINDOW = 0.25  # days from mid-transit that the fold keeps
BIN = 10.0 / 1440.0  # days
FIGURE_SIZE, DPI = (4.6, 3.0), 200


def pkg_resources_stand_in():
    """PyTransit 2.2.0, which TRICERATOPS imports, calls pkg_resources.resource_filename, and setuptools 81 removed that module: where it is missing, a module with that one function is registered."""
    try:
        import pkg_resources  # noqa: F401
    except ImportError:
        module = types.ModuleType("pkg_resources")

        def resource_filename(package, name):
            __import__(package)
            return str(Path(sys.modules[package].__file__).parent / name)

        module.resource_filename = resource_filename
        sys.modules["pkg_resources"] = module


def light_curve():
    """Time (BTJD), flux and flux error of the SPOC 2-minute PDCSAP light curves of the sectors, each normalised and flattened with a running filter that leaves out the transits (a mask of 4.8 hours around each)."""
    import lightkurve

    found = lightkurve.search_lightcurve(f"TIC {TIC}", author="SPOC", exptime=120, sector=SECTORS)
    collection = found.download_all(flux_column="pdcsap_flux", quality_bitmask="default")
    time, flux, error = [], [], []
    for curve in collection:
        curve = curve.remove_nans().normalize()
        phase = ((curve.time.value - MID_TRANSIT) / PERIOD + 0.5) % 1 - 0.5
        flat = curve.flatten(window_length=721, mask=np.abs(phase * PERIOD) < 0.1)
        time.append(flat.time.value)
        flux.append(flat.flux.value)
        error.append(flat.flux_err.value)
    return np.concatenate(time), np.concatenate(flux), np.concatenate(error)


def fold_and_bin(time, flux, period, mid_transit, window=WINDOW, width=BIN, minimum=6):
    """The fold within `window` days of mid-transit in bins of `width` days: the mean time and flux of each bin with at least `minimum` points, and the error of its mean, the scatter of the points more than 0.1 days from mid-transit over the square root of the points of the bin. The depth is the flux deficit of the bins within 0.03 days of mid-transit."""
    offset = (((np.asarray(time) - mid_transit) / period + 0.5) % 1 - 0.5) * period
    flux = np.asarray(flux)
    scatter = np.std(flux[np.abs(offset) > 0.1])
    keep = np.abs(offset) < window
    offset, selected = offset[keep], flux[keep]
    index = np.digitize(offset, np.arange(-window, window + width / 2, width)) - 1
    counts = np.bincount(index, minlength=index.max() + 1)
    bins = [i for i in range(len(counts)) if counts[i] >= minimum]
    centres = np.array([offset[index == i].mean() for i in bins])
    means = np.array([selected[index == i].mean() for i in bins])
    errors = scatter / np.sqrt(counts[bins])
    depth = 1.0 - means[np.abs(centres) < 0.03].mean()
    return centres, means, errors, depth


def build_field(cache):
    """The TRICERATOPS target of the star (needs the network for the TIC and Gaia catalogs and the TESS cutouts of the sectors)."""
    import lightkurve

    pkg_resources_stand_in()
    from triceratops import triceratops

    original = lightkurve.search_tesscut

    def search(*arguments, **keywords):  # a cutout can lack the keywords 1CRV4P and 2CRV4P, and the package then ignores its sector; set them to 0, the corner of the cutout, which the default 5 by 5 pixel aperture does not use
        found = original(*arguments, **keywords)
        download_all = found.download_all

        def downloaded(*more, **options):
            cutouts = download_all(*more, **options)
            for cutout in cutouts:
                for key in ("1CRV4P", "2CRV4P"):
                    if key not in cutout.hdu[1].header:
                        cutout.hdu[1].header[key] = 0
            return cutouts

        found.download_all = downloaded
        return found

    lightkurve.search_tesscut = search
    here = Path.cwd()
    os.chdir(cache)  # the package writes the background population of the star into the current directory
    try:
        target = triceratops.target(ID=TIC, sectors=np.array(SECTORS), search_radius=10)
    finally:
        os.chdir(here)
        lightkurve.search_tesscut = original
    if target.trilegal_fname and not os.path.isabs(target.trilegal_fname):
        target.trilegal_fname = os.path.join(cache, target.trilegal_fname)
    return target


class Recorder:
    """The arguments and the result of every call of the three likelihood functions of TRICERATOPS' marginal likelihoods, in the order of the calls (the order of the rows of its table of scenarios), copied at the call: the package changes its draws in place afterwards."""

    NAMES = ("lnL_TP_p", "lnL_EB_p", "lnL_EB_twin_p")

    def __init__(self):
        self.calls = []

    @contextlib.contextmanager
    def watching(self):
        import triceratops.marginal_likelihoods as module

        originals = {name: getattr(module, name) for name in self.NAMES}

        def copy(value):
            return np.array(value, copy=True) if hasattr(value, "shape") else value

        def wrap(name):
            def wrapper(*arguments, **keywords):
                received = (tuple(copy(v) for v in arguments), {key: copy(v) for key, v in keywords.items()})  # copied before the call because TRICERATOPS converts the inclination of its draws from degrees to radians in place afterwards
                result = originals[name](*arguments, **keywords)
                self.calls.append((name, *received, np.array(result, copy=True)))
                return result

            return wrapper

        for name in self.NAMES:
            setattr(module, name, wrap(name))
        try:
            yield
        finally:
            for name, function in originals.items():
                setattr(module, name, function)


def best_draw(call):
    """The draw with the smallest half chi-squared of a recorded call: its index, its half chi-squared, the scenario arguments after time, flux and sigma and the keywords, each with that draw's value."""
    _, arguments, keywords, result = call
    index = int(np.argmin(np.where(np.isfinite(result), result, np.inf)))

    def pick(value):
        return value[index] if hasattr(value, "shape") and value.shape[:1] == result.shape else value

    return index, float(result[index]), [pick(v) for v in arguments[3:]], {key: pick(v) for key, v in keywords.items()}


def scenario_curve(call, time):
    """The light curve of the best draw of a recorded call at `time` (days from mid-transit), from velociraptor's light-curve functions with the 20 midpoints of the exposure that TRICERATOPS uses. The inclination is in degrees, as the draws enter the likelihood."""
    import torch

    from velociraptor import torch_likelihood

    name = call[0]
    _, _, scenario, keywords = best_draw(call)
    options = dict(companion_fluxratio=keywords["companion_fluxratio"], companion_is_host=keywords.get("companion_is_host", False), exposure=keywords["exptime"], samples=int(keywords["nsamples"]))
    simulate = torch_likelihood.simulate_tp_flux if name == "lnL_TP_p" else torch_likelihood.simulate_eb_flux
    flux = simulate(torch.as_tensor(time), *scenario, **options)
    return (flux[0] if isinstance(flux, tuple) else flux)[0].numpy()


def half_chi_squared(flux, model, sigma):
    """Half the chi-squared of `flux` against `model` for the uncertainty `sigma`."""
    return 0.5 * float(np.sum((flux - model) ** 2 / sigma**2))


def main(output, cache, draws, seed, stock):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    time, flux, _ = light_curve()
    centres, means, errors, depth = fold_and_bin(time, flux, PERIOD, MID_TRANSIT)
    sigma = float(np.median(errors))
    print(f"TIC {TIC}, sectors {', '.join(str(s) for s in SECTORS)}: {len(time)} points of 2-minute PDCSAP flux, {len(centres)} bins of {BIN * 1440:.0f} minutes within {WINDOW} days of mid-transit, error of a bin {1e6 * sigma:.0f} ppm (median), transit depth {1e6 * depth:.0f} ppm")
    target = build_field(cache)
    print(f"field: {len(target.stars)} stars within 10 pixels")

    from velociraptor.triceratops import install

    device = "cuda" if torch.cuda.is_available() else "cpu"
    recorder = Recorder()
    started = clock.perf_counter()
    np.random.seed(seed)
    with install(device, overlap=1), recorder.watching():  # one call at a time, so a seed gives the same draws and the likelihood calls come in the order of the rows
        target.calc_depths(tdepth=depth)
        target.calc_probs(time=centres, flux_0=means, flux_err_0=sigma, P_orb=PERIOD, N=draws, parallel=True, verbose=0)
    seconds = clock.perf_counter() - started
    table = target.probs[["ID", "scenario", "prob"]].reset_index(drop=True)
    print(f"velociraptor, install({device!r}, overlap=1): seed {seed}, {draws} draws, {seconds:.1f} s: FPP {float(target.FPP):.4g}, NFPP {float(target.NFPP):.3g}")
    print(f"published (Giacalone et al. 2022): FPP {PUBLISHED['FPP']:.3g} +- {PUBLISHED['FPP_error']:.2g}, NFPP {PUBLISHED['NFPP']:g}")
    planet_row = 0
    strongest = max((row for row in range(len(table)) if row not in (0, 3, 9)), key=lambda row: table.prob[row])  # FPP = 1 - (TP + PTP + DTP): rows 0, 3 and 9
    for row in (planet_row, strongest):
        print(f"scenario {table.scenario[row]} of star {table.ID[row]}: probability {table.prob[row]:.4g}")
    if len(recorder.calls) != len(table):
        raise SystemExit(f"{len(recorder.calls)} likelihood calls for {len(table)} scenarios: the order of the calls is not that of the rows")

    if stock:
        started = clock.perf_counter()
        np.random.seed(seed)
        target.calc_probs(time=centres, flux_0=means, flux_err_0=sigma, P_orb=PERIOD, N=draws, parallel=True, verbose=0)
        print(f"TRICERATOPS 1.1.0 as it is: same seed and draws, {clock.perf_counter() - started:.1f} s: FPP {float(target.FPP):.4g}, NFPP {float(target.NFPP):.3g}")

    grid = np.linspace(-WINDOW, WINDOW, 500)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.errorbar(centres * 24, 1e6 * (means - 1.0), yerr=1e6 * errors, fmt="o", color="#555555", ms=2.8, lw=0.7, label="TESS, 10-minute bins")
    style = {planet_row: ("#0072B2", "-"), strongest: ("#D55E00", "--")}
    names = {"TP": "Planet on the target", "STP": "Planet on an unresolved companion"}
    for row in (planet_row, strongest):
        call = recorder.calls[row]
        recorded = best_draw(call)[1]
        recomputed = half_chi_squared(means, scenario_curve(call, centres), sigma)
        print(f"best draw of {table.scenario[row]}: half chi-squared {recorded:.3f} recorded, {recomputed:.3f} recomputed at the {len(centres)} bins")
        ax.plot(grid * 24, 1e6 * (scenario_curve(call, grid) - 1.0), color=style[row][0], ls=style[row][1], lw=1.6, label=f"{names.get(table.scenario[row], table.scenario[row])}: {table.prob[row]:.3f}")
    ax.set_xlabel("Hours from mid-transit", color="black")
    ax.set_ylabel("Relative flux (ppm)", color="black")
    ax.tick_params(colors="black")
    for side in ax.spines.values():
        side.set_color("black")
    ax.set_xlim(-WINDOW * 24, WINDOW * 24)
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.45 * (high - low))
    ax.legend(loc="upper center", title="Scenario at its best draw: probability", title_fontsize=6.5, fontsize=6.5, frameon=True, facecolor="white", edgecolor="#999999", labelcolor="black")
    fig.tight_layout()
    fig.savefig(output, facecolor="white")
    print(f"wrote {output}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", type=Path, default=Path(__file__).with_name("figure.png"))
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--draws", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-stock", action="store_true")
    options = parser.parse_args()
    with tempfile.TemporaryDirectory() as scratch:
        main(options.output, options.cache or Path(scratch), options.draws, options.seed, not options.no_stock)
