"""The shift of the evidence of the three scenario likelihoods between precision="mixed" and float64, against the scatter of the same evidence over random draws, on the synthetic input of bench.py.

usage: python benchmarks/precision.py [--draws 10000] [--points 1000] [--seeds 10] [--noise-scale 1] [--json PATH]

The light curve is a transit of a planet of 8 Earth radii (impact parameter 0.3, 3 days) around a Sun-like star, seen through the exposure with the model of the package, plus Gaussian noise of 5e-4 drawn from the seed. The draws are those of bench.make_input.

For each scenario and each of `seeds` seeds, the log-likelihood of every draw is computed with the Triton kernel in float64 and with precision="mixed" (3 exposure nodes). The evidence of the scenario is ln Z = ln mean exp(lnL) over the draws. For each scenario the output has four numbers:
  the largest shift of ln Z between the two precisions over the seeds
  the largest shift of the log-likelihood of one draw
  the standard deviation of ln Z over the seeds in float64 (the Monte Carlo noise of the evidence)
  the ratio of the first to the third
Needs a device. Prints the hardware and versions of bench.py."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src")]


def load_bench():
    """benchmarks/bench.py imported as a module, for its input and timing helpers."""
    spec = importlib.util.spec_from_file_location("velociraptor_bench", ROOT / "benchmarks" / "bench.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["velociraptor_bench"] = module
    spec.loader.exec_module(module)
    return module


def synthetic_flux(bench, time_axis, seed, noise_scale=1.0):
    """The light curve: the package's transit model of the reference planet plus noise."""
    from velociraptor import transit_flux

    star = 1.0
    period = bench.PERIOD
    a_rs = bench.semi_major_axis(star, period) / (star * bench.RSUN)
    k = 8.0 * 0.0091577 / star
    inc = np.arccos(0.3 / a_rs)
    model = transit_flux(time_axis, k, 0.4, 0.22, period, a_rs, inc, 0.0, np.pi / 2, 0.0, bench.EXPOSURE, bench.SAMPLES)
    return np.asarray(model) + np.random.default_rng(seed).normal(0.0, bench.NOISE * noise_scale, time_axis.size)


def log_mean_exp(values):
    """The logarithm of the mean of exp(values) over all values, a value that is not finite adding nothing to the sum; -inf when none is finite."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -np.inf
    top = finite.max()
    return float(top + np.log(np.exp(finite - top).sum() / values.size))


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--points", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--noise-scale", type=float, default=1.0, help="multiplies the noise of the light curve and the sigma of the likelihood; the eclipsing-binary likelihood is infinite for a draw whose secondary eclipse is 1.5 sigma deep or more, so at the default noise it can have no finite draw")
    parser.add_argument("--json", type=Path)
    options = parser.parse_args(arguments)
    bench = load_bench()
    from velociraptor import triton_analytic
    from velociraptor.torch_likelihood import lnl_eb, lnl_eb_twin, lnl_tp
    from velociraptor.triceratops import _bound

    previous = triton_analytic.PRECISION
    functions = [_bound(function, "cuda", "triton", 3) for function in (lnl_tp, lnl_eb, lnl_eb_twin)]
    names = ("tp", "eb", "twin")
    evidence = {name: {"float64": [], "mixed": []} for name in names}
    largest_draw_shift = {name: 0.0 for name in names}
    for seed in range(options.seeds):
        time_axis, _, sigma, scenarios = bench.make_input(options.draws, options.points, seed=bench.SEED + seed)
        data = (time_axis, synthetic_flux(bench, time_axis, bench.SEED + seed, options.noise_scale), sigma * options.noise_scale, scenarios)
        results = {}
        for precision in ("float64", "mixed"):
            triton_analytic.PRECISION = precision
            results[precision] = bench.calls(functions, data)
        triton_analytic.PRECISION = previous
        for name, low, high in zip(names, results["float64"], results["mixed"]):
            low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
            both = np.isfinite(low) & np.isfinite(high)
            largest_draw_shift[name] = max(largest_draw_shift[name], float(np.abs(low[both] - high[both]).max()) if both.any() else 0.0)
            evidence[name]["float64"].append(log_mean_exp(low))
            evidence[name]["mixed"].append(log_mean_exp(high))
    table = {}
    for name in names:
        full, mixed = np.array(evidence[name]["float64"]), np.array(evidence[name]["mixed"])
        if not (np.isfinite(full).all() and np.isfinite(mixed).all()):
            table[name] = {"note": "the evidence is not finite for some seed (no draw has a finite likelihood): no row"}
            print(f"{name}: {table[name]['note']}")
            continue
        shift = float(np.abs(full - mixed).max())
        scatter = float(full.std(ddof=1))
        table[name] = {"largest_shift_of_ln_Z": shift, "largest_shift_of_a_draw": largest_draw_shift[name], "scatter_of_ln_Z_over_seeds": scatter, "shift_over_scatter": shift / scatter if scatter > 0 else None}
        print(f"{name}: largest shift of ln Z {shift:.3g}, of one draw's log-likelihood {largest_draw_shift[name]:.3g}, scatter of ln Z over {options.seeds} seeds {scatter:.3g}, ratio {table[name]['shift_over_scatter']}")
    session = {"command": " ".join(sys.argv), "draws": options.draws, "points": options.points, "seeds": options.seeds, "noise_scale": options.noise_scale, "hardware": bench.hardware(), "versions": bench.versions(), "table": table}
    print(json.dumps({key: session[key] for key in ("hardware", "versions")}, indent=1))
    if options.json:
        options.json.write_text(json.dumps(session, indent=1) + "\n")


if __name__ == "__main__":
    main()
