"""The pairs (draw, point) that the first pass of the device kernel leaves to the closed form in each scenario of the input of bench.py, at the 20 midpoints of the exposure -> benchmarks/flagged.json.

usage: python benchmarks/flagged.py [--draws 10000] [--points 1000]

The closed form costs more than the series that the first pass uses for a planet wholly inside the disc. The share of flagged pairs of a scenario sets the work of its likelihood on the device. A scenario with a large radius ratio flags more pairs."""
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


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--points", type=int, default=1000)
    parser.add_argument("--json", type=Path, default=ROOT / "benchmarks" / "flagged.json")
    options = parser.parse_args(arguments)
    bench = load_bench()
    from velociraptor import triton_analytic
    from velociraptor.torch_likelihood import lnl_eb, lnl_eb_twin, lnl_tp
    from velociraptor.triceratops import _bound

    data = bench.make_input(options.draws, options.points)
    time_axis, flux, sigma, scenarios = data
    rows = []
    for name, label, function in (("tp", "planet", lnl_tp), ("eb", "eclipsing binary", lnl_eb), ("twin", "twin at twice the period", lnl_eb_twin)):
        bound = _bound(function, "cuda", "triton", None)
        arguments_ = lambda: [np.array(each) if isinstance(each, np.ndarray) else each for each in scenarios[name]["arguments"]]
        bound(np.array(time_axis), np.array(flux), sigma, *arguments_(), companion_fluxratio=np.array(scenarios[name]["comp"]), exptime=bench.EXPOSURE, nsamples=bench.SAMPLES)
        triton_analytic.TRACE = []
        bound(np.array(time_axis), np.array(flux), sigma, *arguments_(), companion_fluxratio=np.array(scenarios[name]["comp"]), exptime=bench.EXPOSURE, nsamples=bench.SAMPLES)
        calls = [entry for entry in triton_analytic.TRACE if entry["samples"] == bench.SAMPLES]
        triton_analytic.TRACE = None
        rows.append({"scenario": label, "pairs": sum(entry["pairs"] for entry in calls), "flagged": sum(entry["flagged_pairs"] for entry in calls)})
        print(rows[-1], flush=True)
    result = {"command": " ".join(["flagged.py", *sys.argv[1:]]), "draws": options.draws, "points": options.points, "samples": bench.SAMPLES, "rows": rows}
    options.json.write_text(json.dumps(result, indent=1) + "\n")
    return result


if __name__ == "__main__":
    main()
