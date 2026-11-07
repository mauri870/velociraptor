"""Seconds of the three scenario likelihoods of TRICERATOPS 1.1.0, as the package computes them and as TRICERATOPS does, on the same draws.

usage: python benchmarks/bench.py [--draws 10000 100000] [--points 1000] [--repeats 5] [--threads 1 2 4 8 16 32] [--cpu-only] [--allow-busy] [--json PATH]
       python benchmarks/bench.py --publish PATH.json     adds the tables of a session to benchmarks/results.json

One call is lnL_TP_p, lnL_EB_p and lnL_EB_twin_p on `draws` draws of the parameters and `points` points of one light curve (README.md in this directory has the input and the rows). The input is made here from a fixed seed. The methods:
  triceratops-numba      the functions of triceratops.likelihoods, at the numba thread count that was fastest in the scan
  velociraptor-cpu       velociraptor.triceratops.install("cpu"), torch with the same number of threads
  velociraptor-device-default install("cuda"), what the package runs without arguments: 3 Gauss-Legendre nodes and the first pass in float32, inside the ring guard
  velociraptor-device-g3 the same as the 20 midpoints in float64 with exposure_nodes=3, the default of the package's calculations (the per-element cost of the 20 midpoints is 20/3 times larger)
  velociraptor-device-mixed the same 20 midpoints with the exposure nodes of the first pass in float32 (triton_analytic.PRECISION = "mixed")
  velociraptor-device    install("cuda", kernel="triton", exposure_nodes=None) (the 20 midpoints of the exposure, the rule of the recorded results), inside the ring guard; refused while another process holds the compute device
Every row has one call that is not counted, then --repeats timed calls. Writes the session (hardware, versions, command, scan, rows, agreement) to --json."""
import argparse
import contextlib
import glob
import json
import math
import os
import platform
import re
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "benchmarks")]

SEED = 20261031
PERIOD = 3.0  # days
NOISE = 5e-4  # normalized flux
DEPTH, HALF_DURATION = 1e-3, 0.06  # the box in the light curve, days
WINDOW = 0.3  # the times run over -WINDOW to WINDOW days from the transit midpoint
EXPOSURE = 0.00139  # days, the exposure time of TRICERATOPS's default
SAMPLES = 20
GM_SUN = 1.32712440018e26  # cm^3 s^-2
RSUN = 6.957e10  # cm
MAX_ARRAY_GIB = 2.0
METHODS = ("triceratops-numba", "velociraptor-cpu", "velociraptor-device", "velociraptor-device-mixed", "velociraptor-device-g3", "velociraptor-device-default")
DRAW_DISTRIBUTIONS = (
    "radius of the star 0.6 to 1.4 solar, mass equal to the radius in solar units, impact parameter 0 to 1, eccentricity 0 to 0.2, argument of periastron 0 to 360 degrees, limb darkening u1 0.3 to 0.5 and u2 0.15 to 0.3, "
    "flux share of the unresolved companion 0 to 0.3, planet radius 1 to 15 Earth radii, eclipsing binary radius 0.1 to 1 solar, its flux ratio 0.01 to 0.5; the period is 3 days and twice that for the twin"
)


def semi_major_axis(radius, period):
    """cm, from Kepler's third law for a star of mass equal to its radius in solar units."""
    return (GM_SUN * radius * (period * 86400.0) ** 2 / (4.0 * math.pi**2)) ** (1.0 / 3.0)


def make_input(draws, points, seed=SEED):
    """The light curve (time, flux, sigma) and the draws of each scenario, from `seed` alone: {"tp", "eb", "twin"} each (arguments after `sigma` in the order of triceratops.likelihoods, period, companion flux share)."""
    rng = np.random.default_rng(seed)
    time_axis = np.linspace(-WINDOW, WINDOW, points)
    flux = 1.0 + rng.normal(0.0, NOISE, points) - DEPTH * (np.abs(time_axis) < HALF_DURATION)
    star = rng.uniform(0.6, 1.4, draws)
    impact = rng.uniform(0.0, 1.0, draws)
    common = {"star": star, "ecc": rng.uniform(0.0, 0.2, draws), "argp": rng.uniform(0.0, 360.0, draws), "u1": rng.uniform(0.3, 0.5, draws), "u2": rng.uniform(0.15, 0.3, draws), "comp": rng.uniform(0.0, 0.3, draws)}
    planet = rng.uniform(1.0, 15.0, draws)
    binary, ratio = rng.uniform(0.1, 1.0, draws), rng.uniform(0.01, 0.5, draws)

    def orbit(period):
        axis = semi_major_axis(star, period)
        return axis, np.degrees(np.arccos(np.clip(impact * star * RSUN / axis, 0.0, 1.0)))

    scenarios = {}
    for name, period, first in (("tp", PERIOD, (planet,)), ("eb", PERIOD, (binary, ratio)), ("twin", 2 * PERIOD, (binary, ratio))):
        axis, inclination = orbit(period)
        scenarios[name] = {"arguments": (*first, period, inclination, axis, star, common["u1"], common["u2"], common["ecc"], common["argp"]), "comp": common["comp"]}
    return time_axis, flux, NOISE, scenarios


def calls(functions, data):
    """The three likelihoods on the data, as one list of numpy arrays: `functions` is (tp, eb, twin) with the signature of triceratops.likelihoods."""
    time_axis, flux, sigma, scenarios = data
    out = []
    for function, name in zip(functions, ("tp", "eb", "twin")):
        arguments = [np.array(each) if isinstance(each, np.ndarray) else each for each in scenarios[name]["arguments"]]  # simulate_TP_transit_p and its siblings convert `inc` from degrees to radians in place, so each call gets copies
        out.append(function(np.array(time_axis), np.array(flux), sigma, *arguments, companion_fluxratio=np.array(scenarios[name]["comp"]), exptime=EXPOSURE, nsamples=SAMPLES))
    return out


def timed(call, repeats):
    """(outputs of the first call, seconds of that call, seconds of each of `repeats` further calls): the first call compiles and warms the caches and is not counted."""
    began = time.perf_counter()
    first = call()
    first_seconds = time.perf_counter() - began
    seconds = []
    for _ in range(repeats):
        began = time.perf_counter()
        call()
        seconds.append(time.perf_counter() - began)
    return first, first_seconds, seconds


def agreement(reference, other):
    """Largest |difference| over max(|reference|, 1) of the finite entries of each function, and the number of entries where only one of the two is infinite."""
    largest, mismatched = 0.0, 0
    for left, right in zip(reference, other):
        left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
        finite = np.isfinite(left) & np.isfinite(right)
        mismatched += int(np.sum(np.isfinite(left) != np.isfinite(right)))
        if finite.any():
            largest = max(largest, float(np.max(np.abs(left[finite] - right[finite]) / np.maximum(np.abs(left[finite]), 1.0))))
    return {"largest_difference": largest, "infinity_mismatches": mismatched}


def summary(seconds):
    """The median and the first and third quartiles of the timed calls, with the seconds of each call."""
    quartiles = statistics.quantiles(seconds, n=4) if len(seconds) > 1 else [seconds[0], seconds[0], seconds[0]]
    return {"median": statistics.median(seconds), "q1": quartiles[0], "q3": quartiles[2], "seconds": seconds}


def other_device_jobs(proc="/proc", own=None):
    """Process ids other than this one and its descendants that hold /dev/kfd open: a render node is held by every desktop program, the compute device only by a compute job."""
    own = os.getpid() if own is None else own

    def descends(pid):
        for _ in range(64):
            try:
                parent = int(open(os.path.join(proc, str(pid), "stat")).read().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                return False
            if parent == own:
                return True
            if parent <= 1:
                return False
            pid = parent
        return False

    found = set()
    for path in glob.glob(os.path.join(proc, "[0-9]*", "fd", "*")):
        try:
            if os.readlink(path) != "/dev/kfd":
                continue
        except OSError:
            continue
        pid = int(path.split(os.sep)[-3])
        if pid != own and not descends(pid):
            found.add(pid)
    return sorted(found)


def hardware():
    """The CPU, the cores the process may use, the GPU and its runtime when torch sees one, and the load average at the start."""
    cpu = next((line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name")), platform.processor()) if os.path.exists("/proc/cpuinfo") else platform.processor()
    found = {"cpu": cpu, "logical_cpus": os.cpu_count(), "cpus_allowed": len(os.sched_getaffinity(0)), "gpu": None, "gpu_runtime": None, "load_average_at_start": os.getloadavg()[0]}
    try:
        import torch

        if torch.cuda.is_available():
            found["gpu"] = torch.cuda.get_device_name(0)
            found["gpu_runtime"] = f"ROCm {torch.version.hip}" if torch.version.hip else f"CUDA {torch.version.cuda}"
    except ImportError:
        pass
    return found


def versions():
    """The versions of python and of the installed packages that the timings depend on; a package that is not installed is left out."""
    from importlib import metadata

    found = {"python": platform.python_version()}
    for name in ("numpy", "numba", "torch", "triton-rocm", "triton", "triceratops", "pytransit", "velociraptor"):
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return found


def refuse_large(draws, points, limit_gib):
    """Exit with the reason when an array of `draws` by `points` float64 values is larger than `limit_gib` GiB, since TRICERATOPS holds several such arrays."""
    gib = draws * points * 8 / 2**30
    if gib > limit_gib:
        raise SystemExit(f"{draws} draws by {points} points is an array of {gib:.1f} GiB of float64 and TRICERATOPS holds several; the limit is --max-array-gib {limit_gib}")


def cpus_for(count, logical=None):
    """The logical CPUs for `count` threads on a machine whose CPU n and n + cores are the two threads of core n: one thread to a core up to the number of cores, in the order of the core numbers (so a count up to half the cores stays in the first group of cores
    that shares a cache), and every logical CPU above it, where the second thread of a core is used."""
    logical = logical or sorted(os.sched_getaffinity(0))
    cores = len(logical) // 2 if len(logical) > 1 else 1
    return set(logical[:count]) if count <= cores else set(logical)


def pin(cpus):
    """Restrict every thread of the process, the workers of numba and torch included, to `cpus` (sched_setaffinity(0, ...) would restrict only the calling thread)."""
    for thread in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(thread), cpus)
        except (ProcessLookupError, PermissionError):
            pass


def scan_threads(data, counts, repeats_discarded=1):
    """Seconds of one call of the numba path at each thread count, after a call that is not counted at that count; the fastest count and the seconds of each."""
    import numba
    from triceratops import likelihoods

    functions = (likelihoods.lnL_TP_p, likelihoods.lnL_EB_p, likelihoods.lnL_EB_twin_p)
    seconds = {}
    allowed = sorted(os.sched_getaffinity(0))
    for count in counts:
        numba.set_num_threads(count)
        pin(cpus_for(count, allowed))
        for _ in range(repeats_discarded):
            calls(functions, data)
        began = time.perf_counter()
        calls(functions, data)
        seconds[count] = time.perf_counter() - began
    return min(seconds, key=seconds.get), seconds


def thread_counts(requested):
    """The requested thread counts that the process can run: at most the cores it may use and the threads numba started with; [1] when none fits."""
    import numba

    limit = min(len(os.sched_getaffinity(0)), numba.config.NUMBA_NUM_THREADS)
    return [count for count in requested if count <= limit] or [1]


def pkg_resources_stand_in():
    """PyTransit 2.2.0, which TRICERATOPS imports, calls pkg_resources.resource_filename, and setuptools 81 removed that module. Where it is missing, a module with that one function is registered."""
    import types

    try:
        import pkg_resources  # noqa: F401
    except ImportError:
        module = types.ModuleType("pkg_resources")

        def resource_filename(package, name):
            __import__(package)
            return str(Path(sys.modules[package].__file__).parent / name)

        module.resource_filename = resource_filename
        sys.modules["pkg_resources"] = module


@contextlib.contextmanager
def device_precision(precision, exposure_nodes):
    """install("cuda", kernel="triton", exposure_nodes=exposure_nodes) with the precision of the first pass set for the block and put back after it, also after an error."""
    from velociraptor import triton_analytic
    from velociraptor.triceratops import install

    previous = triton_analytic.PRECISION
    triton_analytic.PRECISION = precision
    try:
        with install("cuda", kernel="triton", exposure_nodes=exposure_nodes):
            yield
    finally:
        triton_analytic.PRECISION = previous


def mixed_precision():
    """The 20 midpoints with the exposure nodes of the first pass in float32."""
    return device_precision("mixed", None)


def session(options):
    """Time every row of the options and return the record of the run: its hardware, versions, command, input and rows, with the rows named by --skip recorded as skipped."""
    import numba
    import torch

    pkg_resources_stand_in()
    from triceratops import likelihoods, marginal_likelihoods

    from velociraptor.triceratops import install

    device = not options.cpu_only and torch.cuda.is_available()
    stock = (likelihoods.lnL_TP_p, likelihoods.lnL_EB_p, likelihoods.lnL_EB_twin_p)
    report = {"skipped": [], "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]), "hardware": hardware(), "versions": versions(), "seed": SEED, "input": {"points": options.points, "nsamples": SAMPLES, "exposure_days": EXPOSURE, "period_days": PERIOD, "noise": NOISE, "draw_distributions": DRAW_DISTRIBUTIONS}, "rows": []}
    for draws in options.draws:
        refuse_large(draws, options.points, options.max_array_gib)
    largest = make_input(max(options.draws), options.points)
    initial_cpus = os.sched_getaffinity(0)
    best, seconds = scan_threads(largest, thread_counts(options.threads))
    numba.set_num_threads(best)
    torch.set_num_threads(best)
    pin(cpus_for(best, sorted(initial_cpus)))
    report["scan"] = {"draws": max(options.draws), "seconds_by_threads": {str(count): value for count, value in seconds.items()}, "fastest_threads": best, "cpus_by_threads": {str(count): sorted(cpus_for(count, sorted(initial_cpus))) for count in seconds}}
    functions = {"velociraptor-cpu": lambda: install("cpu"), "velociraptor-device": lambda: device_precision("float64", None), "velociraptor-device-mixed": mixed_precision, "velociraptor-device-g3": lambda: device_precision("float64", 3), "velociraptor-device-default": lambda: install("cuda")}
    names = ("lnL_TP_p", "lnL_EB_p", "lnL_EB_twin_p")
    for draws in options.draws:
        data = make_input(draws, options.points)
        reference = None
        for method in METHODS:
            if f"{method}:{draws}" in options.skip:
                report["skipped"].append({"method": method, "draws": draws, "reason": options.skip_reason})
                continue
            if method.startswith("velociraptor-device") and not device:
                continue
            if method.startswith("velociraptor-device") and not options.allow_busy and other_device_jobs():
                raise SystemExit(f"the compute device is held by process {other_device_jobs()}; no device row is run (--allow-busy overrides)")
            if method == "triceratops-numba":
                outputs, first, repeated = timed(lambda: calls(stock, data), options.repeats)
                reference = outputs
            else:
                from ring_guard import ring_guard

                with functions[method](), (ring_guard() if method.startswith("velociraptor-device") else contextlib.nullcontext()):
                    patched = tuple(getattr(marginal_likelihoods, name) for name in names)
                    outputs, first, repeated = timed(lambda: calls(patched, data), options.repeats)
            row = {"method": method, "draws": draws, "points": options.points, "threads": best if not method.startswith("velociraptor-device") else None, "first_call_seconds": first, **summary(repeated), "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
            if method != "triceratops-numba":
                row["agreement"] = agreement(reference, outputs)
            if method.startswith("velociraptor-device"):
                from velociraptor import triton_model

                row["slices"] = {"elements": triton_model.ELEMENTS, "per_call": [len(triton_model.plan_slices(draws * options.points, 3 if method.endswith("g3") else SAMPLES))] * 3}
            report["rows"].append(row)
            print(json.dumps({key: row[key] for key in ("method", "draws", "median")}), flush=True)
    report["results"] = tables_of(report)
    return report


def caption_command(command):
    """The command of a session as the caption of its table shows it: without the text of --skip-reason (the table names the reason under it) and with the file of --json left out."""
    command = re.sub(r" --skip-reason .*?(?= --[a-z]|$)", "", command)
    return re.sub(r"--json \S+", "--json ...", command)


def tables_of(report):
    """The entries of benchmarks/results.json of the session, one for each size, in the shape that tables.py reads."""
    hardware_text = report["hardware"]["cpu"] + (f", {report['hardware']['gpu']} ({report['hardware']['gpu_runtime']})" if report["hardware"]["gpu"] else "")
    versions_text = ", ".join(f"{name} {value}" for name, value in report["versions"].items())
    labels = {"triceratops-numba": "TRICERATOPS 1.1.0, numba path, {threads} threads", "velociraptor-cpu": "velociraptor, torch on the CPU, {threads} threads", "velociraptor-device": "velociraptor, fused kernel on the device, 20 midpoints", "velociraptor-device-mixed": "velociraptor, fused kernel on the device, 20 midpoints, first pass in float32", "velociraptor-device-g3": "velociraptor, fused kernel on the device, 3 Gauss-Legendre nodes in the exposure", "velociraptor-device-default": "velociraptor, install(\"cuda\") with no arguments: 3 Gauss-Legendre nodes, first pass in float32"}
    out = []
    threads = next((row["threads"] for row in report["rows"] if row.get("threads")), None)
    for draws in sorted({row["draws"] for row in report["rows"]}):
        rows = [row for row in report["rows"] if row["draws"] == draws]
        out.append({
            "label": f"The three scenario likelihoods of TRICERATOPS, {draws} draws and {report['input']['points']} points each",
            "machine": hardware_text,
            "command": caption_command(report["command"]),
            "versions": versions_text,
            "rows": [{"method": labels[row["method"]].format(threads=row["threads"]), "scenarios": 3 * draws, "seconds": row["median"]} for row in rows],
            "not_run": [{"method": labels[skip["method"]].format(threads=threads), "reason": skip["reason"]} for skip in report.get("skipped", []) if skip["draws"] == draws],
        })
    return out


def publish(path):
    """The entries of a session replace those of results.json with the same label and machine, and the others are added."""
    target = ROOT / "benchmarks" / "results.json"
    existing = json.loads(target.read_text())
    session = json.loads(Path(path).read_text())
    added = tables_of(session) if "rows" in session else session["results"]  # the tables of a full session are regenerated, so a change of the generator reaches the stored ones
    keys = {(entry["label"], entry["machine"]) for entry in added}
    existing["results"] = [entry for entry in existing["results"] if (entry["label"], entry["machine"]) not in keys] + added
    target.write_text(json.dumps(existing, indent=1) + "\n")


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--draws", type=int, nargs="+", default=[10000, 100000])
    parser.add_argument("--points", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--cpu-only", action="store_true", help="no device row, and torch is not asked for a device")
    parser.add_argument("--allow-busy", action="store_true", help="run the device row although another process holds the compute device")
    parser.add_argument("--skip", nargs="+", default=[], metavar="METHOD:DRAWS", help="rows not run, for example velociraptor-cpu:100000; the session records them as skipped, with --skip-reason")
    parser.add_argument("--skip-reason", default="", help="why the --skip rows were not run")
    parser.add_argument("--max-array-gib", type=float, default=MAX_ARRAY_GIB)
    parser.add_argument("--json")
    parser.add_argument("--publish")
    options = parser.parse_args(arguments)
    if options.publish:
        return publish(options.publish)
    report = session(options)
    if options.json:
        Path(options.json).write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    main()
