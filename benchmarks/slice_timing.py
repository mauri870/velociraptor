"""Seconds of one launch of the fused kernel for one slice, stepping the slice up from a few thousand elements while the previous step stays under the limit, and the slice size that gives the target time.

usage: python benchmarks/slice_timing.py [--samples 20] [--target 0.010] [--stop-at 0.005] [--json PATH]
Each step launches one slice of ELEMENTS sub-sample elements (pairs times sub-samples) through transit_flux_triton and synchronizes. The time of a step is the median of 5 launches after one that is not counted. The run is inside the ring guard and stops
after the first step that takes longer than --stop-at seconds."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "benchmarks")]
from ring_guard import ring_guard

from velociraptor import triton_model


def one_slice(elements, samples):
    """Time the Triton model on a launch of `elements` sub-sample elements with `samples` sub-samples a point; returns the number of points and the median of 5 timed launches in seconds."""
    pairs = max(1, elements // samples)
    triton_model.ELEMENTS = pairs * samples
    generator = np.random.default_rng(0)
    time_axis = torch.as_tensor(generator.uniform(-0.3, 0.3, pairs), device="cuda")
    run = lambda: triton_model.transit_flux_triton(time_axis, 0.03, 0.4, 0.25, 4.0, 8.0, 1.55, 0.1, 1.0, exposure=0.00139, samples=samples)
    run()
    torch.cuda.synchronize()
    seconds = []
    for _ in range(5):
        began = time.perf_counter()
        run()
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - began)
    return pairs, statistics.median(seconds)


def main(samples, target, stop_at, path):
    rows = []
    elements = 1 << 12
    with ring_guard():
        while elements <= (1 << 22):
            pairs, seconds = one_slice(elements, samples)
            rows.append({"elements": pairs * samples, "pairs": pairs, "seconds": seconds})
            print(json.dumps(rows[-1]), flush=True)
            if seconds > stop_at:
                break
            elements *= 2
    last = rows[-1]
    out = {"samples": samples, "target_seconds": target, "rows": rows, "elements_for_target": int(last["elements"] * target / last["seconds"]), "torch": torch.__version__, "device": torch.cuda.get_device_name(0)}
    print(json.dumps(out, indent=1))
    if path:
        Path(path).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--target", type=float, default=0.010)
    parser.add_argument("--stop-at", type=float, default=0.005)
    parser.add_argument("--json")
    arguments = parser.parse_args()
    main(arguments.samples, arguments.target, arguments.stop_at, arguments.json)
