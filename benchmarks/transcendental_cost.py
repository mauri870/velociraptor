"""Nanoseconds per evaluation of a float64 square root, atan2, sine, cosine, division and multiply-add on the device, each alone. In the kernel every lane applies the function OPS times to its own value in registers (a dependent chain). The kernel is launched
with ELEMENTS lanes. The output also gives the cost above a chain of multiply-adds of the same length.

usage: python benchmarks/transcendental_cost.py [--json PATH]   (inside the ring guard; each launch is a few milliseconds)"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "benchmarks")]
from ring_guard import ring_guard

OPS = 32
ELEMENTS = 1 << 17
BLOCK = 128  # 4 warps of 32 lanes: one lane to a thread
NAMES = ("multiply_add", "sqrt", "atan2", "sin", "cos", "divide")


@triton.jit
def _chain(x_ptr, out_ptr, n, WHICH: tl.constexpr, OPS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    x = tl.load(x_ptr + i, mask=m, other=0.5)
    y = x
    for _ in range(OPS):
        if WHICH == 0:
            y = y * x + 0.25
        elif WHICH == 1:
            y = tl.sqrt(y + x)
        elif WHICH == 2:
            y = libdevice.atan2(y, x + 0.5) + x
        elif WHICH == 3:
            y = tl.sin(y) + x
        elif WHICH == 4:
            y = tl.cos(y) + x
        else:
            y = x / (y + 0.75)
    tl.store(out_ptr + i, y, mask=m)


def seconds(which, x, out):
    """The median of 7 timed launches of the chain kernel for the operation `which` on `x`, in seconds, with the device synchronized around each launch."""
    run = lambda: _chain[(triton.cdiv(x.numel(), BLOCK),)](x, out, x.numel(), WHICH=which, OPS=OPS, BLOCK=BLOCK)
    run()
    torch.cuda.synchronize()
    times = []
    for _ in range(7):
        began = time.perf_counter()
        run()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - began)
    return statistics.median(times)


def main(path):
    x = torch.rand(ELEMENTS, dtype=torch.float64, device="cuda") + 0.1
    out = torch.empty_like(x)
    total = {name: seconds(which, x, out) for which, name in enumerate(NAMES)}
    base = total["multiply_add"]
    result = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0), "elements_per_launch": ELEMENTS, "evaluations_per_lane": OPS,
              "launch_seconds": total, "nanoseconds_per_evaluation": {name: (total[name] - 0.0) / (ELEMENTS * OPS) * 1e9 for name in NAMES},
              "nanoseconds_per_evaluation_above_multiply_add": {name: (total[name] - base) / (ELEMENTS * OPS) * 1e9 for name in NAMES if name != "multiply_add"}}
    print(json.dumps(result, indent=1))
    if path:
        Path(path).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--json")
    with ring_guard():
        main(parser.parse_args().json)
