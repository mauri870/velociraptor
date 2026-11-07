"""The closed-form flux on a device against the numpy closed form, on the grids of the tests. It compares the torch flux over the radius ratios, separations and limb-darkening pairs of the validation grid, the Triton kernel over orbits, exposures and radius ratios, and the scenario fluxes of the TRICERATOPS likelihoods on the device against the CPU.

usage: python benchmarks/device_agreement.py [--device cuda] [--json PATH]   (cuda is torch's name for both CUDA and ROCm devices)"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT / "benchmarks")]
from ring_guard import ring_guard
from test_analytic_torch import K, U, separations
from test_torch import ORBITS
from test_torch_likelihood import PERIOD, TIME, draws

from velociraptor import quadratic_flux, transit_flux
from velociraptor.analytic_torch import quadratic_flux_analytic_torch
from velociraptor.torch_likelihood import simulate_eb_flux, simulate_tp_flux

LIMITS = {"flux_grid": 1e-12, "triton_orbits_and_exposure": 1e-11, "planet_scenarios_torch_device_against_cpu": 1e-12, "eclipsing_binary_scenarios_torch_device_against_cpu": 1e-12, "planet_scenarios_triton_against_cpu": 1e-11}


def main(device, path):
    out = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0) if device != "cpu" else "cpu", "limits": LIMITS}
    worst = 0.0
    for k in K:
        z = separations(k)
        for u1, u2 in U:
            got = quadratic_flux_analytic_torch(torch.as_tensor(z, device=device), k, u1, u2).cpu().numpy()
            worst = max(worst, float(np.max(np.abs(got - quadratic_flux(z, k, u1, u2)))))
    out["flux_grid"] = worst
    worst = 0.0
    time = np.linspace(-0.3, 0.3, 301)
    if device != "cpu":
        from velociraptor.triton_model import transit_flux_triton

        for k in (0.001, 0.05, 0.3, 1.5):
            for period, a_rs, inc, ecc, omega in ORBITS:
                for exposure, samples in ((0.0, 1), (0.00139, 20), (0.02, 7)):
                    got = transit_flux_triton(torch.as_tensor(time, device=device), k, 0.4, 0.25, period, a_rs, inc, ecc, omega, exposure=exposure, samples=samples).cpu().numpy()
                    worst = max(worst, float(np.max(np.abs(got - transit_flux(time, k, 0.4, 0.25, period, a_rs, inc, ecc, omega, exposure=exposure, samples=samples)))))
        out["triton_orbits_and_exposure"] = worst
    d = draws(40)
    t = {key: torch.as_tensor(value, device=device) for key, value in d.items()}
    on_cpu = {key: torch.as_tensor(value) for key, value in d.items()}
    time_t = torch.as_tensor(TIME, device=device)
    args = lambda s: (s["radius"], PERIOD, s["inc"], s["a"], s["star"], s["u1"], s["u2"], s["ecc"], s["argp"], s["comp"])
    eb = lambda s: (s["eb_radius"], s["eb_flux"], PERIOD, s["inc"], s["a"], s["star"], s["u1"], s["u2"], s["ecc"], s["argp"], s["comp"], False)
    out["planet_scenarios_torch_device_against_cpu"] = float((simulate_tp_flux(time_t, *args(t)).cpu() - simulate_tp_flux(torch.as_tensor(TIME), *args(on_cpu))).abs().max())
    flux_d, depth_d = simulate_eb_flux(time_t, *eb(t))
    flux_c, depth_c = simulate_eb_flux(torch.as_tensor(TIME), *eb(on_cpu))
    out["eclipsing_binary_scenarios_torch_device_against_cpu"] = float(max((flux_d.cpu() - flux_c).abs().max(), (depth_d.cpu() - depth_c).abs().max()))
    if device != "cpu":
        out["planet_scenarios_triton_against_cpu"] = float((simulate_tp_flux(time_t, *args(t), kernel="triton").cpu() - simulate_tp_flux(torch.as_tensor(TIME), *args(on_cpu))).abs().max())
    out["holds"] = all(out[key] < limit for key, limit in LIMITS.items() if key in out)
    print(json.dumps(out, indent=1))
    if path:
        Path(path).write_text(json.dumps(out, indent=1) + "\n")
    return 0 if out["holds"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json")
    arguments = parser.parse_args()
    with ring_guard():
        code = main(arguments.device, arguments.json)
    sys.exit(code)
