"""The scenario light curves and log-likelihoods of TRICERATOPS 1.1.0 (Giacalone et al. 2021, AJ 161, 24, arXiv:2002.00691; MIT licence, THIRD_PARTY/TRICERATOPS-LICENSE) on a device, with the transit model of velociraptor.

The functions take the draws of a scenario as 1-d tensors, one entry a draw, and the times of the observation as a 1-d tensor. They give float64 tensors of draws by points, as the `_p` functions of triceratops.likelihoods do. The argument names are
snake_case and the units are those of TRICERATOPS: radii in Earth or solar radii, the semi-major axis in cm, angles in degrees. The input tensors are not modified. Dilution, the minimum of the secondary eclipse and the chi-squared are computed on the device of
the inputs. The likelihood functions return one number a draw.

The functions re-express simulate_TP_transit_p, simulate_EB_transit_p, lnL_TP_p, lnL_EB_p and lnL_EB_twin_p of triceratops.likelihoods in release 1.1.0 of TRICERATOPS, which is released under the MIT License:
Copyright (c) 2020 Steven Giacalone. The text of the licence is THIRD_PARTY/TRICERATOPS-LICENSE.
"""

import math

import numpy as np
import torch

from velociraptor.torch_model import CHUNK

RSUN = 6.957e10  # cm, IAU 2015 nominal solar radius; astropy.constants.R_sun in cgs
REARTH = 6.3781e8  # cm, IAU 2015 nominal equatorial Earth radius; astropy.constants.R_earth in cgs
SECONDARY_TIMES = np.linspace(-0.05, 0.05, 25)  # the points at which the depth of the secondary eclipse is measured


def _model(kernel):
    """The transit flux function: "torch" (the closed form on any device) or "triton" (the fused kernel, on a CUDA or ROCm device)."""
    if kernel == "triton":
        from velociraptor.triton_model import transit_flux_triton

        return transit_flux_triton
    from velociraptor.analytic_torch import transit_flux_analytic_torch

    return transit_flux_analytic_torch


DEFAULT_NODES = 3  # Gauss-Legendre nodes of the exposure integral in the Triton kernel; exposure_nodes=None selects the package's equal parts (20 midpoints in TRICERATOPS)
DEFAULT = "default"


def _extra(kernel, chunk, exposure_nodes=DEFAULT):
    """The keywords of the transit function for a kernel. The Triton kernel integrates the exposure with DEFAULT_NODES nodes unless `exposure_nodes` is an integer or None (the midpoints of `samples` equal parts). The torch kernel has only the midpoints."""
    if kernel != "triton":
        if exposure_nodes not in (DEFAULT, None):
            raise ValueError("exposure_nodes is an option of the Triton kernel")
        return {"chunk": chunk}
    return {"exposure_nodes": DEFAULT_NODES if exposure_nodes == DEFAULT else exposure_nodes}


def _column(value, like):
    return torch.as_tensor(value, dtype=torch.float64, device=like.device).reshape(-1, 1)


def _guard(k):
    """Multiplies a radius ratio below 1 + 1e-6 by 0.999, as TRICERATOPS 1.1.0 does for an eclipsing binary."""
    return torch.where(k - 1.0 < 1e-6, k * 0.999, k)


def simulate_tp_flux(time, radius, period, inc, a, star_radius, u1, u2, ecc, argp, companion_fluxratio, companion_is_host=False, exposure=0.00139, samples=20, chunk=CHUNK, kernel="torch", exposure_nodes=DEFAULT):
    """Counterpart of simulate_TP_transit_p: the diluted flux of a transiting planet of `radius` Earth radii, draws by points."""
    time = torch.as_tensor(time, dtype=torch.float64)
    radius, period, inc, a, star_radius, u1, u2, ecc, argp, comp = (_column(v, time) for v in (radius, period, inc, a, star_radius, u1, u2, ecc, argp, companion_fluxratio))
    f_comp = comp / (1 - comp)
    flux = _model(kernel)(time, radius * REARTH / (star_radius * RSUN), u1, u2, period, a / (star_radius * RSUN), inc * (math.pi / 180.0), ecc, (90 - argp) * (math.pi / 180.0), exposure=exposure, samples=samples, **_extra(kernel, chunk, exposure_nodes))
    dilute = 1.0 / f_comp if companion_is_host else f_comp
    return (flux + dilute) / (1 + dilute)


def simulate_eb_flux(time, eb_radius, eb_fluxratio, period, inc, a, star_radius, u1, u2, ecc, argp, companion_fluxratio, companion_is_host=False, exposure=0.00139, samples=20, chunk=CHUNK, kernel="torch", exposure_nodes=DEFAULT):
    """Counterpart of simulate_EB_transit_p: the diluted flux of an eclipsing binary of `eb_radius` solar radii (draws by points) and the depth of its secondary eclipse (draws by 1)."""
    time = torch.as_tensor(time, dtype=torch.float64)
    eb_radius, eb_fluxratio, period, inc, a, star_radius, u1, u2, ecc, argp, comp = (_column(v, time) for v in (eb_radius, eb_fluxratio, period, inc, a, star_radius, u1, u2, ecc, argp, companion_fluxratio))
    f_comp = comp / (1 - comp)
    f_eb = eb_fluxratio / (1 - eb_fluxratio)
    a = a / (star_radius * RSUN)
    inc = inc * (math.pi / 180.0)
    flux = _model(kernel)(time, _guard(eb_radius / star_radius), u1, u2, period, a, inc, ecc, (90 - argp) * (math.pi / 180.0), exposure=exposure, samples=samples, **_extra(kernel, chunk, exposure_nodes))
    secondary_time = torch.as_tensor(SECONDARY_TIMES, dtype=torch.float64, device=time.device)
    sec_flux = _model(kernel)(secondary_time, _guard(star_radius / eb_radius), u1, u2, period, a, inc, ecc, (90 - argp + 180) * (math.pi / 180.0), **_extra(kernel, chunk, exposure_nodes)).min(dim=1, keepdim=True).values
    if companion_is_host:
        flux = (flux + f_eb / f_comp) / (1 + f_eb / f_comp)
        sec_flux = (sec_flux + f_comp / f_eb) / (1 + f_comp / f_eb)
        dilute = 1.0 / (f_comp + f_eb)
    else:
        flux = (flux + f_eb) / (1 + f_eb)
        sec_flux = (sec_flux + 1.0 / f_eb) / (1 + 1.0 / f_eb)
        dilute = f_comp / (1 + f_eb)
    flux = (flux + dilute) / (1 + dilute)
    return flux, 1 - (sec_flux + dilute) / (1 + dilute)


BLOCK = 32  # pairs of a draw and a point in a block of the Triton kernel (velociraptor.triton_analytic.BLOCK)
MAX_ELEMENTS = 1 << 26  # draws times points of one chunk of a likelihood; a call holds three float64 arrays of draws by points on the device at its peak (24 bytes an element), so a chunk of 2^26 elements is 1.6 GiB whatever the number of draws


def _per_draw(value, count):
    return hasattr(value, "shape") and len(value.shape) >= 1 and value.shape[0] == count


def _in_chunks(function):
    """`function` (a likelihood of the arguments `time, flux, sigma, *scenario`) on chunks of draws with at most MAX_ELEMENTS draws times points each, the results joined. The draws are independent, so the values are those of the whole, bit for bit. The device memory of a call depends on the chunk and not on the number of draws."""

    def wrapper(time, flux, sigma, *scenario, **options):
        first_argument = scenario[0]
        count = first_argument.shape[0] if hasattr(first_argument, "shape") and len(first_argument.shape) >= 1 else 1
        points = max(int(time.numel() if hasattr(time, "numel") else len(time)), 1)
        rows = max(1, MAX_ELEMENTS // points)
        align = BLOCK // math.gcd(points, BLOCK)  # a chunk starts at a multiple of the kernel's block of pairs, so its blocks are those of the whole
        if rows >= align:
            rows -= rows % align
        if count <= rows:
            return function(time, flux, sigma, *scenario, **options)
        parts = []
        for first in range(0, count, rows):
            last = min(first + rows, count)
            parts.append(function(time, flux, sigma, *[value[first:last] if _per_draw(value, count) else value for value in scenario], **options))
        return torch.cat(parts)

    wrapper.__name__, wrapper.__doc__ = function.__name__, function.__doc__
    return wrapper


def _chi_squared(flux, model, sigma):
    flux = torch.as_tensor(flux, dtype=torch.float64, device=model.device)
    return 0.5 * torch.sum((flux - model) ** 2 / sigma**2, dim=1)


@_in_chunks
def lnl_tp(time, flux, sigma, *scenario, **options):
    """Counterpart of lnL_TP_p: half the chi-squared of the planet scenario against `flux`, one number a draw. `scenario` is the arguments of simulate_tp_flux after `time`."""
    return _chi_squared(flux, simulate_tp_flux(time, *scenario, **options), sigma)


@_in_chunks
def lnl_eb(time, flux, sigma, *scenario, **options):
    """Counterpart of lnL_EB_p: half the chi-squared, and infinity for a draw whose secondary eclipse is 1.5 sigma deep or more. `scenario` is the arguments of simulate_eb_flux after `time`."""
    model, depth = simulate_eb_flux(time, *scenario, **options)
    chi = _chi_squared(flux, model, sigma)
    return torch.where(depth[:, 0] < 1.5 * sigma, chi, torch.full_like(chi, math.inf))


@_in_chunks
def lnl_eb_twin(time, flux, sigma, *scenario, **options):
    """Counterpart of lnL_EB_twin_p: the chi-squared without the secondary-eclipse mask (the caller gives twice the period)."""
    model, _ = simulate_eb_flux(time, *scenario, **options)
    return _chi_squared(flux, model, sigma)
