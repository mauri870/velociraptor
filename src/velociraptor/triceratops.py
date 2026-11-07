"""TRICERATOPS 1.1.0 with the log-likelihoods of the parallel path computed by velociraptor on a device.

    from velociraptor.triceratops import install
    with install("cuda"):
        target.calc_probs(time=time, flux_0=flux, flux_err_0=flux_err, P_orb=period, N=100000, parallel=True)

Inside the block `triceratops.marginal_likelihoods.lnL_TP_p`, `lnL_EB_p` and `lnL_EB_twin_p` are the functions of velociraptor.torch_likelihood, taking and returning numpy arrays. The package's own are put back on leaving the block. The seed, the draws and the scenario probabilities are the package's.
Needs the torch extra and TRICERATOPS 1.1.0.

The functions re-express simulate_TP_transit_p, simulate_EB_transit_p, lnL_TP_p, lnL_EB_p and lnL_EB_twin_p of triceratops.likelihoods in release 1.1.0 of TRICERATOPS, which is released under the MIT License:
Copyright (c) 2020 Steven Giacalone. The text of the licence is THIRD_PARTY/TRICERATOPS-LICENSE.
"""
import contextlib
import inspect
import zlib

import numpy as np
import torch

from velociraptor.overlap import overlapped_calc_probs
from velociraptor.splines import released_splines
from velociraptor.torch_likelihood import DEFAULT, lnl_eb, lnl_eb_twin, lnl_tp

NAMES = ("lnL_TP_p", "lnL_EB_p", "lnL_EB_twin_p")


def _bound(function, device, kernel, exposure_nodes):
    def wrapper(time, flux, sigma, *scenario, companion_fluxratio=None, companion_is_host=False, exptime=0.00139, nsamples=20, **unknown):
        if unknown:
            raise TypeError(f"unexpected keyword argument(s): {', '.join(unknown)}")
        if companion_fluxratio is None:
            raise TypeError("companion_fluxratio is required, by keyword")
        converted = [torch.as_tensor(value, device=device) if hasattr(value, "shape") else value for value in scenario]
        converted.append(torch.as_tensor(companion_fluxratio, device=device))
        return function(torch.as_tensor(time, device=device), torch.as_tensor(flux, device=device), float(sigma), *converted, companion_is_host=companion_is_host, exposure=exptime, samples=nsamples, kernel=kernel, exposure_nodes=exposure_nodes).cpu().numpy()

    return wrapper


STOCK_LOOKUP = """    for i, (comp_Teff, comp_logg) in enumerate(
            zip(rounded_Teffs_comp, rounded_loggs_comp)
            ):
        mask = (Teffs_at_Z == comp_Teff) & (loggs_at_Z == comp_logg)
        u1s_comp[i], u2s_comp[i] = u1s_at_Z[mask].item(), u2s_at_Z[mask].item()
"""
GATHERED_LOOKUP = """    pairs, inverse = np.unique(np.stack([rounded_Teffs_comp, rounded_loggs_comp]), axis=1, return_inverse=True)
    found = np.array([(u1s_at_Z[(Teffs_at_Z == t) & (loggs_at_Z == g)].item(), u2s_at_Z[(Teffs_at_Z == t) & (loggs_at_Z == g)].item()) for t, g in pairs.T])
    u1s_comp, u2s_comp = found[inverse.ravel(), 0], found[inverse.ravel(), 1]
"""
LOOKUP_NAMES = ("lnZ_STP", "lnZ_SEB")


def with_gathered_lookup(function):
    """`function` (lnZ_STP or lnZ_SEB of TRICERATOPS 1.1.0) with the limb-darkening lookup of its drawn companions done once for each distinct (Teff, logg) pair. The package loops over the draws, masks the table and calls .item() twice for each draw.
    The values are the same, element by element. The draws and their order are untouched. A pair that is not in the table raises the same ValueError of .item(). The function is built from the installed source with the loop replaced.
    A source that does not hold the loop exactly once raises RuntimeError. The function runs in the namespace of the package's module, so a later replacement of its likelihood functions takes effect in it."""
    source = inspect.getsource(function)
    if source.count(STOCK_LOOKUP) != 1:
        raise RuntimeError(f"{function.__name__} of the installed TRICERATOPS does not hold the limb-darkening loop that this replacement was written for exactly once")
    name = f"_{function.__name__}_gathered_lookup"
    patched = source.replace(STOCK_LOOKUP, GATHERED_LOOKUP).replace(f"def {function.__name__}(", f"def {name}(", 1)
    scope = function.__globals__
    exec(compile(patched, inspect.getsourcefile(function), "exec"), scope)
    built = scope.pop(name)
    built.__name__, built.__qualname__ = function.__name__, function.__name__
    return built


@contextlib.contextmanager
def remembered_splines():
    """Inside the block a call of a scipy InterpolatedUnivariateSpline on an array it has already been called on, with the same arguments, returns a copy of the earlier values. TRICERATOPS evaluates its stellar-relation splines repeatedly on the same arrays of masses. The key is the spline and a CRC-32 of the bytes of the input, so a changed array is a different key. The values are exactly the ones the spline gives. Arrays under 1000 elements are not remembered."""
    from scipy.interpolate import InterpolatedUnivariateSpline

    original = InterpolatedUnivariateSpline.__call__
    memory = {}

    def call(self, x, nu=0, ext=None):
        array = np.ascontiguousarray(x, dtype=float) if isinstance(x, np.ndarray) else None
        if array is None or array.size < 1000:
            return original(self, x, nu=nu, ext=ext)
        key = (id(self), array.shape, zlib.crc32(array.view(np.uint8)), nu, ext)
        if key not in memory:
            memory[key] = original(self, x, nu=nu, ext=ext)
        return memory[key].copy()

    InterpolatedUnivariateSpline.__call__ = call
    try:
        yield memory
    finally:
        InterpolatedUnivariateSpline.__call__ = original


def default_kernel(device):
    """The kernel install() uses when none is named: the Triton kernel on a device ("cuda", "hip"), the torch kernel on the CPU."""
    return "torch" if str(device).split(":")[0] == "cpu" else "triton"


@contextlib.contextmanager
def install(device, kernel=None, exposure_nodes=DEFAULT, gathered_lookup=True, remember_splines=True, overlap=3, release_splines=True):
    """Replace the three likelihood functions of triceratops.marginal_likelihoods by the ones of velociraptor on `device` ("cpu", "cuda") for the block.

    `kernel` is "torch" or "triton". The default is "triton" on a device and "torch" on the CPU.
    `remember_splines`: the stellar-relation splines of the package are not evaluated twice on the same array (see remembered_splines).
    `gathered_lookup`: the limb-darkening lookup of the companions in lnZ_STP and lnZ_SEB is made once for each distinct pair, which gives the same numbers (see with_gathered_lookup).
    `release_splines`: those splines are evaluated by velociraptor.splines.evaluate, which gives the same values to the last bit and holds no lock, so the overlap can run it beside another scenario.
    `overlap`: calc_probs runs the sampling of a scenario while the likelihood of the one before it is on the device, with the draws, their order and the results of the sequential run (see velociraptor.overlap). 0 or 1 keeps the package's own order.
    `exposure_nodes`: the Triton kernel integrates the exposure with this many Gauss-Legendre nodes (3 by default). None takes the package's `nsamples` equal parts (20 in TRICERATOPS), the setting to use when comparing with stock TRICERATOPS. The torch kernel has only the equal parts."""
    from triceratops import marginal_likelihoods
    from triceratops import triceratops as package

    kernel = default_kernel(device) if kernel is None else kernel
    original = {name: getattr(marginal_likelihoods, name) for name in NAMES}
    for name, function in zip(NAMES, (lnl_tp, lnl_eb, lnl_eb_twin)):
        setattr(marginal_likelihoods, name, _bound(function, device, kernel, exposure_nodes))
    lookups = {}
    if gathered_lookup:  # calc_probs finds the lnZ functions by name in triceratops.triceratops, where they were imported
        lookups = {name: (getattr(marginal_likelihoods, name), getattr(package, name)) for name in LOOKUP_NAMES}
        for name, (function, _) in lookups.items():
            patched = with_gathered_lookup(function)
            setattr(marginal_likelihoods, name, patched)
            setattr(package, name, patched)
    try:
        with (released_splines() if release_splines else contextlib.nullcontext()), (remembered_splines() if remember_splines else contextlib.nullcontext()), overlapped_calc_probs(overlap):
            yield
    finally:
        for name, function in original.items():
            setattr(marginal_likelihoods, name, function)
        for name, (module_function, package_function) in lookups.items():
            setattr(marginal_likelihoods, name, module_function)
            setattr(package, name, package_function)
