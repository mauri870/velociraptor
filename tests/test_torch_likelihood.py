"""The device likelihoods against an independent numpy composition of velociraptor.transit_flux (tight tolerance) and against TRICERATOPS 1.1.0 with PyTransit (loose tolerance, because PyTransit itself deviates by up to 5e-6 in flux)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from velociraptor import transit_flux
from velociraptor.torch_likelihood import (
    REARTH,
    RSUN,
    lnl_eb,
    lnl_eb_twin,
    lnl_tp,
    simulate_eb_flux,
    simulate_tp_flux,
)

TIME = np.linspace(-0.25, 0.25, 121)
PERIOD = 4.2
SIGMA = 2e-4


def draws(n, seed=3):
    g = np.random.default_rng(seed)
    return {
        "radius": g.uniform(0.5, 12.0, n),  # Earth radii for a planet
        "inc": g.uniform(86.0, 90.0, n),
        "a": g.uniform(10.0, 16.0, n) * (star := g.uniform(0.8, 1.2, n)) * RSUN,  # a/R* from 10 and e up to 0.1, where the orbit of PyTransit agrees with the exact one to 5e-6
        "star": star,
        "u1": g.uniform(0.2, 0.6, n),
        "u2": g.uniform(0.1, 0.3, n),
        "ecc": g.uniform(0.0, 0.1, n),
        "argp": g.uniform(0.0, 360.0, n),
        "comp": g.uniform(0.0, 0.4, n),
        "eb_radius": g.uniform(0.3, 1.5, n),  # solar radii
        "eb_flux": 10 ** g.uniform(-5.0, -0.3, n),  # from a faint companion with no visible secondary eclipse to a bright one
    }


def reference_tp(d, host):
    k = d["radius"] * REARTH / (d["star"] * RSUN)
    out = []
    for i in range(len(k)):
        flux = transit_flux(TIME, k[i], d["u1"][i], d["u2"][i], PERIOD, d["a"][i] / (d["star"][i] * RSUN), d["inc"][i] * np.pi / 180, d["ecc"][i], (90 - d["argp"][i]) * np.pi / 180, exposure=0.00139, samples=20)
        f_comp = d["comp"][i] / (1 - d["comp"][i])
        dilute = 1 / f_comp if host else f_comp
        out.append((flux + dilute) / (1 + dilute))
    return np.array(out)


def pyt_shim():
    """Supplies the pkg_resources function that PyTransit 2.2.0 imports and setuptools 81 removed, when the module is absent."""
    import importlib.util
    import pathlib
    import sys
    import types

    if "pkg_resources" not in sys.modules and importlib.util.find_spec("pkg_resources") is None:
        shim = types.ModuleType("pkg_resources")
        shim.resource_filename = lambda package, name: str(pathlib.Path(importlib.util.find_spec(package).origin).parent / name)
        sys.modules["pkg_resources"] = shim


def as_t(d):
    return {key: torch.as_tensor(value) for key, value in d.items()}


@pytest.mark.parametrize("host", [False, True])
def test_the_planet_flux_equals_the_numpy_composition_to_1e_12(host):
    d = draws(12)
    t = as_t(d)
    got = simulate_tp_flux(torch.as_tensor(TIME), t["radius"], PERIOD, t["inc"], t["a"], t["star"], t["u1"], t["u2"], t["ecc"], t["argp"], t["comp"], host)
    assert got.shape == (12, len(TIME))
    assert np.max(np.abs(got.numpy() - reference_tp(d, host))) < 1e-12


def test_the_lnl_of_a_planet_is_half_the_chi_squared():
    d = draws(5)
    t = as_t(d)
    model = reference_tp(d, False)
    flux = model[0] + 1e-4 * np.sin(TIME * 40)
    got = lnl_tp(torch.as_tensor(TIME), torch.as_tensor(flux), SIGMA, t["radius"], PERIOD, t["inc"], t["a"], t["star"], t["u1"], t["u2"], t["ecc"], t["argp"], t["comp"])
    want = 0.5 * np.sum((flux - model) ** 2 / SIGMA**2, axis=1)
    assert np.max(np.abs(got.numpy() / want - 1)) < 1e-10


def test_the_eclipsing_binary_masks_by_the_secondary_depth_and_scales_every_ratio_under_one():
    d = draws(40)
    t = as_t(d)
    args = (torch.as_tensor(TIME), t["eb_radius"], t["eb_flux"], PERIOD, t["inc"], t["a"], t["star"], t["u1"], t["u2"], t["ecc"], t["argp"], t["comp"], False)
    model, depth = simulate_eb_flux(*args)
    flux = torch.ones(len(TIME), dtype=torch.float64)
    lnl = lnl_eb(torch.as_tensor(TIME), flux, SIGMA, *args[1:])
    masked = (depth[:, 0] >= 1.5 * SIGMA)
    assert masked.any() and (~masked).any(), "the draws must exercise both sides of the secondary-eclipse mask"
    assert torch.isinf(lnl[masked]).all() and torch.isfinite(lnl[~masked]).all()
    twin = lnl_eb_twin(torch.as_tensor(TIME), flux, SIGMA, *args[1:])
    assert torch.isfinite(twin).all()
    assert torch.equal(twin[~masked], lnl[~masked])


def test_the_planet_radius_ratio_guard_is_the_packages_scaling_of_every_ratio_below_one_plus_1e_6():
    # TRICERATOPS multiplies k = R_EB / R_s by 0.999 whenever k - 1 < 1e-6, which is every ratio under 1. An EB of ratio 0.5 therefore has the flux of ratio 0.4995.
    d = draws(1)
    d["eb_radius"] = d["star"] * 0.5
    t = as_t(d)
    got, _ = simulate_eb_flux(torch.as_tensor(TIME), t["eb_radius"], t["eb_flux"], PERIOD, t["inc"], t["a"], t["star"], t["u1"], t["u2"], t["ecc"], t["argp"], t["comp"], False)
    f_eb = d["eb_flux"][0] / (1 - d["eb_flux"][0])
    flux = transit_flux(TIME, 0.5 * 0.999, d["u1"][0], d["u2"][0], PERIOD, d["a"][0] / (d["star"][0] * RSUN), d["inc"][0] * np.pi / 180, d["ecc"][0], (90 - d["argp"][0]) * np.pi / 180, exposure=0.00139, samples=20)
    flux = (flux + f_eb) / (1 + f_eb)
    f_comp = d["comp"][0] / (1 - d["comp"][0])
    dilute = f_comp / (1 + f_eb)
    assert np.max(np.abs(got[0].numpy() - (flux + dilute) / (1 + dilute))) < 1e-12


def test_the_constants_are_the_values_of_astropy():
    constants = pytest.importorskip("astropy.constants")
    assert RSUN == constants.R_sun.cgs.value and REARTH == constants.R_earth.cgs.value


def test_against_triceratops_with_pytransit_as_a_black_box():
    pyt_shim()
    pytest.importorskip("triceratops")
    from triceratops import likelihoods

    d = draws(30, seed=9)
    t = as_t(d)
    flux = np.ones(len(TIME))
    ours = lnl_tp(torch.as_tensor(TIME), torch.as_tensor(flux), SIGMA, t["radius"], PERIOD, t["inc"], t["a"], t["star"], t["u1"], t["u2"], t["ecc"], t["argp"], t["comp"]).numpy()
    theirs = likelihoods.lnL_TP_p(TIME, flux, SIGMA, d["radius"], PERIOD, d["inc"].copy(), d["a"], d["star"], d["u1"], d["u2"], d["ecc"], d["argp"], d["comp"])
    assert np.max(np.abs(ours - theirs)) < 5e-6 * np.max(np.abs(theirs)) + 1e-2, (ours - theirs)
    args = (d["eb_radius"], d["eb_flux"], PERIOD, d["inc"].copy(), d["a"], d["star"], d["u1"], d["u2"], d["ecc"], d["argp"], d["comp"])
    model, depth = likelihoods.simulate_EB_transit_p(TIME, *args)
    got, got_depth = simulate_eb_flux(torch.as_tensor(TIME), t["eb_radius"], t["eb_flux"], PERIOD, t["inc"], t["a"], t["star"], t["u1"], t["u2"], t["ecc"], t["argp"], t["comp"], False)
    # the bounds allow for PyTransit's loss of accuracy at radius ratios up to 1.9
    assert np.max(np.abs(got.numpy() - model)) < 2e-5 and np.max(np.abs(got_depth.numpy() - depth)) < 5e-6


def test_a_period_given_as_an_array_of_the_draws_equals_the_same_scalar_period():
    d = draws(6)
    t = as_t(d)
    args = (t["radius"], None, t["inc"], t["a"], t["star"], t["u1"], t["u2"], t["ecc"], t["argp"], t["comp"])
    scalar = simulate_tp_flux(torch.as_tensor(TIME), *(PERIOD if a is None else a for a in args))
    array = simulate_tp_flux(torch.as_tensor(TIME), *(torch.full((6,), PERIOD, dtype=torch.float64) if a is None else a for a in args))
    assert torch.equal(scalar, array)


def test_the_installed_functions_take_and_give_numpy_and_follow_the_package_to_pytransits_accuracy():
    pyt_shim()
    pytest.importorskip("triceratops")
    from triceratops import marginal_likelihoods

    from velociraptor.triceratops import install

    d = draws(20, seed=4)
    flux = np.ones(len(TIME))
    period = np.full(20, PERIOD)
    package = marginal_likelihoods.lnL_TP_p
    reference = package(TIME, flux, SIGMA, d["radius"], period, d["inc"].copy(), d["a"], d["star"], d["u1"], d["u2"], d["ecc"], d["argp"], d["comp"])
    with install("cpu"):
        assert marginal_likelihoods.lnL_TP_p is not package
        got = marginal_likelihoods.lnL_TP_p(TIME, flux, SIGMA, d["radius"], period, d["inc"].copy(), d["a"], d["star"], d["u1"], d["u2"], d["ecc"], d["argp"], companion_fluxratio=d["comp"])
        eb = marginal_likelihoods.lnL_EB_p(TIME, flux, SIGMA, d["eb_radius"], d["eb_flux"], period, d["inc"].copy(), d["a"], d["star"], d["u1"], d["u2"], d["ecc"], d["argp"], companion_fluxratio=d["comp"], companion_is_host=False)
    assert marginal_likelihoods.lnL_TP_p is package
    assert isinstance(got, np.ndarray) and got.shape == (20,) and isinstance(eb, np.ndarray)
    assert np.max(np.abs(got - reference) / np.maximum(reference, 1.0)) < 1e-4


def test_the_three_functions_follow_the_package_for_both_hosts_with_the_same_draws_masked():
    pyt_shim()
    pytest.importorskip("triceratops")
    from triceratops import marginal_likelihoods as package

    from velociraptor.triceratops import install

    d = draws(60, seed=11)
    flux = 1 - 1.5e-3 * np.exp(-0.5 * (TIME / 0.05) ** 2)
    period = np.full(60, PERIOD)
    tp = lambda host: dict(args=(TIME, flux, SIGMA, d["radius"], period, d["inc"].copy(), d["a"], d["star"], d["u1"], d["u2"], d["ecc"], d["argp"]), kwargs=dict(companion_fluxratio=d["comp"], companion_is_host=host))
    eb = lambda host: dict(args=(TIME, flux, SIGMA, d["eb_radius"], d["eb_flux"], period, d["inc"].copy(), d["a"], d["star"], d["u1"], d["u2"], d["ecc"], d["argp"]), kwargs=dict(companion_fluxratio=d["comp"], companion_is_host=host))
    cases = [("lnL_TP_p", tp(False)), ("lnL_TP_p", tp(True)), ("lnL_EB_p", eb(False)), ("lnL_EB_p", eb(True)), ("lnL_EB_twin_p", eb(False)), ("lnL_EB_twin_p", eb(True))]
    reference = {}
    for index, (name, case) in enumerate(cases):
        reference[index] = getattr(package, name)(*(a.copy() if hasattr(a, "copy") else a for a in case["args"]), **case["kwargs"])
    originals = {name: getattr(package, name) for name in ("lnL_TP_p", "lnL_EB_p", "lnL_EB_twin_p")}
    with install("cpu"):
        for index, (name, case) in enumerate(cases):
            found = getattr(package, name)(*(a.copy() if hasattr(a, "copy") else a for a in case["args"]), **case["kwargs"])
            expected = reference[index]
            assert found.shape == expected.shape and np.array_equal(np.isinf(found), np.isinf(expected)), (name, case["kwargs"]["companion_is_host"])
            finite = np.isfinite(expected)
            assert np.max(np.abs(found[finite] - expected[finite]) / np.maximum(expected[finite], 10.0)) < 1e-3, (name, case["kwargs"]["companion_is_host"])
    assert all(getattr(package, name) is function for name, function in originals.items())
    assert any(np.isinf(reference[i]).any() for i in (2, 3)) and any(np.isfinite(reference[i]).any() for i in (2, 3)), "the draws must exercise both sides of the secondary-eclipse mask"


def test_the_files_that_re_express_triceratops_carry_its_licence_notice():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    assert (root / "THIRD_PARTY" / "TRICERATOPS-LICENSE").read_text().startswith("MIT License") or "MIT" in (root / "THIRD_PARTY" / "TRICERATOPS-LICENSE").read_text()
    for name in ("torch_likelihood.py", "triceratops.py"):
        head = (root / "src" / "velociraptor" / name).read_text()[:2500]
        assert "MIT" in head and "THIRD_PARTY/TRICERATOPS-LICENSE" in head and "Copyright (c) 2020 Steven Giacalone" in head, name


def test_a_keyword_the_package_does_not_have_is_refused_and_a_time_on_another_device_is_followed():
    pyt_shim()
    pytest.importorskip("triceratops")
    from triceratops import marginal_likelihoods as package

    from velociraptor.triceratops import install

    d = draws(3)
    arguments = (TIME, np.ones(len(TIME)), SIGMA, d["radius"], np.full(3, PERIOD), d["inc"], d["a"], d["star"], d["u1"], d["u2"], d["ecc"], d["argp"])
    with install("cpu"):
        with pytest.raises(TypeError, match="exptim"):
            package.lnL_TP_p(*arguments, companion_fluxratio=d["comp"], exptim=0.0)
        with pytest.raises(TypeError, match="companion_fluxratio"):
            package.lnL_TP_p(*arguments)


def test_the_kernel_choice_changes_the_arguments_passed_to_the_flux_function():
    from velociraptor.torch_likelihood import _extra

    assert _extra("triton", 100) == {"exposure_nodes": 3}  # the default rule of the Triton kernel
    assert _extra("triton", 100, None) == {"exposure_nodes": None}  # the 20 midpoints, the rule to compare with stock
    assert _extra("torch", 100, None) == {"chunk": 100}
    assert _extra("triton", 100, 3) == {"exposure_nodes": 3}
    assert _extra("torch", 100) == {"chunk": 100}
    with pytest.raises(ValueError, match="exposure_nodes"):
        _extra("torch", 100, 3)


CASES = (
    ("planet", lambda t: lnl_tp, lambda t: (t["radius"], PERIOD, t["inc"], t["a"], t["star"], t["u1"], t["u2"], t["ecc"], t["argp"], t["comp"], False)),
    ("binary", lambda t: lnl_eb, lambda t: (t["eb_radius"], t["eb_flux"], PERIOD, t["inc"], t["a"], t["star"], t["u1"], t["u2"], t["ecc"], t["argp"], t["comp"], False)),
    ("twin", lambda t: lnl_eb_twin, lambda t: (t["eb_radius"], t["eb_flux"], PERIOD, t["inc"], t["a"], t["star"], t["u1"], t["u2"], t["ecc"], t["argp"], t["comp"], False)),
)


@pytest.mark.parametrize("name,function,arguments", CASES)
def test_a_likelihood_in_chunks_of_draws_gives_the_values_of_the_whole_bit_for_bit(monkeypatch, name, function, arguments):
    """The log-likelihood of a draw does not depend on the other draws. Splitting the draws into chunks, which bounds the device memory (draws by points), must not change a bit, including the infinity of a masked eclipsing binary and a period given per draw."""
    from velociraptor import torch_likelihood

    d = draws(61)
    t = as_t(d)
    flux = torch.as_tensor(1.0 + 1e-4 * np.sin(TIME * 40))
    args = arguments(t)
    monkeypatch.setattr(torch_likelihood, "MAX_ELEMENTS", 1 << 40)
    whole = function(t)(torch.as_tensor(TIME), flux, SIGMA, *args)
    monkeypatch.setattr(torch_likelihood, "MAX_ELEMENTS", len(TIME) * 7)  # 7 draws a chunk
    chunked = function(t)(torch.as_tensor(TIME), flux, SIGMA, *args)
    assert whole.shape == chunked.shape == (61,)
    assert torch.equal(whole, chunked), (whole, chunked)


@pytest.mark.parametrize("draws_count", [100, 1000])
def test_no_call_of_the_flux_function_gets_more_draws_than_the_element_bound_allows_whatever_the_draw_count(monkeypatch, draws_count):
    from velociraptor import torch_likelihood

    seen = []
    original = torch_likelihood.simulate_tp_flux

    def watching(time, radius, *rest, **options):
        seen.append(int(torch.as_tensor(radius).numel()))
        return original(time, radius, *rest, **options)

    monkeypatch.setattr(torch_likelihood, "simulate_tp_flux", watching)
    monkeypatch.setattr(torch_likelihood, "MAX_ELEMENTS", len(TIME) * 7)
    t = as_t(draws(draws_count))
    lnl_tp(torch.as_tensor(TIME), torch.ones(len(TIME), dtype=torch.float64), SIGMA, t["radius"], PERIOD, t["inc"], t["a"], t["star"], t["u1"], t["u2"], t["ecc"], t["argp"], t["comp"], False)
    assert max(seen) <= 7 and sum(seen) == draws_count, (max(seen), sum(seen))


@pytest.mark.parametrize("precision,exact", [("mixed", True), ("float64", False)])
def test_on_the_device_the_peak_memory_of_a_likelihood_is_bounded_by_the_element_bound_and_the_values_are_those_of_the_whole(monkeypatch, precision, exact):
    """With the mixed first pass (the default) the chunked values equal the whole's bit for bit. The float64 kernel depends on how the draws are laid out in a launch, so its chunked values are compared to a relative 1e-11 of the log-likelihood. Slicing one launch by `ELEMENTS` changes no bit in either precision."""
    if not torch.cuda.is_available():
        pytest.skip("the peak of the device memory needs a CUDA or ROCm device")
    pytest.importorskip("triton")
    from velociraptor import torch_likelihood, triton_analytic

    monkeypatch.setattr(triton_analytic, "PRECISION", precision)
    n = 200_000
    t = {key: torch.as_tensor(value, device="cuda") for key, value in draws(n).items()}
    time = torch.as_tensor(TIME, device="cuda")
    flux = torch.ones(len(TIME), dtype=torch.float64, device="cuda")
    args = (t["radius"], PERIOD, t["inc"], t["a"], t["star"], t["u1"], t["u2"], t["ecc"], t["argp"], t["comp"], False)

    def run(bound):
        monkeypatch.setattr(torch_likelihood, "MAX_ELEMENTS", bound)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = lnl_tp(time, flux, SIGMA, *args, kernel="triton")
        torch.cuda.synchronize()
        return out, torch.cuda.max_memory_allocated() - before

    whole, whole_peak = run(1 << 40)
    bound = 1 << 22
    chunked, chunked_peak = run(bound)
    if exact:
        assert torch.equal(whole, chunked)
    else:
        assert float(((whole - chunked).abs() / whole.abs().clamp_min(1e-300)).max()) < 1e-11
    assert chunked_peak <= 4 * bound * 8 + 64 * 2**20, (chunked_peak, whole_peak)  # three arrays of the bound and the result, plus room for the kernel's buffers
    assert chunked_peak < 0.4 * whole_peak, (chunked_peak, whole_peak)
