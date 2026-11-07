"""The planet-scenario light curve and log-likelihood against a 50-digit reference, and against TRICERATOPS 1.1.0 where the two curves agree.

The case is a heavy draw of the planet scenario of TIC 201354238 (k 0.146, e 0.52, a/R* 4.0, P 0.713 d, the companion as host). The fixture holds its arguments, the light curve it was scored against, and a reference light curve. The reference solves Kepler's equation in 50 digits, uses the 20 midpoints of the exposure and the closed-form flux of velociraptor.analytic (checked in validation/results.json). TRICERATOPS 1.1.0 through PyTransit 2.2.0 gives this draw a second dip of the depth of the transit, 0.07 to 0.09 days from mid-transit, which the reference does not have (up to 7.0e-4 in flux and 19.6 in log-likelihood). No test asks for agreement with it where it differs from the reference."""
from pathlib import Path

import numpy as np
import pytest

CASE = np.load(Path(__file__).parent / "fixtures" / "ghost_transit_draw.npz")
RSUN, REARTH = 6.957e10, 6.3781e8


def parameters():
    c = CASE
    k = float(c["R_p"]) * REARTH / (float(c["R_s"]) * RSUN)
    a_rs = float(c["a"]) / (float(c["R_s"]) * RSUN)
    comp = float(c["comp"])
    return k, a_rs, comp / (1 - comp)


def window():
    return np.asarray(CASE["window"])


def test_the_numpy_model_with_the_exact_orbit_is_within_1e_12_of_the_50_digit_reference_on_the_stored_draw():
    from velociraptor.quadratic import transit_flux

    k, a_rs, f_comp = parameters()
    c = CASE
    flux = transit_flux(c["time"], k, float(c["u1"]), float(c["u2"]), float(c["P_orb"]), a_rs, np.radians(float(c["inc"])), float(c["ecc"]), (90 - float(c["argp"])) * np.pi / 180, 0.0, float(c["exptime"]), 20)
    diluted = (flux + 1.0 / f_comp) / (1 + 1.0 / f_comp)  # the companion is the host: the dilution of lnL_TP_p with companion_is_host
    difference = np.abs(diluted - c["reference"])[window()]
    assert difference.max() <= 1e-12, f"largest difference {difference.max():.3g} over {window().size} points"
    assert 1 - c["reference"].min() > 6e-4, "the stored reference has the transit in it"


def test_the_torch_likelihood_on_the_cpu_gives_the_log_likelihood_of_the_reference_curve_on_the_stored_draw():
    pytest.importorskip("torch")
    from velociraptor.torch_likelihood import lnl_tp
    from velociraptor.triceratops import _bound

    c = CASE
    given = _bound(lnl_tp, "cpu", "torch", None)
    ours = float(given(c["time"], c["flux"], float(c["sigma"]), np.array([c["R_p"]]), float(c["P_orb"]), np.array([c["inc"]]), np.array([c["a"]]), np.array([c["R_s"]]), np.array([c["u1"]]), np.array([c["u2"]]), np.array([c["ecc"]]), np.array([c["argp"]]), companion_fluxratio=np.array([c["comp"]]), companion_is_host=True, exptime=float(c["exptime"]), nsamples=20)[0])
    reference = 0.5 * float(np.sum((c["flux"] - c["reference"]) ** 2 / float(c["sigma"]) ** 2))
    assert abs(ours - reference) <= 1e-8 * reference, f"{ours!r} against {reference!r}"


def test_the_device_kernel_at_20_midpoints_is_within_1e_9_of_the_50_digit_reference_on_the_stored_draw():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("the kernel needs a CUDA or ROCm device")
    from velociraptor.triton_analytic import transit_flux_triton_analytic

    k, a_rs, f_comp = parameters()
    c = CASE
    flux = transit_flux_triton_analytic(torch.as_tensor(c["time"], device="cuda"), k, float(c["u1"]), float(c["u2"]), float(c["P_orb"]), a_rs, float(np.radians(float(c["inc"]))), float(c["ecc"]), float((90 - float(c["argp"])) * np.pi / 180), exposure=float(c["exptime"]), samples=20, exposure_nodes=None).cpu().numpy()
    diluted = (flux + 1.0 / f_comp) / (1 + 1.0 / f_comp)
    difference = np.abs(diluted - c["reference"])[window()]
    assert difference.max() <= 1e-9, f"largest difference {difference.max():.3g}"


def prior_draws(n=300):
    """n planet draws of a seeded prior: the period of the stored draw, e to 0.85, a/R* from the stellar mass and radius, impact parameter to 1.1, planets of 1 to 15 Earth radii, a companion holding up to 30 percent of the light."""
    c = CASE
    rng = np.random.default_rng(6)
    star = rng.uniform(0.6, 1.4, n)
    period = float(c["P_orb"])
    a = (6.674e-8 * star * 1.989e33 * (period * 86400.0) ** 2 / (4 * np.pi**2)) ** (1 / 3)
    ecc, argp, b = rng.uniform(0, 0.85, n), rng.uniform(0, 360, n), rng.uniform(0, 1.1, n)
    w = np.radians(90 - argp)
    inc = np.degrees(np.arccos(np.clip(b * (1 + ecc * np.sin(w)) / ((1 - ecc**2) * a / (star * RSUN)), 0, 1)))
    return dict(radius=rng.uniform(1, 15, n), period=period, inc=inc, a=a, star=star, u1=np.full(n, float(c["u1"])), u2=np.full(n, float(c["u2"])), ecc=ecc, argp=argp, comp=rng.uniform(0.0, 0.3, n))


def both_packages(companion_is_host):
    """(stock curves, stock log-likelihoods, velociraptor curves, velociraptor log-likelihoods) of the seeded draws on the stored light curve."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("triceratops")
    from triceratops import likelihoods

    from velociraptor.torch_likelihood import lnl_tp, simulate_tp_flux
    from velociraptor.triceratops import _bound

    c, d = CASE, prior_draws()
    time, flux, sigma, exptime = c["time"], c["flux"], float(c["sigma"]), float(c["exptime"])
    arguments = (d["radius"], d["period"], d["inc"].copy(), d["a"], d["star"], d["u1"], d["u2"], d["ecc"], d["argp"])
    stock_curve = likelihoods.simulate_TP_transit_p(time, *arguments[:2], arguments[2].copy(), *arguments[3:], d["comp"], companion_is_host, exptime, 20)
    n = len(d["radius"])
    ours_curve = simulate_tp_flux(torch.as_tensor(time), torch.as_tensor(d["radius"]), torch.as_tensor(np.full(n, d["period"])), torch.as_tensor(d["inc"]), torch.as_tensor(d["a"]), torch.as_tensor(d["star"]), torch.as_tensor(d["u1"]), torch.as_tensor(d["u2"]), torch.as_tensor(d["ecc"]), torch.as_tensor(d["argp"]), torch.as_tensor(d["comp"]), companion_is_host, exposure=exptime, samples=20).numpy()
    stock_lnl = likelihoods.lnL_TP_p(time, flux, sigma, arguments[0], arguments[1], arguments[2].copy(), *arguments[3:], companion_fluxratio=d["comp"], companion_is_host=companion_is_host, exptime=exptime, nsamples=20)
    ours_lnl = _bound(lnl_tp, "cpu", "torch", None)(time, flux, sigma, arguments[0], arguments[1], arguments[2].copy(), *arguments[3:], companion_fluxratio=d["comp"], companion_is_host=companion_is_host, exptime=exptime, nsamples=20)
    return stock_curve, np.asarray(stock_lnl), ours_curve, np.asarray(ours_lnl)


@pytest.mark.parametrize("companion_is_host", [False, True])
def test_each_package_reports_the_chi_squared_of_its_own_curve_so_that_any_difference_of_the_log_likelihoods_is_a_difference_of_the_curves(companion_is_host):
    """The log-likelihood of each package equals half the chi-squared of its own light curve against the stored data to 1e-9 relative, on 300 draws of a seeded prior. The dilution and the chi-squared are then the same in both, and the packages differ only in the light curve."""
    stock_curve, stock_lnl, ours_curve, ours_lnl = both_packages(companion_is_host)
    flux, sigma = CASE["flux"], float(CASE["sigma"])
    for curve, reported in ((stock_curve, stock_lnl), (ours_curve, ours_lnl)):
        formula = 0.5 * np.sum((flux - curve) ** 2 / sigma**2, axis=1)
        finite = np.isfinite(reported)
        assert finite.sum() > 250
        assert np.max(np.abs(formula - reported)[finite] / reported[finite]) <= 1e-9


@pytest.mark.parametrize("companion_is_host", [False, True])
def test_where_the_curves_agree_to_1e_6_and_the_fit_is_at_the_noise_floor_the_log_likelihoods_agree_to_0_03(companion_is_host):
    """A curve difference delta moves the log-likelihood by at most sqrt(2 lnL) sqrt(N) delta / sigma (Cauchy-Schwarz). That is 2 for delta = 1e-6 at lnL 500, N = 1000 points and sigma 5.08e-4. The draws that fit (stock lnL up to 500) with curves within 1e-6 agree to 0.03 in log-likelihood, and the failure message counts them. Draws with curves farther apart than 1e-6 are excluded, because PyTransit's series orbit has a second dip on short eccentric orbits (up to 7.0e-4 in flux and 19.6 in log-likelihood on the stored draw)."""
    stock_curve, stock_lnl, ours_curve, ours_lnl = both_packages(companion_is_host)
    agree = np.abs(stock_curve - ours_curve).max(axis=1) <= 1e-6
    covered = agree & (stock_lnl <= 500) & np.isfinite(ours_lnl)
    assert covered.sum() >= 15, f"only {covered.sum()} draws are covered"
    largest = np.max(np.abs(stock_lnl - ours_lnl)[covered])
    assert largest <= 0.03, f"largest difference {largest:.3g} over {int(covered.sum())} draws ({int((~agree).sum())} of 300 have curves farther apart than 1e-6)"
