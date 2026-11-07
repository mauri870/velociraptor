"""The quadratic limb-darkened transit model against the closed-form limits and against PyTransit, which is a test-only reference and is skipped when it is not installed.

The CPU group that the CI installs has no PyTransit, so the CI skips those comparisons. To run them in the environment of validation/pytransit_env.md:

    PYTHONPATH=src .oracle/bin/python -m pytest tests/test_quadratic.py -q
"""

import numpy as np
import pytest

from velociraptor import quadratic_flux, quadrature_flux

U = [(0.0, 0.0), (0.4, 0.25), (0.6, 0.1), (0.1, 0.5), (1.0, 0.0), (0.3, 0.7)]
K = [0.0, 1e-4, 0.01, 0.05, 0.1, 0.3, 0.5, 0.99, 1.0, 1.5, 3.0]


def test_a_planet_off_the_disc_removes_nothing():
    z = np.array([1.2, 2.0, 5.0])
    assert quadratic_flux(z, 0.2, 0.4, 0.25).tolist() == [1.0, 1.0, 1.0]


def test_a_planet_with_no_radius_removes_nothing():
    z = np.linspace(0.0, 2.0, 21)
    assert quadratic_flux(z, 0.0, 0.4, 0.25).tolist() == [1.0] * 21


def test_a_planet_wider_than_the_star_and_covering_it_removes_everything():
    assert quadratic_flux(np.array([0.0, 0.5]), 3.0, 0.4, 0.25).tolist() == [0.0, 0.0]


def test_uniform_disc_deficit_is_the_area_ratio_inside_the_disc():
    # u1 = u2 = 0: the deficit is the area of the planet inside the star over pi (Mandel and Agol 2002, eq. 1).
    k = 0.3
    assert 1 - quadratic_flux(np.array([0.0]), k, 0.0, 0.0)[0] == pytest.approx(k**2, abs=1e-12)
    assert 1 - quadratic_flux(np.array([0.5]), k, 0.0, 0.0)[0] == pytest.approx(k**2, abs=1e-12)
    # centre of the planet on the limb: half of its disc is inside, to the first order in k; use the lens area formula for the exact value
    z = 1.0
    lens = k**2 * np.arccos(k / 2) + np.arccos(1 - k**2 / 2) - 0.5 * k * np.sqrt(4 - k**2)
    assert 1 - quadratic_flux(np.array([z]), k, 0.0, 0.0)[0] == pytest.approx(lens / np.pi, abs=1e-10)


def test_central_depth_of_a_small_planet_is_k_squared_times_the_intensity_at_the_centre():
    k, u1, u2 = 1e-3, 0.4, 0.25
    centre = (1 - u1 * 0 - u2 * 0) / (1 - u1 / 3 - u2 / 6)  # I(mu = 1) over the mean intensity
    assert 1 - quadratic_flux(np.array([0.0]), k, u1, u2)[0] == pytest.approx(k**2 * centre, rel=1e-6)


def test_the_flux_does_not_depend_on_the_sign_of_z_or_on_the_order_of_the_points():
    z = np.array([0.1, 0.9, 1.1, 0.5])
    forward = quadratic_flux(z, 0.1, 0.4, 0.25)
    assert np.array_equal(forward[::-1], quadratic_flux(z[::-1], 0.1, 0.4, 0.25))
    assert np.array_equal(forward, quadratic_flux(-z, 0.1, 0.4, 0.25))


def lens_area(z, k):
    """The area of the overlap of two discs of radii 1 and k at the distance z (the standard lens formula), in 50-digit arithmetic: in double precision the arc cosines lose half their digits at the tangencies."""
    mpmath = pytest.importorskip("mpmath")
    mpmath.mp.dps = 50
    out = []
    for value in np.asarray(z, dtype=float):
        d, r = mpmath.mpf(float(value)), mpmath.mpf(float(k))
        if d >= 1 + r:
            out.append(0.0)
        elif d <= abs(1 - r):
            out.append(float(mpmath.pi * min(1, r) ** 2))
        else:
            first = mpmath.acos((d**2 + 1 - r**2) / (2 * d))
            second = r**2 * mpmath.acos((d**2 + r**2 - 1) / (2 * d * r))
            root = mpmath.sqrt((-d + 1 + r) * (d + 1 - r) * (d - 1 + r) * (d + 1 + r)) / 2
            out.append(float(first + second - root))
    return np.array(out)


@pytest.mark.parametrize("k", [1e-3, 0.01, 0.1, 0.3, 0.5, 0.99, 1.0, 1.5, 3.0])
def test_uniform_disc_matches_the_lens_area_at_every_separation_and_tangency_to_1e_14(k):
    z = np.unique(np.concatenate([np.linspace(0.0, 1 + k + 0.2, 301), abs(1 - k) + np.array([-1e-6, -1e-9, 0.0, 1e-9, 1e-6]), 1 + k + np.array([-1e-6, -1e-9, 0.0, 1e-9, 1e-6])]))
    z = z[z >= 0]
    got = 1 - quadratic_flux(z, k, 0.0, 0.0)
    assert np.max(np.abs(got - lens_area(z, k) / np.pi)) < 1e-14


@pytest.mark.parametrize("k", [0.01, 0.1, 0.3, 0.99, 1.0, 1.5, 3.0])
def test_the_closed_form_agrees_with_the_quadrature_of_768_nodes_with_limb_darkening(k):
    """The closed form and the quadrature share only the intensity law. The bound allows for the worst difference, 7e-10, at the radius ratio 0.5 within 1e-4 of the separation 0.5, where two special cases of the closed form meet. The quadrature is converged to 3e-15 there."""
    z = np.unique(np.concatenate([EDGE(k), abs(1 - k) + np.logspace(-9, -1, 25), 1 + k - np.logspace(-9, -1, 25)]))
    z = z[z >= 0]
    for u1, u2 in U:
        assert np.max(np.abs(quadratic_flux(z, k, u1, u2) - quadrature_flux(z, k, u1, u2, nodes=768))) < 1e-9, (k, u1, u2)


def test_the_bulirsch_integral_matches_carlson_s_forms_of_scipy():
    scipy_special = pytest.importorskip("scipy.special")
    from velociraptor.analytic import cel

    rng = np.random.default_rng(0)
    for _ in range(300):
        kc, p, a, b = 10 ** rng.uniform(-9, 0), 10 ** rng.uniform(-2, 2), *rng.normal(size=2) * 3
        expected = a * scipy_special.elliprf(0, kc**2, 1) + (b - a * p) / 3 * scipy_special.elliprj(0, kc**2, 1, p)
        assert float(cel(kc, p, a, b)) == pytest.approx(expected, rel=1e-11)


def test_the_quadrature_is_converged_at_the_default_nodes():
    # The same model with 8 times the nodes: the difference is the error of the default, over the radius ratios, separations and limb darkening of the tests.
    worst = 0.0
    for k in (0.01, 0.1, 0.3, 0.99, 1.0, 1.5):
        z = EDGE(k)
        for u1, u2 in U:
            worst = max(worst, float(np.max(np.abs(quadrature_flux(z, k, u1, u2) - quadrature_flux(z, k, u1, u2, nodes=384)))))
    assert worst < 1e-9, worst


def test_a_planet_exactly_the_size_of_the_star_centred_on_it_removes_everything():
    assert quadratic_flux(np.array([0.0]), 1.0, 0.3, 0.7)[0] == pytest.approx(0.0, abs=1e-12)


def oracle(z, k, u1, u2):
    """PyTransit's quadratic model (its public QuadraticModel) at the separations z of a circular edge-on orbit of a/R* = 20: the times are chosen so that z = a sin(phase)."""
    pytransit = pytest.importorskip("pytransit")
    period, a = 1.0, 20.0
    phase = np.arcsin(np.asarray(z) / a)
    time = phase * period / (2 * np.pi)
    model = pytransit.QuadraticModel(interpolate=False)
    model.set_data(time)
    return model.evaluate(k, [u1, u2], 0.0, period, a, np.pi / 2)


def grid(k_values, z_of_k):
    for k in k_values:
        for u1, u2 in U:
            yield k, u1, u2, z_of_k(k)


EDGE = lambda k: np.unique(np.clip(np.concatenate([[0.0, 1e-9, 1e-6, 1e-3, 0.5, 1.0], np.abs(1 - k) + np.array([-1e-6, -1e-9, 0.0, 1e-9, 1e-6]), 1 + k + np.array([-1e-6, -1e-9, 0.0, 1e-9, 1e-6]), np.linspace(0.0, 1 + k + 0.1, 41)]), 0.0, 3.0))


@pytest.mark.parametrize("k", [0.01, 0.05, 0.1, 0.3, 0.5, 0.99, 1.0, 1.5])
def test_agrees_with_pytransit_on_a_grid_of_radius_ratio_separation_and_limb_darkening(k):
    worst = 0.0
    for u1, u2 in U:
        z = EDGE(k)
        z = z[(z < 19.0) & ~((k == 1.0) & (z < 1e-5))]  # PyTransit returns 0.018 to 0.036 for a planet the size of the star centred on it, where the flux is 0 (the test above); the corner is left out of the comparison
        worst = max(worst, float(np.max(np.abs(quadratic_flux(z, k, u1, u2) - oracle(z, k, u1, u2)))))
    assert worst < 5e-6, f"k {k}: largest difference {worst:.2e}"  # PyTransit's own deviation from the exact lens area (u1 = u2 = 0) and from a 2-D integral is 1e-8 to 4e-7 on this grid, 5e-6 at k 1.5 beyond the star's edge


def test_arrays_of_radius_ratio_and_limb_darkening_broadcast_against_the_separations():
    z = np.linspace(0.0, 1.6, 17)[:, None]
    k = np.array([0.05, 0.3, 1.2])[None, :]
    u1 = np.array([0.4, 0.1, 0.6])[None, :]
    u2 = np.array([0.25, 0.5, 0.1])[None, :]
    together = quadratic_flux(z, k, u1, u2)
    assert together.shape == (17, 3)
    for j in range(3):
        assert np.array_equal(together[:, j], quadratic_flux(z[:, 0], float(k[0, j]), float(u1[0, j]), float(u2[0, j])))


from velociraptor import separation, transit_flux


def test_the_separation_at_mid_transit_is_the_impact_parameter_of_winn_2010_eq_7():
    for ecc, omega in ((0.0, 0.0), (0.3, 1.0), (0.7, 4.0), (0.9, 2.5)):
        a, inc = 12.0, np.radians(87.0)
        z, front = separation(np.array([0.0]), 5.0, a, inc, ecc, omega)
        assert front[0]
        assert z[0] == pytest.approx(a * np.cos(inc) * (1 - ecc**2) / (1 + ecc * np.sin(omega)), abs=1e-12)


def test_the_planet_is_behind_the_star_half_an_orbit_after_a_circular_transit_and_in_front_at_the_transit():
    z, front = separation(np.array([0.0, 2.5, 5.0]), 5.0, 10.0, np.radians(89.0))
    assert front.tolist() == [True, False, True]


def test_kepler_residual_is_at_rounding_for_every_eccentricity():
    # M = E - e sin E recovered from the separation's true anomaly: z at the node crossing times must repeat every period.
    for ecc in (0.0, 0.5, 0.9, 0.99):
        t = np.linspace(-1.013, 0.987, 41)  # off the instants where the planet crosses the plane of the sky, which a rounding error can put on either side
        z1, f1 = separation(t, 4.0, 9.0, np.radians(88.0), ecc, 1.2)
        z2, f2 = separation(t + 4.0, 4.0, 9.0, np.radians(88.0), ecc, 1.2)
        assert np.max(np.abs(z1 - z2)) < 1e-9 and np.array_equal(f1, f2)


def test_a_long_exposure_is_the_mean_of_the_flux_over_equal_sub_intervals_taken_at_their_middles():
    t = np.linspace(-0.1, 0.1, 21)
    args = (0.1, 0.4, 0.25, 3.0, 8.0, np.radians(88.5))
    mean = np.mean([transit_flux(t + o, *args) for o in ((np.arange(4) + 0.5) / 4 - 0.5) * 0.02], axis=0)
    assert np.max(np.abs(transit_flux(t, *args, exposure=0.02, samples=4) - mean)) < 1e-13  # the sub-samples start from the centre's solution of Kepler's equation, not from their own


def test_a_batch_of_scenarios_has_one_row_each():
    t = np.linspace(-0.1, 0.1, 21)[None, :]
    periods = np.array([2.0, 3.0, 5.0])[:, None]
    batch = transit_flux(t, 0.1, 0.4, 0.25, periods, 8.0, np.radians(88.0))
    assert batch.shape == (3, 21)
    for row, period in enumerate(periods[:, 0]):
        assert np.array_equal(batch[row], transit_flux(t[0], 0.1, 0.4, 0.25, period, 8.0, np.radians(88.0)))


def test_agrees_with_pytransit_on_random_orbits_with_exposure_where_pytransit_s_own_orbit_is_exact():
    """Random circular and eccentric orbits (e up to 0.3, a/R* from 8 to 40), 5e-6. PyTransit's separation departs from the exact one as a/R* falls and as e rises (1e-5 at a/R* 5 and e 0.14, 2e-4 at e 0 and a/R* 2.1,
    and 0.2 at e 0.87, where this model agrees with a two-dimensional integral to the integral's 4e-9), so those are not compared."""
    oracle(np.array([0.1]), 0.1, 0.4, 0.25)
    import pytransit

    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(150):
        period, a = rng.uniform(0.5, 30), rng.uniform(8, 40)
        inc = np.arccos(rng.uniform(0, 1.1) / a)
        ecc, omega, k = rng.choice([0.0, rng.uniform(0, 0.3)]), rng.uniform(0, 2 * np.pi), rng.uniform(0.01, 0.5)
        time = np.linspace(-0.3, 0.3, 121) * period / a * 2
        exposure, samples = rng.choice([0.0, 0.0208333]), int(rng.choice([1, 5, 20]))
        model = pytransit.QuadraticModel(interpolate=False)
        model.set_data(time, exptimes=exposure, nsamples=samples)
        expected = model.evaluate_ps(k, [0.4, 0.25], 0.0, period, a, inc, ecc, omega)
        got = transit_flux(time, k, 0.4, 0.25, period, a, inc, ecc, omega, exposure=exposure, samples=samples if exposure else 1)
        worst = max(worst, float(np.max(np.abs(got - expected))))
    assert worst < 5e-6, worst  # PyTransit's orbit is approximate at the 1e-6 level up to a/R* of about 10


def test_skipping_the_points_that_cannot_overlap_changes_no_value():
    """transit_flux evaluates only the points whose exposure can reach the disc; the same orbits evaluated sub-sample by sub-sample over all the points, each with its own full solution of Kepler's equation, give the same fluxes to 1e-13, long exposures and eccentric orbits included."""
    rng = np.random.default_rng(11)
    for _ in range(20):
        period, a, ecc, omega = rng.uniform(0.5, 30), rng.uniform(2, 30), rng.choice([0.0, rng.uniform(0, 0.9)]), rng.uniform(0, 2 * np.pi)
        inc, k = np.arccos(rng.uniform(0, 1.1) / a), rng.uniform(0.01, 1.3)
        time = np.linspace(-0.5, 0.5, 301) * period
        exposure, samples = rng.choice([0.0208, 0.3]), int(rng.choice([2, 7, 20]))
        offsets = ((np.arange(samples) + 0.5) / samples - 0.5) * exposure
        z, front = separation(time[:, None] + offsets, period, a, inc, ecc, omega)
        every = np.where(front & (z < 1 + k), quadratic_flux(np.where(front & (z < 1 + k), z, 5.0), k, 0.4, 0.25), 1.0).mean(axis=1)
        assert np.max(np.abs(transit_flux(time, k, 0.4, 0.25, period, a, inc, ecc, omega, exposure=exposure, samples=samples) - every)) < 1e-13


def test_a_long_input_gives_the_values_of_its_pieces_and_stays_in_memory(monkeypatch):
    import velociraptor.quadratic as module

    z = np.linspace(0.0, 1.6, 1000)
    whole = quadrature_flux(z, 0.1, 0.4, 0.25)
    monkeypatch.setattr(module, "CHUNK_ELEMENTS", 64)
    assert np.array_equal(module.quadrature_flux(z, 0.1, 0.4, 0.25), whole)
    assert module.quadrature_flux(z.reshape(10, 100), 0.1, 0.4, 0.25).shape == (10, 100)


@pytest.mark.parametrize("radius_offset", [0.0, 1e-12, 1e-9, 1e-7, -1e-9, -1e-7])
def test_the_closed_form_holds_near_the_double_degeneracy_of_radius_one_half_and_separation_one_half(radius_offset):
    """The special cases b = r and b + r = 1 of the paper meet at r = b = 1/2, where cancellation in b - r and b + r loses digits unless the closed form uses the products (b - r)(b + r) and (max - 1) + min. Points near k = 0.5, z = 0.5 are compared with 768 nodes of the quadrature to 1e-13."""
    k = 0.5 + radius_offset
    z = 0.5 + np.array([1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, -1e-9, -1e-7, -1e-5])
    for u1, u2 in U:
        assert np.max(np.abs(quadratic_flux(z, k, u1, u2) - quadrature_flux(z, k, u1, u2, nodes=768))) < 1e-13, (k, u1, u2)


def test_the_taylor_series_of_the_orbit_reproduces_the_exact_orbit_near_mid_transit_at_high_order():
    from velociraptor.quadratic import _series_project, orbit_series

    rng = np.random.default_rng(2)
    for _ in range(30):
        period, a, ecc, omega = rng.uniform(0.5, 30), rng.uniform(3, 40), rng.choice([0.0, rng.uniform(0, 0.5)]), rng.uniform(0, 2 * np.pi)
        inc = np.arccos(rng.uniform(0, 1.0) / a)
        offset = np.linspace(-0.01, 0.01, 21) * period  # the radius of convergence shrinks as the angular speed at mid-transit rises with e
        X, Y = orbit_series(period, a, ecc, omega, order=14)
        z_series, front_series = _series_project(offset, X[:, None], Y[:, None], inc, omega)
        z_exact, front_exact = separation(offset, period, a, inc, ecc, omega)
        assert np.max(np.abs(z_series - z_exact)) < 1e-10 and np.array_equal(front_series, front_exact), (period, a, ecc)


def test_the_series_orbit_of_order_4_stays_within_1e_6_of_the_exact_flux_for_a_over_r_star_of_8_and_more():
    rng = np.random.default_rng(4)
    worst = 0.0
    for _ in range(40):
        period, a = rng.uniform(0.5, 30), rng.uniform(8, 40)
        inc, ecc, omega, k = np.arccos(rng.uniform(0, 1.0) / a), rng.choice([0.0, rng.uniform(0, 0.5)]), rng.uniform(0, 2 * np.pi), rng.uniform(0.01, 0.4)
        time = np.linspace(-0.6, 0.6, 121) * period / a
        exact = transit_flux(time, k, 0.4, 0.25, period, a, inc, ecc, omega, exposure=0.00139, samples=5)
        series = transit_flux(time, k, 0.4, 0.25, period, a, inc, ecc, omega, exposure=0.00139, samples=5, orbit="series")
        worst = max(worst, float(np.max(np.abs(exact - series))))
    assert worst < 1e-6, worst


def test_the_series_orbit_is_out_of_transit_beyond_a_quarter_of_the_period():
    time = np.array([0.0, 0.3, 0.6, 0.9]) * 4.0
    series = transit_flux(time, 0.1, 0.4, 0.25, 4.0, 3.0, np.radians(89.0), orbit="series", order=4)
    assert series[2] == series[3] == 1.0 and series[0] < 1.0


def test_the_finite_difference_series_with_a_small_step_is_the_series_of_the_two_body_orbit():
    from velociraptor import orbit_taylor
    from velociraptor.quadratic import _series_project, _taylor_project, orbit_series

    rng = np.random.default_rng(6)
    for _ in range(10):
        period, a, ecc, omega = rng.uniform(0.5, 30), rng.uniform(4, 40), rng.uniform(0, 0.6), rng.uniform(0, 2 * np.pi)
        inc = np.arccos(rng.uniform(0, 1.0) / a)
        offset = np.linspace(-0.01, 0.01, 11) * period
        X, Y = orbit_series(period, a, ecc, omega, order=4)
        across, along = orbit_taylor(period, a, ecc, omega, step=1e-3 * period)
        z_series, _ = _series_project(offset, X[:, None], Y[:, None], inc, omega)
        z_taylor, front = _taylor_project(offset, across[:, None], along[:, None], inc)
        assert np.max(np.abs(z_series - z_taylor)) < 1e-6 * a and front.all()


def test_the_taylor_orbit_reproduces_pytransit_where_the_exact_orbit_does_not():
    """PyTransit 2.2.0 computes the sky position by a fourth-order Taylor series about mid-transit, with seven-point finite differences of step 0.02 d. On five orbits of period 0.61 to 2 d, a/R* 3.2 to 8 and e 0.3 to 0.7, the exact orbit differs from it by 1.0e-5 to 6.3e-4, and the series with that step is within 1e-7."""
    oracle(np.array([0.1]), 0.1, 0.4, 0.25)
    import pytransit

    for period, a, ecc, omega, b in ((0.61, 4.0, 0.6, 1.1, 0.2), (0.61, 4.0, 0.3, 2.0, 0.3), (0.61, 6.0, 0.5, 0.4, 0.1), (0.61, 3.2, 0.6, 5.0, 0.3), (2.0, 8.0, 0.7, 1.0, 0.2)):
        inc = np.arccos(b / (a * (1 - ecc**2) / (1 + ecc * np.sin(omega))))
        time = np.linspace(-0.05, 0.05, 401) * period / 0.61 * (4.0 / a)
        model = pytransit.QuadraticModel(interpolate=False)
        model.set_data(time)
        stock = model.evaluate_ps(0.1, [0.4, 0.25], 0.0, period, a, inc, ecc, omega)
        taylor = transit_flux(time, 0.1, 0.4, 0.25, period, a, inc, ecc, omega, orbit="taylor")
        exact = transit_flux(time, 0.1, 0.4, 0.25, period, a, inc, ecc, omega)
        assert np.max(np.abs(taylor - stock)) < 1e-7 < np.max(np.abs(exact - stock)), (period, a, ecc, omega)


def test_the_gauss_legendre_rule_is_computed_once_for_each_count_and_is_read_only():
    from velociraptor.quadratic import gauss_legendre

    x, w = gauss_legendre(7)
    assert gauss_legendre(7)[0] is x and gauss_legendre(7)[1] is w
    want_x, want_w = np.polynomial.legendre.leggauss(7)
    assert np.array_equal(x, want_x) and np.array_equal(w, want_w)
    assert not x.flags.writeable and not w.flags.writeable
    one_x, one_w = gauss_legendre(1)  # the edge: a single node is the midpoint with the whole weight
    assert one_x[0] == 0.0 and one_w[0] == 2.0
