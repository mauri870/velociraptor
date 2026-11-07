"""The closed-form Triton kernel against the numpy model where a CUDA or ROCm device is available."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from velociraptor import transit_flux

ORBITS = [(3.0, 8.0, 1.5, 0.0, np.pi / 2), (10.0, 20.0, 1.55, 0.3, 1.0), (0.5, 3.0, 1.45, 0.6, 4.0), (30.0, 40.0, 1.5707, 0.05, 2.5), (4.2, 9.0, 1.53, 0.9, 5.5)]


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("the kernel needs a CUDA or ROCm device")
    return "cuda"


def test_the_closed_form_kernel_agrees_with_the_numpy_model_on_orbits_exposures_and_radius_ratios(device):
    from velociraptor.triton_analytic import transit_flux_triton_analytic

    worst, where = 0.0, None
    time = np.linspace(-0.3, 0.3, 301)
    for k in (0.001, 0.05, 0.3, 0.5, 0.99, 1.0, 1.5, 3.0):
        for orbit in ORBITS:
            period, a_rs, inc, ecc, omega = orbit
            for exposure, samples in ((0.0, 1), (0.00139, 20), (0.02, 7)):
                got = transit_flux_triton_analytic(torch.as_tensor(time, device=device), k, 0.4, 0.25, period, a_rs, inc, ecc, omega, exposure=exposure, samples=samples, exposure_nodes=None).cpu().numpy()
                want = transit_flux(time, k, 0.4, 0.25, period, a_rs, inc, ecc, omega, exposure=exposure, samples=samples)
                difference = float(np.max(np.abs(got - want)))
                if difference > worst:
                    worst, where = difference, (k, orbit, exposure, samples)
    assert worst < 1e-11, (worst, where)


def test_a_batch_of_scenarios_in_one_call_equals_the_scenarios_one_by_one(device):
    from velociraptor.triton_analytic import transit_flux_triton_analytic

    time = torch.as_tensor(np.tile(np.linspace(-0.2, 0.2, 51), (4, 1)), device=device)
    k = torch.as_tensor(np.array([0.02, 0.1, 0.3, 1.1])[:, None], device=device)
    period = torch.as_tensor(np.array([2.0, 5.0, 11.0, 0.7])[:, None], device=device)
    a_rs = torch.as_tensor(np.array([6.0, 12.0, 25.0, 3.0])[:, None], device=device)
    got = transit_flux_triton_analytic(time, k, 0.4, 0.25, period, a_rs, 1.52, 0.2, 1.0, exposure=0.00139, samples=5, exposure_nodes=None).cpu().numpy()
    for i in range(4):
        want = transit_flux(time[i].cpu().numpy(), float(k[i, 0]), 0.4, 0.25, float(period[i, 0]), float(a_rs[i, 0]), 1.52, 0.2, 1.0, exposure=0.00139, samples=5)
        assert np.max(np.abs(got[i] - want)) < 1e-11


def test_the_series_path_agrees_with_the_closed_form_path_of_the_kernel_for_small_planets(device):
    from velociraptor.triton_analytic import transit_flux_triton_analytic

    time = torch.as_tensor(np.tile(np.linspace(-0.4, 0.4, 801), (6, 1)), device=device)
    k = torch.as_tensor(np.array([0.0005, 0.002, 0.005, 0.02, 0.05, 0.08])[:, None], device=device)
    period = torch.as_tensor(np.array([3.0, 5.0, 11.0, 2.0, 7.0, 4.0])[:, None], device=device)
    a_rs = torch.as_tensor(np.array([8.0, 12.0, 25.0, 6.0, 15.0, 9.0])[:, None], device=device)
    with_series = transit_flux_triton_analytic(time, k, 0.4, 0.25, period, a_rs, 1.53, 0.15, 1.0, exposure=0.00139, samples=20, series=True)
    without = transit_flux_triton_analytic(time, k, 0.4, 0.25, period, a_rs, 1.53, 0.15, 1.0, exposure=0.00139, samples=20, series=False)
    assert float((with_series - without).abs().max()) < 1e-13
    assert float(without.min()) < 1.0 - 1e-6, "the planets must transit"


def test_gauss_legendre_nodes_in_the_exposure_agree_with_the_midpoints_of_the_package_to_the_error_of_the_rule(device):
    from velociraptor.triton_analytic import transit_flux_triton_analytic

    time = torch.as_tensor(np.tile(np.linspace(-0.4, 0.4, 801), (4, 1)), device=device)
    k = torch.as_tensor(np.array([0.002, 0.01, 0.05, 0.3])[:, None], device=device)
    period = torch.as_tensor(np.array([3.0, 5.0, 11.0, 2.0])[:, None], device=device)
    a_rs = torch.as_tensor(np.array([8.0, 12.0, 25.0, 6.0])[:, None], device=device)
    args = (time, k, 0.4, 0.25, period, a_rs, 1.53, 0.15, 1.0)
    midpoints = transit_flux_triton_analytic(*args, exposure=0.00139, samples=20, exposure_nodes=None)
    # the limits allow for points where the ingress falls inside one exposure of 0.00139 d, where the rule is least accurate
    for nodes, limit in ((3, 5e-6), (5, 2e-6), (8, 1e-6), (64, 5e-7)):
        gauss = transit_flux_triton_analytic(*args, exposure=0.00139, samples=20, exposure_nodes=nodes)
        assert float((gauss - midpoints).abs().max()) < limit, nodes
    # the converged integral against the 20 midpoints
    converged = transit_flux_triton_analytic(*args, exposure=0.00139, samples=20, exposure_nodes=64)
    assert 0.0 < float((converged - midpoints).abs().max()) < 5e-7


def test_the_exposure_rule_of_the_kernel_is_three_gauss_legendre_nodes_and_the_midpoints_are_the_explicit_option(device):
    """The 20-midpoint path (exposure_nodes=None) is the one that agrees with the numpy model and with TRICERATOPS's own rule; the default is the 3-node rule."""
    from velociraptor.triton_analytic import transit_flux_triton_analytic

    time = torch.as_tensor(np.tile(np.linspace(-0.4, 0.4, 801), (2, 1)), device=device)
    args = (time, 0.05, 0.4, 0.25, 3.0, 8.0, 1.53, 0.15, 1.0)
    default = transit_flux_triton_analytic(*args, exposure=0.00139, samples=20)
    three = transit_flux_triton_analytic(*args, exposure=0.00139, samples=20, exposure_nodes=3)
    midpoints = transit_flux_triton_analytic(*args, exposure=0.00139, samples=20, exposure_nodes=None)
    assert torch.equal(default, three)
    assert float((default - midpoints).abs().max()) > 0.0, "the two rules are different quadratures"
    want = transit_flux(time[0].cpu().numpy(), 0.05, 0.4, 0.25, 3.0, 8.0, 1.53, 0.15, 1.0, exposure=0.00139, samples=20)
    assert float(np.max(np.abs(midpoints[0].cpu().numpy() - want))) < 1e-11


def kernel_batch():
    """96 seeded draws of 40 points over eccentric, short and grazing orbits. The flux in tests/fixtures/triton_kernel_golden.json was recorded for these draws with 3 Gauss-Legendre nodes and with 20 midpoints."""
    rng = np.random.default_rng(728)
    n, points = 96, 40
    period = np.exp(rng.uniform(np.log(0.5), np.log(40.0), n))
    a_rs = 4.2 * period ** (2 / 3) * rng.uniform(0.5, 2.0, n)
    b = rng.uniform(0.0, 1.3, n)
    inc = np.arccos(np.clip(b / a_rs, -1, 1))
    ecc = np.where(np.arange(n) % 3 == 0, rng.uniform(0.3, 0.8, n), rng.uniform(0.0, 0.3, n))
    k = np.exp(rng.uniform(np.log(0.003), np.log(0.6), n))
    columns = dict(k=k, u1=rng.uniform(0.2, 0.5, n), u2=rng.uniform(0.1, 0.3, n), period=period, a_rs=a_rs, inc=inc, ecc=ecc, omega=rng.uniform(0, 2 * np.pi, n), t0=rng.uniform(-0.05, 0.05, n))
    return np.tile(np.linspace(-0.3, 0.3, points), (n, 1)), {name: value[:, None] for name, value in columns.items()}


@pytest.mark.parametrize("label,options,binned,tolerance", [("nodes3", dict(exposure_nodes=3), True, 0.0), ("midpoints20", dict(exposure_nodes=None, samples=20), False, 0.0), ("midpoints20", dict(exposure_nodes=None, samples=20), True, 1e-15)])
def test_the_flux_is_the_flux_recorded_before_the_terms_of_a_draw_were_made_once(device, label, options, binned, tolerance, monkeypatch):
    """Bit for bit with the second pass inside the main kernel, and at 3 nodes, where the binned pass is not used. With the binned second pass at 20 midpoints the closed form is compiled into a kernel of its own and rounds some pairs differently in the last bit, by at most 1e-15 in the flux."""
    import json
    import pathlib

    from velociraptor import triton_analytic
    from velociraptor.triton_analytic import transit_flux_triton_analytic

    monkeypatch.setattr(triton_analytic, "BINNED", binned)
    monkeypatch.setattr(triton_analytic, "INTERPOLATE", False)  # the golden records the Kepler solve at each node
    monkeypatch.setattr(triton_analytic, "COMPACT", False)  # and the single-stage first pass
    recorded = json.loads((pathlib.Path(__file__).parent / "fixtures" / "triton_kernel_golden.json").read_text())[label]
    time, c = kernel_batch()
    got = transit_flux_triton_analytic(torch.as_tensor(time, device=device), c["k"], c["u1"], c["u2"], c["period"], c["a_rs"], c["inc"], c["ecc"], c["omega"], t0=c["t0"], exposure=0.00139, **options).cpu().numpy()
    want = np.array([[float.fromhex(v) for v in row] for row in recorded])
    largest = float(np.max(np.abs(got - want)))
    assert largest <= tolerance, f"{int((got != want).sum())} of {got.size} values differ, the largest by {largest:.3g}"


def test_the_binned_second_pass_is_used_at_20_midpoints_and_not_at_3_nodes(device, monkeypatch):
    from velociraptor import triton_analytic

    calls = []
    original = triton_analytic._second_pass_binned
    monkeypatch.setattr(triton_analytic, "_second_pass_binned", lambda *a, **k: calls.append(1) or original(*a, **k))
    time, c = kernel_batch()
    args = (torch.as_tensor(time, device=device), c["k"], c["u1"], c["u2"], c["period"], c["a_rs"], c["inc"], c["ecc"], c["omega"])
    triton_analytic.transit_flux_triton_analytic(*args, t0=c["t0"], exposure=0.00139, exposure_nodes=3)
    assert calls == []
    triton_analytic.transit_flux_triton_analytic(*args, t0=c["t0"], exposure=0.00139, exposure_nodes=None, samples=20)
    assert calls == [1]


def test_the_terms_of_a_draw_equal_the_numpy_expressions(device):
    from velociraptor.triton_analytic import row_terms

    rng = np.random.default_rng(1)
    n = 500
    columns = [rng.uniform(0.01, 0.5, n), rng.uniform(0.2, 0.5, n), rng.uniform(0.1, 0.3, n), rng.uniform(0.5, 30, n), rng.uniform(2, 50, n), rng.uniform(1.3, 1.5707, n), rng.uniform(0, 0.8, n), rng.uniform(0, 2 * np.pi, n), rng.uniform(-0.1, 0.1, n), np.full(n, 0.00139)]
    made = row_terms([torch.as_tensor(c, device=device) for c in columns])
    k, u1, u2, period, a, inc, ecc, omega, t0, exposure = columns
    start = np.pi / 2 - omega
    anomaly0 = 2 * np.arctan2(np.sqrt(1 - ecc) * np.sin(start / 2), np.sqrt(1 + ecc) * np.cos(start / 2))
    want = [k, u1, u2, period, a, np.cos(inc), ecc, np.sin(omega), t0, exposure, np.cos(omega), anomaly0 - ecc * np.sin(anomaly0), np.sqrt(1 - ecc * ecc), 1.0 / (1 - u1 / 3 - u2 / 6)]
    for index, (got, expected) in enumerate(zip(made, want)):
        assert np.max(np.abs(got.cpu().numpy() - expected)) < 1e-13, index


def test_leaving_out_the_blocks_that_are_too_far_in_phase_changes_no_value_including_eccentric_short_orbits(device, monkeypatch):
    """The first pass skips a block of pairs when every pair is too far in phase from conjunction to come within 1 + k + reach of the star. Their flux is 1 either way, so the output must be the same bit for bit with the skip on and off, on draws with eccentricities up to 0.9, periods from 0.3 days, large and small radius ratios and a window of 0.3 days."""
    from velociraptor import triton_analytic

    rng = np.random.default_rng(5)
    n, points = 3000, 120
    period = np.exp(rng.uniform(np.log(0.3), np.log(40.0), n))
    a_rs = 3.0 * period ** (2 / 3) * rng.uniform(0.4, 3.0, n)
    ecc = np.where(np.arange(n) % 2 == 0, rng.uniform(0.0, 0.9, n), rng.uniform(0.0, 0.2, n))
    omega = rng.uniform(0, 2 * np.pi, n)
    b = rng.uniform(0.0, 1.4, n)
    inc = np.arccos(np.clip(b / (a_rs * (1 - ecc ** 2) / (1 + ecc * np.sin(omega))), -1, 1))
    k = np.exp(rng.uniform(np.log(0.003), np.log(0.9), n))
    time = np.tile(np.linspace(-0.3, 0.3, points), (n, 1))
    args = (torch.as_tensor(time, device=device), k[:, None], 0.4, 0.25, period[:, None], a_rs[:, None], inc[:, None], ecc[:, None], omega[:, None])
    results = {}
    for skip in (False, True):
        monkeypatch.setattr(triton_analytic, "PHASE_SKIP", skip)
        results[skip] = triton_analytic.transit_flux_triton_analytic(*args, t0=rng.uniform(-0.05, 0.05, n)[:, None] * 0 + 0.01, exposure=0.00139, exposure_nodes=3).cpu().numpy()
    assert np.array_equal(results[True], results[False]), f"{int((results[True] != results[False]).sum())} of {results[True].size} values differ"
    assert (results[False] < 1).mean() > 0.01, "the batch has transits in it"


@pytest.mark.parametrize("options", [dict(exposure_nodes=3), dict(exposure_nodes=None, samples=20)])
def test_the_mixed_precision_flux_is_within_1e_6_of_float64_and_does_not_depend_on_the_phase_skip(device, options, monkeypatch):
    """precision="mixed" puts the exposure nodes of the first pass in float32: on the golden batch (eccentric, short and grazing orbits) its flux is within 1e-6 of the float64 flux, and with the phase skip on or off it is the same bit for bit."""
    from velociraptor import triton_analytic
    from velociraptor.triton_analytic import transit_flux_triton_analytic

    time, c = kernel_batch()
    args = (torch.as_tensor(time, device=device), c["k"], c["u1"], c["u2"], c["period"], c["a_rs"], c["inc"], c["ecc"], c["omega"])
    run = lambda precision: transit_flux_triton_analytic(*args, t0=c["t0"], exposure=0.00139, precision=precision, **options).cpu().numpy()
    full = run("float64")
    mixed = run("mixed")
    assert float(np.max(np.abs(mixed - full))) <= 1e-6
    assert (full < 1).mean() > 0.01 and not np.array_equal(mixed, full), "the batch has transits and the two precisions are different code"
    monkeypatch.setattr(triton_analytic, "PHASE_SKIP", False)
    assert np.array_equal(run("mixed"), mixed)


@pytest.mark.parametrize("binned,options", [(False, dict(exposure_nodes=3)), (True, dict(exposure_nodes=None, samples=20))])
def test_the_float32_closed_form_of_the_second_pass_stays_within_1e_6_and_changes_the_flagged_pairs(device, binned, options, monkeypatch):
    """With precision="mixed" and SINGLE_CLOSED the closed form of the second pass (in the main kernel at 3 nodes, in the binned pass at 20 midpoints) is float32 from the float64 sky separation: the golden batch stays within 1e-6 of float64, and the flux differs from the float64 closed form of the same mixed run in the pairs that go to the second pass."""
    from velociraptor import triton_analytic
    from velociraptor.triton_analytic import transit_flux_triton_analytic

    monkeypatch.setattr(triton_analytic, "BINNED", binned)
    time, c = kernel_batch()
    args = (torch.as_tensor(time, device=device), c["k"], c["u1"], c["u2"], c["period"], c["a_rs"], c["inc"], c["ecc"], c["omega"])
    run = lambda precision: transit_flux_triton_analytic(*args, t0=c["t0"], exposure=0.00139, precision=precision, **options).cpu().numpy()
    full = run("float64")
    monkeypatch.setattr(triton_analytic, "SINGLE_CLOSED", False)
    double_closed = run("mixed")
    monkeypatch.setattr(triton_analytic, "SINGLE_CLOSED", True)
    single_closed = run("mixed")
    assert float(np.max(np.abs(single_closed - full))) <= 1e-6
    assert not np.array_equal(single_closed, double_closed), "the float32 closed form changed nothing: it was not used"


def closed_form_at(b, k, u1, u2, single, device):
    """The kernel's own closed form (velociraptor.triton_analytic._flux_closed) on arrays of separations, in float64 or, with `single`, in float32 from the same float64 inputs."""
    import triton
    import triton.language as tl

    from velociraptor.triton_analytic import _flux_closed

    @triton.jit
    def closed_kernel(b_ptr, k_ptr, u1_ptr, u2_ptr, out_ptr, count, SINGLE: tl.constexpr, BLOCK: tl.constexpr):
        lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = lane < count
        pi = tl.full([BLOCK], 3.141592653589793, dtype=tl.float64)
        value = _flux_closed(tl.load(b_ptr + lane, mask=valid, other=0.3), tl.load(k_ptr + lane, mask=valid, other=0.2), tl.load(u1_ptr + lane, mask=valid, other=0.3), tl.load(u2_ptr + lane, mask=valid, other=0.2), pi, 8, 1.4901161193847656e-08, SINGLE)
        tl.store(out_ptr + lane, value, mask=valid)

    tensors = [torch.as_tensor(np.asarray(v, dtype=np.float64), device=device) for v in (b, k, u1, u2)]
    out = torch.empty_like(tensors[0])
    closed_kernel[(triton.cdiv(len(b), 32),)](*tensors, out, len(b), SINGLE=single, BLOCK=32, num_warps=1)
    return out.cpu().numpy()


def boundary_points():
    """Separations within a few float32 steps of every branch boundary of the closed form, for radius ratios on both sides of 1: the outer contact b = 1 + k, the inner contact b = |1 - k| (b + k = 1 for k < 1, b = k - 1 for k > 1), b = k, b = 0, and k = 1 itself; only the points that float64 hands to the closed form (over the disc, the planet not covering the star), and the case b = 1 + k - 1e-8 at k = 1.7."""
    points = []
    for k in (0.003, 0.2, 0.5, 0.9999999, 1.0, 1.0000001, 1.3, 1.7):
        for centre in (1 + k, abs(1 - k), k, 0.0):
            step = float(np.spacing(np.float32(max(centre, 1e-3))))
            for n in range(-4, 5):
                points.append((centre + n * step, k))
    points.append((1.7 + 1.0 - 1e-8, 1.7))
    b, k = (np.array(v) for v in zip(*points))
    keep = (b > 0) & (b < 1 + k) & ~((k >= 1) & (b <= k - 1))
    keep |= (b == 0) & (k < 1)
    return b[keep], k[keep]


def test_the_float32_closed_form_is_within_1e_6_of_float64_at_every_branch_boundary_including_the_one_that_failed_on_the_rows(device):
    """A separation that float64 puts over the disc can round across a contact in float32, where the partial-overlap formulas do not hold. The guard in _flux_closed gives 1 past the outer contact and 0 where the planet covers the star. The float32 and float64 closed forms are compared at points within four float32 steps of each boundary."""
    b, k = boundary_points()
    assert b.size > 100
    for u1, u2 in ((0.3, 0.2), (0.0, 0.0), (0.8, 0.0)):
        full = closed_form_at(b, k, np.full_like(b, u1), np.full_like(b, u2), False, device)
        low = closed_form_at(b, k, np.full_like(b, u1), np.full_like(b, u2), True, device)
        worst = int(np.argmax(np.abs(low - full)))
        assert np.all(np.isfinite(low))
        assert float(np.abs(low - full).max()) <= 1e-6, f"b={b[worst]!r} k={k[worst]!r} u=({u1}, {u2}): float64 {full[worst]!r}, float32 {low[worst]!r}"


def test_the_interpolation_table_reproduces_a_polynomial_of_degree_five_at_the_nodes_and_the_ends():
    from velociraptor.triton_analytic import interpolation_table

    fractions = (np.arange(20) + 0.5) / 20 - 0.5
    table = interpolation_table(fractions, "cpu").numpy()
    anchors, ends, nodes = table[:6], table[6:18].reshape(2, 6), table[18:].reshape(20, 6)
    polynomial = np.polynomial.Polynomial([0.3, -1.2, 0.7, 2.0, -0.9, 1.1])
    assert np.allclose(nodes @ polynomial(anchors), polynomial(fractions), rtol=0, atol=1e-13)
    assert np.allclose(ends @ polynomial(anchors), polynomial(np.array([-0.5, 0.5])), rtol=0, atol=1e-13)
    assert np.allclose(nodes.sum(axis=1), 1.0, rtol=0, atol=1e-14) and len(table) == 18 + 20 * 6


@pytest.mark.parametrize("exposure,options", [(0.00139, dict(exposure_nodes=None, samples=20)), (0.00694, dict(exposure_nodes=None, samples=20)), (0.00694, dict(exposure_nodes=20))])
def test_the_interpolated_exposure_nodes_agree_with_the_kepler_solve_at_each_node_in_the_flux_including_eccentric_short_orbits(device, exposure, options, monkeypatch):
    """The nodes of an exposure from the polynomial through six Kepler solves, used where it agrees with the equation at the ends of the exposure, on draws with eccentricities up to 0.9 and periods from 0.3 days at the exposures of 2 and 10 minutes: the flux differs from the exact solve at each node by at most 1e-9, against a limit of 1e-6 for the package, and the interpolation changes some values."""
    from velociraptor import triton_analytic

    rng = np.random.default_rng(11)
    n, points = 3000, 120
    period = np.exp(rng.uniform(np.log(0.3), np.log(40.0), n))
    a_rs = 3.0 * period ** (2 / 3) * rng.uniform(0.4, 3.0, n)
    ecc = np.where(np.arange(n) % 2 == 0, rng.uniform(0.0, 0.9, n), rng.uniform(0.0, 0.2, n))
    omega = rng.uniform(0, 2 * np.pi, n)
    b = rng.uniform(0.0, 1.4, n)
    inc = np.arccos(np.clip(b / (a_rs * (1 - ecc ** 2) / (1 + ecc * np.sin(omega))), -1, 1))
    k = np.exp(rng.uniform(np.log(0.003), np.log(0.9), n))
    time = np.tile(np.linspace(-0.3, 0.3, points), (n, 1))
    args = (torch.as_tensor(time, device=device), k[:, None], 0.4, 0.25, period[:, None], a_rs[:, None], inc[:, None], ecc[:, None], omega[:, None])
    results = {}
    for interpolate in (False, True):
        monkeypatch.setattr(triton_analytic, "INTERPOLATE", interpolate)
        results[interpolate] = triton_analytic.transit_flux_triton_analytic(*args, t0=0.01, exposure=exposure, **options).cpu().numpy()
    largest = float(np.max(np.abs(results[True] - results[False])))
    assert largest <= 1e-9, f"the flux differs by {largest:.3g} at {int((np.abs(results[True] - results[False]) > 1e-9).sum())} of {results[True].size} values"
    assert (results[True] != results[False]).any(), "the interpolation was never used"
    assert (results[False] < 1).mean() > 0.01, "the batch has transits in it"


def test_a_planet_that_passes_just_outside_the_disc_and_one_just_inside_the_edge_at_20_midpoints_match_the_numpy_model(device):
    """Blocks of pairs whose nodes are all off the disc take no tier test: a pass at an impact parameter of 1 + k + 0.002 (flux 1 at every point, the exposure reaching within the contact distance) and one at 1 - k - 0.002 (the series), 32 points each across the transit."""
    from velociraptor.triton_analytic import transit_flux_triton_analytic

    k, period, a_rs = 0.05, 3.0, 8.0
    time = np.tile(np.linspace(-0.12, 0.12, 64), (2, 1))
    b = np.array([1 + k + 0.002, 1 - k - 0.002])[:, None]
    inc = np.arccos(b / a_rs)
    got = transit_flux_triton_analytic(torch.as_tensor(time, device=device), k, 0.4, 0.25, period, a_rs, inc, 0.0, np.pi / 2, 0.0, exposure=0.0139, exposure_nodes=None, samples=20).cpu().numpy()
    want = transit_flux(time, k, 0.4, 0.25, period, a_rs, inc, 0.0, np.pi / 2, 0.0, 0.0139, 20)
    assert np.max(np.abs(got[0] - 1.0)) < 1e-12, "a pass outside the disc has the flux 1 at every point, up to the rounding of the sum of the 20 weights"
    assert np.max(np.abs(got - want)) < 1e-6, np.max(np.abs(got - want))
    assert got[1].min() < 1 - 1e-3, "the pass inside the edge has a transit"


@pytest.mark.parametrize("exposure,options", [(0.00139, dict(exposure_nodes=None, samples=20)), (0.00694, dict(exposure_nodes=None, samples=20)), (0.00694, dict(exposure_nodes=20))])
def test_the_first_pass_over_the_list_of_near_pairs_agrees_with_the_single_stage_pass_including_eccentric_short_orbits(device, exposure, options, monkeypatch):
    """The node loop run over the compacted list of the pairs whose nodes are to be evaluated against the loop over every pair, on draws with eccentricities up to 0.9 and periods from 0.3 days at the exposures of 2 and 10 minutes: the flux differs by at most 1e-9, because the blocks of the list choose their path for the eccentric anomaly from other pairs and the last bits can differ. A batch with no near pair gives the flux 1."""
    from velociraptor import triton_analytic

    rng = np.random.default_rng(13)
    n, points = 3000, 120
    period = np.exp(rng.uniform(np.log(0.3), np.log(40.0), n))
    a_rs = 3.0 * period ** (2 / 3) * rng.uniform(0.4, 3.0, n)
    ecc = np.where(np.arange(n) % 2 == 0, rng.uniform(0.0, 0.9, n), rng.uniform(0.0, 0.2, n))
    omega = rng.uniform(0, 2 * np.pi, n)
    b = rng.uniform(0.0, 1.4, n)
    inc = np.arccos(np.clip(b / (a_rs * (1 - ecc ** 2) / (1 + ecc * np.sin(omega))), -1, 1))
    k = np.exp(rng.uniform(np.log(0.003), np.log(0.9), n))
    time = np.tile(np.linspace(-0.3, 0.3, points), (n, 1))
    args = (torch.as_tensor(time, device=device), k[:, None], 0.4, 0.25, period[:, None], a_rs[:, None], inc[:, None], ecc[:, None], omega[:, None])
    results = {}
    for compact in (False, True):
        monkeypatch.setattr(triton_analytic, "COMPACT", compact)
        results[compact] = triton_analytic.transit_flux_triton_analytic(*args, t0=0.01, exposure=exposure, **options).cpu().numpy()
    largest = float(np.max(np.abs(results[True] - results[False])))
    assert largest <= 1e-9, f"the flux differs by {largest:.3g} at {int((np.abs(results[True] - results[False]) > 1e-9).sum())} of {results[True].size} values"
    assert (results[False] < 1).mean() > 0.01, "the batch has transits in it"
    far = triton_analytic.transit_flux_triton_analytic(torch.as_tensor(np.full((1, 64), 1.3), device=device), 0.05, 0.4, 0.25, 3.0, 8.0, np.pi / 2 - 0.0001, 0.0, np.pi / 2, 0.0, exposure=exposure, **options).cpu().numpy()
    assert np.max(np.abs(far - 1.0)) < 1e-12, "a window with no near pair has the flux 1"


def test_the_second_pass_in_chunks_of_flagged_pairs_agrees_with_one_chunk(device, monkeypatch):
    """The chunks split the flagged pairs and nothing else: with chunks of 1000 flagged pairs the flux differs from one chunk of all of them by at most 1e-14. The closed form stops iterating when every lane of a block has converged, and the blocks of a chunk hold other nodes. The draws have large radius ratios so that many pairs are flagged."""
    from velociraptor import triton_analytic

    rng = np.random.default_rng(17)
    n, points = 1500, 120
    period = np.exp(rng.uniform(np.log(0.5), np.log(20.0), n))
    a_rs = 3.0 * period ** (2 / 3) * rng.uniform(0.6, 2.0, n)
    ecc = rng.uniform(0.0, 0.3, n)
    omega = rng.uniform(0, 2 * np.pi, n)
    b = rng.uniform(0.0, 1.2, n)
    inc = np.arccos(np.clip(b / (a_rs * (1 - ecc ** 2) / (1 + ecc * np.sin(omega))), -1, 1))
    k = rng.uniform(0.1, 0.9, n)
    time = np.tile(np.linspace(-0.3, 0.3, points), (n, 1))
    args = (torch.as_tensor(time, device=device), k[:, None], 0.4, 0.25, period[:, None], a_rs[:, None], inc[:, None], ecc[:, None], omega[:, None])
    results = {}
    for chunk in (1 << 40, 1000):
        monkeypatch.setattr(triton_analytic, "FLAGGED_CHUNK", chunk)
        triton_analytic.TRACE = []
        results[chunk] = triton_analytic.transit_flux_triton_analytic(*args, t0=0.01, exposure=0.00139, exposure_nodes=None, samples=20).cpu().numpy()
        flagged = triton_analytic.TRACE[-1]["flagged_pairs"]
        triton_analytic.TRACE = None
    assert flagged > 5000, f"only {flagged} flagged pairs: the batch must need several chunks"
    largest = float(np.max(np.abs(results[1000] - results[1 << 40])))
    assert largest <= 1e-14, f"the flux differs by {largest:.3g}"


def test_a_call_of_a_million_draws_with_every_pair_flagged_stays_under_8_gib_of_device_memory(device):
    """The size of TRICERATOPS's default number of draws (10^6) with 100 points and radius ratios of 0.5 and more, where the series cannot be used: the peak device memory above what the call's inputs and outputs hold stays under 8 GiB because the second pass works in chunks."""
    from velociraptor import triton_analytic

    free, _ = torch.cuda.mem_get_info()
    if free < 12 * 2**30:
        pytest.skip("the call needs 12 GiB free on the device")
    n, points = 1_000_000, 100
    rng = np.random.default_rng(3)
    torch.cuda.empty_cache()
    columns = [torch.as_tensor(v, dtype=torch.float64, device=device)[:, None] for v in (rng.uniform(0.5, 0.9, n), np.full(n, 0.4), np.full(n, 0.25), np.full(n, 3.0), np.full(n, 8.0), np.arccos(rng.uniform(0.0, 1.0, n) / 8.0), np.zeros(n), np.full(n, np.pi / 2))]
    time = torch.as_tensor(np.linspace(-0.2, 0.2, points), device=device)
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    triton_analytic.TRACE = []
    flux = triton_analytic.transit_flux_triton_analytic(time, *columns, exposure=0.00139, exposure_nodes=None, samples=20)
    flagged = triton_analytic.TRACE[-1]["flagged_pairs"]
    triton_analytic.TRACE = None
    peak = torch.cuda.max_memory_allocated() - before
    assert flagged > 0.2 * n * points, f"only {flagged} of {n * points} pairs flagged"
    assert flux.shape == (n, points) and bool((flux <= 1.0 + 1e-12).all())
    assert peak < 8 * 2**30, f"peak {peak / 2**30:.2f} GiB above the inputs"


def test_the_shipped_precision_is_mixed_and_float64_stays_selectable():
    """The source declares "mixed" as the module's precision, and transit_flux_triton_analytic takes it when no precision is given; the other value the function accepts is "float64"."""
    import inspect
    import re

    from velociraptor import triton_analytic

    source = inspect.getsource(triton_analytic)
    assert re.search(r'^PRECISION = "mixed"', source, re.MULTILINE)
    assert "precision = PRECISION if precision is None else precision" in source
    assert inspect.signature(triton_analytic.transit_flux_triton_analytic).parameters["precision"].default is None


@pytest.mark.default_precision
@pytest.mark.parametrize("options", [dict(exposure_nodes=3), dict(exposure_nodes=None, samples=20)])
def test_without_a_precision_the_device_gives_the_mixed_flux_bit_for_bit_and_float64_is_a_different_flux(device, options):
    from velociraptor.triton_analytic import transit_flux_triton_analytic

    time, c = kernel_batch()
    args = (torch.as_tensor(time, device=device), c["k"], c["u1"], c["u2"], c["period"], c["a_rs"], c["inc"], c["ecc"], c["omega"])
    run = lambda **precision: transit_flux_triton_analytic(*args, t0=c["t0"], exposure=0.00139, **precision, **options).cpu().numpy()
    default = run()
    assert np.array_equal(default, run(precision="mixed"))
    assert not np.array_equal(default, run(precision="float64"))
