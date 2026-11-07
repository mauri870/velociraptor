"""The series for a small planet wholly inside the disc against the closed form of velociraptor.analytic."""

import numpy as np
import pytest

from velociraptor import quadratic_flux
from velociraptor.series import (
    ABSOLUTE_FLUX_ERROR,
    ORDERS,
    series_flux,
    series_order,
    uses_series,
)

U = ((0.0, 0.0), (0.4, 0.25), (0.6, 0.1), (0.1, 0.5), (1.0, 0.0), (0.3, 0.7))


def draws(n, seed):
    g = np.random.default_rng(seed)
    k = 10 ** g.uniform(-4, np.log10(0.08), n)
    b = g.uniform(0.0, 1.0, n) * (1 - k)
    return b, k


@pytest.mark.parametrize("u1,u2", U)
def test_the_series_agrees_with_the_closed_form_where_it_is_used(u1, u2):
    b, k = draws(200_000, 5)
    order = series_order(b, k, u1, u2)
    assert (order > 0).sum() > 0.5 * len(b)
    for n in ORDERS:
        use = order == n
        if not use.any():  # a uniform disc has c1 = 0, so its bounds vanish and the lowest order is used throughout
            assert (u1, u2) == (0.0, 0.0)
            continue
        difference = np.abs(series_flux(b[use], k[use], u1, u2, order=n) - quadratic_flux(b[use], k[use], u1, u2))
        assert difference.max() < 5 * ABSOLUTE_FLUX_ERROR, (n, difference.max(), b[use][np.argmax(difference)], k[use][np.argmax(difference)])


def test_a_typical_small_planet_takes_the_series_nearly_everywhere_inside_the_disc():
    b = np.linspace(0.0, 0.97, 1000)
    assert uses_series(b, 0.0022, 0.4, 0.25).mean() > 0.99


def test_the_series_is_not_used_where_the_planet_touches_the_limb_or_is_large_against_the_limb_scale():
    assert not uses_series(np.array([0.99]), 0.02, 0.4, 0.25)[0]  # b + k > 1
    assert not uses_series(np.array([0.9999]), 0.00008, 0.4, 0.25)[0]  # k / mu^2 is 0.4: the tail of the series is above the tolerance
    assert not uses_series(np.array([0.3]), 0.5, 0.4, 0.25)[0]  # a large planet
    assert not uses_series(np.array([0.5]), 0.0, 0.4, 0.25)[0]  # no planet


def test_a_uniform_disc_loses_k_squared_of_the_flux():
    b = np.linspace(0.0, 0.9, 10)
    assert np.allclose(series_flux(b, 0.003, 0.0, 0.0), 1 - 0.003**2, atol=1e-16, rtol=0)


def test_an_empty_input_gives_an_empty_output():
    empty = np.zeros(0)
    assert series_flux(empty, empty, 0.4, 0.25).shape == (0,) and uses_series(empty, empty, 0.4, 0.25).shape == (0,)


def test_the_higher_order_takes_the_planets_the_lower_order_refuses():
    b = np.linspace(0.0, 0.5, 11)
    assert (series_order(b, 0.04, 0.4, 0.25) == ORDERS[1]).any() and (series_order(b, 0.0015, 0.4, 0.25) == ORDERS[0]).all()
    assert (series_order(b, 0.3, 0.4, 0.25) == 0).all()


def test_the_generated_coefficients_are_the_output_of_the_generator():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("generate_series_terms", root / "validation" / "generate_series_terms.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (root / "src" / "velociraptor" / "_series_terms.py").read_text() == module.text()
    assert (root / "src" / "velociraptor" / "_series_triton.py").read_text() == module.triton_text()


def _generated_triton_namespace():
    """Runs the generated Triton functions on numpy, with `triton.jit` as the identity and the few `triton.language` functions they use replaced by numpy's, so that their arithmetic is checked on the CPU."""
    from pathlib import Path
    from types import SimpleNamespace

    text = (Path(__file__).resolve().parent.parent / "src" / "velociraptor" / "_series_triton.py").read_text()
    language = SimpleNamespace(full=lambda shape, value, dtype=None: np.full(shape, value), sqrt=np.sqrt, abs=np.abs, maximum=np.maximum, where=np.where, float64=None)
    namespace = {"triton": SimpleNamespace(jit=lambda function: function), "tl": language}
    exec(text.replace("import triton\nimport triton.language as tl\n", ""), namespace)
    return namespace


def test_the_generated_triton_series_is_the_series_of_order_9_and_its_switch_is_the_switch():
    namespace = _generated_triton_namespace()
    b, k = draws(200_000, 11)
    for u1, u2 in U[1:]:
        normalisation = 1.0 / (1 - u1 / 3 - u2 / 6)
        order = series_order(b, k, u1, u2)
        low, high = namespace["_series_tier"](b, k, np.full_like(b, u1), np.full_like(b, u2), normalisation, np.ones(len(b), dtype=bool), ABSOLUTE_FLUX_ERROR / 2)
        assert np.array_equal(low, order == ORDERS[0]) and np.array_equal(high, order == ORDERS[1])
        use = order == ORDERS[1]
        got = namespace["_series_high"](b[use], k[use], np.full(use.sum(), u1), np.full(use.sum(), u2), normalisation)
        assert np.max(np.abs(got - series_flux(b[use], k[use], u1, u2, order=9))) < 1e-15
