"""velociraptor.splines.evaluate returns the same values as scipy's spline evaluation for any degree, inside and outside the knots, and is compiled without the interpreter lock."""
import numpy as np
import pytest

pytest.importorskip("numba")
pytest.importorskip("scipy")


@pytest.mark.parametrize("count,degree", [(200, 3), (50, 3), (30, 1), (80, 5), (10, 2), (5, 3)])
def test_the_values_are_scipys_to_the_last_bit(count, degree):
    from scipy.interpolate import InterpolatedUnivariateSpline

    from velociraptor.splines import evaluate

    rng = np.random.default_rng(count * 10 + degree)
    x = np.sort(rng.uniform(0, 1, count))
    spline = InterpolatedUnivariateSpline(x, np.sin(7 * x) + 0.1 * rng.normal(size=count), k=degree)
    points = np.concatenate([rng.uniform(-0.3, 1.3, 100000), x, [x[0], x[-1], np.nan]])  # inside, outside the knots, on every knot, on both ends, not a number
    assert np.array_equal(spline(points), evaluate(spline, points), equal_nan=True)


def test_a_two_dimensional_input_keeps_its_shape_and_values():
    from scipy.interpolate import InterpolatedUnivariateSpline

    from velociraptor.splines import evaluate

    x = np.linspace(0, 1, 40)
    spline = InterpolatedUnivariateSpline(x, x ** 2, k=3)
    points = np.random.default_rng(0).uniform(0, 1, (7, 5))
    assert np.array_equal(evaluate(spline, points), spline(points)) and evaluate(spline, points).shape == (7, 5)


def test_the_replacement_applies_to_large_plain_calls_only_and_is_put_back():
    from scipy.interpolate import InterpolatedUnivariateSpline

    from velociraptor import splines

    x = np.linspace(0, 1, 40)
    spline = InterpolatedUnivariateSpline(x, np.cos(x), k=3)
    original = InterpolatedUnivariateSpline.__call__
    big, small = np.random.default_rng(1).uniform(0, 1, 5000), np.array([0.2, 0.7])
    with splines.released_splines():
        assert InterpolatedUnivariateSpline.__call__ is not original
        assert np.array_equal(spline(big), original(spline, big))
        assert np.array_equal(spline(small), original(spline, small))
        assert np.array_equal(spline(big, nu=1), original(spline, big, nu=1))
        assert np.array_equal(spline(big, ext=1), original(spline, big, ext=1))
    assert InterpolatedUnivariateSpline.__call__ is original


def test_the_evaluation_is_compiled_without_the_lock():
    from velociraptor.splines import _compile

    assert _compile().targetoptions.get("nogil") is True
