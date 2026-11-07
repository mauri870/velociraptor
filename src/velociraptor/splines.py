"""The evaluation of a scipy InterpolatedUnivariateSpline (cubic or any degree, no derivative, the default extrapolation) without the global interpreter lock.

scipy's evaluation (FITPACK's splev) holds the lock for its whole run, so threads cannot overlap the sampling of different TRICERATOPS scenarios (velociraptor.overlap). `evaluate` is the recurrence of de Boor, "On calculating with B-splines", J. Approx. Theory 6 (1972) 50-62, with the same operations in the same order as FITPACK, so the values are identical to the last bit. It is compiled by numba with the lock released.
"""
import contextlib

import numpy as np

_COMPILED = None


def _compile():
    global _COMPILED
    if _COMPILED is None:
        from numba import njit

        @njit(nogil=True, cache=True)
        def evaluate(knots, coefficients, degree, points, out):
            count = knots.shape[0]
            order = degree + 1
            low = knots[degree]
            weights = np.empty(order)
            previous = np.empty(order)
            for index in range(points.shape[0]):
                x = points[index]
                if np.isnan(x):
                    out[index] = np.nan
                    continue
                span = degree  # the interval with knots[span] <= x < knots[span + 1]
                last = count - order - 1
                if x >= knots[last]:
                    span = last
                elif x > low:
                    left, right = degree, last
                    while right - left > 1:
                        middle = (left + right) // 2
                        if x >= knots[middle]:
                            left = middle
                        else:
                            right = middle
                    span = left
                weights[0] = 1.0
                for j in range(1, degree + 1):
                    for i in range(j):
                        previous[i] = weights[i]
                    weights[0] = 0.0
                    for i in range(j):
                        upper = span + 1 + i
                        lower = upper - j
                        f = previous[i] / (knots[upper] - knots[lower])
                        weights[i] = weights[i] + f * (knots[upper] - x)
                        weights[i + 1] = f * (x - knots[lower])
                total = 0.0
                for j in range(order):
                    total = total + coefficients[span - degree + j] * weights[j]
                out[index] = total
            return out

        _COMPILED = evaluate
    return _COMPILED


def evaluate(spline, x):
    """The values of `spline` at `x` (an array of any shape), identical to spline(x) to the last bit."""
    knots, coefficients, degree = spline._eval_args
    array = np.ascontiguousarray(x, dtype=float)
    out = np.empty(array.size)
    _compile()(np.ascontiguousarray(knots, dtype=float), np.ascontiguousarray(coefficients, dtype=float), int(degree), array.reshape(-1), out)
    return out.reshape(array.shape)


@contextlib.contextmanager
def released_splines(minimum=1000):
    """Inside the block, a call of an InterpolatedUnivariateSpline on an array of at least `minimum` values, with the default extrapolation and no derivative, runs `evaluate`. Any other call runs scipy's."""
    from scipy.interpolate import InterpolatedUnivariateSpline

    original = InterpolatedUnivariateSpline.__call__

    def call(self, x, nu=0, ext=None):
        if nu != 0 or (ext if ext is not None else self.ext) != 0 or not isinstance(x, np.ndarray) or x.size < minimum or x.dtype.kind != "f":
            return original(self, x, nu=nu, ext=ext)
        return evaluate(self, x)

    InterpolatedUnivariateSpline.__call__ = call
    try:
        yield
    finally:
        InterpolatedUnivariateSpline.__call__ = original
