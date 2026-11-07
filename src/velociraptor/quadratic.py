"""Quadratic limb-darkened transit model, written from Mandel and Agol (2002, ApJ 580, L171, arXiv:astro-ph/0210099).

The star has the intensity I(mu) = 1 - u1 (1 - mu) - u2 (1 - mu)^2 with mu = sqrt(1 - r^2) at the radius r (in stellar radii) from its centre (their eq. 1 and 2); the planet is a dark disc of radius k at the
separation z of the two centres. The flux removed is the integral of I over the area of the planet inside the star, divided by the flux of the whole star, pi (1 - u1/3 - u2/6).

On the circle of radius r about the star's centre, the arc inside the planet has the half-angle theta(r) = arccos((r^2 + z^2 - k^2) / (2 r z)) for |z - k| < r < z + k, the whole circle for r < k - z and none elsewhere.
The removed flux is therefore

    pi-weighted:  the integral of 2 pi r I dr for r from 0 to k - z (a polynomial in mu, done in closed form) plus the integral of 2 theta(r) r I(r) dr for r from |z - k| to min(1, z + k).

The second integral is done by Gauss-Legendre quadrature after the substitution r = m - h cos phi, with m and h the middle and half-width of the range: the square-root behaviour of theta at both ends of the range and of mu at the limb
becomes a smooth function of phi, so a fixed number of nodes gives a fixed accuracy without branches on the case of Mandel and Agol's Table 1.
"""

import functools

import numpy as np

from velociraptor.analytic import quadratic_flux

CHUNK_ELEMENTS = 100_000  # elements of quadrature_flux evaluated at once; it bounds the memory of the intermediates
KEPLER_ITERATIONS = 12  # the most Newton steps for Kepler's equation; from M + e sin M it converges to 1e-15 in a few steps for e up to 0.99
NODES = 48  # the nodes of the quadrature


def _polynomial_part(radius, u1, u2):
    """The integral of 2 pi r I(r) dr from 0 to `radius` (at most 1), the flux behind a disc of that radius centred on the star."""
    mu = np.sqrt(np.maximum(1.0 - radius**2, 0.0))
    first = (1 - mu**2) / 2
    second = (1 - mu**3) / 3
    third = (1 - mu**4) / 4
    weight_zero = first
    weight_one = first - second  # integral of mu (1 - mu)
    weight_two = first - 2 * second + third  # integral of mu (1 - mu)^2
    return 2 * np.pi * (weight_zero - u1 * weight_one - u2 * weight_two)


@functools.cache
def gauss_legendre(nodes):
    """The nodes and weights of the Gauss-Legendre rule of `nodes` points on [-1, 1], computed once for each count and returned read-only."""
    x, w = np.polynomial.legendre.leggauss(nodes)
    x.flags.writeable = w.flags.writeable = False
    return x, w


def quadrature_flux(z, k, u1, u2, nodes=NODES):
    """The relative flux of the star, 1 when the planet is off its disc, at the separations z (an array, in stellar radii; the sign of z does not matter) of a planet of radius ratio k, with the quadratic
    limb-darkening coefficients u1 and u2."""
    z, k, u1, u2 = np.broadcast_arrays(np.abs(np.asarray(z, dtype=float)), *(np.asarray(value, dtype=float) for value in (k, u1, u2)))
    if z.size > CHUNK_ELEMENTS:  # the node axis multiplies every intermediate by `nodes`, so a long input goes through in pieces of CHUNK_ELEMENTS
        flat = [value.reshape(-1) for value in (z, k, u1, u2)]
        out = np.empty(z.size)
        for start in range(0, z.size, CHUNK_ELEMENTS):
            part = slice(start, start + CHUNK_ELEMENTS)
            out[part] = quadrature_flux(*(value[part] for value in flat), nodes=nodes)
        return out.reshape(z.shape)
    total = np.pi * (1 - u1 / 3 - u2 / 6)
    low = np.abs(z - k)
    high = np.minimum(1.0, z + k)
    removed = np.where(k > z, _polynomial_part(np.minimum(k - z, 1.0), u1, u2), 0.0)
    wide = high > low
    x, w = gauss_legendre(nodes)
    phi = (x + 1) * np.pi / 2  # phi from 0 to pi
    weight = w * np.pi / 2
    half = np.where(wide, (high - low) / 2, 0.0)[..., None]
    radius = ((low + high) / 2)[..., None] - half * np.cos(phi)
    z_, k_, u1_, u2_ = z[..., None], k[..., None], u1[..., None], u2[..., None]
    high_ = high[..., None]
    from_low = half * (1 - np.cos(phi))  # r - low, without the cancellation of r - |z - k| near the ends of the range
    from_high = half * (1 + np.cos(phi))  # high - r
    # tan(theta / 2)^2 = (k^2 - (r - z)^2) / ((r + z)^2 - k^2), each factor written so that no difference of nearly equal numbers is formed at the tangencies z = |1 - k|, z = 1 + k and r = |z - k|
    numerator = ((z_ + k_ - high_) + from_high) * (2 * np.maximum(k_ - z_, 0.0) + from_low)
    denominator = (from_low + 2 * np.maximum(z_ - k_, 0.0)) * (radius + z_ + k_)
    theta = 2 * np.arctan2(np.sqrt(np.maximum(numerator, 0.0)), np.sqrt(np.maximum(denominator, 1e-300)))
    mu = np.sqrt(np.maximum(1.0 - radius**2, 0.0))
    intensity = 1 - u1_ * (1 - mu) - u2_ * (1 - mu) ** 2
    jacobian = half * np.sin(phi)
    arc = np.sum(weight * 2 * theta * radius * intensity * jacobian, axis=-1)
    removed = removed + np.where(wide, arc, 0.0)
    return 1.0 - removed / total


def _mean_anomaly(time, period, ecc, omega, t0):
    """The mean anomaly at `time` for the orbit whose transit centre is at t0, where the true anomaly is pi/2 - omega (Winn 2010, arXiv:1001.2010, eq. 1 to 7)."""
    start = np.pi / 2 - omega
    anomaly0 = 2 * np.arctan2(np.sqrt(1 - ecc) * np.sin(start / 2), np.sqrt(1 + ecc) * np.cos(start / 2))  # eccentric anomaly at t0
    return anomaly0 - ecc * np.sin(anomaly0) + 2 * np.pi * (time - t0) / period


def _kepler(mean, ecc, guess=None):
    """Kepler's equation E - e sin E = M solved by Newton's method from `guess` (default M + e sin M) until no step is over 1e-15, at most KEPLER_ITERATIONS steps."""
    eccentric = mean + ecc * np.sin(mean) if guess is None else guess
    for _ in range(KEPLER_ITERATIONS):
        sine, cosine = np.sin(eccentric), np.cos(eccentric)
        step = (eccentric - ecc * sine - mean) / (1 - ecc * cosine)
        eccentric = eccentric - step
        if np.max(np.abs(step), initial=0.0) < 1e-15:
            break
    return eccentric


def _project(eccentric, a_rs, inc, ecc, omega):
    """The sky separation and whether the planet is in front, from the eccentric anomaly. In the orbital plane the planet is at X = a (cos E - e), Y = a sqrt(1 - e^2) sin E. The position along
    the line of sight is S = sin(omega) X + cos(omega) Y (in front of the star where S > 0), and the other sky component is cos(omega) X - sin(omega) Y. The sky separation is
    the hypotenuse of that component and S cos(i), a sum of squares with no cancellation near mid-transit (Winn 2010, arXiv:1001.2010, eq. 1 to 7)."""
    x = a_rs * (np.cos(eccentric) - ecc)
    y = a_rs * np.sqrt(1 - ecc**2) * np.sin(eccentric)
    along = np.sin(omega) * x + np.cos(omega) * y
    across = np.cos(omega) * x - np.sin(omega) * y
    return np.hypot(across, along * np.cos(inc)), along > 0


def separation(time, period, a_rs, inc, ecc=0.0, omega=np.pi / 2, t0=0.0):
    """The sky-projected separation of the centres of planet and star in stellar radii, and whether the planet is in front of the star (a transit), at `time` (days) for the orbit with the period `period` (days), a/R* `a_rs`,
    inclination `inc` (radians), eccentricity `ecc` and argument of periastron `omega` (radians, of the planet's orbit as in Winn 2010, arXiv:1001.2010, eq. 1 to 7), with the transit centre at `t0`. Returns (z, in_front).
    Kepler's equation is solved by Newton's method. All arguments broadcast together."""
    time, period, a_rs, inc, ecc, omega, t0 = np.broadcast_arrays(*(np.asarray(value, dtype=float) for value in (time, period, a_rs, inc, ecc, omega, t0)))
    return _project(_kepler(_mean_anomaly(time, period, ecc, omega, t0), ecc), a_rs, inc, ecc, omega)


def orbit_series(period, a_rs, ecc, omega, order=4):
    """The Taylor coefficients about mid-transit (time in days) of the position (X, Y) of the planet in its orbital plane, in stellar radii, as two arrays of shape (order + 1,) + the shape of the broadcast arguments.

    The planet obeys r'' = -mu r / |r|^3 with mu = n^2 a^3 and n = 2 pi / P. Its position and velocity at mid-transit, where the true anomaly is pi/2 - omega, give the coefficients of order 0 and 1. The others follow
    from the series of the right-hand side: with w the series of |r|^2 and s that of |r|^-3 = w^(-3/2), the coefficients of a power y = w^p obey n w_0 y_n = sum over k = 1..n of (p k - (n - k)) w_k y_(n-k) (Taylor
    arithmetic of the power function, from w y' = p w' y), and r_(m+2) (m + 2)(m + 1) = -mu (sum over j of r_j s_(m-j))."""
    period, a_rs, ecc, omega = np.broadcast_arrays(*(np.asarray(value, dtype=float) for value in (period, a_rs, ecc, omega)))
    n = 2 * np.pi / period
    mu = n**2 * a_rs**3
    f0 = np.pi / 2 - omega
    distance = a_rs * (1 - ecc**2) / (1 + ecc * np.cos(f0))
    speed = n * a_rs / np.sqrt(1 - ecc**2)
    X = np.zeros((order + 1,) + period.shape)
    Y = np.zeros_like(X)
    X[0], Y[0] = distance * np.cos(f0), distance * np.sin(f0)
    X[1], Y[1] = -speed * np.sin(f0), speed * (ecc + np.cos(f0))
    w = np.zeros_like(X)
    s = np.zeros_like(X)
    for m in range(order - 1):
        w[m] = sum(X[j] * X[m - j] + Y[j] * Y[m - j] for j in range(m + 1))
        s[m] = w[0] ** -1.5 if m == 0 else sum((-1.5 * k - (m - k)) * w[k] * s[m - k] for k in range(1, m + 1)) / (m * w[0])
        X[m + 2] = -mu * sum(X[j] * s[m - j] for j in range(m + 1)) / ((m + 2) * (m + 1))
        Y[m + 2] = -mu * sum(Y[j] * s[m - j] for j in range(m + 1)) / ((m + 2) * (m + 1))
    return X, Y


def _series_project(offset, X, Y, inc, omega):
    """The sky separation and whether the planet is in front at `offset` days from mid-transit, from the series of orbit_series evaluated by Horner's rule (coefficients on the first axis)."""
    x = np.zeros(np.broadcast(offset, X[0]).shape)
    y = np.zeros_like(x)
    for c in range(len(X) - 1, -1, -1):
        x = x * offset + X[c]
        y = y * offset + Y[c]
    along = np.sin(omega) * x + np.cos(omega) * y
    across = np.cos(omega) * x - np.sin(omega) * y
    return np.hypot(across, along * np.cos(inc)), along > 0


TAYLOR_STEP = 0.02  # days: the step of the central differences of orbit_taylor (Parviainen and Korth 2020, arXiv:2009.09965, appendix)


def orbit_taylor(period, a_rs, ecc, omega, step=TAYLOR_STEP, order=4):
    """The coefficients about mid-transit of the sky position of the planet as the Taylor series of Parviainen and Korth (2020, MNRAS 499, 3356, arXiv:2009.09965). The derivatives of the
    position are central finite differences of seven positions of the exact Kepler orbit at the times k * step from mid-transit, k = -3..3 (the polynomial through the seven
    positions, truncated to `order`). Returns (across, along), each of shape (order + 1,) + the shape of the broadcast arguments: the component of the position that carries the transit (the sky x) and the one along the
    line of sight (its sky y is along times cos i, and the planet is in front where it is positive). With a step of 1e-3 d it is
    the series of orbit_series. The step of 0.02 d (their appendix A), the seven nodes (section 2) and the fourth order (their eq. 8) are those of Parviainen and Korth (2020)."""
    period, a_rs, ecc, omega = np.broadcast_arrays(*(np.asarray(value, dtype=float) for value in (period, a_rs, ecc, omega)))
    nodes = np.arange(-3, 4) * step
    weights = np.linalg.inv(np.vander(nodes, N=7, increasing=True))  # polynomial coefficients from the seven samples: c = inverse(Vandermonde) @ samples
    mean = _mean_anomaly(nodes[(slice(None),) + (None,) * period.ndim], period, ecc, omega, 0.0)
    eccentric = _kepler(mean, ecc)
    x = a_rs * (np.cos(eccentric) - ecc)
    y = a_rs * np.sqrt(1 - ecc**2) * np.sin(eccentric)
    along = np.sin(omega) * x + np.cos(omega) * y
    across = np.cos(omega) * x - np.sin(omega) * y
    return np.tensordot(weights[: order + 1], across, axes=(1, 0)), np.tensordot(weights[: order + 1], along, axes=(1, 0))


def _taylor_project(offset, across, along, inc):
    """The sky separation and whether the planet is in front from the coefficients of orbit_taylor, by Horner's rule."""
    x = np.zeros(np.broadcast(offset, across[0]).shape)
    y = np.zeros_like(x)
    for c in range(len(across) - 1, -1, -1):
        x = x * offset + across[c]
        y = y * offset + along[c]
    return np.hypot(x, y * np.cos(inc)), y > 0


def transit_flux(time, k, u1, u2, period, a_rs, inc, ecc=0.0, omega=np.pi / 2, t0=0.0, exposure=0.0, samples=1, orbit="exact", order=4):
    """The relative flux of the star at `time` (days) with the planet on the orbit of `separation`, for a radius ratio `k` and quadratic limb darkening `u1`, `u2`.

    A point of exposure `exposure` (days) is the mean over `samples` equal sub-intervals of its exposure, each taken at its middle, the convention that TRICERATOPS asks of its transit model for a long exposure (PyTransit 2.2.0 with
    exptimes and nsamples uses it). The arrays broadcast with time as the last axis, so the parameters of a batch of scenarios have the shape (n, 1) beside a time of shape (n, points).

    orbit="exact" solves Kepler's equation at the centre of each exposure, and for the sub-samples of the points that can reach the disc by Newton's method from the centre's solution carried by the mean
    motion. orbit="series" replaces the orbit by its Taylor series of the given order about mid-transit (orbit_series), valid within a quarter of the period of mid-transit and taken as out of transit beyond. orbit="taylor" is the
    series with finite-difference derivatives (orbit_taylor), the orbit of PyTransit 2.2.0."""
    raw = [np.asarray(value, dtype=float) for value in (period, a_rs, ecc, omega)]
    parameters = np.broadcast_arrays(*(np.asarray(value, dtype=float) for value in (time, k, u1, u2, period, a_rs, inc, ecc, omega, t0, exposure)))
    time, k, u1, u2, period, a_rs, inc, ecc, omega, t0, exposure = parameters
    if orbit in ("series", "taylor"):
        X, Y = orbit_series(*raw, order=order) if orbit == "series" else orbit_taylor(*raw, order=order)
        X, Y = (np.broadcast_to(value.reshape(value.shape + (1,) * (time.ndim - value.ndim + 1)), (order + 1,) + time.shape) for value in (X, Y))
        offset = time - t0
        centre, ahead = _series_project(offset, X, Y, inc, omega) if orbit == "series" else _taylor_project(offset, X, Y, inc)
        centre = np.where(ahead & (np.abs(offset) < period / 4), centre, np.inf)
    else:
        mean = _mean_anomaly(time, period, ecc, omega, t0)
        centre_anomaly = _kepler(mean, ecc)
        centre, _ = _project(centre_anomaly, a_rs, inc, ecc, omega)
    if samples > 1:
        # no sub-sample of a point can come within 1 + k of the star if the centre of the exposure is farther than that by more than the planet moves on the sky in half the exposure: its speed is at most 2 pi a (1 + e) / (P sqrt(1 - e^2))
        reach = 2 * np.pi * a_rs * (1 + ecc) / (period * np.sqrt(1 - ecc**2)) * exposure / 2
        near = centre - reach <= 1.0 + k
    else:
        near = centre < 1.0 + k
    flux = np.ones(time.shape)
    if not near.any():
        return flux
    select = lambda value: value[near][:, None]
    if orbit in ("series", "taylor"):
        sub = select(offset) + (((np.arange(samples) + 0.5) / samples - 0.5) * select(exposure) if samples > 1 else 0.0)
        if orbit == "series":
            z, in_front = _series_project(sub, X[:, near][:, :, None], Y[:, near][:, :, None], select(inc), select(omega))
        else:
            z, in_front = _taylor_project(sub, X[:, near][:, :, None], Y[:, near][:, :, None], select(inc))
        in_front = in_front & (np.abs(sub) < select(period) / 4)
    else:
        if samples > 1:
            shift = 2 * np.pi * ((np.arange(samples) + 0.5) / samples - 0.5) * select(exposure / period)  # the change of mean anomaly to each sub-sample
            mean_sub = select(mean) + shift
            guess = select(centre_anomaly) + shift / (1 - select(ecc) * np.cos(select(centre_anomaly)))
            anomaly = _kepler(mean_sub, select(ecc), guess)
        else:
            anomaly = select(centre_anomaly)
        z, in_front = _project(anomaly, select(a_rs), select(inc), select(ecc), select(omega))
    k_, u1_, u2_ = (np.broadcast_to(select(value), z.shape) for value in (k, u1, u2))
    overlap = in_front & (z < 1.0 + k_)  # the flux is 1 wherever the planet does not overlap the disc
    sub = np.ones(z.shape)
    sub[overlap] = quadratic_flux(z[overlap], k_[overlap], u1_[overlap], u2_[overlap])
    flux[near] = sub.mean(axis=-1)
    return flux
