"""The references of README.md: lens area, two-dimensional integral, convergence and orbit; writes validation/results.json (the key of each reference).

usage: python validation/grid.py lens|integral|convergence|orbit|all   (needs mpmath and scipy besides numpy)
"""
import json
import sys
from pathlib import Path

import numpy as np

from velociraptor import quadratic_flux, quadrature_flux, separation

HERE = Path(__file__).parent
RESULTS = HERE / "results.json"
K = (0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.99, 1.0, 1.5, 3.0)
U = ((0.0, 0.0), (0.4, 0.25), (0.6, 0.1), (0.1, 0.5), (1.0, 0.0), (0.3, 0.7))


def separations(k):
    """The separations that the validation visits for radius ratio k: 0, 1e-9, 1e-6, 1e-3, 0.5 and 1, the contacts |1 - k| and 1 + k with neighbours at 1e-6 and 1e-9 on each side, and 41 evenly spaced values from 0 to 1 + k + 0.1."""
    edges = [0.0, 1e-9, 1e-6, 1e-3, 0.5, 1.0]
    for centre in (abs(1 - k), 1 + k):
        edges += [centre - 1e-6, centre - 1e-9, centre, centre + 1e-9, centre + 1e-6]
    z = np.concatenate([edges, np.linspace(0.0, 1 + k + 0.1, 41)])
    return np.unique(z[z >= 0])


def lens_area(z, k):
    """The overlap of two discs of radii 1 and k at the distance z, in 50-digit arithmetic."""
    import mpmath

    mpmath.mp.dps = 50
    out = []
    for value in z:
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


def integral(z, k, u1, u2):
    """The flux by integrating the intensity over the planet's disc inside the star, in polar coordinates about the planet's centre. A ray from the planet's centre at the angle phi is inside the star between
    rho = -z cos(phi) -+ sqrt(1 - z^2 sin^2(phi)) (when that is real). The radial limits are the overlap of that interval with [0, k]. The integrand is smooth between the angles where a limit changes."""
    from scipy import integrate

    def limits(angle):
        discriminant = 1.0 - (z * np.sin(angle)) ** 2
        if discriminant <= 0.0:
            return 0.0, 0.0
        root = np.sqrt(discriminant)
        low = max(0.0, -z * np.cos(angle) - root)
        high = min(k, -z * np.cos(angle) + root)
        return (low, high) if high > low else (0.0, 0.0)

    def inner(rho, angle):
        x, y = z + rho * np.cos(angle), rho * np.sin(angle)
        mu = np.sqrt(max(1.0 - x * x - y * y, 0.0))
        return (1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2) * rho

    def over_angle(angle):
        low, high = limits(angle)
        if high <= low:
            return 0.0
        return integrate.quad(inner, low, high, args=(angle,), epsabs=1e-13, epsrel=1e-12, limit=200)[0]

    # the angles where a radial limit changes character: where the ray is tangent to the star (z sin = 1) and where the circle of radius k meets the star's limb
    breaks = [0.0, np.pi]
    if z > 1.0:
        breaks.append(np.arcsin(1.0 / z))
    if abs(1 - k) < z < 1 + k:
        breaks.append(np.arccos(np.clip((1 - z * z - k * k) / (2 * z * k), -1, 1)))  # where rho = k meets the limb, measured from the direction away from the star
        breaks.append(np.pi - breaks[-1])
    points = sorted({float(angle) for angle in breaks if 0.0 <= angle <= np.pi})
    value = sum(integrate.quad(over_angle, left, right, epsabs=1e-13, epsrel=1e-12, limit=200)[0] for left, right in zip(points[:-1], points[1:]))
    return 1.0 - 2 * value / (np.pi * (1 - u1 / 3 - u2 / 6))


def lens():
    """The largest difference of the closed form from the lens-area flux at u1 = u2 = 0 for each radius ratio, with that of the quadrature of 48 nodes and the separations where they occur."""
    rows = {}
    for k in K:
        z = separations(k)
        reference = lens_area(z, k) / np.pi
        difference = np.abs((1 - quadratic_flux(z, k, 0.0, 0.0)) - reference)
        by_quadrature = np.abs((1 - quadrature_flux(z, k, 0.0, 0.0)) - reference)
        rows[str(k)] = {"points": int(z.size), "largest_difference": float(difference.max()), "at_separation": float(z[difference.argmax()]),
                        "quadrature_48_nodes_largest_difference": float(by_quadrature.max()), "quadrature_at_separation": float(z[by_quadrature.argmax()])}
    return rows


def integral_reference():
    """The largest difference of the closed form from the numerical integral of the intensity over the planet's disc, for each radius ratio, on the first six separations and every fourth of the others, for three limb-darkening pairs of U."""
    rows = {}
    for k in K:
        z = separations(k)
        z = np.concatenate([z[:6], z[6:][::4]])
        worst, where = 0.0, None
        for u1, u2 in U[1:4]:
            mine = quadratic_flux(z, k, u1, u2)
            for value, ours in zip(z, mine):
                if value >= 1 + k:
                    continue
                difference = abs(ours - integral(float(value), k, u1, u2))
                if difference > worst:
                    worst, where = difference, [float(value), u1, u2]
        rows[str(k)] = {"largest_difference": worst, "at_separation_u1_u2": where}
    return rows


def convergence():
    """The closed form against the quadrature of 768 nodes on the whole grid, and the quadrature of fewer nodes against the 768."""
    rows = {}
    for k in K:
        z = separations(k)
        worst, where = 0.0, None
        for u1, u2 in U:
            difference = np.abs(quadratic_flux(z, k, u1, u2) - quadrature_flux(z, k, u1, u2, nodes=768))
            if difference.max() > worst:
                worst, where = float(difference.max()), [float(z[difference.argmax()]), u1, u2]
        rows[str(k)] = {"largest_difference": worst, "at_separation_u1_u2": where, "points": int(z.size) * len(U)}
        for n in (24, 48):
            rows[str(k)][f"quadrature_{n}_nodes_against_768"] = max(float(np.max(np.abs(quadrature_flux(z, k, u1, u2, nodes=n) - quadrature_flux(z, k, u1, u2, nodes=768)))) for u1, u2 in U)
    return rows


def kepler_reference(mean, ecc):
    """The eccentric anomaly E with E - ecc sin E = mean (modulo 2 pi), by bisection in 50-digit arithmetic."""
    import mpmath

    mpmath.mp.dps = 50
    low, high = mpmath.mpf(0), mpmath.mpf(2) * mpmath.pi
    mean = mpmath.mpf(float(mean)) % (2 * mpmath.pi)
    ecc = mpmath.mpf(float(ecc))
    for _ in range(200):
        middle = (low + high) / 2
        if middle - ecc * mpmath.sin(middle) < mean:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def orbit():
    """The largest difference of the sky separation of `separation` from a 50-digit evaluation of the same projection, by band of eccentricity, over 300 seeded orbits; also the number of times that the front flag disagrees with the sign of the exact line-of-sight component."""
    import mpmath

    rng = np.random.default_rng(1)
    worst = {}
    for _ in range(300):
        period, a, inc = rng.uniform(0.5, 30), rng.uniform(2, 40), np.arccos(rng.uniform(0, 1.1) / 40)
        ecc, omega = rng.uniform(0, 0.9), rng.uniform(0, 2 * np.pi)
        t0 = 0.0
        time = rng.uniform(-0.5, 0.5, 8) * period
        z, front = separation(time, period, a, inc, ecc, omega, t0)
        start = np.pi / 2 - omega
        e0 = 2 * mpmath.atan2(mpmath.sqrt(1 - ecc) * mpmath.sin(start / 2), mpmath.sqrt(1 + ecc) * mpmath.cos(start / 2))
        for moment, ours, ahead in zip(time, z, front):
            mean = e0 - ecc * mpmath.sin(e0) + 2 * mpmath.pi * moment / period
            low, high = mpmath.mpf(-100), mpmath.mpf(100)
            target = mean
            for _ in range(300):
                middle = (low + high) / 2
                if middle - ecc * mpmath.sin(middle) < target:
                    low = middle
                else:
                    high = middle
            eccentric = (low + high) / 2
            true = 2 * mpmath.atan2(mpmath.sqrt(1 + ecc) * mpmath.sin(eccentric / 2), mpmath.sqrt(1 - ecc) * mpmath.cos(eccentric / 2))
            distance = a * (1 - ecc**2) / (1 + ecc * mpmath.cos(true))
            along = mpmath.sin(omega + true)
            exact = distance * mpmath.sqrt(1 - (along * mpmath.sin(inc)) ** 2)
            band = f"e_{int(ecc * 10) / 10:.1f}"
            worst[band] = max(worst.get(band, 0.0), float(abs(exact - ours)))
            if (float(along) > 0) != bool(ahead) and abs(float(along)) > 1e-9:
                worst["front_flag_disagreements"] = worst.get("front_flag_disagreements", 0) + 1
    return worst


def main():
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    wanted = sys.argv[1]
    for name, function in (("lens_area", lens), ("two_dimensional_integral", integral_reference), ("convergence", convergence), ("orbit_50_digit", orbit)):
        if wanted in ("all", {"lens_area": "lens", "two_dimensional_integral": "integral", "convergence": "convergence", "orbit_50_digit": "orbit"}[name]):
            results[name] = function()
            print(name, json.dumps(results[name])[:400], flush=True)
    RESULTS.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
