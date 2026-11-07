"""Quadratic limb-darkened transit flux in closed form, written from Agol, Luger and Foreman-Mackey (2020, AJ 159, 123, arXiv:1908.03222), sections III to V, and Bulirsch (1969, Numerische Mathematik 13, 305) for the general complete elliptic integral.

With the intensity I(mu) = 1 - u1 (1 - mu) - u2 (1 - mu)^2 = (1 - u1 - u2) + (u1 + 2 u2) mu - u2 mu^2, the flux visible with the planet at separation b and radius ratio r is the sum of the visible fluxes of the three terms:
the uniform term s0 = pi - A_lens (their eq. 34, with the overlap area of eq. 33 written with a Kahan-stable kite area), the term in mu, s1 = (2 pi / 3)(1 - 3 Lambda / 2 - Theta(r - b)) (eq. 36, with Lambda in eq. 37 and its stable form with cel in eq. 47), and the term in mu^2,
s0 + pi eta - pi / 2, with eta from their eq. 53 (the quadratic basis function 4 mu^2 - 2 has the visible flux s2 = 2 s0 + 4 pi eta - 2 pi, their eq. 54). The unocculted flux is pi (1 - u1 / 3 - u2 / 6).
"""

import numpy as np

CEL_ITERATIONS = 24  # the most Landen steps cel takes; each step squares the error
CEL_TOLERANCE = np.sqrt(np.finfo(float).eps)


def cel(kc, p, a, b):
    """Bulirsch's general complete elliptic integral, the integral from 0 to pi/2 of (a cos^2 + b sin^2) / ((cos^2 + p sin^2) sqrt(cos^2 + kc^2 sin^2)) (Agol et al. 2020, eq. 40), for kc != 0 and p > 0 by the
    Landen iteration of Bulirsch (1969); the arrays broadcast. Each element stops at its own convergence."""
    kc, p, a, b = np.broadcast_arrays(*(np.asarray(value, dtype=float) for value in (kc, p, a, b)))
    shape = kc.shape
    qc = np.abs(kc).reshape(-1).copy()
    p = np.sqrt(p.reshape(-1))
    a = a.reshape(-1).copy()
    b = b.reshape(-1) / p
    e = qc.copy()
    em = np.ones_like(qc)
    out = np.empty(qc.size)
    index = np.arange(qc.size)
    for _ in range(CEL_ITERATIONS):
        f = a
        a = a + b / p
        g = e / p
        b = 2 * (b + f * g)
        p = g + p
        g_old = em
        em = qc + em
        finished = np.abs(g_old - qc) <= g_old * CEL_TOLERANCE
        if finished.any():
            out[index[finished]] = np.pi / 2 * (b[finished] + a[finished] * em[finished]) / (em[finished] * (em[finished] + p[finished]))
            keep = ~finished
            if not keep.any():
                return out.reshape(shape)
            index, qc, e, em, a, b, p = index[keep], qc[keep], e[keep], em[keep], a[keep], b[keep], p[keep]
        qc = 2 * np.sqrt(e)
        e = qc * em
    out[index] = np.pi / 2 * (b + a * em) / (em * (em + p))
    return out.reshape(shape)


def _kite(r, b):
    """Area of the kite of the two circles' centres and their intersections (Agol et al. 2020, eq. 31), by Kahan's ordering of the sides 1, r, b."""
    sides = np.sort(np.stack(np.broadcast_arrays(np.ones_like(r), r, b)), axis=0)
    c, mid, a = sides[0], sides[1], sides[2]
    return 0.5 * np.sqrt(np.maximum((a + (mid + c)) * (c - (a - mid)) * (c + (a - mid)) * (a + (mid - c)), 0.0))


def _excess(r, b):
    """b + r - 1 without the rounding of b + r. The larger of the two minus 1 is exact (Sterbenz), and adding the smaller rounds once, at the size of the result."""
    return (np.maximum(r, b) - 1) + np.minimum(r, b)


def _lambda(r, b):
    """Lambda(r, b) of Agol et al. 2020, eq. 47 (eq. 37 in K, E and Pi), in the order of cases the paper gives. r and b are arrays of one shape; only the cases the inputs fall in are evaluated."""
    pi = np.pi
    out = np.zeros(r.shape)
    done = (r == 0) | (np.abs(r - b) >= 1)

    def take(mask, build):
        nonlocal done
        mask = mask & ~done
        if mask.any():
            out[mask] = build(r[mask], b[mask])
        done = done | mask

    take(b == 0, lambda r, b: -2.0 / 3.0 * (1 - r**2) ** 1.5)
    take((b == r) & (r == 0.5), lambda r, b: np.full(r.shape, 1.0 / 3.0 - 4.0 / (9 * pi)))

    def equal_small(r, b):
        mk = 4 * r**2
        return 1.0 / 3.0 + 2.0 / (9 * pi) * cel(np.sqrt(1 - mk), 1.0, mk - 3, (1 - mk) * (2 * mk - 3))

    def equal_large(r, b):
        mk = 1.0 / (4 * r**2)
        return 1.0 / 3.0 + 4 * r / (9 * pi) * cel(np.sqrt(1 - mk), 1.0, 1 - 3 * mk, mk - 1)

    take((b == r) & (r < 0.5), equal_small)
    take((b == r) & (r > 0.5), equal_large)
    take(b + r == 1, lambda r, b: 2.0 / (9 * pi) * (3 * np.arccos(1 - 2 * r) - 2 * (3 + 2 * r - 8 * r**2) * np.sqrt(r * b) - 3 * pi * (r > 0.5)))

    def ingress(r, b):  # k^2 < 1: b + r > 1
        kc2 = _excess(r, b) * (b + r + 1) / (4 * b * r)  # 1 - m_k, computed directly because 1 - (1 - kc2) loses it as b + r approaches 1
        kc = np.sqrt(kc2)
        return (1 - (b - r) ** 2) / (9 * pi * np.sqrt(b * r)) * (
            kc2 * ((b - r) * (b + r)) * cel(kc, (b - r) ** 2 * kc2, 0.0, 3.0)
            - (3 - 6 * r**2 - 2 * b * r) * cel(kc, 1.0, 1.0, 0.0)
            - 4 * b * r * cel(kc, 1.0, 1.0, kc2)
        )

    def inside(r, b):  # k^2 > 1: b + r < 1
        deficit = -_excess(r, b) * (b + r + 1)  # 1 - (b + r)^2
        kc = np.sqrt(deficit / (1 - (b - r) ** 2))
        q = 3 * (b - r) / ((b + r) * (1 - (b - r) ** 2))
        p = ((b - r) / (b + r)) ** 2 * deficit / (1 - (b - r) ** 2)
        return 2 * np.sqrt(1 - (b - r) ** 2) / (9 * pi) * (deficit * cel(kc, p, 1 + q, p + q) - (4 - 7 * r**2 - b**2) * cel(kc, 1.0, 1.0, kc**2))

    take(b + r > 1, ingress)
    take(b + r < 1, inside)
    return out


def quadratic_flux(z, k, u1, u2):
    """The relative flux of the star, 1 when the planet is off its disc and 0 when it covers the star, for radius ratio k at separation z. The arrays broadcast."""
    b, r, u1, u2 = np.broadcast_arrays(np.abs(np.asarray(z, dtype=float)), *(np.asarray(value, dtype=float) for value in (k, u1, u2)))
    b, r = b.copy(), r.copy()
    pi = np.pi
    total = pi * (1 - u1 / 3 - u2 / 6)
    flux = np.ones(b.shape)
    covered = (r >= 1) & (b <= r - 1)
    flux[covered] = 0.0
    live = (b < 1 + r) & ~covered
    if not live.any():
        return flux
    bl, rl, u1l, u2l = b[live], r[live], u1[live], u2[live]
    partial = (np.abs(1 - rl) < bl) & (bl < 1 + rl)
    s0 = np.empty(bl.shape)
    eta = np.empty(bl.shape)
    kite = _kite(np.where(partial, rl, 0.5), np.where(partial, bl, 0.5))
    kappa0 = np.arctan2(2 * kite, (rl - 1) * (rl + 1) + bl**2)
    kappa1 = np.arctan2(2 * kite, (1 - rl) * (1 + rl) + bl**2)
    s0 = np.where(partial, pi - kappa1 - rl**2 * kappa0 + kite, pi * (1 - rl**2))
    eta = np.where(partial, (kappa1 + rl**2 * (rl**2 + 2 * bl**2) * kappa0 - 0.5 * (1 + 5 * rl**2 + bl**2) * kite) / (2 * pi), rl**2 * (rl**2 + 2 * bl**2) / 2)
    lam = _lambda(rl, bl)
    s1 = 2 * pi / 3 * (1 - 1.5 * lam - (rl > bl))
    second = s0 + pi * eta - pi / 2
    flux[live] = ((1 - u1l - u2l) * s0 + (u1l + 2 * u2l) * s1 - u2l * second) / (pi * (1 - u1l / 3 - u2l / 6))
    return flux
