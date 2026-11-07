"""velociraptor.analytic in torch: the closed-form flux of Agol, Luger and Foreman-Mackey (2020, AJ 159, 123, arXiv:1908.03222) with Bulirsch's cel in a fixed number of Landen steps, evaluated for every element with masks and no branch on the data.
Every case is computed on safe inputs for the elements outside it and selected with torch.where, in the order of the cases of velociraptor.analytic."""

import math

import torch

CEL_STEPS = 8  # an element that has converged keeps its values
CEL_TOLERANCE = math.sqrt(2.220446049250313e-16)
PI = math.pi


def cel_torch(kc, p, a, b):
    """Bulirsch's cel (velociraptor.analytic.cel) in CEL_STEPS Landen steps with a done flag."""
    kc, p, a, b = torch.broadcast_tensors(*(torch.as_tensor(v, dtype=torch.float64) for v in (kc, p, a, b)))
    qc = kc.abs()
    e = qc
    em = torch.ones_like(qc)
    p = torch.sqrt(p)
    b = b / p
    done = torch.zeros(qc.shape, dtype=torch.bool, device=qc.device)
    for _ in range(CEL_STEPS):
        f = a
        a_new = a + b / p
        g = e / p
        b_new = 2 * (b + f * g)
        p_new = g + p
        em_new = qc + em
        step = (em - qc).abs() <= em * CEL_TOLERANCE
        a, b, p, em_next = torch.where(done, a, a_new), torch.where(done, b, b_new), torch.where(done, p, p_new), torch.where(done, em, em_new)
        done = done | step
        qc_new = 2 * torch.sqrt(e)
        e_new = qc_new * em_next
        qc, e, em = torch.where(done, qc, qc_new), torch.where(done, e, e_new), em_next
    return PI / 2 * (b + a * em) / (em * (em + p))


def _kite(r, b):
    sides = torch.sort(torch.stack(torch.broadcast_tensors(torch.ones_like(r), r, b)), dim=0).values
    c, mid, a = sides[0], sides[1], sides[2]
    return 0.5 * torch.sqrt(torch.clamp((a + (mid + c)) * (c - (a - mid)) * (c + (a - mid)) * (a + (mid - c)), min=0.0))


def _excess(r, b):
    """b + r - 1 without the rounding of b + r (velociraptor.analytic._excess)."""
    return (torch.maximum(r, b) - 1) + torch.minimum(r, b)


def _lambda(r, b):
    ingress_mask, inside_mask = b + r > 1, b + r < 1
    # ingress, k^2 < 1, on safe inputs where the element is not in it
    ri, bi = torch.where(ingress_mask, r, torch.full_like(r, 0.7)), torch.where(ingress_mask, b, torch.full_like(b, 0.8))
    kc2 = _excess(ri, bi) * (bi + ri + 1) / (4 * bi * ri)
    kc = torch.sqrt(kc2)
    ingress = (1 - (bi - ri) ** 2) / (9 * PI * torch.sqrt(bi * ri)) * (
        kc2 * ((bi - ri) * (bi + ri)) * cel_torch(kc, (bi - ri) ** 2 * kc2, torch.zeros_like(kc), torch.full_like(kc, 3.0))
        - (3 - 6 * ri**2 - 2 * bi * ri) * cel_torch(kc, torch.ones_like(kc), torch.ones_like(kc), torch.zeros_like(kc))
        - 4 * bi * ri * cel_torch(kc, torch.ones_like(kc), torch.ones_like(kc), kc2)
    )
    # inside, k^2 > 1
    rn, bn = torch.where(inside_mask, r, torch.full_like(r, 0.2)), torch.where(inside_mask, b, torch.full_like(b, 0.3))
    deficit = -_excess(rn, bn) * (bn + rn + 1)
    kcn = torch.sqrt(deficit / (1 - (bn - rn) ** 2))
    q = 3 * (bn - rn) / ((bn + rn) * (1 - (bn - rn) ** 2))
    pn = ((bn - rn) / (bn + rn)) ** 2 * deficit / (1 - (bn - rn) ** 2)
    ones = torch.ones_like(kcn)
    inside = 2 * torch.sqrt(1 - (bn - rn) ** 2) / (9 * PI) * (deficit * cel_torch(kcn, pn, 1 + q, pn + q) - (4 - 7 * rn**2 - bn**2) * cel_torch(kcn, ones, ones, kcn**2))
    lam = torch.where(inside_mask, inside, torch.zeros_like(r))
    lam = torch.where(ingress_mask, ingress, lam)
    # b + r == 1
    touch = 2.0 / (9 * PI) * (3 * torch.acos(torch.clamp(1 - 2 * r, -1.0, 1.0)) - 2 * (3 + 2 * r - 8 * r**2) * torch.sqrt(r * b) - 3 * PI * (r > 0.5).to(torch.float64))
    lam = torch.where(b + r == 1, touch, lam)
    equal = b == r
    r_small = torch.where(equal & (r < 0.5), r, torch.full_like(r, 0.25))
    mk = 4 * r_small**2
    small = 1.0 / 3.0 + 2.0 / (9 * PI) * cel_torch(torch.sqrt(1 - mk), torch.ones_like(mk), mk - 3, (1 - mk) * (2 * mk - 3))
    r_large = torch.where(equal & (r > 0.5), r, torch.full_like(r, 1.0))
    mk = 1.0 / (4 * r_large**2)
    large = 1.0 / 3.0 + 4 * r_large / (9 * PI) * cel_torch(torch.sqrt(1 - mk), torch.ones_like(mk), 1 - 3 * mk, mk - 1)
    lam = torch.where(equal & (r > 0.5), large, lam)
    lam = torch.where(equal & (r < 0.5), small, lam)
    lam = torch.where(equal & (r == 0.5), torch.full_like(r, 1.0 / 3.0 - 4.0 / (9 * PI)), lam)
    lam = torch.where(b == 0, -2.0 / 3.0 * torch.clamp(1 - r**2, min=0.0) ** 1.5, lam)
    return torch.where((r == 0) | ((r - b).abs() >= 1), torch.zeros_like(r), lam)


def quadratic_flux_analytic_torch(z, k, u1, u2):
    """velociraptor.analytic.quadratic_flux on tensors: the relative flux, 1 off the disc and 0 where the planet covers the star; the arguments broadcast."""
    z = torch.as_tensor(z, dtype=torch.float64)
    b, r, u1, u2 = torch.broadcast_tensors(z.abs(), *(torch.as_tensor(v, dtype=torch.float64, device=z.device) for v in (k, u1, u2)))
    total = PI * (1 - u1 / 3 - u2 / 6)
    covered = (r >= 1) & (b <= r - 1)
    live = (b < 1 + r) & ~covered
    partial = (torch.abs(1 - r) < b) & (b < 1 + r)
    kite = _kite(torch.where(partial, r, torch.full_like(r, 0.5)), torch.where(partial, b, torch.full_like(b, 0.5)))
    kappa0 = torch.atan2(2 * kite, (r - 1) * (r + 1) + b**2)
    kappa1 = torch.atan2(2 * kite, (1 - r) * (1 + r) + b**2)
    s0 = torch.where(partial, PI - kappa1 - r**2 * kappa0 + kite, PI * (1 - r**2))
    eta = torch.where(partial, (kappa1 + r**2 * (r**2 + 2 * b**2) * kappa0 - 0.5 * (1 + 5 * r**2 + b**2) * kite) / (2 * PI), r**2 * (r**2 + 2 * b**2) / 2)
    lam = _lambda(torch.where(live, r, torch.full_like(r, 0.2)), torch.where(live, b, torch.full_like(b, 0.3)))
    s1 = 2 * PI / 3 * (1 - 1.5 * lam - (r > b).to(torch.float64))
    second = s0 + PI * eta - PI / 2
    flux = ((1 - u1 - u2) * s0 + (u1 + 2 * u2) * s1 - u2 * second) / total
    flux = torch.where(live, flux, torch.ones_like(flux))
    return torch.where(covered, torch.zeros_like(flux), flux)


def transit_flux_analytic_torch(time, k, u1, u2, period, a_rs, inc, ecc=0.0, omega=PI / 2, t0=0.0, exposure=0.0, samples=1, chunk=1 << 17):
    """velociraptor.transit_flux in torch with the closed-form flux. Points whose exposure cannot reach the disc are dropped before the sub-samples are made (the bound of velociraptor.transit_flux), and the others are processed `chunk` sub-sample
    elements at a time. The arguments broadcast with time as the last axis. The result has the broadcast shape and lives on the device of `time`."""
    from velociraptor.torch_model import _pace, separation_torch

    time = torch.as_tensor(time, dtype=torch.float64)
    values = torch.broadcast_tensors(time, *(torch.as_tensor(v, dtype=torch.float64, device=time.device) for v in (k, u1, u2, period, a_rs, inc, ecc, omega, t0, exposure)))
    shape = values[0].shape
    time, k, u1, u2, period, a_rs, inc, ecc, omega, t0, exposure = (v.reshape(-1) for v in values)
    centre, _ = separation_torch(time, period, a_rs, inc, ecc, omega, t0)
    if samples > 1:
        reach = 2 * PI * a_rs * (1 + ecc) / (period * torch.sqrt(1 - ecc**2)) * exposure / 2
        near = centre - reach <= 1.0 + k
    else:
        near = centre < 1.0 + k
    flux = torch.ones_like(time)
    index = torch.nonzero(near).reshape(-1)
    offsets = ((torch.arange(samples, dtype=torch.float64, device=time.device) + 0.5) / samples - 0.5) if samples > 1 else torch.zeros(1, dtype=torch.float64, device=time.device)
    rows = max(1, chunk // len(offsets))
    for number, start in enumerate(range(0, index.numel(), rows)):
        _pace(time.device, number)
        part = index[start : start + rows]
        select = lambda value: value[part][:, None]
        z, in_front = separation_torch(select(time) + offsets * select(exposure), select(period), select(a_rs), select(inc), select(ecc), select(omega), select(t0))
        kk, uu1, uu2 = (select(value).expand_as(z) for value in (k, u1, u2))
        overlap = in_front & (z < 1.0 + kk)
        sub = torch.ones_like(z)
        sub[overlap] = quadratic_flux_analytic_torch(z[overlap], kk[overlap], uu1[overlap], uu2[overlap])
        flux[part] = sub.mean(dim=-1)
    return flux.reshape(shape)
