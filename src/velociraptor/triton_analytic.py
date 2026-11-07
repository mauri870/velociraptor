"""The closed-form transit flux of velociraptor.analytic (Agol, Luger and Foreman-Mackey 2020, AJ 159, 123; Bulirsch 1969) in Triton kernels, for ROCm and CUDA devices.

Each lane handles one (row, point) pair: Kepler's equation and the sky separation for each exposure sub-sample, then the stellar flux. The elliptic integrals are Bulirsch's cel in at most CEL_STEPS Landen steps.
Where the planet is small and wholly inside the disc the flux is the series of velociraptor.series and no elliptic integral is evaluated. Kernels are launched in slices (velociraptor.triton_model.plan_slices). The results agree with velociraptor.analytic to rounding."""

import math

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from velociraptor._series_triton import _series_high, _series_tier
from velociraptor.quadratic import KEPLER_ITERATIONS, gauss_legendre
from velociraptor.series import ABSOLUTE_FLUX_ERROR
from velociraptor.triton_model import plan_slices

BLOCK = 32
PHASE_SKIP = True  # the first pass skips blocks of pairs too far in phase from conjunction to come near the star (their flux is 1); the output is unchanged
SINGLE_CLOSED = True  # with precision="mixed" the closed form of the second pass runs in float32 from the float64 sky separation; False keeps it in float64
PRECISION = "mixed"  # the default of transit_flux_triton_analytic(precision=None): "mixed" puts the exposure nodes of the first pass in float32 (velociraptor.triton_mixed); "float64" keeps every step in float64
BINNED = True  # the second pass runs on a list of nodes sorted by regime (velociraptor.triton_binned); False runs it inside the main kernel
BINNED_MIN_SAMPLES = 8  # with fewer exposure nodes the second pass stays inside the main kernel, because the extra launches cost more than the closed form saves
FLAGGED_CHUNK = 1 << 21  # flagged pairs in one chunk of the second pass; it bounds the device memory of the pass
CLOSED_ELEMENTS = 1 << 20  # closed-form nodes in one launch of the sorted list
LAUNCH_OPTIONS = {}  # extra options of the Triton launch of the main kernel (for example waves_per_eu on AMD), for scans of launch shapes
WARPS = 1  # one wave for the 32 lanes of a block; with 4 warps three quarters of the threads of every block repeat the work of the others
COMPACT = True  # with 8 exposure nodes or more the first pass has two stages: the first finds the pairs whose nodes need evaluating, the second runs the node loop over that list so that its blocks are full; the flux differs from the single stage in the last bits, because a block's path for the eccentric anomaly depends on the pairs it holds
INTERPOLATE = True  # with BINNED_MIN_SAMPLES nodes or more, the exposure nodes come from Kepler solutions at six points of the exposure where they agree with the equation at its two ends (INTERPOLATION_TOLERANCE in b), and from an exact solve at each node elsewhere
INTERPOLATION_TOLERANCE = tl.constexpr(1e-10)  # the largest difference of the separation b between the polynomial and Kepler's equation at the ends of the exposure
CEL_STEPS = 8
TOLERANCE = math.sqrt(2.220446049250313e-16)


@triton.jit
def _cel_pair(kc, p, a, b1, b2, STEPS: tl.constexpr, TOL: tl.constexpr, PI):
    """Bulirsch's cel for two values of b with the same kc, p and a: (integral for b1, integral for b2). The recurrences of qc, e, em and p do not depend on a and b, so the two integrals share them. The loop ends when every lane of the block has converged."""
    qc = tl.abs(kc)
    e = qc
    em = tl.zeros_like(qc) + 1.0
    p = tl.sqrt(p)
    b1 = b1 / p
    b2 = b2 / p
    a1 = a
    a2 = a
    done = tl.zeros_like(qc) > 1.0
    count = 0
    pending = tl.sum((~done).to(tl.int32))
    while (count < STEPS) & (pending > 0):
        g = e / p
        a1_new = a1 + b1 / p
        a2_new = a2 + b2 / p
        b1_new = 2 * (b1 + a1 * g)
        b2_new = 2 * (b2 + a2 * g)
        p_new = g + p
        em_new = qc + em
        step = tl.abs(em - qc) <= em * TOL
        a1 = tl.where(done, a1, a1_new)
        a2 = tl.where(done, a2, a2_new)
        b1 = tl.where(done, b1, b1_new)
        b2 = tl.where(done, b2, b2_new)
        p = tl.where(done, p, p_new)
        em = tl.where(done, em, em_new)
        done = done | step
        qc_new = 2 * tl.sqrt(e)
        e_new = qc_new * em
        qc = tl.where(done, qc, qc_new)
        e = tl.where(done, e, e_new)
        count += 1
        pending = tl.sum((~done).to(tl.int32))
    return PI / 2 * (b1 + a1 * em) / (em * (em + p)), PI / 2 * (b2 + a2 * em) / (em * (em + p))


@triton.jit
def _flux_closed_body(b, r, u1, u2, PI, STEPS: tl.constexpr, TOL: tl.constexpr, DTYPE: tl.constexpr):
    """The relative flux of the star for the separation b and radius ratio r (velociraptor.analytic.quadratic_flux), for lanes over the disc; other lanes give an arbitrary finite number. Every value has the type DTYPE."""
    one = tl.zeros_like(b) + 1.0
    partial = (tl.abs(1 - r) < b) & (b < 1 + r)
    rk = tl.where(partial, r, 0.5)
    bk = tl.where(partial, b, 0.5)
    low = tl.minimum(tl.minimum(one, rk), bk)
    high = tl.maximum(tl.maximum(one, rk), bk)
    mid = tl.maximum(tl.minimum(one, rk), tl.minimum(tl.maximum(one, rk), bk))
    kite = 0.5 * tl.sqrt(tl.maximum((high + (mid + low)) * (low - (high - mid)) * (low + (high - mid)) * (high + (mid - low)), 0.0))
    kappa0 = libdevice.atan2(2 * kite, (r - 1) * (r + 1) + b * b)
    kappa1 = libdevice.atan2(2 * kite, (1 - r) * (1 + r) + b * b)
    s0 = tl.where(partial, PI - kappa1 - r * r * kappa0 + kite, PI * (1 - r * r))
    eta = tl.where(partial, (kappa1 + r * r * (r * r + 2 * b * b) * kappa0 - 0.5 * (1 + 5 * r * r + b * b) * kite) / (2 * PI), r * r * (r * r + 2 * b * b) / 2)
    excess = (tl.maximum(r, b) - 1) + tl.minimum(r, b)
    ingress_mask = excess > 0
    inside_mask = excess < 0
    lam = tl.zeros_like(b)
    if tl.sum(ingress_mask.to(tl.int32)) > 0:
        ri = tl.where(ingress_mask, r, 0.7)
        bi = tl.where(ingress_mask, b, 0.8)
        kc2 = ((tl.maximum(ri, bi) - 1) + tl.minimum(ri, bi)) * (bi + ri + 1) / (4 * bi * ri)
        kc = tl.sqrt(kc2)
        zero = tl.zeros_like(b)
        first_value, _ = _cel_pair(kc, (bi - ri) * (bi - ri) * kc2, zero, zero + 3.0, zero + 3.0, STEPS, TOL, PI)
        second_value, third_value = _cel_pair(kc, one, one, zero, kc2, STEPS, TOL, PI)
        ingress = (1 - (bi - ri) * (bi - ri)) / (9 * PI * tl.sqrt(bi * ri)) * (kc2 * ((bi - ri) * (bi + ri)) * first_value - (3 - 6 * ri * ri - 2 * bi * ri) * second_value - 4 * bi * ri * third_value)
        lam = tl.where(ingress_mask, ingress, lam)
    if tl.sum(inside_mask.to(tl.int32)) > 0:
        rn = tl.where(inside_mask, r, 0.2)
        bn = tl.where(inside_mask, b, 0.3)
        deficit = -((tl.maximum(rn, bn) - 1) + tl.minimum(rn, bn)) * (bn + rn + 1)
        kcn = tl.sqrt(deficit / (1 - (bn - rn) * (bn - rn)))
        q = 3 * (bn - rn) / ((bn + rn) * (1 - (bn - rn) * (bn - rn)))
        pn = ((bn - rn) / (bn + rn)) * ((bn - rn) / (bn + rn)) * deficit / (1 - (bn - rn) * (bn - rn))
        zero = tl.zeros_like(b)
        value_a, _ = _cel_pair(kcn, pn, 1 + q, pn + q, pn + q, STEPS, TOL, PI)
        value_b, _ = _cel_pair(kcn, one, one, kcn * kcn, kcn * kcn, STEPS, TOL, PI)
        inside = 2 * tl.sqrt(1 - (bn - rn) * (bn - rn)) / (9 * PI) * (deficit * value_a - (4 - 7 * rn * rn - bn * bn) * value_b)
        lam = tl.where(inside_mask, inside, lam)
    special = (excess == 0) | (b == r) | (b == 0)
    if tl.sum(special.to(tl.int32)) > 0:
        touch = 2.0 / (9 * PI) * (3 * libdevice.acos(tl.minimum(tl.maximum(1 - 2 * r, -1.0), 1.0)) - 2 * (3 + 2 * r - 8 * r * r) * tl.sqrt(r * b) - 3 * PI * (r > 0.5).to(DTYPE))
        lam = tl.where(excess == 0, touch, lam)
        equal = b == r
        r_small = tl.where(equal & (r < 0.5), r, 0.25)
        mk = 4 * r_small * r_small
        small_value, _ = _cel_pair(tl.sqrt(1 - mk), one, mk - 3, (1 - mk) * (2 * mk - 3), (1 - mk) * (2 * mk - 3), STEPS, TOL, PI)
        small = 1.0 / 3.0 + 2.0 / (9 * PI) * small_value
        r_large = tl.where(equal & (r > 0.5), r, 1.0)
        mk2 = 1.0 / (4 * r_large * r_large)
        large_value, _ = _cel_pair(tl.sqrt(1 - mk2), one, 1 - 3 * mk2, mk2 - 1, mk2 - 1, STEPS, TOL, PI)
        large = 1.0 / 3.0 + 4 * r_large / (9 * PI) * large_value
        lam = tl.where(equal & (r > 0.5), large, lam)
        lam = tl.where(equal & (r < 0.5), small, lam)
        lam = tl.where(equal & (r == 0.5), 1.0 / 3.0 - 4.0 / (9 * PI), lam)
        lam = tl.where(b == 0, -2.0 / 3.0 * tl.sqrt(tl.maximum(1 - r * r, 0.0)) * tl.maximum(1 - r * r, 0.0), lam)
    lam = tl.where((r == 0) | (tl.abs(r - b) >= 1), 0.0, lam)
    s1 = 2 * PI / 3 * (1 - 1.5 * lam - (r > b).to(DTYPE))
    second = s0 + PI * eta - PI / 2
    return ((1 - u1 - u2) * s0 + (u1 + 2 * u2) * s1 - u2 * second) / (PI * (1 - u1 / 3 - u2 / 6))


@triton.jit
def _flux_closed(b, r, u1, u2, PI, STEPS: tl.constexpr, TOL: tl.constexpr, SINGLE: tl.constexpr):
    """_flux_closed_body in float64, or with SINGLE in float32 from float32 copies of its arguments. The flux is returned as float64."""
    if SINGLE:
        b32 = b.to(tl.float32)
        r32 = r.to(tl.float32)
        value = _flux_closed_body(b32, r32, u1.to(tl.float32), u2.to(tl.float32), PI.to(tl.float32), STEPS, 3.4526698e-4, tl.float32)  # the square root of the float32 epsilon; the float64 tolerance would never be met
        # in float32 a lane can round onto the far side of a contact, where the formulas of the partial overlap do not hold; there the flux is 1 off the disc and 0 when the planet covers the star
        value = tl.where(b32 >= 1 + r32, 1.0, value)
        value = tl.where((r32 > 1) & (b32 <= r32 - 1), 0.0, value)
        return value.to(tl.float64)
    else:
        return _flux_closed_body(b, r, u1, u2, PI, STEPS, TOL, tl.float64)


@triton.jit
def _sky(sine, cosine, ecc, root, a, cos_inc, sin_omega, cos_omega):
    """The sky-projected separation and the coordinate along the line of sight at the eccentric anomaly with the given sine and cosine. With x = a (cos E - e) and y = a sqrt(1 - e^2) sin E in the orbital plane, the planet is in front of the star where Y = x sin(omega) + y cos(omega) is positive, and the separation is sqrt(X^2 + (Y cos i)^2)."""
    x = a * (cosine - ecc)
    y = a * root * sine
    big_x = x * cos_omega - y * sin_omega
    big_y = x * sin_omega + y * cos_omega
    return tl.sqrt(big_x * big_x + big_y * big_y * cos_inc * cos_inc), big_y


@triton.jit
def _small_angle(d):
    """sin(d) and cos(d) - 1 for |d| of at most 0.1 by their series in Horner form. The factors are float64 constants multiplied in, because a division costs several multiplications on the device and a Python float in a kernel is float32."""
    d2 = d * d
    sine = d * (1 - d2 * tl.full(d.shape, 1.0 / 6.0, tl.float64) * (1 - d2 * tl.full(d.shape, 1.0 / 20.0, tl.float64) * (1 - d2 * tl.full(d.shape, 1.0 / 42.0, tl.float64) * (1 - d2 * tl.full(d.shape, 1.0 / 72.0, tl.float64) * (1 - d2 * tl.full(d.shape, 1.0 / 110.0, tl.float64) * (1 - d2 * tl.full(d.shape, 1.0 / 156.0, tl.float64)))))))
    cosine_minus_one = -d2 * tl.full(d.shape, 0.5, tl.float64) * (1 - d2 * tl.full(d.shape, 1.0 / 12.0, tl.float64) * (1 - d2 * tl.full(d.shape, 1.0 / 30.0, tl.float64) * (1 - d2 * tl.full(d.shape, 1.0 / 56.0, tl.float64) * (1 - d2 * tl.full(d.shape, 1.0 / 90.0, tl.float64) * (1 - d2 * tl.full(d.shape, 1.0 / 132.0, tl.float64))))))
    return sine, cosine_minus_one


@triton.jit
def _small_solve(shift, ecc, sine_centre, cosine_centre):
    """The sine and cosine of the eccentric anomaly `shift` of mean anomaly after the centre of the exposure, for a change of the eccentric anomaly of at most 0.1: d solves d - e (sin E_c (cos d - 1) + cos E_c sin d) = shift. Three Newton steps from the first-order start and the angle-addition formulas avoid the sine and cosine of a large argument."""
    d = shift / (1 - ecc * cosine_centre)
    for _ in range(3):
        sd, cm = _small_angle(d)
        d = d - (d - ecc * (sine_centre * cm + cosine_centre * sd) - shift) / (1 - ecc * (cosine_centre * (cm + 1) - sine_centre * sd))
    sd, cm = _small_angle(d)
    return sine_centre * (cm + 1) + cosine_centre * sd, cosine_centre * (cm + 1) - sine_centre * sd


@triton.jit
def _interpolation(table_ptr, exposure, period, ecc, sine_centre, cosine_centre, root, a, cos_inc, sin_omega, cos_omega, PI):
    """The sine and cosine of the eccentric anomaly at the six Chebyshev points of the exposure from Kepler's equation, and whether the polynomial through them agrees with the equation at the two ends of the exposure. Returns the six sines, the six cosines and the agreement: the largest difference of the separation b at the two ends is at most INTERPOLATION_TOLERANCE.

    The table holds the six anchor fractions, then the weights of the two ends and then, for each exposure node, the six weights of the Lagrange polynomial (interpolation_table)."""
    s0, c0 = _small_solve(2 * PI * (tl.load(table_ptr + 0) * exposure) / period, ecc, sine_centre, cosine_centre)
    s1, c1 = _small_solve(2 * PI * (tl.load(table_ptr + 1) * exposure) / period, ecc, sine_centre, cosine_centre)
    s2, c2 = _small_solve(2 * PI * (tl.load(table_ptr + 2) * exposure) / period, ecc, sine_centre, cosine_centre)
    s3, c3 = _small_solve(2 * PI * (tl.load(table_ptr + 3) * exposure) / period, ecc, sine_centre, cosine_centre)
    s4, c4 = _small_solve(2 * PI * (tl.load(table_ptr + 4) * exposure) / period, ecc, sine_centre, cosine_centre)
    s5, c5 = _small_solve(2 * PI * (tl.load(table_ptr + 5) * exposure) / period, ecc, sine_centre, cosine_centre)
    agree = tl.full(exposure.shape, 1, tl.int1)
    for end in tl.static_range(2):
        w0 = tl.load(table_ptr + 6 + 6 * end)
        w1 = tl.load(table_ptr + 7 + 6 * end)
        w2 = tl.load(table_ptr + 8 + 6 * end)
        w3 = tl.load(table_ptr + 9 + 6 * end)
        w4 = tl.load(table_ptr + 10 + 6 * end)
        w5 = tl.load(table_ptr + 11 + 6 * end)
        sine_end, cosine_end = _small_solve(2 * PI * ((end - 0.5) * exposure) / period, ecc, sine_centre, cosine_centre)
        b_exact, _ = _sky(sine_end, cosine_end, ecc, root, a, cos_inc, sin_omega, cos_omega)
        b_fit, _ = _sky(w0 * s0 + w1 * s1 + w2 * s2 + w3 * s3 + w4 * s4 + w5 * s5, w0 * c0 + w1 * c1 + w2 * c2 + w3 * c3 + w4 * c4 + w5 * c5, ecc, root, a, cos_inc, sin_omega, cos_omega)
        agree = agree & (tl.abs(b_fit - b_exact) <= INTERPOLATION_TOLERANCE)
    return s0, s1, s2, s3, s4, s5, c0, c1, c2, c3, c4, c5, agree


@triton.jit
def _series(b, k, u1, u2, normalisation):
    """velociraptor.series.series_flux at the order 3 (the lowest order of ORDERS), in float64."""
    mu = tl.sqrt(1 - b * b)
    m1 = 1.0 / mu
    m2 = m1 * m1
    m3 = m1 * m2
    m5 = m3 * m2
    m7 = m5 * m2
    m9 = m7 * m2
    m11 = m9 * m2
    l0 = (1 - u1 - u2) + (u1 + 2 * u2) * mu - u2 * mu * mu
    l1 = u1 * (-m1 - m3) + u2 * (4 - 2 * m1 - 2 * m3)
    l2 = u1 * (m3 + 6 * m5 - 15 * m7) + u2 * (2 * m3 + 12 * m5 - 30 * m7)
    l3 = u1 * (-9 * m5 - 135 * m7 + 945 * m9 - 945 * m11) + u2 * (-18 * m5 - 270 * m7 + 1890 * m9 - 1890 * m11)
    k2 = k * k
    mean = l0 + k2 * (l1 * tl.full(b.shape, 0.125, tl.float64) + k2 * (l2 * tl.full(b.shape, 1.0 / 192.0, tl.float64) + k2 * l3 * tl.full(b.shape, 1.0 / 9216.0, tl.float64)))
    return 1 - k2 * mean * normalisation


@triton.jit
def _sample_flux(b, along, k, u1, u2, normalisation, active, PI, STEPS: tl.constexpr, TOL: tl.constexpr, SERIES: tl.constexpr, BOUND: tl.constexpr, CLOSED: tl.constexpr, SINGLE: tl.constexpr, BLOCK: tl.constexpr):
    """The flux of one sub-sample and the lanes that need the closed form. The flux is 1 where the planet is behind the star or off its disc, 0 where it covers the star, the series where the planet is small and wholly inside the disc (velociraptor.series.uses_series), and the closed form elsewhere.
    With CLOSED false the closed form is not compiled in, the flux of those lanes is 1 and the second result marks them."""
    covered = (k >= 1) & (b <= k - 1)
    live = active & (along > 0) & (b < 1 + k) & ~covered
    flux = tl.zeros([BLOCK], dtype=tl.float64) + 1.0
    closed = live
    if SERIES:
        small_low = live & ~live
        small_high = live & ~live
        if tl.sum(live.to(tl.int32)) > 0:  # a block with no lane in front of the disc takes no tier test
            small_low, small_high = _series_tier(b, k, u1, u2, normalisation, live, BOUND * 0.5)
        if tl.sum(small_low.to(tl.int32)) > 0:
            flux = tl.where(small_low, _series(tl.where(small_low, b, 0.3), tl.where(small_low, k, 0.01), u1, u2, normalisation), flux)
        if tl.sum(small_high.to(tl.int32)) > 0:
            flux = tl.where(small_high, _series_high(tl.where(small_high, b, 0.3), tl.where(small_high, k, 0.01), u1, u2, normalisation), flux)
        closed = live & ~small_low & ~small_high
    if CLOSED & (tl.sum(closed.to(tl.int32)) > 0):
        value = _flux_closed(tl.where(closed, b, 0.3), tl.where(closed, k, 0.2), u1, u2, PI, STEPS, TOL, SINGLE)
        flux = tl.where(closed, value, flux)
    return tl.where(active & covered & (along > 0), 0.0, flux), closed


@triton.jit
def _kernel(time_ptr, k_ptr, u1_ptr, u2_ptr, period_ptr, a_ptr, cos_inc_ptr, ecc_ptr, sin_omega_ptr, t0_ptr, exposure_ptr, cos_omega_ptr, mean0_ptr, root_ptr, norm_ptr, mlow_ptr, mhigh_ptr, asinc_ptr, out_ptr, index_ptr, flag_ptr, node_ptr, weight_ptr, table_ptr,
            offset_pairs, pairs, points, SAMPLES: tl.constexpr, KEPLER: tl.constexpr, STEPS: tl.constexpr, TOL: tl.constexpr, SERIES: tl.constexpr, BOUND: tl.constexpr, CLOSED: tl.constexpr, GAUSS: tl.constexpr, INTERP: tl.constexpr, INDEXED: tl.constexpr, FLAGS: tl.constexpr, PHASE_SKIP: tl.constexpr, SINGLE: tl.constexpr, NEAR_ONLY: tl.constexpr, BLOCK: tl.constexpr):
    lane = offset_pairs + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = lane < pairs
    if INDEXED:
        pair = tl.load(index_ptr + lane, mask=valid, other=0)
    else:
        pair = lane
    row = pair // points
    time = tl.load(time_ptr + pair, mask=valid, other=0.0)
    k = tl.load(k_ptr + row, mask=valid, other=0.0)
    u1 = tl.load(u1_ptr + row, mask=valid, other=0.0)
    u2 = tl.load(u2_ptr + row, mask=valid, other=0.0)
    period = tl.load(period_ptr + row, mask=valid, other=1.0)
    a = tl.load(a_ptr + row, mask=valid, other=1.0)
    cos_inc = tl.load(cos_inc_ptr + row, mask=valid, other=0.0)
    ecc = tl.load(ecc_ptr + row, mask=valid, other=0.0)
    sin_omega = tl.load(sin_omega_ptr + row, mask=valid, other=0.0)
    cos_omega = tl.load(cos_omega_ptr + row, mask=valid, other=1.0)
    t0 = tl.load(t0_ptr + row, mask=valid, other=0.0)
    exposure = tl.load(exposure_ptr + row, mask=valid, other=0.0)
    mean0 = tl.load(mean0_ptr + row, mask=valid, other=0.0)  # the terms of the draw, made once in row_terms()
    root = tl.load(root_ptr + row, mask=valid, other=1.0)
    normalisation = tl.load(norm_ptr + row, mask=valid, other=1.0)
    # a Python float in a kernel is a float32 constant: pi is built as float64
    PI = tl.full([BLOCK], 3.141592653589793, dtype=tl.float64)
    # Newton's iteration for Kepler's equation at the exposure's centre, from the usual start, ended when every lane has converged
    if PHASE_SKIP:
        # a pair is too far in phase from conjunction to come within 1 + k + reach of the star when the sky separation, at least a (1 - e) |cos(true anomaly + omega)|, stays above that. The true anomaly gains at least m = sqrt(1 - e) / (1 + e)^1.5 and at most M = (1 + e)^2 / (1 - e^2)^1.5 per radian of mean anomaly, so |anomaly from conjunction| lies in [m |delta|, M |delta|] for the phase delta, and outside the windows of both conjunctions when m |delta| > asin(c) and M |delta| < pi - asin(c), c = (1 + k + reach) / (a (1 - e)). Such a pair has the flux 1, as the exact test below finds, and a block of them takes no orbit.
        m_low = tl.load(mlow_ptr + row, mask=valid, other=0.0)
        m_high = tl.load(mhigh_ptr + row, mask=valid, other=1.0)
        asin_c = tl.load(asinc_ptr + row, mask=valid, other=10.0)
        phase = 2 * PI * (time - t0) / period
        delta = tl.abs(phase - 2 * PI * libdevice.rint(phase / (2 * PI)))
        far = (m_low * delta > asin_c + 1e-9) & (m_high * delta < PI - asin_c - 1e-9)
        live_lanes = valid & ~far
    else:
        live_lanes = valid
    result = tl.zeros([BLOCK], dtype=tl.float64) + 1.0
    refused = tl.zeros([BLOCK], dtype=tl.int32)
    if tl.sum(live_lanes.to(tl.int32)) > 0:
        mean = mean0 + 2 * PI * (time - t0) / period
        eccentric = mean + ecc * tl.sin(mean)
        count = 0
        pending = tl.sum(valid.to(tl.int32))
        while (count < KEPLER) & (pending > 0):
            step = (eccentric - ecc * tl.sin(eccentric) - mean) / (1 - ecc * tl.cos(eccentric))
            eccentric = eccentric - step
            pending = tl.sum((tl.abs(step) > 1e-15).to(tl.int32))
            count += 1
        sine_centre = tl.sin(eccentric)
        cosine_centre = tl.cos(eccentric)
        centre, along_centre = _sky(sine_centre, cosine_centre, ecc, root, a, cos_inc, sin_omega, cos_omega)
        if SAMPLES == 1:
            result, closed_lanes = _sample_flux(centre, along_centre, k, u1, u2, normalisation, valid, PI, STEPS, TOL, SERIES, BOUND, CLOSED, SINGLE, BLOCK)
            refused = closed_lanes.to(tl.int32)
        else:
            # no sub-sample can come within 1 + k of the star if the centre is farther than that by more than the planet moves on the sky in half the exposure (its speed is at most 2 pi a (1 + e) / (P sqrt(1 - e^2)))
            reach = 2 * PI * a * (1 + ecc) / (period * root) * exposure / 2
            near = valid & (centre - reach <= 1.0 + k)
            accumulated = tl.zeros([BLOCK], dtype=tl.float64) + 1.0  # the weights sum to 1, and a pair out of reach has the flux 1
            refused = tl.zeros([BLOCK], dtype=tl.int32)
            if NEAR_ONLY:  # first stage of the compacted first pass: the flag only marks pairs whose nodes need evaluating, and no lane takes the node loop
                refused = near.to(tl.int32)
                near = near & ~near
            if tl.sum(near.to(tl.int32)) > 0:
                accumulated = tl.zeros([BLOCK], dtype=tl.float64)
                swing = 2 * PI * (exposure / 2) / (period * (1 - ecc))  # bound on the change of the eccentric anomaly inside the exposure (dE/dM is at most 1 / (1 - e))
                small = tl.sum((near & (swing > 0.1)).to(tl.int32)) == 0
                if INTERP:
                    s0, s1, s2, s3, s4, s5, c0, c1, c2, c3, c4, c5, agree = _interpolation(table_ptr, exposure, period, ecc, sine_centre, cosine_centre, root, a, cos_inc, sin_omega, cos_omega, PI)
                    interpolate = small & (tl.sum((near & ~agree).to(tl.int32)) == 0)
                else:
                    s0 = sine_centre
                    s1 = sine_centre
                    s2 = sine_centre
                    s3 = sine_centre
                    s4 = sine_centre
                    s5 = sine_centre
                    c0 = cosine_centre
                    c1 = cosine_centre
                    c2 = cosine_centre
                    c3 = cosine_centre
                    c4 = cosine_centre
                    c5 = cosine_centre
                    interpolate = small & (tl.sum(near.to(tl.int32)) < 0)
                for sample in range(SAMPLES):
                    if GAUSS:
                        fraction = tl.load(node_ptr + sample)  # the node, in units of the exposure from its centre, and its weight
                        weight = tl.load(weight_ptr + sample)
                    else:
                        fraction = (sample.to(tl.float64) + 0.5) / SAMPLES - 0.5
                        weight = tl.full([BLOCK], 1.0 / SAMPLES, tl.float64)
                    shift = 2 * PI * (fraction * exposure) / period
                    if interpolate:
                        base = 18 + 6 * sample
                        sine_now = tl.load(table_ptr + base) * s0 + tl.load(table_ptr + base + 1) * s1 + tl.load(table_ptr + base + 2) * s2 + tl.load(table_ptr + base + 3) * s3 + tl.load(table_ptr + base + 4) * s4 + tl.load(table_ptr + base + 5) * s5
                        cosine_now = tl.load(table_ptr + base) * c0 + tl.load(table_ptr + base + 1) * c1 + tl.load(table_ptr + base + 2) * c2 + tl.load(table_ptr + base + 3) * c3 + tl.load(table_ptr + base + 4) * c4 + tl.load(table_ptr + base + 5) * c5
                    elif small:
                        sine_now, cosine_now = _small_solve(shift, ecc, sine_centre, cosine_centre)
                    else:
                        guess = eccentric + shift / (1 - ecc * cosine_centre)
                        for _ in range(3):
                            guess = guess - (guess - ecc * tl.sin(guess) - (mean + shift)) / (1 - ecc * tl.cos(guess))
                        sine_now = tl.sin(guess)
                        cosine_now = tl.cos(guess)
                    b, along = _sky(sine_now, cosine_now, ecc, root, a, cos_inc, sin_omega, cos_omega)
                    value, closed_lanes = _sample_flux(b, along, k, u1, u2, normalisation, near, PI, STEPS, TOL, SERIES, BOUND, CLOSED, SINGLE, BLOCK)
                    accumulated += weight * tl.where(near, value, 1.0)
                    refused = refused | (closed_lanes & near).to(tl.int32)
            result = accumulated
    tl.store(out_ptr + pair, result, mask=valid)
    if FLAGS:
        tl.store(flag_ptr + pair, (refused > 0).to(tl.int8), mask=valid)


TRACE = None  # a list to which each call appends the pairs, the flagged pairs and the seconds of its two passes; tracing synchronizes the device between the passes


def _tick(device):
    import time

    torch.cuda.synchronize(device)
    return time.perf_counter()


@triton.jit
def _per_draw(u1_ptr, u2_ptr, inc_ptr, ecc_ptr, omega_ptr, cos_inc_ptr, sin_omega_ptr, cos_omega_ptr, mean0_ptr, root_ptr, norm_ptr, rows, BLOCK: tl.constexpr):
    """The terms of a draw that the kernel needs for each of its points, computed once per draw."""
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = row < rows
    u1 = tl.load(u1_ptr + row, mask=valid, other=0.0)
    u2 = tl.load(u2_ptr + row, mask=valid, other=0.0)
    inc = tl.load(inc_ptr + row, mask=valid, other=0.0)
    ecc = tl.load(ecc_ptr + row, mask=valid, other=0.0)
    omega = tl.load(omega_ptr + row, mask=valid, other=0.0)
    PI = tl.full([BLOCK], 3.141592653589793, dtype=tl.float64)
    start = PI / 2 - omega
    anomaly0 = 2 * libdevice.atan2(tl.sqrt(1 - ecc) * tl.sin(start / 2), tl.sqrt(1 + ecc) * tl.cos(start / 2))
    tl.store(mean0_ptr + row, anomaly0 - ecc * tl.sin(anomaly0), mask=valid)
    tl.store(root_ptr + row, tl.sqrt(1 - ecc * ecc), mask=valid)
    tl.store(norm_ptr + row, 1.0 / (1 - u1 / 3 - u2 / 6), mask=valid)
    tl.store(cos_inc_ptr + row, tl.cos(inc), mask=valid)
    tl.store(sin_omega_ptr + row, tl.sin(omega), mask=valid)
    tl.store(cos_omega_ptr + row, tl.cos(omega), mask=valid)


def interpolation_table(fractions, device):
    """The float64 table `_interpolation` reads: the six Chebyshev points of the exposure (in units of the exposure from its centre), then the weights of the Lagrange polynomial through them at the two ends of the exposure and at each of `fractions`, the nodes."""
    anchors = 0.5 * np.cos((2 * np.arange(6) + 1) * np.pi / 12)

    def weights(at):
        out = np.ones((len(at), 6))
        for i in range(6):
            for j in range(6):
                if i != j:
                    out[:, i] *= (at - anchors[j]) / (anchors[i] - anchors[j])
        return out

    table = np.concatenate([anchors, weights(np.array([-0.5, 0.5])).reshape(-1), weights(np.asarray(fractions, dtype=np.float64)).reshape(-1)])
    return torch.as_tensor(table, dtype=torch.float64, device=device)


def row_terms(columns):
    """The arguments of the kernel from the ten columns of a draw (k, u1, u2, period, a, inc, ecc, omega, t0, exposure). The terms that depend on the draw alone (cos inc, sin and cos omega, the mean anomaly at the start of the transit,
    sqrt(1 - e^2) and the limb-darkening normalisation) are made once per draw in float64."""
    k, u1, u2, period, a, inc, ecc, omega, t0, exposure = columns
    rows = k.numel()
    made = [torch.empty_like(k) for _ in range(6)]
    block = 128
    _per_draw[(triton.cdiv(rows, block),)](u1, u2, inc, ecc, omega, *made, rows, BLOCK=block, num_warps=1)
    cos_inc, sin_omega, cos_omega, mean0, root, normalisation = made
    reach = 2 * math.pi * a * (1 + ecc) / (period * root) * exposure / 2
    c = (1 + k + reach) / (a * (1 - ecc))
    phase_low = torch.sqrt(1 - ecc) / (1 + ecc) ** 1.5
    phase_high = (1 + ecc) ** 2 / (1 - ecc * ecc) ** 1.5
    asin_c = torch.where(c < 1, torch.asin(torch.clamp(c, max=1.0)), torch.full_like(c, 10.0))  # 10 marks draws with c of 1 or more, whose pairs are never skipped
    return [k, u1, u2, period, a, cos_inc, ecc, sin_omega, t0, exposure, cos_omega, mean0, root, normalisation, phase_low, phase_high, asin_c]


def _second_pass_binned(flat, per_row, out, indices, nodes, weights, table, interpolate, points, samples, gauss, device, trace=None, single=False):
    """The flagged pairs through velociraptor.triton_binned in chunks of FLAGGED_CHUNK pairs, which bounds the device memory of the pass. The numbers equal those of a single chunk."""
    for start in range(0, indices.numel(), FLAGGED_CHUNK):
        part = {} if trace is not None else None
        _second_pass_chunk(flat, per_row, out, indices[start:start + FLAGGED_CHUNK], nodes, weights, table, interpolate, points, samples, gauss, device, part, single)
        if trace is not None:
            total = trace.setdefault("binned", {})
            for key, value in part["binned"].items():
                total[key] = total.get(key, 0) + value


def _second_pass_chunk(flat, per_row, out, indices, nodes, weights, table, interpolate, points, samples, gauss, device, trace=None, single=False):
    """The flagged pairs through velociraptor.triton_binned: the orbit and the nodes of each pair, the closed form on the nodes that need it sorted by regime and modulus, and the sum of the nodes in their order. The numbers equal the main kernel's."""
    from velociraptor.triton_binned import _accumulate, _closed_nodes, _prepare

    flagged = indices.numel()
    began = _tick(device) if trace is not None else 0.0
    code = torch.empty(flagged * samples, dtype=torch.int8, device=device)
    separation = torch.empty(flagged * samples, dtype=torch.float64, device=device)
    k, u1, u2, period, a, cos_inc, ecc, sin_omega, t0, exposure, cos_omega, mean0, root = per_row[:13]
    for number, (first, count) in enumerate(plan_slices(flagged, samples)):
        _prepare[(triton.cdiv(count, BLOCK),)](flat, k, period, a, cos_inc, ecc, sin_omega, t0, exposure, cos_omega, mean0, root, code, separation, indices, nodes, weights, table, first, first + count, points, SAMPLES=samples, KEPLER=KEPLER_ITERATIONS, GAUSS=gauss, INTERP=interpolate, BLOCK=BLOCK, num_warps=WARPS)
        if number % 8 == 7 and device.type != "cpu":
            torch.cuda.synchronize(device)
    prepared = _tick(device) if trace is not None else 0.0
    listed = torch.nonzero(code == 2).reshape(-1)
    flux = torch.empty(flagged * samples, dtype=torch.float64, device=device)
    if listed.numel():
        row = indices[listed // samples] // points
        kk, bb = k[row], separation[listed]
        excess = (torch.maximum(kk, bb) - 1) + torch.minimum(kk, bb)
        ingress = excess > 0
        listed = torch.cat([listed[ingress], listed[~ingress]])  # sorted by regime so that the lanes of a block take the same branch of the closed form
        sorted_at = _tick(device) if trace is not None else 0.0
        for number, first in enumerate(range(0, listed.numel(), CLOSED_ELEMENTS)):
            count = min(CLOSED_ELEMENTS, listed.numel() - first)
            _closed_nodes[(triton.cdiv(count, BLOCK),)](listed, code, separation, flux, k, u1, u2, indices, first, first + count, points, SAMPLES=samples, STEPS=CEL_STEPS, TOL=TOLERANCE, SINGLE=single, BLOCK=BLOCK, num_warps=WARPS)
            if number % 8 == 7 and device.type != "cpu":
                torch.cuda.synchronize(device)
    closed_at = _tick(device) if trace is not None else 0.0
    for number, (first, count) in enumerate(plan_slices(flagged, samples)):
        _accumulate[(triton.cdiv(count, BLOCK),)](code, flux, out, indices, nodes, weights, first, first + count, SAMPLES=samples, GAUSS=gauss, BLOCK=BLOCK, num_warps=WARPS)
        if number % 8 == 7 and device.type != "cpu":
            torch.cuda.synchronize(device)
    if trace is not None:
        finished = _tick(device)
        trace["binned"] = {"orbit_and_nodes_s": prepared - began, "list_and_partition_s": (sorted_at - prepared) if listed.numel() else 0.0, "closed_form_s": closed_at - (sorted_at if listed.numel() else prepared), "sum_s": finished - closed_at, "closed_form_nodes": int(listed.numel())}


def transit_flux_triton_analytic(time, k, u1, u2, period, a_rs, inc, ecc=0.0, omega=math.pi / 2, t0=0.0, exposure=0.0, samples=1, series=True, exposure_nodes=3, precision=None):
    """velociraptor.transit_flux on a device with the closed-form kernel, in float64. The arguments broadcast with time as the last axis, and the parameters of a row are taken from its first point.

    With `series` the first pass has no elliptic integrals. It gives the flux of every (row, point) pair whose sub-samples all take the series and flags the others (the points of ingress and egress, and the planets too large for the series). A second pass evaluates the flagged pairs with the closed form alone."""
    precision = PRECISION if precision is None else precision
    time = torch.as_tensor(time, dtype=torch.float64)
    device = time.device
    values = torch.broadcast_tensors(time, *(torch.as_tensor(value, dtype=torch.float64, device=device) for value in (k, u1, u2, period, a_rs, inc, ecc, omega, t0, exposure)))
    shape = values[0].shape
    points = shape[-1] if len(shape) else 1
    flat = values[0].reshape(-1).contiguous()
    per_row = row_terms([(value[..., 0] if value.dim() else value).reshape(-1).contiguous() for value in values[1:]])
    out = torch.empty_like(flat)
    nothing = torch.empty(1, dtype=torch.int64, device=device)
    nothing_float = torch.empty(1, dtype=torch.float64, device=device)
    flags = torch.zeros(flat.numel(), dtype=torch.int8, device=device) if series else torch.empty(1, dtype=torch.int8, device=device)
    samples = max(int(samples), 1)
    gauss = exposure_nodes is not None
    if gauss:  # Gauss-Legendre in the exposure window in place of the midpoints of `samples` equal parts
        x, w = gauss_legendre(int(exposure_nodes))
        nodes = torch.as_tensor(x / 2, dtype=torch.float64, device=device)
        weights = torch.as_tensor(w / 2, dtype=torch.float64, device=device)
        samples = int(exposure_nodes)
    else:
        nodes = weights = torch.empty(1, dtype=torch.float64, device=device)
    interpolate = bool(INTERPOLATE and samples >= BINNED_MIN_SAMPLES)
    if interpolate:
        fractions = nodes.cpu().numpy() if gauss else (np.arange(samples) + 0.5) / samples - 0.5
        table = interpolation_table(fractions, device)
    else:
        table = nothing_float

    def launch(total, indices, closed_form, use_series, flagged, near_only=False, flag_buffer=None):
        kernel = _kernel
        if precision == "mixed" and use_series and not closed_form and samples > 1 and not near_only:  # first pass with the exposure nodes in float32 (velociraptor.triton_mixed); the second pass stays float64
            from velociraptor.triton_mixed import _kernel_mixed as kernel
        for number, (first, count) in enumerate(plan_slices(total, samples)):
            kernel[(triton.cdiv(count, BLOCK),)](flat, *per_row, out, indices if indices is not None else nothing, flag_buffer if flag_buffer is not None else flags, nodes, weights, table, first, first + count, points, SAMPLES=samples, KEPLER=KEPLER_ITERATIONS, STEPS=CEL_STEPS, TOL=TOLERANCE, SERIES=use_series,
                                                  BOUND=ABSOLUTE_FLUX_ERROR, CLOSED=closed_form, GAUSS=gauss, INTERP=interpolate, INDEXED=indices is not None, FLAGS=flagged, PHASE_SKIP=bool(PHASE_SKIP and use_series and not closed_form and indices is None), SINGLE=bool(closed_form and precision == "mixed" and SINGLE_CLOSED), NEAR_ONLY=near_only, BLOCK=BLOCK, num_warps=WARPS, **LAUNCH_OPTIONS)
            if number % 8 == 7 and device.type != "cpu":
                torch.cuda.synchronize(device)

    trace = {"pairs": flat.numel(), "samples": samples} if TRACE is not None else None
    if not series:
        launch(flat.numel(), None, True, False, False)
    else:
        began = _tick(device) if trace is not None else 0.0
        if COMPACT and samples >= BINNED_MIN_SAMPLES:
            out.fill_(1.0)
            near_flags = torch.zeros(flat.numel(), dtype=torch.int8, device=device)
            launch(flat.numel(), None, False, True, True, near_only=True, flag_buffer=near_flags)
            listed = torch.nonzero(near_flags).reshape(-1)
            del near_flags
            if listed.numel():
                launch(listed.numel(), listed, False, True, True)
            del listed
        else:
            launch(flat.numel(), None, False, True, True)
        indices = torch.nonzero(flags).reshape(-1)
        if trace is not None:
            middle = _tick(device)
            trace.update(first_pass_seconds=middle - began, flagged_pairs=int(indices.numel()))
        if indices.numel():
            if BINNED and samples >= BINNED_MIN_SAMPLES:
                _second_pass_binned(flat, per_row, out, indices, nodes, weights, table, interpolate, points, samples, gauss, device, trace, single=precision == "mixed" and SINGLE_CLOSED)
            else:
                launch(indices.numel(), indices, True, False, False)
        if trace is not None:
            trace["second_pass_seconds"] = _tick(device) - middle
    if trace is not None:
        TRACE.append(trace)
    return out.reshape(shape)
