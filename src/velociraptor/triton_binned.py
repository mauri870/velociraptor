"""The second pass of the closed-form kernel in three steps. The closed form runs on a list of nodes sorted by regime instead of inside the orbit loop of the flagged pair (see transit_flux_triton_analytic).

1. _prepare: for each flagged pair, the orbit and the exposure nodes as the main kernel makes them (same expressions, same blocks of 32 flagged pairs, same path for the change of the anomaly). For each node it stores the separation b and a code: 2 the planet is over the disc and needs the closed form, 1 it covers the star (flux 0), 0 anything else (flux 1).
2. The nodes with code 2 are listed and partitioned by regime (ingress, or inside the disc), so that the lanes of a block take the same branch of the closed form. _closed_nodes evaluates the closed form on the list, one node a lane.
3. _accumulate adds weight x flux over the nodes of each flagged pair in the order of the nodes, from 0.0, as the main kernel does. The closed form compiled here is not rounded exactly like the one in the main kernel, so a few pairs differ in the last bits of the flux.
"""
import triton
import triton.language as tl

from velociraptor.triton_analytic import (
    _flux_closed,
    _interpolation,
    _sky,
    _small_solve,
)


@triton.jit
def _prepare(time_ptr, k_ptr, period_ptr, a_ptr, cos_inc_ptr, ecc_ptr, sin_omega_ptr, t0_ptr, exposure_ptr, cos_omega_ptr, mean0_ptr, root_ptr, code_ptr, b_ptr, index_ptr, node_ptr, weight_ptr, table_ptr,
             offset_pairs, pairs, points, SAMPLES: tl.constexpr, KEPLER: tl.constexpr, GAUSS: tl.constexpr, INTERP: tl.constexpr, BLOCK: tl.constexpr):
    lane = offset_pairs + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = lane < pairs
    pair = tl.load(index_ptr + lane, mask=valid, other=0)
    row = pair // points
    time = tl.load(time_ptr + pair, mask=valid, other=0.0)
    k = tl.load(k_ptr + row, mask=valid, other=0.0)
    period = tl.load(period_ptr + row, mask=valid, other=1.0)
    a = tl.load(a_ptr + row, mask=valid, other=1.0)
    cos_inc = tl.load(cos_inc_ptr + row, mask=valid, other=0.0)
    ecc = tl.load(ecc_ptr + row, mask=valid, other=0.0)
    sin_omega = tl.load(sin_omega_ptr + row, mask=valid, other=0.0)
    cos_omega = tl.load(cos_omega_ptr + row, mask=valid, other=1.0)
    t0 = tl.load(t0_ptr + row, mask=valid, other=0.0)
    exposure = tl.load(exposure_ptr + row, mask=valid, other=0.0)
    mean0 = tl.load(mean0_ptr + row, mask=valid, other=0.0)
    root = tl.load(root_ptr + row, mask=valid, other=1.0)
    PI = tl.full([BLOCK], 3.141592653589793, dtype=tl.float64)
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
    reach = 2 * PI * a * (1 + ecc) / (period * root) * exposure / 2
    near = valid & (centre - reach <= 1.0 + k)
    if tl.sum(near.to(tl.int32)) > 0:
        swing = 2 * PI * (exposure / 2) / (period * (1 - ecc))
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
                fraction = tl.load(node_ptr + sample)
            else:
                fraction = (sample.to(tl.float64) + 0.5) / SAMPLES - 0.5
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
            covered = (k >= 1) & (b <= k - 1)
            live = near & (along > 0) & (b < 1 + k) & ~covered
            code = tl.where(live, 2, tl.where(near & covered & (along > 0), 1, 0))
            tl.store(code_ptr + lane * SAMPLES + sample, code.to(tl.int8), mask=valid)
            tl.store(b_ptr + lane * SAMPLES + sample, b, mask=valid)
    else:
        for sample in range(SAMPLES):
            tl.store(code_ptr + lane * SAMPLES + sample, tl.zeros([BLOCK], dtype=tl.int8), mask=valid)


@triton.jit
def _closed_nodes(list_ptr, code_ptr, b_ptr, flux_ptr, k_ptr, u1_ptr, u2_ptr, index_ptr, offset_nodes, nodes, points, SAMPLES: tl.constexpr, STEPS: tl.constexpr, TOL: tl.constexpr, SINGLE: tl.constexpr, BLOCK: tl.constexpr):
    lane = offset_nodes + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = lane < nodes
    element = tl.load(list_ptr + lane, mask=valid, other=0)
    flagged = element // SAMPLES
    row = tl.load(index_ptr + flagged, mask=valid, other=0) // points
    k = tl.load(k_ptr + row, mask=valid, other=0.2)
    u1 = tl.load(u1_ptr + row, mask=valid, other=0.4)
    u2 = tl.load(u2_ptr + row, mask=valid, other=0.25)
    b = tl.load(b_ptr + element, mask=valid, other=0.3)
    PI = tl.full([BLOCK], 3.141592653589793, dtype=tl.float64)
    tl.store(flux_ptr + element, _flux_closed(b, k, u1, u2, PI, STEPS, TOL, SINGLE), mask=valid)


@triton.jit
def _accumulate(code_ptr, flux_ptr, out_ptr, index_ptr, node_ptr, weight_ptr, offset_pairs, pairs, SAMPLES: tl.constexpr, GAUSS: tl.constexpr, BLOCK: tl.constexpr):
    lane = offset_pairs + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = lane < pairs
    accumulated = tl.zeros([BLOCK], dtype=tl.float64)
    for sample in range(SAMPLES):
        if GAUSS:
            weight = tl.load(weight_ptr + sample)
        else:
            weight = tl.full([BLOCK], 1.0 / SAMPLES, tl.float64)
        code = tl.load(code_ptr + lane * SAMPLES + sample, mask=valid, other=0)
        flux = tl.load(flux_ptr + lane * SAMPLES + sample, mask=valid, other=1.0)
        value = tl.where(code == 2, flux, tl.where(code == 1, 0.0, 1.0))
        accumulated += weight * value
    pair = tl.load(index_ptr + lane, mask=valid, other=0)
    tl.store(out_ptr + pair, accumulated, mask=valid)
