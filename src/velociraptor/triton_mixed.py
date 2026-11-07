"""The first pass of the closed-form kernel with the exposure nodes in float32 (the default precision; see transit_flux_triton_analytic(precision=...), where "float64" keeps every step in float64).

Float64: the orbit of the centre of the exposure (the Kepler solution, the sky coordinates of the centre), the weights and nodes, the tests that decide which pairs the first pass keeps, the limb-darkening series tiers, the sum of the exposure into the flux, and the second pass (the closed form for the flagged pairs).
Float32: for each node of the exposure, the small change of the eccentric anomaly from the centre (a few Newton steps on an angle of at most 0.1 rad, which keeps its digits because it is relative to the centre) and the change it makes in the sky position, and the order-3 series flux of a small planet wholly inside the disc. The series is
formed as a deficit (k^2 times the mean intensity ratio, not 1 minus a number close to 1) and summed as a deficit, which keeps about 1e-10 of absolute precision in the flux. A pair is flagged for the float64 second pass when a node is not on the order-3 series (ingress and egress, a larger planet, the band within 1e-4 of the contact b + k = 1) or when the exposure moves the eccentric anomaly by more than 0.1 rad."""
import triton
import triton.language as tl
from triton.language.extra import libdevice

from velociraptor.triton_analytic import _series_tier


@triton.jit
def _small_angle32(d):
    """sin(d) and cos(d) - 1 for |d| of at most 0.1, as _small_angle in float32."""
    d2 = d * d
    sine = d * (1 - d2 * (1.0 / 6.0) * (1 - d2 * (1.0 / 20.0) * (1 - d2 * (1.0 / 42.0) * (1 - d2 * (1.0 / 72.0)))))
    cosine_minus_one = -d2 * 0.5 * (1 - d2 * (1.0 / 12.0) * (1 - d2 * (1.0 / 30.0) * (1 - d2 * (1.0 / 56.0) * (1 - d2 * (1.0 / 90.0)))))
    return sine, cosine_minus_one


@triton.jit
def _series_deficit32(b, k, u1, u2, normalisation):
    """The deficit 1 - flux of the order-3 series, in float32."""
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
    mean = l0 + k2 * (l1 * 0.125 + k2 * (l2 * (1.0 / 192.0) + k2 * l3 * (1.0 / 9216.0)))
    return k2 * mean * normalisation


@triton.jit
def _kernel_mixed(time_ptr, k_ptr, u1_ptr, u2_ptr, period_ptr, a_ptr, cos_inc_ptr, ecc_ptr, sin_omega_ptr, t0_ptr, exposure_ptr, cos_omega_ptr, mean0_ptr, root_ptr, norm_ptr, mlow_ptr, mhigh_ptr, asinc_ptr, out_ptr, index_ptr, flag_ptr, node_ptr, weight_ptr, table_ptr,
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
    mean0 = tl.load(mean0_ptr + row, mask=valid, other=0.0)
    root = tl.load(root_ptr + row, mask=valid, other=1.0)
    normalisation = tl.load(norm_ptr + row, mask=valid, other=1.0)
    PI = tl.full([BLOCK], 3.141592653589793, dtype=tl.float64)
    if PHASE_SKIP:  # the bound of triton_analytic._kernel: pairs too far in phase from conjunction have flux 1, and a block of them takes no orbit
        m_low = tl.load(mlow_ptr + row, mask=valid, other=0.0)
        m_high = tl.load(mhigh_ptr + row, mask=valid, other=1.0)
        asin_c = tl.load(asinc_ptr + row, mask=valid, other=10.0)
        phase = 2 * PI * (time - t0) / period
        delta = tl.abs(phase - 2 * PI * libdevice.rint(phase / (2 * PI)))
        far = (m_low * delta > asin_c + 1e-9) & (m_high * delta < PI - asin_c - 1e-9)
        live_lanes = valid & ~far
    else:
        live_lanes = valid
    deficit = tl.zeros([BLOCK], dtype=tl.float32)
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
        x_c = a * (cosine_centre - ecc)
        y_c = a * root * sine_centre
        big_x_c = x_c * cos_omega - y_c * sin_omega
        big_y_c = x_c * sin_omega + y_c * cos_omega
        centre = tl.sqrt(big_x_c * big_x_c + big_y_c * big_y_c * cos_inc * cos_inc)
        reach = 2 * PI * a * (1 + ecc) / (period * root) * exposure / 2
        near = valid & (centre - reach <= 1.0 + k)
        if tl.sum(near.to(tl.int32)) > 0:
            swing = 2 * PI * (exposure / 2) / (period * (1 - ecc))
            refused = (near & (swing > 0.1)).to(tl.int32)  # an exposure that moves the eccentric anomaly by more than 0.1 rad goes to the second pass
            usable = near & (swing <= 0.1)
            e32 = ecc.to(tl.float32)
            a32 = a.to(tl.float32)
            root32 = root.to(tl.float32)
            sc32 = sine_centre.to(tl.float32)
            cc32 = cosine_centre.to(tl.float32)
            so32 = sin_omega.to(tl.float32)
            co32 = cos_omega.to(tl.float32)
            ci32 = cos_inc.to(tl.float32)
            x_centre = big_x_c.to(tl.float32)
            y_centre = big_y_c.to(tl.float32)
            y_scaled_centre = (big_y_c * cos_inc).to(tl.float32)
            k32 = k.to(tl.float32)
            u132 = u1.to(tl.float32)
            u232 = u2.to(tl.float32)
            n32 = normalisation.to(tl.float32)
            for sample in range(SAMPLES):
                if GAUSS:
                    fraction = tl.load(node_ptr + sample)
                    weight = tl.load(weight_ptr + sample)
                else:
                    fraction = (sample.to(tl.float64) + 0.5) / SAMPLES - 0.5
                    weight = tl.full([BLOCK], 1.0 / SAMPLES, tl.float64)
                shift = (2 * PI * (fraction * exposure) / period).to(tl.float32)
                d = shift / (1 - e32 * cc32)
                for _ in range(3):
                    sd, cm = _small_angle32(d)
                    d = d - (d - e32 * (sc32 * cm + cc32 * sd) - shift) / (1 - e32 * (cc32 * (cm + 1) - sc32 * sd))
                sd, cm = _small_angle32(d)
                delta_x = a32 * (cc32 * cm - sc32 * sd)
                delta_y = a32 * root32 * (sc32 * cm + cc32 * sd)
                sky_x = x_centre + (delta_x * co32 - delta_y * so32)
                sky_y = y_centre + (delta_x * so32 + delta_y * co32)
                scaled_y = y_scaled_centre + (delta_x * so32 + delta_y * co32) * ci32
                b = tl.sqrt(sky_x * sky_x + scaled_y * scaled_y)
                b64 = b.to(tl.float64)
                covered = (k32 >= 1) & (b <= k32 - 1)
                live = usable & (sky_y > 0) & (b < 1 + k32) & ~covered
                small_low, small_high = _series_tier(b64, k, u1, u2, normalisation, live, BOUND * 0.5)
                edge = live & (tl.abs(b + k32 - 1) < 1e-4)  # the band around the contact, where the float32 b can fall on the wrong side of the test b + k < 1
                on_series = small_low & ~edge
                node_deficit = tl.where(on_series, _series_deficit32(tl.where(on_series, b, 0.3), tl.where(on_series, k32, 0.01), u132, u232, n32), 0.0)
                node_deficit = tl.where(usable & covered & (sky_y > 0), 1.0, node_deficit)
                deficit += weight.to(tl.float32) * node_deficit
                refused = refused | (live & ~on_series).to(tl.int32)
    tl.store(out_ptr + pair, 1.0 - deficit.to(tl.float64), mask=valid)
    if FLAGS:
        tl.store(flag_ptr + pair, (refused > 0).to(tl.int8), mask=valid)
