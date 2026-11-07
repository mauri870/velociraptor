"""The launch plan of the Triton kernel of the package and its entry point.

The kernel is launched in slices of at most ELEMENTS sub-sample elements (a pair of a point and a scenario, times its exposure sub-samples). A short dispatch does not starve the display or trip the driver's ring timeout. The kernel itself, the
closed-form flux, is velociraptor.triton_analytic."""

ELEMENTS = 1 << 20  # sub-sample elements of one launch; sized so that a launch of the slowest pass takes a few milliseconds of the device


def plan_slices(pairs, samples):
    """The (first, count) of each launch for `pairs` pairs of `samples` sub-samples: slices of at most ELEMENTS sub-sample elements (at least one pair), which cover the pairs once and in order. A slice is a multiple of 32 pairs where the bound allows it. The kernel works in blocks of 32 consecutive pairs and makes some choices per block (the number of Newton steps of Kepler's equation, the path of the change of the anomaly). When every slice but the last is a multiple of 32, the blocks and the last bits of the flux do not depend on ELEMENTS."""
    size = ELEMENTS // max(int(samples), 1)
    size = size // 32 * 32 if size >= 32 else max(1, size)
    return [(first, min(size, pairs - first)) for first in range(0, pairs, size)]


def transit_flux_triton(*arguments, **options):
    """velociraptor.transit_flux on a device with the closed-form kernel (velociraptor.triton_analytic.transit_flux_triton_analytic)."""
    from velociraptor.triton_analytic import transit_flux_triton_analytic

    return transit_flux_triton_analytic(*arguments, **options)
