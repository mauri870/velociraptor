# Validation of the transit model

How close `quadratic_flux`, `separation` and `transit_flux` are to values that do not come from this package, over the range of radius ratio, separation, limb darkening and orbit that a validation tool draws.

## Question

For which radius ratios, separations, limb-darkening pairs and orbits does the model differ from an independent reference by more than the tolerance stated here, and by how much does PyTransit 2.2.0, the model that TRICERATOPS 1.1.0 uses, differ from the same references?

## Method

References. Each is computed outside the model under test.

1. Lens area. For u1 = u2 = 0 the flux removed is the area of the overlap of two discs over pi. The area is the standard lens formula evaluated in 50-digit arithmetic (mpmath), which does not lose digits at the tangencies as double precision does.
2. Two-dimensional integral. For limb-darkened cases, the integral of the intensity over the part of the planet's disc inside the star, in polar coordinates about the planet's centre, by nested `scipy.integrate.quad` calls (the radius inside, the angle outside, split at the angles where a radial limit changes character) at a relative tolerance of 1e-12 and an absolute tolerance of 1e-13, divided by the flux of the star. It shares the intensity law with the model and nothing else.
3. Convergence. The closed form against the quadrature of 768 nodes (`quadrature_flux` with `nodes=768`) on the whole grid, and the quadratures of 24 and 48 nodes against the 768.
4. Orbit. The separation by the model against a 50-digit solution of Kepler's equation by bisection followed by the same projection formulas, at random orbits.
5. PyTransit 2.2.0 (`QuadraticModel(interpolate=False)`), through its public interface, in a separate environment (`validation/pytransit_env.md`); its flux at the separations of the grid, and its light curves on random circular and eccentric orbits with exposure, against the model.

Grid. Radius ratio k in {0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.99, 1.0, 1.5, 3.0}; six limb-darkening pairs (0, 0), (0.4, 0.25), (0.6, 0.1), (0.1, 0.5), (1, 0), (0.3, 0.7); separations at 0, 1e-9, 1e-6, 1e-3, 0.5, 1, at |1 - k| and 1 + k and each side of them at 1e-9 and 1e-6, and 41 points evenly spaced from 0 to 1 + k + 0.1. Orbits: seeded draws (seed 1), the number of the loop in `grid.orbit` for the orbit reference and the `count` of `pytransit_grid.orbits` for PyTransit, with a/R* from 2 to 40, eccentricity 0 to 0.9, impact parameter 0 to 1.1 and, for PyTransit, exposure 0 or 30 minutes with 1, 5 or 20 sub-samples.

Reading rules.
- V1. The model agrees with the lens area to 1e-11 at every point of the grid with u1 = u2 = 0. A point that fails is listed.
- V2. The model agrees with the two-dimensional integral to the integral's own accuracy, taken as 1e-8, on a subset of the grid (the first six separations and the tangencies, three limb-darkening pairs).
- V3. The closed form agrees with the quadrature of 768 nodes to 1e-9 on the whole grid.
- V4. The separation agrees with the 50-digit reference to 1e-12 for eccentricity up to 0.9.
- V5. PyTransit is compared as it is: the largest difference by radius ratio and limb-darkening pair, and by a/R* and eccentricity for the orbits. No tolerance is set for it; the table is the result.


## Results

Source: `results.json`, written by `grid.py` and `pytransit_grid.py` (`nodes_scan.py` for the node table of the quadrature).

V1 holds: the closed form agrees with the lens area to 2.8e-16 at the worst of the 552 cases (rule: 1e-11). The quadrature of 48 nodes, which the closed form replaced, read 1.2e-10 at the worst case, 1e-6 beyond the internal tangency, and so did not hold it.

V2 holds: the closed form agrees with the two-dimensional integral to 1.5e-13 (rule: 1e-8) on the subset of the grid.

V3 holds: the closed form agrees with the quadrature of 768 nodes to 6.4e-15 (rule: 1e-9) at the worst of the whole grid (3,312 cases).

V4 holds: the separation of an orbit agrees with the 50-digit solution within the rule (1e-12) at the 300 random orbits of eccentricity up to 0.9 and a/R* from 2 to 40; the largest difference of each band of eccentricity is under `orbit_50_digit` in `results.json`.

V5, PyTransit 2.2.0: the flux at a separation differs by at most 1.5e-05 for every radius ratio of the grid except 1.0, and by up to 3.6e-02 at a radius ratio of 1.0, where the table shows the case: at a separation of 0 PyTransit returns a negative flux when limb darkening is on (the flux is 0). The light curves of random orbits differ by an amount that rises as a/R* falls and as the eccentricity rises: the median of the largest difference of an orbit is 5.0e-09 at a/R* 20 to 40 with e under 0.3, 1.2e-07 at a/R* 5 to 10, and 1.7e-01 at a/R* under 3 with e over 0.6 (one orbit); the full table is in `results.json`. The reference for these differences is the closed form, which agrees with the 50-digit separation and with the two-dimensional integral at the points where PyTransit departs (V1, V2, V4).

## Limits

The two-dimensional integral and the quadrature share the intensity law with the closed form and nothing else, but they share it. The radius ratios of the grid stop at 3 and the limb-darkening pairs are six. The PyTransit comparison draws 300 orbits, of which few fall in the cells of small a/R* (the counts are in `results.json`).
