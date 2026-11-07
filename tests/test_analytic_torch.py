"""The closed-form model in torch against the closed-form model in numpy (velociraptor.analytic), on a grid of radius ratios, separations and limb-darkening coefficients, on the CPU."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from velociraptor.analytic import cel as cel_numpy
from velociraptor.analytic import quadratic_flux as flux_numpy
from velociraptor.analytic_torch import cel_torch, quadratic_flux_analytic_torch

K = (0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.99, 1.0, 1.5, 3.0)
U = ((0.0, 0.0), (0.4, 0.25), (0.6, 0.1), (0.1, 0.5), (1.0, 0.0), (0.3, 0.7))


def separations(k):
    edges = [0.0, 1e-9, 1e-6, 1e-3, 0.5, 1.0]
    for centre in (abs(1 - k), 1 + k):
        edges += [centre - 1e-6, centre - 1e-9, centre, centre + 1e-9, centre + 1e-6]
    z = np.concatenate([edges, np.linspace(0.0, 1 + k + 0.1, 41)])
    return np.unique(z[z >= 0])


@pytest.mark.parametrize("k", K)
def test_the_torch_flux_agrees_with_the_numpy_flux_to_1e_12_on_the_validation_grid(k):
    z = separations(k)
    worst = 0.0
    for u1, u2 in U:
        got = quadratic_flux_analytic_torch(torch.as_tensor(z), k, u1, u2).numpy()
        worst = max(worst, float(np.max(np.abs(got - flux_numpy(z, k, u1, u2)))))
    assert worst < 1e-12, (k, worst)


def test_cel_agrees_with_the_numpy_cel_in_8_steps():
    g = np.random.default_rng(1)
    kc = g.uniform(1e-4, 1.0, 20000)
    p = g.uniform(0.01, 5.0, 20000)
    a, b = g.uniform(-2, 2, 20000), g.uniform(-2, 2, 20000)
    got = cel_torch(torch.as_tensor(kc), torch.as_tensor(p), torch.as_tensor(a), torch.as_tensor(b)).numpy()
    assert np.max(np.abs(got - cel_numpy(kc, p, a, b)) / np.maximum(np.abs(got), 1.0)) < 1e-13


def test_broadcasting_and_empty_input():
    z = torch.linspace(0.0, 1.6, 17, dtype=torch.float64)[:, None]
    k = torch.tensor([0.05, 0.3, 1.2], dtype=torch.float64)[None, :]
    out = quadratic_flux_analytic_torch(z, k, 0.4, 0.25)
    assert out.shape == (17, 3)
    assert quadratic_flux_analytic_torch(torch.zeros(0, dtype=torch.float64), 0.1, 0.4, 0.25).shape == (0,)


def test_the_torch_transit_flux_agrees_with_the_numpy_transit_flux_to_1e_12_and_skips_the_points_out_of_reach():
    from velociraptor import transit_flux
    from velociraptor.analytic_torch import transit_flux_analytic_torch

    time = np.linspace(-1.5, 1.5, 1201)
    for period, a_rs, inc, ecc, omega, k, exposure, samples in ((3.0, 8.0, 1.5, 0.0, np.pi / 2, 0.1, 0.00139, 20), (10.0, 20.0, 1.55, 0.3, 1.0, 1.2, 0.02, 7), (0.5, 3.0, 1.45, 0.6, 4.0, 0.3, 0.0, 1)):
        got = transit_flux_analytic_torch(torch.as_tensor(time), k, 0.4, 0.25, period, a_rs, inc, ecc, omega, exposure=exposure, samples=samples, chunk=500).numpy()
        want = transit_flux(time, k, 0.4, 0.25, period, a_rs, inc, ecc, omega, exposure=exposure, samples=samples)
        assert np.max(np.abs(got - want)) < 1e-12
    out_of_reach = transit_flux_analytic_torch(torch.as_tensor(np.linspace(0.4, 0.6, 11) * 4.0), 0.1, 0.4, 0.25, 4.0, 8.0, 1.55, exposure=0.00139, samples=20)
    assert out_of_reach.tolist() == [1.0] * 11
