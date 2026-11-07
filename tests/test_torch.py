"""The orbit in torch against the numpy orbit, and a long queue of chunks on a device where one is available."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from velociraptor import separation
from velociraptor.torch_model import separation_torch

ORBITS = [(3.0, 8.0, 1.5, 0.0, np.pi / 2), (10.0, 20.0, 1.55, 0.3, 1.0), (0.5, 3.0, 1.45, 0.6, 4.0), (30.0, 40.0, 1.5707, 0.05, 2.5)]


@pytest.mark.parametrize("period,a_rs,inc,ecc,omega", ORBITS)
def test_separation_agrees_with_the_numpy_model_to_1e_12(period, a_rs, inc, ecc, omega):
    time = np.linspace(-0.4, 0.4, 801) * period / 3
    z, front = separation_torch(torch.as_tensor(time), period, a_rs, inc, ecc, omega, 0.01)
    z_ref, front_ref = separation(time, period, a_rs, inc, ecc, omega, 0.01)
    assert np.max(np.abs(z.numpy() - z_ref)) < 1e-12
    assert np.array_equal(front.numpy(), front_ref)


def test_a_long_queue_of_chunks_completes_on_a_device_where_one_is_available():
    # 1,000 scenarios of 1,000 points with 20 sub-samples make 150 chunks. A queue that deep crashed on ROCm when the host did not wait for the device.
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA or ROCm device")
    from velociraptor.analytic_torch import transit_flux_analytic_torch

    time = torch.linspace(-0.2, 0.2, 1000, dtype=torch.float64, device="cuda").expand(1000, 1000)
    k = torch.linspace(0.01, 0.1, 1000, dtype=torch.float64, device="cuda")[:, None]
    flux = transit_flux_analytic_torch(time, k, 0.4, 0.25, 4.0, 8.0, 1.55, exposure=0.00139, samples=20)
    assert flux.shape == (1000, 1000) and torch.isfinite(flux).all() and float(flux.min()) < 1.0
