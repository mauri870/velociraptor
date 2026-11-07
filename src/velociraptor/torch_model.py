"""The orbit of velociraptor.quadratic in torch, for the CPU and for CUDA and ROCm devices, and the pace at which the host queues chunks of work on a device."""

import numpy as np
import torch

from velociraptor.quadratic import KEPLER_ITERATIONS

PACE = 16  # chunks queued on the device before the host waits for it
CHUNK = 1 << 17  # elements per pass of the torch flux


def _pace(device, number):
    """Wait for the device after every PACE chunks. A queue of hundreds of chunks ahead of the device causes a segmentation fault on ROCm 7.2 (torch 2.12)."""
    if device.type != "cpu" and number % PACE == PACE - 1:
        torch.cuda.synchronize(device)


def separation_torch(time, period, a_rs, inc, ecc=0.0, omega=np.pi / 2, t0=0.0):
    """velociraptor.separation in torch: the separation in stellar radii and whether the planet is in front, on the device of `time`. All arguments broadcast together. Kepler's equation is solved in KEPLER_ITERATIONS Newton steps (the numpy
    version stops when no step is over 1e-15; further steps change nothing). The separation is the hypotenuse of the two sky components, which has no cancellation near mid-transit."""
    time = torch.as_tensor(time, dtype=torch.float64)
    period, a_rs, inc, ecc, omega, t0 = (torch.as_tensor(value, dtype=torch.float64, device=time.device) for value in (period, a_rs, inc, ecc, omega, t0))
    start = np.pi / 2 - omega
    anomaly0 = 2 * torch.atan2(torch.sqrt(1 - ecc) * torch.sin(start / 2), torch.sqrt(1 + ecc) * torch.cos(start / 2))
    mean = anomaly0 - ecc * torch.sin(anomaly0) + 2 * np.pi * (time - t0) / period
    eccentric = mean + ecc * torch.sin(mean)
    for _ in range(KEPLER_ITERATIONS):
        eccentric = eccentric - (eccentric - ecc * torch.sin(eccentric) - mean) / (1 - ecc * torch.cos(eccentric))
    x = a_rs * (torch.cos(eccentric) - ecc)
    y = a_rs * torch.sqrt(1 - ecc**2) * torch.sin(eccentric)
    along = torch.sin(omega) * x + torch.cos(omega) * y
    across = torch.cos(omega) * x - torch.sin(omega) * y
    return torch.hypot(across, along * torch.cos(inc)), along > 0
