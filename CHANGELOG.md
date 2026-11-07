# Changes

## 0.1.0 - 2026-11-07

- `velociraptor.allocation.batch_effective_sample_size` and `evidence_report`: the number of batches that carry a call's evidence, the call that carries the scatter of the outputs, its batch effective sample size and the ratio of that to its number of batches; `batch_estimates` returns the same three numbers beside the error. They are reported with no verdict.
- `quadratic_flux(z, k, u1, u2)`: the flux of a star with quadratic limb darkening behind a dark disc in closed form, from Agol, Luger and Foreman-Mackey (2020), tested against 50-digit references.
- `separation` and `transit_flux`: the sky separation of the planet on an orbit, and the flux over time with an exposure time, written from Mandel and Agol (2002) as the cross-check of the closed form.
- Triton kernels of the closed-form flux for CUDA and ROCm devices, with the exposure integrated by 3 Gauss-Legendre nodes or by the equal parts that TRICERATOPS uses (`exposure_nodes=None`).
- The default precision of the device kernel is `"mixed"` (the exposure nodes of the first pass in float32, the closed form near contact and the chi-squared sum in float64). `triton_analytic.PRECISION = "float64"` or `precision="float64"` keeps every step in float64.
- `velociraptor.triceratops.install(device, ...)`: runs the three likelihood functions of TRICERATOPS 1.1.0 on a device; the kernel defaults to Triton on a device and to torch on the CPU.
- `install` overlaps the sampling of the scenario functions with the likelihoods of the calls before them (`overlap`, three calls in flight by default; `overlap=1` leaves the package's own `calc_probs` in place).
- The splines of TRICERATOPS's stellar relations are evaluated without the interpreter lock and not twice on the same array (`released_splines`, `remember_splines`), and the limb-darkening lookup of the companions is made once for each distinct pair (`gathered_lookup`).
- `velociraptor.streams.scenario_streams`: a random stream of its own for each call of `calc_probs`, fixed by a seed.
- `velociraptor.allocation`: `batches` and `Bank` run seeded batches of draws for each scenario call, `effective_sample_size` gives the effective sample size of a call, and `batch_estimates` and `grouped_estimates` give the probabilities of batches and groups of batches with the scatter between batches.
- The likelihoods run on chunks of draws, so that their device memory is bounded by the chunk and not by the number of draws.
