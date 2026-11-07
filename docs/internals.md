# velociraptor internals

## Exposure integral

`install("cuda")` runs the Triton kernel and integrates each exposure with 3 Gauss-Legendre nodes. `exposure_nodes=None` takes the equal parts of the exposure that TRICERATOPS uses (`nsamples`, 20); use it to compare with TRICERATOPS 1.1.0.

## Precision

The device kernel's first pass runs the exposure nodes in float32 by default (`"mixed"`: the closed form near contact and the chi-squared sum stay in float64). `float64` keeps every step in float64:

```python
from velociraptor import triton_analytic

triton_analytic.PRECISION = "float64"
```

With `PRECISION = "float64"` a chunked likelihood can differ from an unchunked call in the last bit on a small share of the draws (about 1 percent), because the chunks change the order of the sums; the default precision is bit for bit.

## Random streams

TRICERATOPS draws every scenario from one global NumPy stream, so leaving one scenario out changes the numbers of all the others. `scenario_streams` gives each call of `calc_probs` a stream of its own, fixed by the seed, the function and the index of the call; outside the block TRICERATOPS is unchanged.

```python
from velociraptor.streams import scenario_streams

with scenario_streams(seed=7):
    target.calc_probs(time=time, flux_0=flux, flux_err_0=flux_err, P_orb=period, N=100000)
```

## Batches and the error of a probability

TRICERATOPS reports one value of FPP and NFPP from one run of random draws. `velociraptor.allocation` runs a fixed number of seeded batches, gives the effective sample size of each scenario call and the scatter of the probabilities between batches:

```python
from velociraptor.allocation import Bank, batch_estimates, batches, effective_sample_size, evidence_report
from velociraptor.triceratops import install

def run():
    target.calc_probs(time=time, flux_0=flux, flux_err_0=flux_err, P_orb=period, N=100000, parallel=True)
    return target.FPP, target.NFPP

bank = Bank()
with install("cuda"), batches(bank, lambda call: 10, seed=7):
    run()

effective_sample_size(bank, 0)  # of the first scenario call, over its 10 batches
batch_estimates(bank, run)      # the probabilities of each batch, pooled, and their error
evidence_report(bank, run)      # the call that carries the scatter, its batch effective sample size and their ratio
```

`evidence_report` names the scenario call whose batches move the output most and gives that call's batch effective sample size, (sum Z)^2 / sum Z^2 over the evidences of its batches, and the ratio of that to the number of its batches. The size is 1 when one batch carries the sum and the number of batches when they are equal. `batch_estimates` returns the same three numbers beside the error. No value of the ratio is given as safe.

`batches` runs the calls of `calc_probs` one at a time, also inside `install`, since each batch is seeded on numpy's one global generator. It suspends the overlap of calls that `install` turns on by default (its `overlap` argument), so a run inside `batches` takes longer than the same `calc_probs` under `install` alone. `batch_estimates` gives `error`, the standard deviation of the probabilities between batches over the root of the number of batches, which every call must hold equally and at least 2. The error is the scatter between batches, not a confidence interval: it can be too small where few draws carry the evidence of a call.

## Light curve

```python
from velociraptor import quadratic_flux

flux = quadratic_flux(z=0.3, k=0.1, u1=0.4, u2=0.2)  # 1 is no transit
```

`z` is the separation of the centres in stellar radii (the sign is ignored), `k` the radius ratio, `u1` and `u2` the quadratic limb-darkening coefficients. Any of them may be an array, and the result has their broadcast shape.

## Speed and accuracy

The tables of speed, of the pairs left to the closed form and of the difference from references are in [README.md](../README.md), written from `benchmarks/results.json` and `validation/results.json` by `benchmarks/tables.py`.

## Differences from TRICERATOPS

TRICERATOPS computes its light curves with PyTransit 2.2.0, which is GPL. The source of velociraptor does not import or copy PyTransit; `tests/test_quadratic.py` and `validation/pytransit_grid.py` import it to compare results, and those two files run only where it is installed. The PyTransit table of the README gives the largest difference between PyTransit and the reference; `validation/pytransit_grid.py` reproduces it.

Where the two light curves differ, the log-likelihoods differ, and a probability whose evidence rests on a few such draws differs with them. `tests/test_stock_reference.py` holds one draw with its 50-digit reference, and compares the log-likelihoods of the two packages on draws where the curves agree.

## Tests

`uv run --group cpu pytest tests -q`
