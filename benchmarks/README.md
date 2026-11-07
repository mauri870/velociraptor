# Benchmarks

`bench.py` times the three scenario likelihoods of TRICERATOPS 1.1.0 (`lnL_TP_p`, `lnL_EB_p`, `lnL_EB_twin_p`, the functions that `velociraptor.triceratops.install` replaces) with TRICERATOPS's own numba code and with velociraptor on the CPU and on the device. `results.json` holds the sessions that were run, and the README tables are written from it by `tables.py`. A figure about speed that appears anywhere else in the repository is copied from `results.json`, with the machine, the command and the library versions beside it.

## Input

Nothing is read from a file or a network. `make_input` draws everything from seed 20261031:

- one light curve of `--points` points (default 1,000) evenly spaced over ±0.3 days from the transit midpoint, flux 1 with 500 ppm of Gaussian noise and a 1,000 ppm box 0.12 days long;
- `--draws` draws (default 10,000 and 100,000) of the parameters of each scenario, a planet around the star, an eclipsing binary, and its twin at twice the period. The star has a radius of 0.6 to 1.4 solar radii and a mass equal to its radius in solar units; the period is 3 days; the impact parameter is 0 to 1 and sets the inclination; the eccentricity is 0 to 0.2; the argument of periastron 0 to 360 degrees; limb darkening u1 0.3 to 0.5 and u2 0.15 to 0.3; the flux share of the unresolved companion 0 to 0.3; the planet radius 1 to 15 Earth radii; the binary radius 0.1 to 1 solar radii with a flux ratio 0.01 to 0.5.

The exposure time is 0.00139 days and the supersampling 20, the defaults of TRICERATOPS. Every method gets copies of the same arrays: TRICERATOPS converts the inclination to radians inside the array it is given.

## Scope

The rows time the likelihoods of one scenario: the model, the dilution and the chi-squared. A whole TRICERATOPS calculation also draws the scenarios, builds the table of probabilities and, for a real target, queries the catalogs for its field, none of which is timed here, so the speedup of a calculation is smaller than the speedup of the rows by the share of the calculation that these functions are. That share is not measured by this script, since a real target needs the network; a table of whole calculations on real stars is a separate table with its own label. Two sessions are compared row by row only when they ran the same arguments (`command` in `results.json`).

## Size

One call of a row is the three functions on `draws` draws and `points` points, which is the work of one scenario of one star in a TRICERATOPS calculation; the table gives the seconds of that call, and the seconds of a field are those of the calls it makes. TRICERATOPS's default is 1,000,000 draws. With `parallel=True` it builds an array of draws by points in float64, 8 GB at 1,000,000 draws and 1,000 points, and holds several of them, so the sizes here are 10,000 and 100,000 draws (0.08 and 0.8 GB an array) and `--max-array-gib` (default 2) refuses a larger one before anything is allocated. The 1,000 points are a choice of the benchmark, one transit window of a binned fold; the seconds of every method depend on the points, so they are an argument of the script and the table states them.

## Rows

| row | what it runs |
| --- | --- |
| `triceratops-numba` | `triceratops.likelihoods`, at the number of numba threads that was fastest in the scan |
| `velociraptor-cpu` | `velociraptor.triceratops.install("cpu")`, torch with the same number of threads |
| `velociraptor-device` | `install("cuda", kernel="triton", exposure_nodes=None)`, the 20 midpoints of the exposure; only when a device is present, and not with `--cpu-only` |
| `velociraptor-device-mixed` | the same with the exposure nodes of the first pass in float32 |
| `velociraptor-device-g3` | the 20-midpoint row's float64 precision with 3 Gauss-Legendre nodes in the exposure |
| `velociraptor-device-default` | `install("cuda")` with no arguments: 3 Gauss-Legendre nodes and the first pass in float32 |

The scan runs the numba path at each of `--threads` (default 1, 2, 4, 8, 16, 32, up to the cores allowed to the process and `NUMBA_NUM_THREADS`) on the largest size, one call that is not counted and one that is, and the fastest count is used for every row. Every row has one call that is not counted, in which numba compiles and the device kernel compiles, then `--repeats` timed calls (default 5); the table gives the median, and `results.json` the quartiles and the seconds of the call that was not counted.

## Timed region

Inside the timer: the three calls with numpy arrays in and numpy arrays out, which for the device rows includes the copy to the device and back, the copies of the arguments made for each call, and the compilation only in the call that is not counted. Outside: the draws and the light curve, the scan.

## Agreement

Each package row records the largest `|difference| / max(|numba value|, 1)` over the finite entries of the three functions and the number of entries that are infinite in one and finite in the other. The differences are the model's, not a tolerance of the benchmark.

## Device

The device row refuses to run when another process holds `/dev/kfd` open (`--allow-busy` overrides), runs inside `ring_guard.py`, which stops the run at the first GPU ring timeout of the kernel log (once the running call returns to Python: it cannot stop a kernel in flight, and the slices are short for that reason), and goes through `plan_slices`, so that a launch is of `triton_model.ELEMENTS` sub-sample elements; the slices of a call are recorded. A session that wants the device alone takes it before it starts; the script checks and does not take a lock.

## Hardware and versions

`results.json` has the CPU, the cores allowed to the process, the load average at the start, the GPU and its runtime, the version of python, numpy, numba, torch, triton, TRICERATOPS and PyTransit, the seed and the command. Run the session on every logical CPU with `NUMBA_NUM_THREADS` set to their number: the scan pins the whole process, numba and torch workers included, for each thread count (`cpus_for`: one thread to a core up to the number of cores, in the order of the core numbers, and every logical CPU above it, where the second thread of a core is used) and the rows run at the fastest count with that pinning, the device row's host side included. The machine of the sessions has CPU n and CPU n + 16 as the two threads of core n and two groups of 8 cores with their own cache (cores 0 to 7 and 8 to 15), so a count up to 8 stays in one group. `results.json` records the CPUs of each count.

## Commands

```
env NUMBA_NUM_THREADS=32 OMP_NUM_THREADS=32 uv run --group rocm python benchmarks/bench.py --json session.json
uv run python benchmarks/bench.py --publish session.json
uv run python benchmarks/tables.py
```

A small session on the CPU, as the tests run it: `uv run --group cpu python benchmarks/bench.py --draws 60 --points 30 --repeats 2 --threads 1 --cpu-only`.
