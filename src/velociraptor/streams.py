"""TRICERATOPS 1.1.0 with a random stream of its own for each scenario call.

    from velociraptor.streams import scenario_streams
    with scenario_streams(seed=7):
        target.calc_probs(time=time, flux_0=flux, flux_err_0=flux_err, P_orb=period, N=100000)

TRICERATOPS draws the planet and star parameters of every scenario from one global NumPy stream. The numbers a scenario receives depend on every call before it, so changing the field or leaving one scenario out changes the lnZ of all the others. Inside the block each `lnZ_*` function that `calc_probs` calls is preceded by `np.random.seed` with a seed from (`seed`, the function's name, the index of the call among the calls of that name). Each call then has the same draws whatever the other calls do. The draws differ from those of the global stream. Outside the block TRICERATOPS is unchanged.
"""
import collections
import contextlib
import importlib
import zlib

import numpy as np

MODULE = "triceratops.triceratops"


def call_seed(seed, name, index):
    """The integer seed of call number `index` of the function `name` under `seed`; the same three arguments give the same seed."""
    return int(np.random.SeedSequence([seed, zlib.crc32(name.encode()), index]).generate_state(1)[0])


@contextlib.contextmanager
def scenario_streams(seed, module=None):
    """Replace the `lnZ_*` functions of `module` (default triceratops.triceratops, where calc_probs finds them) by ones that seed the global stream from (seed, name, index of the call) before each call; the package's own are put back on leaving the block."""
    module = module or importlib.import_module(MODULE)
    names = [name for name in dir(module) if name.startswith("lnZ_") and callable(getattr(module, name))]
    original = {name: getattr(module, name) for name in names}
    counts = collections.Counter()

    def wrap(name):
        def wrapped(*arguments, **keywords):
            index = counts[name]
            counts[name] += 1
            np.random.seed(call_seed(seed, name, index))
            return original[name](*arguments, **keywords)

        return wrapped

    for name in names:
        setattr(module, name, wrap(name))
    try:
        yield
    finally:
        for name, function in original.items():
            setattr(module, name, function)
