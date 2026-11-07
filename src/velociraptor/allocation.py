"""TRICERATOPS 1.1.0 with the number of draws of each scenario call chosen by the caller.

    from velociraptor.allocation import Bank, batches
    bank = Bank()
    with batches(bank, lambda call: 3, seed=7):
        target.calc_probs(time=time, flux_0=flux, flux_err_0=flux_err, P_orb=period, N=100000)
    bank.lnz[(0, 2)]  # the lnZ of the third batch of the first call
    effective_sample_size(bank, 0)  # of the draws of the first call, over its batches

Inside the block each `lnZ_*` function that `calc_probs` calls runs `plan(call)` batches of N draws. N is the argument `calc_probs` passes and `call` counts the calls from 0 in the order of `calc_probs`. A batch is seeded from (seed, call, batch), so it is the same draws whenever it is made. The lnZ returned for the call is the log of the mean of exp(lnZ) over its batches. The rest of the result is that of the first batch. Batches already in the bank are not computed again, so a later block with a larger plan runs only the new ones.

When every call has the same number of batches, `batch_estimates(bank, run)` runs `run` (a function that calls `calc_probs` and returns the probabilities) once for each batch alone and once for all of them pooled. It gives the scatter of the probabilities between batches as their error. The `lnL_*` functions are recorded for the effective sample size. Enter the block inside `velociraptor.triceratops.install` for this, because `install` replaces the `lnL_*` functions that were recorded.
"""
import contextlib
import importlib
import math
import threading

import numpy as np

from velociraptor.overlap import sequential

MODULE = "triceratops.triceratops"
SOURCE = "triceratops.marginal_likelihoods"


class Bank:
    """The lnZ of every batch made (`lnz[(call, batch)]` is a tuple with one value, or two for a call that returns a scenario and its twin) and the result of each call's first batch."""

    def __init__(self):
        self.lnz = {}
        self.first = {}
        self.calls = 0  # how many calls the last block made
        self.moments = {}  # (call, batch) -> [(log sum of w, log sum of w squared)] for each likelihood evaluation of the batch, w = exp(-value) the weight of a draw
        self._local = threading.local()  # the (call, batch) a thread is computing: an overlapped calc_probs runs the calls in threads that overlap in time

    @property
    def current(self):
        return getattr(self._local, "key", None)

    @current.setter
    def current(self, key):
        self._local.key = key


def log_mean_exp(values):
    """The logarithm of the mean of exp(values), computed without overflow; -inf when every value is -inf."""
    values = np.asarray(values, dtype=float)
    top = values.max()
    if top == -math.inf:
        return -math.inf
    return float(top + math.log(np.mean(np.exp(values - top))))


def _log_moments(values):
    """log of the sum of w and of w squared over the draws, w = exp(-value) for the finite values (the functions return half the chi-squared of a draw), as (-inf, -inf) when none is finite."""
    log_weight = -np.asarray(values, dtype=float).ravel()
    log_weight = log_weight[np.isfinite(log_weight)]
    if log_weight.size == 0:
        return -math.inf, -math.inf
    top = log_weight.max()
    shifted = log_weight - top
    return float(top + math.log(np.exp(shifted).sum())), float(2 * top + math.log(np.exp(2 * shifted).sum()))


def effective_sample_size(bank, call):
    """For each likelihood evaluation of a scenario call, (sum w)^2 / sum w^2 over the draws of all its batches, with w = exp(-value) for the finite values and 0 for the others; 0.0 when no draw is finite. None for a call whose likelihood was not recorded."""
    batches_of_call = [moments for (number, _), moments in bank.moments.items() if number == call and moments]
    if not batches_of_call:
        return None
    sizes = []
    for index in range(min(len(moments) for moments in batches_of_call)):
        first = log_mean_exp([moments[index][0] for moments in batches_of_call]) + math.log(len(batches_of_call))
        second = log_mean_exp([moments[index][1] for moments in batches_of_call]) + math.log(len(batches_of_call))
        sizes.append(0.0 if first == -math.inf else float(math.exp(2 * first - second)))
    return tuple(sizes)


def batch_effective_sample_size(bank, call):
    """(sum Z)^2 / sum Z^2 over the batches of a call, with Z the evidence of a batch (the exponential of its lnZ, the evidences of a pair added), so 1 when one batch carries the sum and the number of batches when they are equal. 0.0 when no batch has a finite evidence, None for a call the bank does not hold.

    This counts batches, where `effective_sample_size` counts the draws inside them: a call can have draws that are many and a few batches that carry the sum."""
    values = [bank.lnz[key] for key in sorted(bank.lnz) if key[0] == call]
    if not values:
        return None
    totals = np.array([_log_sum(v) for v in values])
    top = totals.max()
    if top == -math.inf:
        return 0.0
    weight = np.exp(totals - top)
    return float(weight.sum() ** 2 / (weight**2).sum())


def _log_sum(values):
    values = np.asarray(values, dtype=float)
    top = values.max()
    return -math.inf if top == -math.inf else float(top + math.log(np.exp(values - top).sum()))


def _combined(result, lnz):
    if isinstance(result, tuple):
        return tuple(dict(part, lnZ=value) for part, value in zip(result, lnz))
    return dict(result, lnZ=lnz[0])


def _recorded(function, bank):
    def wrapper(*arguments, **keywords):
        result = function(*arguments, **keywords)
        if bank.current is not None:
            bank.moments[bank.current].append(_log_moments(result))
        return result

    return wrapper


def _wrapped(function, counter, counter_lock, draw_lock, bank, plan, seed):
    def wrapper(*arguments, **keywords):
        with counter_lock:
            call = counter[0]
            counter[0] += 1
        count = int(plan(call))
        if count < 1:
            raise ValueError(f"call {call} was given {count} batches")
        with draw_lock:  # a batch is seeded on numpy's one global generator, so the calls of an overlapped calc_probs take turns; a seeding would otherwise cut into the draws of another call
            return _batches_of(function, call, count, bank, seed, arguments, keywords)

    return wrapper


def _batches_of(function, call, count, bank, seed, arguments, keywords):
    for batch in range(count):
        if (call, batch) in bank.lnz:
            continue
        np.random.seed(int(np.random.SeedSequence([seed, call, batch]).generate_state(1)[0]))
        bank.current = (call, batch)
        bank.moments[bank.current] = []
        result = function(*arguments, **keywords)
        parts = result if isinstance(result, tuple) else (result,)
        bank.lnz[(call, batch)] = tuple(float(part["lnZ"]) for part in parts)
        if batch == 0:
            bank.first[call] = result
    lnz = tuple(log_mean_exp([bank.lnz[(call, batch)][k] for batch in range(count)]) for k in range(len(bank.lnz[(call, 0)])))
    return _combined(bank.first[call], lnz)


@contextlib.contextmanager
def batches(bank, plan, seed, module=None, source=None):
    """Replace the `lnZ_*` functions of `module` (default triceratops.triceratops, where calc_probs finds them) by ones that run `plan(call)` batches; the package's own are put back on leaving the block."""
    module = module or importlib.import_module(MODULE)
    names = [name for name in vars(source or importlib.import_module(SOURCE)) if name.startswith("lnZ_")]
    original = {name: getattr(module, name) for name in names}
    likelihoods = {name: function for name, function in vars(source or importlib.import_module(SOURCE)).items() if name.startswith("lnL_")}
    counter, counter_lock, draw_lock = [0], threading.Lock(), threading.Lock()
    for name, function in original.items():
        setattr(module, name, _wrapped(function, counter, counter_lock, draw_lock, bank, plan, seed))
    for name, function in likelihoods.items():
        setattr(source or importlib.import_module(SOURCE), name, _recorded(function, bank))
    try:
        with sequential():
            yield
    finally:
        for name, function in original.items():
            setattr(module, name, function)
        for name, function in likelihoods.items():
            setattr(source or importlib.import_module(SOURCE), name, function)
        bank.current = None
        bank.calls = counter[0]


@contextlib.contextmanager
def _view(bank, module, source, batch, single=None):
    """Inside the block each `lnZ_*` function returns the banked lnZ of one batch (all batches pooled when `batch` is None) with the rest of the result of the first batch. With `single=(call, batch)` that call returns its one batch and every other call all of its batches pooled."""
    module = module or importlib.import_module(MODULE)
    names = [name for name in vars(source or importlib.import_module(SOURCE)) if name.startswith("lnZ_")]
    original = {name: getattr(module, name) for name in names}
    counter = [0]

    def make():
        def wrapper(*arguments, **keywords):
            call = counter[0]
            counter[0] += 1
            count = 1 + max(b for (c, b) in bank.lnz if c == call)
            if single is not None:
                parts = [single[1]] if call == single[0] else range(count)
            else:
                parts = range(count) if batch is None else ([batch] if isinstance(batch, int) else list(batch))
            width = len(bank.lnz[(call, 0)])
            return _combined(bank.first[call], tuple(log_mean_exp([bank.lnz[(call, b)][k] for b in parts]) for k in range(width)))

        return wrapper

    for name in original:
        setattr(module, name, make())
    try:
        yield
    finally:
        for name, function in original.items():
            setattr(module, name, function)


def evidence_report(bank, run, module=None, source=None):
    """Which call carries the scatter of the outputs of `run()`, and how many of its batches carry its evidence.

    For each call with at least 2 batches, `run()` is made once for each batch of that call with every other call pooled, and the variance of each output between those runs is taken. A call's share of an output is its variance over the sum of the calls' variances; the carrying call is the one with the largest mean share over the outputs that vary at all. Returns {"ess", "carrying_call", "carrying_ess", "carrying_ratio"}.

    "ess" is the batch effective sample size of each call, in call order: (sum Z)^2 / sum Z^2 over the evidences of its batches, 1 when one batch carries the sum and the number of batches when they are equal. "carrying_call" is the call named above and "carrying_ess" its entry of "ess". "carrying_ratio" is "carrying_ess" divided by the number of that call's batches, between 0 and 1. With no output that varies, the carrying call, its size and the ratio are None. The numbers are reported without a verdict. The calls may hold different numbers of batches."""
    calls = sorted({call for call, _ in bank.lnz})
    sizes = tuple(batch_effective_sample_size(bank, call) for call in calls)
    variances = {}
    for call in calls:
        count = 1 + max(b for (c, b) in bank.lnz if c == call)
        if count < 2:
            continue
        outputs = []
        for batch in range(count):
            with _view(bank, module, source, None, single=(call, batch)):
                outputs.append(tuple(run()))
        variances[call] = np.array(outputs, dtype=float).var(axis=0, ddof=1)
    if not variances:
        return {"ess": sizes, "carrying_call": None, "carrying_ess": None, "carrying_ratio": None}
    table = np.array([variances[call] for call in variances])
    total = table.sum(axis=0)
    varying = total > 0
    if not varying.any():
        return {"ess": sizes, "carrying_call": None, "carrying_ess": None, "carrying_ratio": None}
    share = (table[:, varying] / total[varying]).mean(axis=1)
    carrying = list(variances)[int(np.argmax(share))]
    size = sizes[calls.index(carrying)]
    held = sum(1 for (c, _) in bank.lnz if c == carrying)
    return {"ess": sizes, "carrying_call": carrying, "carrying_ess": size, "carrying_ratio": size / held}


def batch_estimates(bank, run, module=None, source=None):
    """The probabilities of `run()` (a function that calls `calc_probs` and returns FPP and NFPP) from each batch of the bank alone, from all of them pooled, and the error of the pooled values: the standard deviation between batches over the root of their number.

    Every call must hold the same number of batches, at least 2. The error is the scatter of a fixed number of batches. Stopping on the sample variance would leave the stated error short of its nominal coverage, so no stop rule is offered. Returns {"count", "pooled", "batches" (a dict by batch), "error", and the call that carries the scatter with its batch effective sample size and the ratio of that to its batches, from `evidence_report`}."""
    counts = {max(b for (c, b) in bank.lnz if c == call) + 1 for call in {c for (c, _) in bank.lnz}}
    if len(counts) != 1:
        raise ValueError(f"every call needs the same number of batches, the bank holds {sorted(counts)}")
    count = counts.pop()
    if count < 2:
        raise ValueError(f"an error needs at least 2 batches, the bank holds {count}")
    with _view(bank, module, source, None):
        pooled = tuple(run())
    each = {}
    for batch in range(count):
        with _view(bank, module, source, batch):
            each[batch] = tuple(run())
    values = np.array([each[batch] for batch in range(count)], dtype=float)
    report = evidence_report(bank, run, module, source)
    return {"count": count, "pooled": pooled, "batches": each, "error": values.std(axis=0, ddof=1) / math.sqrt(count), "carrying_call": report["carrying_call"], "carrying_ess": report["carrying_ess"], "carrying_ratio": report["carrying_ratio"]}


def grouped_estimates(bank, run, groups, module=None, source=None):
    """The probabilities of `run()` for each group of batches of the bank pooled (`groups` is a list of tuples of batch numbers, each batch held for every call). A group of one batch is that batch alone. No error is computed; `batch_estimates` gives it over all the batches."""
    held = {max(b for (c, b) in bank.lnz if c == call) + 1 for call in {c for (c, _) in bank.lnz}}
    count = min(held)
    results = []
    for group in groups:
        group = tuple(group)
        if not group or any(not 0 <= b < count for b in group):
            raise ValueError(f"every group needs batches the bank holds (0 to {count - 1}), got {group}")
        with _view(bank, module, source, group):
            results.append(tuple(run()))
    return results
