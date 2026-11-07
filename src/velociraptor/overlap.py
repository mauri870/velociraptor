"""TRICERATOPS 1.1.0 with the sampling of one scenario overlapped with the likelihood of the one before it.

    from velociraptor.overlap import overlapped_calc_probs
    with overlapped_calc_probs(window=2):
        target.calc_probs(...)

calc_probs calls lnZ_TTP, lnZ_TEB, ... in a fixed order and uses each result at once. Each of these functions draws all its random numbers from numpy's global generator before its first likelihood call and none after it. The draws of the next function can therefore start once the likelihood of the one before has begun.

Inside the block calc_probs runs twice. The first run uses stand-ins that record the calls (name and arguments) and return zeros. The second run is real: `window` worker threads run the recorded calls in order, each starting when the call before it has entered its first likelihood function, and the real run receives their results in the order of the calls. The draws, the order in which they are made and the results are those of the sequential run, bit for bit.

The sequential calc_probs runs instead, with the generator state restored, when any of these holds: the real run's calls differ from the recorded ones (name or arguments), a function drew a number after its likelihood call, the stand-ins do not suit the installed calc_probs, or the first run changed the state of the generator. An exception in a function is raised at the call it belongs to, and the generator is left in the state the sequential run would have. No thread is running when the block's calc_probs returns or raises.
"""
import contextlib
import hashlib
import io
import math
import threading
import time
import warnings

import numpy as np

SUSPENDED = [0]  # depth of the sequential() blocks, inside which calc_probs runs one call at a time
LAST = {}  # what the latest calc_probs did: the outcome ("overlapped" or the reason of the fallback), the calls recorded and the seconds of the recording


class _Fallback(Exception):
    """The overlapped run cannot stand for the sequential one."""


class _StandIn(dict):
    """What a scenario function returns during the recording: every key a calc_probs reads is an array of one zero; the log-evidence is a scalar."""

    def __missing__(self, key):
        return np.zeros(1)


def _fingerprint():
    state = np.random.get_state()
    return hashlib.sha1(state[1].tobytes() + repr(state[2:]).encode()).digest()


def _same(first, second):
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return isinstance(first, np.ndarray) and isinstance(second, np.ndarray) and first.shape == second.shape and first.dtype == second.dtype and bool(np.array_equal(first, second, equal_nan=first.dtype.kind == "f"))
    if isinstance(first, float) and isinstance(second, float) and math.isnan(first):
        return math.isnan(second)
    try:
        return type(first) is type(second) and bool(first == second)
    except Exception:  # noqa: BLE001 - any object may refuse ==; two such objects are the same only if they are one object
        return first is second


class _Call:
    def __init__(self, index, name, function, args, kwargs):
        self.index, self.name, self.function, self.args, self.kwargs = index, name, function, args, kwargs
        self.released, self.done = threading.Event(), threading.Event()
        self.result = self.error = self.start_state = None


class _Pipeline:
    def __init__(self, calls, window, likelihood_names):
        self.calls, self.window, self.likelihood_names = calls, window, likelihood_names
        self.lock = threading.Lock()
        self.next = 0
        self.cancelled = False
        self.failed_at = len(calls)
        self.invalid = False
        self.local = threading.local()
        self.threads = []

    def _entered_likelihood(self):
        call = getattr(self.local, "call", None)
        if call is not None:
            call.released.set()

    def guarded(self):
        """The generator's functions of numpy.random with a check. A draw by a thread that is outside the part of a scenario function before its first likelihood call marks the run invalid."""
        from numpy.random import mtrand

        def guard(function):
            def guarded(*args, **kwargs):
                call = getattr(self.local, "call", None)
                if call is None or call.released.is_set():
                    self.invalid = True
                return function(*args, **kwargs)

            return guarded

        return {name: guard(getattr(np.random, name)) for name in dir(np.random) if getattr(getattr(np.random, name), "__self__", None) is mtrand._rand and name not in ("get_state", "set_state", "seed")}

    def _work(self):
        while True:
            with self.lock:
                if self.cancelled or self.next >= len(self.calls):
                    return
                call = self.calls[self.next]
                self.next += 1
            if call.index:
                self.calls[call.index - 1].released.wait()
            if self.cancelled or self.failed_at < call.index:
                call.released.set()
                call.done.set()
                continue
            call.start_state = np.random.get_state()
            self.local.call = call
            try:
                call.result = call.function(*call.args, **call.kwargs)
            except BaseException as error:  # noqa: BLE001 - kept on the call and raised again in the thread that replays it
                call.error = error
                with self.lock:
                    self.failed_at = min(self.failed_at, call.index)
            finally:
                self.local.call = None
                call.released.set()
                call.done.set()

    def start(self):
        self.threads = [threading.Thread(target=self._work, name=f"scenario-{n}") for n in range(self.window)]
        for thread in self.threads:
            thread.start()

    def stop(self):
        """Cancel the calls that have not begun and wait for those that have."""
        with self.lock:
            self.cancelled = True
        for call in self.calls:
            call.released.set()  # a worker waiting for a call that will not run
        for thread in self.threads:
            thread.join()

    def generator_after(self, index):
        """The state numpy's generator has after the draws of the calls up to `index`: the state at which the next call began, or None when it has not begun."""
        following = self.calls[index + 1] if index + 1 < len(self.calls) else None
        return following.start_state if following is not None and following.start_state is not None else None


def _scenario_names(package):
    return sorted(name for name in dir(package) if name.startswith("lnZ_"))


def _recording(package, names):
    calls = []

    def stand_in(name):
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))
            result = _StandIn(lnZ=0.0)
            if name.endswith("EB"):
                return result, _StandIn(lnZ=0.0)
            if name.endswith("TP"):
                return result
            raise _Fallback(f"{name} is a scenario function whose result this module does not know")

        return call

    return calls, {name: stand_in(name) for name in names}


@contextlib.contextmanager
def _patched(module, replacements):
    originals = {name: getattr(module, name) for name in replacements}
    for name, value in replacements.items():
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, value in originals.items():
            setattr(module, name, value)


@contextlib.contextmanager
def sequential():
    """Inside the block an overlapped calc_probs runs the package's own, one call at a time. A caller that seeds numpy's global generator itself (velociraptor.allocation.batches) needs this, because overlapped calls draw in threads on that one generator and would cut into each other's draws."""
    SUSPENDED[0] += 1
    try:
        yield
    finally:
        SUSPENDED[0] -= 1


def overlapped_calc_probs(window=2):
    """A context manager: inside it `triceratops.triceratops.target.calc_probs` is the overlapped one (see the module), with at most `window` calls in flight. A window under 2 leaves the package's function in place."""

    @contextlib.contextmanager
    def manager():
        from triceratops import marginal_likelihoods as likelihoods
        from triceratops import triceratops as package

        original = package.target.calc_probs
        if window < 2:
            yield
            return

        def calc_probs(self, *args, **kwargs):
            if SUSPENDED[0]:
                LAST.update(outcome="sequential: the caller seeds the generator itself")
                return original(self, *args, **kwargs)
            names = _scenario_names(package)
            stock_state = np.random.get_state()
            started_from = _fingerprint()
            began = time.perf_counter()
            try:
                calls, stand_ins = _recording(package, names)
                with _patched(package, stand_ins), warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                    warnings.simplefilter("ignore")
                    original(self, *args, **kwargs)
                if not calls or _fingerprint() != started_from:
                    raise _Fallback("the recording drew random numbers or found no scenario call")
            except Exception as error:  # noqa: BLE001 - fall back to the sequential run, which raises its own error if the call is bad
                LAST.update(outcome=f"recording: {error!r}")
                np.random.set_state(stock_state)
                return original(self, *args, **kwargs)
            except BaseException:
                np.random.set_state(stock_state)
                raise
            LAST.update(outcome="overlapped", calls=len(calls), recording_seconds=time.perf_counter() - began)
            real = {name: getattr(package, name) for name in names}
            pipeline = _Pipeline([_Call(n, name, real[name], args_, kwargs_) for n, (name, args_, kwargs_) in enumerate(calls)], window, likelihood_names=("lnL_TP_p", "lnL_EB_p", "lnL_EB_twin_p"))
            cursor = [0]

            def replay(name):
                def call(*args_, **kwargs_):
                    n = cursor[0]
                    cursor[0] += 1
                    if n >= len(pipeline.calls) or pipeline.calls[n].name != name or len(args_) != len(pipeline.calls[n].args) or kwargs_.keys() != pipeline.calls[n].kwargs.keys() or not all(_same(a, b) for a, b in zip(args_, pipeline.calls[n].args)) or not all(_same(kwargs_[k], pipeline.calls[n].kwargs[k]) for k in kwargs_):
                        raise _Fallback(f"call {n} ({name}) is not the recorded one")
                    pending = pipeline.calls[n]
                    pending.done.wait()
                    if pipeline.invalid:
                        raise _Fallback("a scenario function drew random numbers after its likelihood call")
                    if pending.error is not None:
                        raise pending.error
                    return pending.result

                return call

            wrapped = {}
            for likelihood in pipeline.likelihood_names:
                inner = getattr(likelihoods, likelihood)

                def hooked(*args_, _inner=inner, **kwargs_):
                    pipeline._entered_likelihood()
                    return _inner(*args_, **kwargs_)

                wrapped[likelihood] = hooked
            outcome = None
            try:
                with _patched(likelihoods, wrapped), _patched(np.random, pipeline.guarded()):
                    pipeline.start()
                    with _patched(package, {name: replay(name) for name in names}):
                        original(self, *args, **kwargs)
                if cursor[0] != len(pipeline.calls):
                    raise _Fallback("the real run made fewer scenario calls than the recording")
                if pipeline.invalid:
                    raise _Fallback("a scenario function drew random numbers after its likelihood call")
            except _Fallback as error:
                outcome = "fallback"
                LAST.update(outcome=f"fallback: {error}")
            except BaseException as error:
                pipeline.stop()
                failed = next((call for call in pipeline.calls if call.error is error), None)
                if failed is not None:
                    state = pipeline.generator_after(failed.index)
                    if state is not None:
                        np.random.set_state(state)
                raise
            finally:
                pipeline.stop()
            if outcome == "fallback":
                np.random.set_state(stock_state)
                return original(self, *args, **kwargs)

        package.target.calc_probs = calc_probs
        try:
            yield
        finally:
            package.target.calc_probs = original

    return manager()
