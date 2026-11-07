"""calc_probs with overlapped scenarios gives the sequential results, draws in the same order and the same final state of numpy's generator. An exception surfaces at its call, no thread is left running, and a run that cannot reproduce the sequential one falls back to it."""
import sys
import threading
import time
import types

import numpy as np
import pytest

from velociraptor.overlap import LAST, overlapped_calc_probs


class Fakes:
    """A stand-in for triceratops.triceratops and triceratops.marginal_likelihoods. Its scenario functions draw, call a likelihood function and compute. Its calc_probs calls them in a fixed order and uses each result at once."""

    def __init__(self, monkeypatch, draw_after=None, fail_at=None, sequence=None):
        self.likelihoods = types.ModuleType("triceratops.marginal_likelihoods")
        self.package = types.ModuleType("triceratops.triceratops")
        self.root = types.ModuleType("triceratops")
        self.root.marginal_likelihoods, self.root.triceratops = self.likelihoods, self.package
        self.draw_after, self.fail_at, self.sequence = draw_after, fail_at, sequence
        self.likelihoods.lnL_TP_p = self.likelihood
        self.likelihoods.lnL_EB_p = self.likelihood
        self.likelihoods.lnL_EB_twin_p = self.likelihood
        self.live = []
        for name in ("lnZ_TTP", "lnZ_TEB", "lnZ_PTP", "lnZ_PEB"):
            setattr(self.package, name, self.scenario(name))
        fakes = self

        class target:
            def calc_probs(self, order=("lnZ_TTP", "lnZ_TEB", "lnZ_PTP", "lnZ_PEB") * 3):
                fakes.calls = getattr(fakes, "calls", 0) + 1
                self.lnZ = []
                for position, name in enumerate(order):
                    function = getattr(fakes.package, name)
                    if name.endswith("EB"):
                        first, second = function(position if fakes.sequence is None else fakes.sequence(fakes.calls, position))
                        self.lnZ += [first["lnZ"], second["lnZ"]]
                    else:
                        self.lnZ.append(function(position if fakes.sequence is None else fakes.sequence(fakes.calls, position))["lnZ"])

        self.package.target = target
        for name, module in (("triceratops", self.root), ("triceratops.marginal_likelihoods", self.likelihoods), ("triceratops.triceratops", self.package)):
            monkeypatch.setitem(sys.modules, name, module)

    def likelihood(self, values):
        time.sleep(0.002)
        self.live.append(threading.current_thread().name)
        return values.sum()

    def scenario(self, name):
        def function(position):
            draws = np.random.rand(500)
            draws = draws + np.random.uniform(size=500)
            value = float(self.likelihoods.lnL_TP_p(draws))
            if self.draw_after == name:
                value += float(np.random.rand())
            if self.fail_at == (name, position):
                raise ValueError(f"{name} at {position}")
            result = {"lnZ": value + position}
            return (result, {"lnZ": value - position}) if name.endswith("EB") else result

        return function


def run(fakes, window):
    np.random.seed(7)
    probe = fakes.package.target()
    if window:
        with overlapped_calc_probs(window):
            probe.calc_probs()
    else:
        probe.calc_probs()
    return probe.lnZ, np.random.get_state()


def same_state(first, second):
    return first[0] == second[0] and np.array_equal(first[1], second[1]) and first[2:] == second[2:]


def no_scenario_threads():
    return not [thread for thread in threading.enumerate() if thread.name.startswith("scenario-")]


@pytest.mark.parametrize("window", [2, 3, 4])
def test_results_and_generator_state_are_the_sequential_runs(monkeypatch, window):
    fakes = Fakes(monkeypatch)
    expected = run(fakes, 0)
    got = run(fakes, window)
    assert LAST["outcome"] == "overlapped", LAST
    assert got[0] == expected[0], "a log-evidence differs"
    assert same_state(got[1], expected[1]), "numpy's generator is not where the sequential run leaves it"
    assert no_scenario_threads()


def test_a_window_under_two_leaves_the_package_function(monkeypatch):
    fakes = Fakes(monkeypatch)
    before = fakes.package.target.calc_probs
    with overlapped_calc_probs(1):
        assert fakes.package.target.calc_probs is before
    with overlapped_calc_probs(2):
        assert fakes.package.target.calc_probs is not before
    assert fakes.package.target.calc_probs is before


def test_an_exception_surfaces_at_its_call_with_the_generator_where_the_sequential_run_leaves_it(monkeypatch):
    fakes = Fakes(monkeypatch, fail_at=("lnZ_PTP", 6))
    with pytest.raises(ValueError, match="lnZ_PTP at 6"):
        run(fakes, 0)
    sequential = np.random.get_state()
    with pytest.raises(ValueError, match="lnZ_PTP at 6"):
        run(fakes, 4)
    assert same_state(np.random.get_state(), sequential), "the exception left the generator where later calls had moved it"
    assert no_scenario_threads()


def test_an_exception_of_the_consumer_leaves_no_thread_running(monkeypatch):
    fakes = Fakes(monkeypatch)

    def sequence(call, position):
        if call == 2 and position == 5:
            raise KeyboardInterrupt  # what the ring guard raises in the main thread
        return position

    fakes.sequence = sequence
    np.random.seed(7)
    with pytest.raises(KeyboardInterrupt):
        with overlapped_calc_probs(3):
            fakes.package.target().calc_probs()
    assert no_scenario_threads()


def test_a_draw_after_the_likelihood_call_gives_up_the_overlap_and_the_sequential_result(monkeypatch):
    fakes = Fakes(monkeypatch, draw_after="lnZ_TEB")
    expected = run(fakes, 0)
    got = run(fakes, 3)
    assert LAST["outcome"].startswith("fallback"), LAST
    assert got[0] == expected[0]
    assert same_state(got[1], expected[1])
    assert no_scenario_threads()


def test_calls_that_differ_between_the_recording_and_the_run_give_up_the_overlap(monkeypatch):
    fakes = Fakes(monkeypatch, sequence=lambda call, position: position + (100 if call == 2 else 0))
    np.random.seed(7)
    probe = fakes.package.target()
    with overlapped_calc_probs(3):
        probe.calc_probs()
    overlapped = (probe.lnZ, np.random.get_state())
    assert LAST["outcome"].startswith("fallback"), LAST
    fakes.sequence = None
    np.random.seed(7)
    sequential = fakes.package.target()
    sequential.calc_probs()
    assert overlapped[0] == sequential.lnZ and same_state(overlapped[1], np.random.get_state())
    assert no_scenario_threads()


def test_the_scenarios_run_in_more_than_one_thread(monkeypatch):
    fakes = Fakes(monkeypatch)
    run(fakes, 3)
    assert len(set(fakes.live)) > 1, "the overlap ran every scenario in one thread"
