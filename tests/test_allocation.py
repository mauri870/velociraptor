"""The batches of velociraptor.allocation, with stub scenario functions in place of TRICERATOPS."""

import math
import types

import numpy as np
import pytest

from velociraptor.allocation import Bank, batches, log_mean_exp


def stubs():
    calls = []

    def lnz_single(n):
        calls.append(n)
        return {"lnZ": float(np.log(np.mean(np.random.rand(n)))), "tag": "single"}

    def lnz_pair(n):
        calls.append(n)
        draws = np.random.rand(n)
        return {"lnZ": float(np.log(draws.mean())), "tag": "a"}, {"lnZ": float(np.log(draws.max())), "tag": "b"}

    namespace = types.SimpleNamespace(lnZ_one=lnz_single, lnZ_two=lnz_pair, other=len)
    return namespace, calls


def run(namespace, bank, plan, seed=3):
    with batches(bank, plan, seed, module=namespace, source=namespace):
        return namespace.lnZ_one(50), namespace.lnZ_two(50), namespace.lnZ_one(50)


def test_a_call_gives_the_log_of_the_mean_evidence_of_its_batches_and_the_first_batchs_result():
    namespace, _ = stubs()
    bank = Bank()
    one, (a, b), _ = run(namespace, bank, lambda call: 4)
    assert one["lnZ"] == pytest.approx(log_mean_exp([bank.lnz[(0, k)][0] for k in range(4)]), abs=1e-15)
    assert (a["lnZ"], b["lnZ"]) == pytest.approx(tuple(log_mean_exp([bank.lnz[(1, k)][j] for k in range(4)]) for j in (0, 1)), abs=1e-15)
    assert (one["tag"], a["tag"], b["tag"]) == ("single", "a", "b")
    assert bank.calls == 3


def test_the_same_seed_gives_the_same_batches_and_a_larger_plan_computes_only_the_new_ones():
    namespace, calls = stubs()
    bank = Bank()
    run(namespace, bank, lambda call: 2)
    assert len(calls) == 6
    again, again_calls = stubs()
    other = Bank()
    run(again, other, lambda call: 2)
    assert other.lnz == bank.lnz
    run(namespace, bank, lambda call: 3 if call == 1 else 2)
    assert len(calls) == 7, "only the third batch of the second call is new"
    assert len(bank.lnz) == 7
    assert bank.lnz[(0, 0)] == other.lnz[(0, 0)]


def test_the_package_functions_are_put_back_and_a_plan_below_one_batch_is_refused():
    namespace, _ = stubs()
    before = (namespace.lnZ_one, namespace.lnZ_two)
    with pytest.raises(ValueError, match="0 batches"):
        run(namespace, Bank(), lambda call: 0)
    assert (namespace.lnZ_one, namespace.lnZ_two) == before
    assert namespace.other is len


def test_batches_with_no_valid_draw_give_minus_infinity_and_do_not_poison_the_mean():
    assert log_mean_exp([-math.inf, -math.inf]) == -math.inf
    assert log_mean_exp([-math.inf, 0.0]) == pytest.approx(math.log(0.5))


def likelihood_stubs(spread):
    """A source namespace whose lnZ functions call a likelihood as TRICERATOPS does. lnL_one returns half the chi-squared of each draw, so the weight of a draw is exp(-value), and lnZ is the log of the mean weight."""
    namespace = types.SimpleNamespace()

    def lnl_one(n):
        return namespace.shape(n)

    def lnz_one(n):
        half = namespace.lnL_one(n)
        return {"lnZ": log_mean_exp(-half)}

    namespace.shape = lambda n: spread * np.random.rand(n)
    namespace.lnL_one = lnl_one
    namespace.lnZ_one = lnz_one
    return namespace


def test_the_effective_sample_size_of_a_call_pools_its_batches_and_is_the_squared_sum_over_the_sum_of_squares_of_the_weights():
    from velociraptor.allocation import effective_sample_size

    namespace = likelihood_stubs(spread=2.0)
    bank = Bank()
    with batches(bank, lambda call: 3, seed=5, module=namespace, source=namespace):
        namespace.lnZ_one(40)
    redo = likelihood_stubs(spread=2.0)
    draws = []
    for batch in range(3):
        np.random.seed(int(np.random.SeedSequence([5, 0, batch]).generate_state(1)[0]))
        draws.append(np.exp(-redo.shape(40)))
    weights = np.concatenate(draws)
    assert effective_sample_size(bank, 0) == pytest.approx((weights.sum() ** 2 / (weights**2).sum(),), rel=1e-12)


def test_one_draw_that_carries_the_call_gives_an_effective_sample_size_of_one_and_equal_weights_give_the_draw_count():
    from velociraptor.allocation import effective_sample_size

    namespace = likelihood_stubs(spread=0.0)
    bank = Bank()
    with batches(bank, lambda call: 2, seed=1, module=namespace, source=namespace):
        namespace.lnZ_one(50)
    assert effective_sample_size(bank, 0) == pytest.approx((100.0,))
    namespace.shape = lambda n: np.concatenate([[0.0], np.full(n - 1, 400.0)])
    other = Bank()
    with batches(other, lambda call: 2, seed=1, module=namespace, source=namespace):
        namespace.lnZ_one(50)
    assert effective_sample_size(other, 0)[0] == pytest.approx(2.0, abs=1e-12)  # one dominant draw in each of the two batches


def test_draws_that_are_not_finite_carry_no_weight_and_a_call_with_none_has_a_sample_size_of_zero():
    from velociraptor.allocation import effective_sample_size

    namespace = likelihood_stubs(spread=0.0)
    namespace.shape = lambda n: np.concatenate([np.zeros(n // 2), np.full(n - n // 2, np.inf)])
    bank = Bank()
    with batches(bank, lambda call: 1, seed=1, module=namespace, source=namespace):
        namespace.lnZ_one(10)
    assert effective_sample_size(bank, 0) == pytest.approx((5.0,))
    namespace.shape = lambda n: np.full(n, np.inf)
    empty = Bank()
    with batches(empty, lambda call: 1, seed=1, module=namespace, source=namespace):
        namespace.lnZ_one(10)
    assert effective_sample_size(empty, 0) == (0.0,)


def test_a_call_whose_likelihood_was_not_seen_has_no_sample_size_and_the_likelihoods_are_put_back():
    from velociraptor.allocation import effective_sample_size

    namespace, _ = stubs()
    bank = Bank()
    run(namespace, bank, lambda call: 2)
    assert effective_sample_size(bank, 0) is None
    shaped = likelihood_stubs(spread=1.0)
    before = shaped.lnL_one
    with batches(Bank(), lambda call: 1, seed=1, module=shaped, source=shaped):
        assert shaped.lnL_one is not before
    assert shaped.lnL_one is before


def toy_run(namespace):
    """Stands for calc_probs: FPP and NFPP from the evidences of the two calls, as ratios of exp(lnZ)."""
    first, second = namespace.lnZ_one(30)["lnZ"], namespace.lnZ_two(30)[0]["lnZ"]
    evidence = np.exp([first, second])
    return float(evidence[0] / evidence.sum()), float(evidence[1] / (evidence[0] + 1.0))


def pair_namespace():
    namespace = types.SimpleNamespace()
    namespace.lnZ_one = lambda n: {"lnZ": float(np.log(np.mean(np.random.rand(n))))}
    namespace.lnZ_two = lambda n: ({"lnZ": float(np.log(np.mean(np.random.rand(n)) * 2))}, {"lnZ": 0.0})
    return namespace


def test_the_error_of_a_fixed_number_of_batches_is_the_scatter_of_the_value_of_each_batch_over_the_root_of_their_count():
    from velociraptor.allocation import batch_estimates

    namespace = pair_namespace()
    bank = Bank()
    with batches(bank, lambda call: 5, seed=2, module=namespace, source=namespace):
        pooled = toy_run(namespace)
    estimate = batch_estimates(bank, lambda: toy_run(namespace), module=namespace, source=namespace)
    each = np.array([estimate["batches"][b] for b in range(5)])
    assert estimate["count"] == 5
    assert estimate["pooled"] == pytest.approx(pooled, abs=1e-15)
    assert estimate["error"] == pytest.approx(each.std(axis=0, ddof=1) / np.sqrt(5), rel=1e-12)
    assert (estimate["error"] > 0).all()


def test_the_value_of_a_batch_is_what_a_run_of_that_batch_alone_gives():
    from velociraptor.allocation import batch_estimates

    namespace = pair_namespace()
    bank = Bank()
    with batches(bank, lambda call: 3, seed=2, module=namespace, source=namespace):
        toy_run(namespace)
    estimate = batch_estimates(bank, lambda: toy_run(namespace), module=namespace, source=namespace)
    for batch in range(3):
        one, two = bank.lnz[(0, batch)][0], bank.lnz[(1, batch)][0]
        evidence = np.exp([one, two])
        assert estimate["batches"][batch] == pytest.approx((evidence[0] / evidence.sum(), evidence[1] / (evidence[0] + 1.0)), abs=1e-15)


def test_an_error_needs_two_batches_and_a_bank_that_does_not_hold_the_same_count_for_every_call_is_refused():
    from velociraptor.allocation import batch_estimates

    namespace = pair_namespace()
    bank = Bank()
    with batches(bank, lambda call: 1, seed=2, module=namespace, source=namespace):
        toy_run(namespace)
    with pytest.raises(ValueError, match="at least 2 batches"):
        batch_estimates(bank, lambda: toy_run(namespace), module=namespace, source=namespace)
    uneven = Bank()
    with batches(uneven, lambda call: 2 + call, seed=2, module=namespace, source=namespace):
        toy_run(namespace)
    with pytest.raises(ValueError, match="same number of batches"):
        batch_estimates(uneven, lambda: toy_run(namespace), module=namespace, source=namespace)


def test_two_calls_started_at_once_in_threads_keep_the_weights_of_their_own_likelihood():
    """The calls of an overlapped calc_probs start in threads, and the bank runs them one at a time because the generator is global. The weights of each likelihood evaluation must go to the call whose function made it, whichever thread gets in first."""
    import threading

    from velociraptor.allocation import effective_sample_size

    namespace = types.SimpleNamespace()
    namespace.lnL_flat = lambda n: np.zeros(n)
    namespace.lnL_peak = lambda n: np.concatenate([[0.0], np.full(n - 1, 400.0)])
    namespace.lnZ_flat = lambda n: {"lnZ": log_mean_exp(-namespace.lnL_flat(n))}
    namespace.lnZ_peak = lambda n: {"lnZ": log_mean_exp(-namespace.lnL_peak(n))}
    bank = Bank()
    with batches(bank, lambda call: 1, seed=1, module=namespace, source=namespace):
        threads = [threading.Thread(target=namespace.lnZ_flat, args=(50,)), threading.Thread(target=namespace.lnZ_peak, args=(50,))]
        threads[0].start()
        while not bank.lnz and threads[0].is_alive():
            pass  # the first call has taken the lock and the number 0
        threads[1].start()
        for thread in threads:
            thread.join(10)
    sizes = {call: effective_sample_size(bank, call)[0] for call in (0, 1)}
    assert sizes[0] == pytest.approx(50.0) and sizes[1] == pytest.approx(1.0, abs=1e-9), sizes


def test_a_batch_is_the_same_draws_when_two_calls_seed_and_draw_in_threads_at_once():
    """The calls of an overlapped calc_probs run in threads on numpy's one global generator, and each call seeds it for each batch. A seeding in the middle of another call's draws must not change them, so each (call, batch) value equals the one of a run with one call at a time, bit for bit."""
    import threading
    import time

    def make():
        namespace = types.SimpleNamespace()

        def lnz(n):
            first = np.random.rand(n)
            time.sleep(0.02)  # the other call seeds and draws here
            second = np.random.rand(n)
            return {"lnZ": float(np.log(np.mean(first * second)))}

        namespace.lnZ_a = lnz
        namespace.lnZ_b = lnz
        return namespace

    reference_namespace, bank = make(), Bank()
    with batches(bank, lambda call: 4, seed=9, module=reference_namespace, source=reference_namespace):
        reference_namespace.lnZ_a(30)
        reference_namespace.lnZ_b(30)
    threaded_namespace, threaded = make(), Bank()
    with batches(threaded, lambda call: 4, seed=9, module=threaded_namespace, source=threaded_namespace):
        first = threading.Thread(target=threaded_namespace.lnZ_a, args=(30,))
        first.start()
        time.sleep(0.005)  # the first call is the first to take its number
        second = threading.Thread(target=threaded_namespace.lnZ_b, args=(30,))
        second.start()
        first.join(20)
        second.join(20)
    assert threaded.lnz == bank.lnz, (threaded.lnz, bank.lnz)


def test_the_bank_made_under_an_overlapped_calc_probs_equals_the_bank_of_one_call_at_a_time_bit_for_bit(monkeypatch):
    """When the batch loop of a call draws after its first likelihood, an overlapped calc_probs gives up and runs the calls again one at a time. What the abandoned attempt wrote into the bank must not stay there."""
    import contextlib
    import importlib
    import time

    from velociraptor.overlap import LAST, overlapped_calc_probs

    fakes = importlib.import_module("test_overlap").Fakes(monkeypatch)

    def slow(values):
        time.sleep(0.02)  # the calls that follow start their sampling meanwhile
        return values.sum()

    for name in ("lnL_TP_p", "lnL_EB_p", "lnL_EB_twin_p"):
        setattr(fakes.likelihoods, name, slow)

    def run(window):
        bank, probe = Bank(), fakes.package.target()
        np.random.seed(7)
        with (overlapped_calc_probs(window) if window else contextlib.nullcontext()), batches(bank, lambda call: 3, seed=5):
            probe.calc_probs()
        return bank.lnz, probe.lnZ

    expected = run(0)
    for window in (2, 3):
        assert run(window) == expected, f"window {window}"
        assert LAST["outcome"].startswith("sequential"), LAST  # the overlap is not attempted


def banked(*calls):
    """A bank holding, for each call, the lnZ of its batches (a list of tuples, one value or two for a call that returns a pair)."""
    bank = Bank()
    for call, batches_of_call in enumerate(calls):
        for batch, values in enumerate(batches_of_call):
            bank.lnz[(call, batch)] = tuple(values)
        first = tuple({"lnZ": value} for value in batches_of_call[0])
        bank.first[call] = first if len(first) > 1 else first[0]
    return bank


def evidence_namespace():
    """Two scenario calls; the output is the evidence of the second call alone, so only the second can carry the scatter."""
    namespace = types.SimpleNamespace()
    namespace.lnZ_one = lambda n: {"lnZ": 0.0}
    namespace.lnZ_two = lambda n: ({"lnZ": 0.0}, {"lnZ": 0.0})
    return namespace


def second_call_run(namespace):
    namespace.lnZ_one(1)
    return (float(np.exp(namespace.lnZ_two(1)[0]["lnZ"])),)


def test_the_batch_effective_sample_size_is_the_squared_sum_over_the_sum_of_squares_of_the_evidence_of_the_batches():
    from velociraptor.allocation import batch_effective_sample_size

    assert batch_effective_sample_size(banked([(0.0,)] * 4), 0) == pytest.approx(4.0)
    assert batch_effective_sample_size(banked([(0.0,), (-30.0,), (-30.0,), (-30.0,)]), 0) == pytest.approx(1.0, abs=1e-12)
    values = [(0.0,), (np.log(3.0),)]
    assert batch_effective_sample_size(banked(values), 0) == pytest.approx((1 + 3) ** 2 / (1 + 9))
    assert batch_effective_sample_size(banked([(0.0, 0.0), (0.0, 0.0)]), 0) == pytest.approx(2.0)  # a pair adds its two evidences
    assert batch_effective_sample_size(banked([(0.0,), (-math.inf,)]), 0) == pytest.approx(1.0)
    assert batch_effective_sample_size(banked([(-math.inf,), (-math.inf,)]), 0) == 0.0
    assert batch_effective_sample_size(banked([(0.0,)]), 3) is None


def test_the_call_that_carries_the_scatter_is_named_with_its_batch_effective_sample_size_and_the_ratio_of_that_to_its_batches():
    from velociraptor import allocation
    from velociraptor.allocation import evidence_report

    assert not hasattr(allocation, "MINIMUM_BATCH_ESS")
    namespace = evidence_namespace()
    steady = [(0.0,), (0.1,), (-0.1,), (0.0,)]
    rare = [(0.0, -math.inf)] + [(-30.0, -math.inf)] * 11
    report = evidence_report(banked(steady, rare), lambda: second_call_run(namespace), module=namespace, source=namespace)
    assert report["carrying_call"] == 1
    assert report["carrying_ess"] == pytest.approx(1.0, abs=1e-9)
    assert report["carrying_ratio"] == pytest.approx(1.0 / 12, abs=1e-9)
    assert report["ess"][1] == report["carrying_ess"]
    assert "unreliable" not in report
    spread = [(0.05 * ((-1) ** b), -math.inf) for b in range(12)]
    report = evidence_report(banked(steady, spread), lambda: second_call_run(namespace), module=namespace, source=namespace)
    assert report["carrying_call"] == 1 and report["carrying_ess"] > 11.5
    assert report["carrying_ratio"] == pytest.approx(report["carrying_ess"] / 12)


def test_equal_batches_give_a_ratio_of_one_at_any_count():
    from velociraptor.allocation import evidence_report

    namespace = evidence_namespace()
    steady = [(0.0,)] * 3
    for count in (9, 10, 20):
        wiggle = [(0.0 if b == 0 else 1e-12 * b, -math.inf) for b in range(count)]  # equal to 1e-12, with a scatter so that the call carries
        report = evidence_report(banked(steady, wiggle), lambda: second_call_run(namespace), module=namespace, source=namespace)
        assert report["carrying_ratio"] == pytest.approx(1.0, abs=1e-9), count
        assert report["carrying_ess"] == pytest.approx(float(count), abs=1e-9), count


def test_a_report_takes_calls_with_different_numbers_of_batches_and_leaves_the_likelihoods_as_they_were():
    from velociraptor.allocation import evidence_report

    namespace = evidence_namespace()
    before = (namespace.lnZ_one, namespace.lnZ_two)
    report = evidence_report(banked([(0.0,)] * 2, [(b / 10, -math.inf) for b in range(7)]), lambda: second_call_run(namespace), module=namespace, source=namespace)
    assert report["carrying_call"] == 1 and len(report["ess"]) == 2
    assert (namespace.lnZ_one, namespace.lnZ_two) == before


def test_an_output_that_no_batch_moves_has_no_carrying_call_and_no_ratio():
    from velociraptor.allocation import evidence_report

    namespace = evidence_namespace()
    report = evidence_report(banked([(0.0,)] * 3, [(0.0, -math.inf)] * 3), lambda: second_call_run(namespace), module=namespace, source=namespace)
    assert report["carrying_call"] is None and report["carrying_ess"] is None and report["carrying_ratio"] is None


def test_the_error_of_a_batch_estimate_comes_with_the_report_of_the_call_that_carries_it():
    from velociraptor.allocation import batch_estimates

    namespace = pair_namespace()
    bank = Bank()
    with batches(bank, lambda call: 5, seed=2, module=namespace, source=namespace):
        toy_run(namespace)
    estimate = batch_estimates(bank, lambda: toy_run(namespace), module=namespace, source=namespace)
    assert estimate["carrying_call"] in (0, 1)
    assert estimate["carrying_ess"] <= 5.0 + 1e-9
    assert 0.0 < estimate["carrying_ratio"] <= 1.0 and estimate["carrying_ratio"] == pytest.approx(estimate["carrying_ess"] / 5)
    assert "unreliable" not in estimate
