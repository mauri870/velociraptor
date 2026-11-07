"""The per-call random streams of velociraptor.streams, on a stand-in for triceratops.triceratops whose scenario functions draw from the global stream."""
import types

import numpy as np

from velociraptor.streams import call_seed, scenario_streams


def field(skip=()):
    """A module with three scenario functions of the kind calc_probs calls; the ones named in `skip` do not draw (a scenario left out)."""

    def draws(name):
        def function(count=4):
            return {"lnZ": float(np.log(np.mean(np.random.random(count)))) if name not in skip else float("-inf")}

        return function

    module = types.SimpleNamespace(**{name: draws(name) for name in ("lnZ_TTP", "lnZ_TEB", "lnZ_PTP")})
    module.helper = lambda: None
    return module


def run(module, seed=7, streams=True):
    """The lnZ of the calls TTP, TEB, TEB, PTP in the order calc_probs makes them."""
    np.random.seed(seed)
    calls = (module.lnZ_TTP, module.lnZ_TEB, module.lnZ_TEB, module.lnZ_PTP)
    if streams:
        with scenario_streams(seed, module=module):
            return [f()["lnZ"] for f in (module.lnZ_TTP, module.lnZ_TEB, module.lnZ_TEB, module.lnZ_PTP)]
    return [f()["lnZ"] for f in calls]


def test_a_call_that_is_left_out_changes_no_other_call_with_streams_and_every_later_call_without():
    complete, left_out = field(), field(skip=("lnZ_TTP",))
    with_streams, with_streams_left_out = run(complete), run(left_out)
    assert with_streams[1:] == with_streams_left_out[1:] and with_streams[0] != with_streams_left_out[0]
    stock, stock_left_out = run(complete, streams=False), run(left_out, streams=False)
    assert all(a != b for a, b in zip(stock[1:], stock_left_out[1:]))  # the global stream shifts every call after the one that drew nothing


def test_the_calls_of_one_name_get_different_streams_and_the_same_call_gets_the_same_stream_in_every_run():
    first, second = run(field()), run(field())
    assert first == second
    assert first[1] != first[2]  # two calls of lnZ_TEB
    assert run(field(), seed=8) != first


def test_the_functions_are_put_back_and_the_stock_stream_is_untouched_outside_the_block():
    module = field()
    before = (module.lnZ_TTP, module.lnZ_TEB, module.lnZ_PTP)
    stock = run(module, streams=False)
    with scenario_streams(3, module=module):
        assert module.lnZ_TTP is not before[0]
    assert (module.lnZ_TTP, module.lnZ_TEB, module.lnZ_PTP) == before
    assert run(module, streams=False) == stock  # the stock run after a block is the stock run before it


def test_a_name_that_is_not_a_scenario_function_is_left_alone_and_the_seed_is_a_function_of_its_three_arguments():
    module = field()
    helper = module.helper
    with scenario_streams(1, module=module):
        assert module.helper is helper
    assert call_seed(1, "lnZ_TTP", 0) == call_seed(1, "lnZ_TTP", 0)
    assert len({call_seed(1, "lnZ_TTP", 0), call_seed(1, "lnZ_TTP", 1), call_seed(1, "lnZ_TEB", 0), call_seed(2, "lnZ_TTP", 0)}) == 4
