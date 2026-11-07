"""The fused kernel is launched in bounded slices only. The plan of the slices depends on the sizes alone, so no device is needed."""

import re
from pathlib import Path

import pytest

triton = pytest.importorskip("triton")
torch = pytest.importorskip("torch")

from velociraptor.triton_model import ELEMENTS, plan_slices

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("pairs,samples", [(0, 20), (1, 20), (26_214, 20), (26_215, 20), (21_067_000, 20), (51_493_000, 20), (100_000_000, 1), (5, 10_000_000)])
def test_the_slices_cover_the_pairs_once_in_order_and_none_is_larger_than_the_bound(pairs, samples):
    plan = plan_slices(pairs, samples)
    assert sum(count for _, count in plan) == pairs
    assert [first for first, _ in plan] == [sum(count for _, count in plan[:i]) for i in range(len(plan))]
    assert all(1 <= count and (count * samples <= ELEMENTS or count == 1) for _, count in plan)


def test_the_largest_calls_of_a_calculate_are_many_small_launches():
    plan = plan_slices(51_493 * 1000, 20)
    assert len(plan) >= -(-51_493_000 // (ELEMENTS // 20)) and len(plan) <= -(-51_493_000 // (ELEMENTS // 20 - 32))
    assert max(count for _, count in plan) * 20 <= ELEMENTS


def test_the_kernel_is_launched_from_one_place_and_through_the_plan():
    source = (ROOT / "src" / "velociraptor" / "triton_analytic.py").read_text()
    assert len(re.findall(r"(?<![\w.])kernel\[", source)) == 1  # the kernel of the precision, chosen just above the launch
    assert "plan_slices(total, samples)" in source.split("kernel[(")[0].split("def transit_flux_triton_analytic")[1]
    others = [path for path in (ROOT / "src").rglob("*.py") if path.name != "triton_analytic.py" and re.search(r"\b_kernel\[|_flux_kernel", path.read_text())]
    others += [path for path in (ROOT / "benchmarks").glob("*.py") if re.search(r"\b_kernel\[|_flux_kernel|_chain\[", path.read_text()) and path.name != "transcendental_cost.py"]
    assert not others, others


@pytest.mark.parametrize("pairs,samples", [(21_067_000, 20), (51_493_000, 3), (1_000_000, 1), (100_000, 20)])
def test_every_slice_but_the_last_is_a_multiple_of_32_pairs_so_that_the_blocks_do_not_depend_on_the_bound(pairs, samples):
    plan = plan_slices(pairs, samples)
    assert all(count % 32 == 0 for _, count in plan[:-1])
    assert all(first % 32 == 0 for first, _ in plan)
    assert all(count * samples <= ELEMENTS for _, count in plan)


def test_the_module_still_offers_the_entry_point_that_the_likelihoods_import():
    from velociraptor.triton_model import transit_flux_triton

    assert callable(transit_flux_triton)
