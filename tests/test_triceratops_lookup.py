"""The limb-darkening lookup of the companions of lnZ_STP and lnZ_SEB runs once for each distinct pair. The scenario functions that install() puts in place return exactly the numbers of the stock functions on the same draws."""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("triceratops")


def stock_inputs():
    time = np.linspace(-0.3, 0.3, 120)
    return dict(time=time, flux=np.ones_like(time), sigma=1e-3, P_orb=5.0, M_s=1.0, R_s=1.0, Teff=5778.0, Z=0.0, plx=5.0, N=3000, parallel=True)


def equal(first, second):
    if isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            equal(first[key], second[key])
    elif isinstance(first, tuple):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            equal(a, b)
    else:
        assert np.array_equal(np.asarray(first), np.asarray(second), equal_nan=True), "a value differs"


@pytest.mark.parametrize("name", ["lnZ_STP", "lnZ_SEB"])
def test_the_gathered_lookup_gives_the_numbers_of_the_stock_function_exactly(name):
    """Both run with the package's own likelihood functions, so the lookup is the only difference. install() also replaces the likelihoods, whose values differ from PyTransit's by the quadrature."""
    from triceratops import marginal_likelihoods

    from velociraptor.triceratops import with_gathered_lookup

    stock = getattr(marginal_likelihoods, name)
    patched = with_gathered_lookup(stock)
    assert patched is not stock and patched.__name__ == name
    np.random.seed(267)
    expected = stock(**stock_inputs())
    np.random.seed(267)
    found = patched(**stock_inputs())
    equal(expected, found)


def test_install_puts_the_gathered_functions_where_calc_probs_finds_them_and_puts_the_stock_ones_back():
    from triceratops import marginal_likelihoods
    from triceratops import triceratops as package

    from velociraptor.triceratops import install

    stock = {name: (getattr(marginal_likelihoods, name), getattr(package, name)) for name in ("lnZ_STP", "lnZ_SEB")}
    with install("cpu"):
        for name, (module_function, package_function) in stock.items():
            assert getattr(marginal_likelihoods, name) is getattr(package, name) is not module_function
    for name, (module_function, package_function) in stock.items():
        assert getattr(marginal_likelihoods, name) is module_function and getattr(package, name) is package_function


def test_the_replacement_is_refused_on_a_source_without_the_loop_or_with_it_twice():
    from velociraptor.triceratops import (
        GATHERED_LOOKUP,
        STOCK_LOOKUP,
        with_gathered_lookup,
    )

    def one():
        pass

    with pytest.raises(RuntimeError, match="exactly once"):
        with_gathered_lookup(one)
    assert STOCK_LOOKUP.count("u1s_comp[i]") == 1 and "np.unique" in GATHERED_LOOKUP


def test_the_option_off_leaves_the_stock_functions_in_place():
    from triceratops import marginal_likelihoods
    from triceratops import triceratops as package

    from velociraptor.triceratops import install

    with install("cpu", gathered_lookup=False):
        assert marginal_likelihoods.lnZ_STP is package.lnZ_STP
    assert marginal_likelihoods.lnZ_STP is package.lnZ_STP


def test_a_spline_called_twice_on_the_same_array_is_evaluated_once_and_gives_the_same_values():
    InterpolatedUnivariateSpline = pytest.importorskip("scipy.interpolate").InterpolatedUnivariateSpline
    from velociraptor.triceratops import remembered_splines

    spline = InterpolatedUnivariateSpline(np.linspace(0, 1, 20), np.linspace(0, 1, 20) ** 2)
    x = np.random.default_rng(1).uniform(0, 1, 5000)
    original = InterpolatedUnivariateSpline.__call__
    expected = spline(x)
    with remembered_splines() as memory:
        first, second = spline(x), spline(x)
        assert np.array_equal(first, expected) and np.array_equal(second, expected)
        assert len(memory) == 1, "one evaluation for two calls"
        first[:] = 0.0
        assert np.array_equal(spline(x), expected), "a caller that changes its result does not change the next one"
        changed = x.copy()
        changed[0] += 1e-9
        assert not np.array_equal(spline(changed), expected) and len(memory) == 2, "a changed array is another key"
        small = spline(x[:10])
        assert len(memory) == 2 and np.array_equal(small, expected[:10]), "short arrays are not remembered"
    assert InterpolatedUnivariateSpline.__call__ is original, "the spline class is back as it was"


def test_install_remembers_the_splines_inside_the_block_and_not_after_it():
    InterpolatedUnivariateSpline = pytest.importorskip("scipy.interpolate").InterpolatedUnivariateSpline
    from velociraptor.triceratops import install

    original = InterpolatedUnivariateSpline.__call__
    with install("cpu"):
        assert InterpolatedUnivariateSpline.__call__ is not original
    assert InterpolatedUnivariateSpline.__call__ is original
    with install("cpu", remember_splines=False):
        assert InterpolatedUnivariateSpline.__call__ is not original  # still the evaluation without the lock
    with install("cpu", remember_splines=False, release_splines=False):
        assert InterpolatedUnivariateSpline.__call__ is original
