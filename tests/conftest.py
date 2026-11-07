import os

# Set before numpy is imported, because numpy reads it when it loads its BLAS. With one thread per core the eigenvalue problems of the quadrature tests stall on a busy machine. A value the user set is kept.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import importlib
import importlib.util
import pathlib
import sys
import types

import numpy as np
import pytest


@pytest.fixture(scope="session", autouse=True)
def names_that_pytransit_needs():
    """PyTransit 2.2.0 imports names that current numpy, scipy and setuptools removed (np.int, np.float, np.bool, np.trapz, scipy.integrate.trapz, pkg_resources.resource_filename). This fixture restores them for the session. PyTransit is used only as a reference in the comparisons that import it."""
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy 2 renamed trapz; the oracle environment of validation/pytransit_env.md has numpy 1
    for name, alias in (("int", int), ("float", float), ("bool", bool), ("trapz", trapezoid)):
        if not hasattr(np, name):
            setattr(np, name, alias)
    try:
        integrate = importlib.import_module("scipy.integrate")
    except ImportError:
        integrate = None
    if integrate is not None and not hasattr(integrate, "trapz"):
        integrate.trapz = trapezoid
    if "pkg_resources" not in sys.modules and importlib.util.find_spec("pkg_resources") is None:
        shim = types.ModuleType("pkg_resources")
        shim.resource_filename = lambda package, name: str(pathlib.Path(importlib.util.find_spec(package).origin).parent / name)
        sys.modules["pkg_resources"] = shim


@pytest.fixture(autouse=True)
def float64_unless_the_test_asks_otherwise(request, monkeypatch):
    """The default precision is "mixed", but the goldens and references are float64 and name no precision. Each test runs in float64 unless it is marked `default_precision`. Without Triton there is no precision to set."""
    if request.node.get_closest_marker("default_precision") is not None or importlib.util.find_spec("triton") is None:
        return
    from velociraptor import triton_analytic

    monkeypatch.setattr(triton_analytic, "PRECISION", "float64")
