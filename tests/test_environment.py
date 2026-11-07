"""The thread count of the BLAS under the tests."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

CONFTEST = Path(__file__).resolve().parent / "conftest.py"


def _value_after_conftest(**environment):
    base = {key: value for key, value in os.environ.items() if key != "OPENBLAS_NUM_THREADS"}
    code = f"import os, runpy; runpy.run_path({str(CONFTEST)!r}); print(os.environ['OPENBLAS_NUM_THREADS'])"
    return subprocess.run([sys.executable, "-c", code], env={**base, **environment}, capture_output=True, text=True, check=True).stdout.strip()


def test_the_conftest_sets_one_blas_thread_by_default_and_keeps_the_count_a_user_set():
    assert _value_after_conftest() == "1"
    assert _value_after_conftest(OPENBLAS_NUM_THREADS="4") == "4"


def test_the_blas_of_this_process_runs_on_the_count_in_the_environment():
    threadpoolctl = pytest.importorskip("threadpoolctl")
    import numpy as np

    np.linalg.eigvalsh(np.eye(4))  # threadpoolctl lists a BLAS only once numpy has loaded it
    counts = {pool["num_threads"] for pool in threadpoolctl.threadpool_info() if pool["internal_api"] == "openblas"}
    assert counts == {int(os.environ["OPENBLAS_NUM_THREADS"])}, counts
