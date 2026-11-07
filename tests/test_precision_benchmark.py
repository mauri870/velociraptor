"""benchmarks/precision.py on a device with a few draws: one table row per scenario, float64 restored afterwards, and a shift of ln Z of the planet scenario far below the scatter of ln Z over seeds."""
import importlib.util
import json
import pathlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_the_precision_table_has_a_row_for_each_scenario_and_leaves_the_default_precision(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("the kernel needs a CUDA or ROCm device")
    from velociraptor import triton_analytic

    spec = importlib.util.spec_from_file_location("velociraptor_precision", ROOT / "benchmarks" / "precision.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "precision.json"
    module.main(["--draws", "400", "--points", "120", "--seeds", "4", "--json", str(path)])
    session = json.loads(path.read_text())
    assert set(session["table"]) == {"tp", "eb", "twin"}
    assert triton_analytic.PRECISION == "float64"
    row = session["table"]["tp"]
    assert row["largest_shift_of_ln_Z"] < 0.1 * row["scatter_of_ln_Z_over_seeds"], row
    assert {"hardware", "versions", "command"} <= set(session)


def test_at_ten_times_the_noise_the_eclipsing_binary_has_a_row_within_a_tenth_of_its_scatter(tmp_path):
    """At the default noise every draw of the eclipsing binary has a secondary eclipse of 1.5 sigma or more and an infinite likelihood, so the scenario has no row."""
    if not torch.cuda.is_available():
        pytest.skip("the kernel needs a CUDA or ROCm device")
    spec = importlib.util.spec_from_file_location("velociraptor_precision", ROOT / "benchmarks" / "precision.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "precision.json"
    module.main(["--draws", "2000", "--points", "200", "--seeds", "4", "--noise-scale", "10", "--json", str(path)])
    row = json.loads(path.read_text())["table"]["eb"]
    assert "largest_shift_of_ln_Z" in row, row
    assert row["largest_shift_of_ln_Z"] < 0.1 * row["scatter_of_ln_Z_over_seeds"], row
