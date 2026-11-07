"""benchmarks/flagged.py on a device with a few draws: it writes a row for each scenario of the input of bench.py, and the pairs it counts as flagged are at most the pairs of the call."""
import importlib.util
import json
import pathlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_the_flagged_pairs_of_each_scenario_are_counted_from_a_call_on_the_device(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("the kernel needs a CUDA or ROCm device")
    spec = importlib.util.spec_from_file_location("velociraptor_flagged", ROOT / "benchmarks" / "flagged.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "flagged.json"
    result = module.main(["--draws", "300", "--points", "100", "--json", str(path)])
    assert json.loads(path.read_text()) == result
    assert [row["scenario"] for row in result["rows"]] == ["planet", "eclipsing binary", "twin at twice the period"]
    for row in result["rows"]:
        assert row["pairs"] == 300 * 100 and 0 <= row["flagged"] <= row["pairs"], row
