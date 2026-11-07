"""benchmarks/tables.py writes the tables of README.md from benchmarks/results.json and validation/results.json. The README holds a figure of speed or of difference only inside the generated blocks. The inputs are the files of the repository and small result files in a temporary directory."""
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("tables", ROOT / "benchmarks" / "tables.py")
tables = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tables)


def outside_the_blocks(text):
    return re.sub(r"<!-- generated: (\w+) -->.*?<!-- end generated: \1 -->", "", text, flags=re.DOTALL)


def test_every_generated_block_of_the_readme_and_of_the_internals_is_what_the_script_writes_from_the_files():
    for path, names in tables.FILES:
        text = path.read_text()
        assert tables.rewrite(text, names) == text, f"run `uv run python benchmarks/tables.py` ({path.name})"


def test_each_file_has_its_blocks_each_once():
    for path, names in tables.FILES:
        text = path.read_text()
        for name in names:
            assert text.count(f"<!-- generated: {name} -->") == 1 and text.count(f"<!-- end generated: {name} -->") == 1, (path.name, name)


def test_no_figure_of_speed_or_of_difference_is_typed_outside_the_blocks():
    for path, _ in tables.FILES:
        text = outside_the_blocks(path.read_text())
        found = re.search(r"\d+(\.\d+)?\s*(x|times)\b|\bfaster\b|\bspeed-?up\b|\d+\s*(ms|milliseconds|seconds)\b|\d(\.\d+)?e-\d+|\bagrees? to\b", text, re.IGNORECASE)
        assert not found, (path.name, found and found.group(0))


def test_the_short_table_of_the_readme_is_the_numba_row_and_the_default_row_of_the_last_result_at_100000_draws():
    row = lambda method, seconds: {"method": method, "scenarios": 300000, "seconds": seconds}
    results = {"results": [{"label": "The three scenario likelihoods of TRICERATOPS, 10000 draws and 1000 points each", "machine": "m", "command": "c", "versions": "v", "rows": [row("TRICERATOPS 1.1.0, numba path, 32 threads", 0.5)]},
                           {"label": "The three scenario likelihoods of TRICERATOPS, 100000 draws and 1000 points each", "machine": "card", "command": "c", "versions": "v", "rows": [row("TRICERATOPS 1.1.0, numba path, 32 threads", 5.0), row('velociraptor, install("cuda") with no arguments: 3 Gauss-Legendre nodes, first pass in float32', 1.0)]}]}
    block = tables.speed_short(results)
    assert "| TRICERATOPS 1.1.0, numba path, 32 threads | 5 | 1.00 |" in block and '| velociraptor, install("cuda") with no arguments | 1 | 5.00 |' in block and "(card)" in block
    assert tables.speed_short({"results": results["results"][:1]}) == tables.NOTHING


def test_a_table_of_speed_is_written_from_a_result_and_says_nothing_without_one(tmp_path, monkeypatch):
    monkeypatch.setattr(tables, "ROOT", tmp_path)
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "benchmarks" / "results.json").write_text(json.dumps({"results": []}))
    assert tables.blocks()["speed"] == tables.NOTHING
    row = {"method": "numba path of TRICERATOPS 1.1.0", "scenarios": 1000, "seconds": 12.345}
    (tmp_path / "benchmarks" / "results.json").write_text(json.dumps({"results": [{"label": "a batch", "machine": "m", "command": "c", "versions": "v", "rows": [row]}]}))
    block = tables.blocks()["speed"]
    assert "| numba path of TRICERATOPS 1.1.0 | 1000 | 12.3 |" in block and "a batch (m; `c`; v)" in block


def test_a_table_of_difference_takes_the_largest_over_the_leaves_of_each_shape_of_result_and_is_empty_without_the_key(tmp_path, monkeypatch):
    monkeypatch.setattr(tables, "ROOT", tmp_path)
    (tmp_path / "validation").mkdir()
    (tmp_path / "validation" / "results.json").write_text(json.dumps({
        "lens_area": {"0.1": {"points": 56, "largest_difference": 3e-12}, "0.3": {"points": 56, "largest_difference": 5e-11}},
        "convergence": {"0.1": {"largest_difference": 2e-16, "points": 1}},
        "orbit_50_digit": {"e_0.1": 1.3e-13, "e_0.8": 8.5e-14},
    }))
    block = tables.blocks()
    assert "| Closed form against the lens area, 50 digits (no limb darkening) | 112 | 5.00e-11 |" in block["references"]
    assert "| Closed form against 768 nodes of the quadrature | 1 | 2.00e-16 |" in block["references"]
    assert "| Separation of an orbit, 50 digits (numpy 1.26.4) | 2 | 1.30e-13 |" in block["references"]
    assert block["pytransit"] == tables.NOTHING


def test_the_last_column_of_the_speed_table_is_the_seconds_of_the_numba_row_over_the_seconds_of_the_row(tmp_path, monkeypatch):
    monkeypatch.setattr(tables, "ROOT", tmp_path)
    (tmp_path / "benchmarks").mkdir()
    rows = [{"method": "TRICERATOPS 1.1.0, numba path", "scenarios": 3, "seconds": 2.0}, {"method": "device", "scenarios": 3, "seconds": 0.5}, {"method": "slow", "scenarios": 3, "seconds": 8.0}]
    (tmp_path / "benchmarks" / "results.json").write_text(json.dumps({"results": [{"label": "l", "machine": "m", "command": "c", "versions": "v", "rows": rows}]}))
    block = tables.blocks()["speed"]
    assert "| TRICERATOPS 1.1.0, numba path | 3 | 2 | 1.00 |" in block and "| device | 3 | 0.5 | 4.00 |" in block and "| slow | 3 | 8 | 0.25 |" in block


def test_the_table_of_flagged_pairs_is_written_from_its_file_and_says_nothing_without_one(tmp_path, monkeypatch):
    monkeypatch.setattr(tables, "ROOT", tmp_path)
    (tmp_path / "benchmarks").mkdir()
    assert tables.blocks()["flagged"] == tables.NOTHING
    data = {"command": "flagged.py", "draws": 10, "points": 5, "samples": 20, "rows": [{"scenario": "planet", "pairs": 50, "flagged": 5}]}
    (tmp_path / "benchmarks" / "flagged.json").write_text(json.dumps(data))
    assert "| planet | 50 | 5 | 0.100 |" in tables.blocks()["flagged"]


def test_a_note_of_a_method_is_written_under_the_table_that_has_the_method_and_under_no_other(tmp_path, monkeypatch):
    monkeypatch.setattr(tables, "ROOT", tmp_path)
    (tmp_path / "benchmarks").mkdir()
    row = lambda method: {"method": method, "scenarios": 1, "seconds": 1.0}
    entry = lambda label, method: {"label": label, "machine": "m", "command": "c", "versions": "v", "rows": [row("numba path of TRICERATOPS 1.1.0"), row(method)]}
    results = {"results": [entry("first", "slow path"), entry("second", "fast path")], "notes": {"slow path": "it varies."}}
    (tmp_path / "benchmarks" / "results.json").write_text(json.dumps(results))
    block = tables.blocks()["speed"]
    assert block.count("Note on slow path: it varies.") == 1 and block.index("Note on slow path") < block.index("second (m;")


def test_the_headline_gives_the_ratios_of_the_likelihoods_and_of_the_whole_calculation_from_the_files_and_nothing_without_them():
    row = lambda method, seconds: {"method": method, "scenarios": 300000, "seconds": seconds}
    results = {"results": [{"label": "The three scenario likelihoods of TRICERATOPS, 100000 draws and 1000 points each", "machine": "card", "command": "c", "versions": "v", "rows": [
        row("TRICERATOPS 1.1.0, numba path, 32 threads", 5.0), row("velociraptor, fused kernel on the device, 20 midpoints", 4.0), row('velociraptor, install("cuda") with no arguments: 3 nodes', 0.5)]}]}
    whole = {"source": "Table 4 of the paper", "reference_is": "the published code with its lookups removed", "draws": 100000, "fpp_shift_3_nodes_vs_64": 9.1e-6, "rows": [
        {"tic": 1, "reference": 4.0, "f64_20_midpoints": 2.0, "default": 0.5}, {"tic": 2, "reference": 3.0, "f64_20_midpoints": 2.0, "default": 0.75}]}
    text = tables.headline(results, whole)
    assert "1.25 times as fast" in text and "10.00 times as fast" in text and "1.50 to 2.00 times as fast" in text and "4.00 to 8.00 times as fast" in text
    assert "9.1e-06" in text and "2 stars" in text and "Table 4 of the paper" in text and "the published code with its lookups removed" in text
    assert tables.headline({"results": []}, whole) == tables.NOTHING and tables.headline(results, {}) == tables.NOTHING
