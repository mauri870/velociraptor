"""benchmarks/bench.py: the input, the refusal of a busy device and of a size that does not fit in memory, the discarded first call, the tables it writes for benchmarks/tables.py, and a small session on the CPU."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "benchmarks")]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "benchmarks" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load("bench")
tables = load("tables")


def test_the_input_is_made_from_its_seed_alone_and_every_draw_has_a_valid_orbit():
    first, again, other = bench.make_input(50, 30), bench.make_input(50, 30), bench.make_input(50, 30, seed=bench.SEED + 1)
    assert np.array_equal(first[1], again[1]) and np.array_equal(first[3]["tp"]["arguments"][0], again[3]["tp"]["arguments"][0])
    assert not np.array_equal(first[1], other[1])
    assert first[0].shape == first[1].shape == (30,)
    for scenario in first[3].values():
        inclination, axis = scenario["arguments"][-7], scenario["arguments"][-6]
        assert np.all(np.isfinite(inclination)) and np.all(inclination <= 90.0) and np.all(axis > 0), scenario["arguments"][-8]
        assert scenario["comp"].shape == (50,)
    assert first[3]["twin"]["arguments"][-8] == 2 * bench.PERIOD and first[3]["eb"]["arguments"][-8] == bench.PERIOD


def test_a_function_that_changes_its_arrays_in_place_does_not_change_the_data_the_next_call_reads():
    """The likelihood functions of TRICERATOPS convert the inclination to radians in place. A benchmark that passed the same array twice would time a light curve without a transit from the second call on."""
    data = bench.make_input(20, 10)

    def spoil(time_axis, flux, sigma, *arguments, **options):
        for each in (time_axis, flux, *arguments, options["companion_fluxratio"]):
            if isinstance(each, np.ndarray):
                each *= 0.0
        return np.zeros(1)

    before = [np.array(each) for each in data[3]["tp"]["arguments"] if isinstance(each, np.ndarray)] + [np.array(data[0]), np.array(data[1])]
    bench.calls((spoil, spoil, spoil), data)
    after = [each for each in data[3]["tp"]["arguments"] if isinstance(each, np.ndarray)] + [data[0], data[1]]
    assert all(np.array_equal(left, right) for left, right in zip(before, after))


def test_the_first_call_is_not_counted_and_each_repeat_is_one_call():
    count = []
    first, first_seconds, seconds = bench.timed(lambda: count.append(1) or len(count), 3)
    assert len(count) == 4 and first == 1 and len(seconds) == 3 and first_seconds >= 0


def test_a_single_repeat_has_quartiles_equal_to_its_value():
    row = bench.summary([2.0])
    assert row["median"] == row["q1"] == row["q3"] == 2.0


def test_agreement_is_the_largest_relative_difference_of_the_finite_entries_and_counts_an_infinity_in_one_of_the_two():
    reference = [np.array([10.0, np.inf, 4.0]), np.array([0.5])]
    other = [np.array([10.5, 3.0, 4.0]), np.array([0.5])]
    found = bench.agreement(reference, other)
    assert found["infinity_mismatches"] == 1 and found["largest_difference"] == pytest.approx(0.05), found


def test_a_size_whose_array_does_not_fit_is_refused_with_its_size_and_the_limit_named():
    with pytest.raises(SystemExit, match=r"--max-array-gib"):
        bench.refuse_large(1_000_000, 1000, 2.0)
    bench.refuse_large(100_000, 1000, 2.0)


def make_proc(root, entries):
    """A directory shaped like /proc: entries is {pid: (parent pid, target of its file descriptor 3)}."""
    for pid, (parent, target) in entries.items():
        (root / str(pid) / "fd").mkdir(parents=True)
        (root / str(pid) / "stat").write_text(f"{pid} (name) S {parent} 0 0\n")
        os.symlink(target, root / str(pid) / "fd" / "3")


def test_a_process_that_holds_the_compute_device_is_found_and_this_one_and_its_children_and_a_render_node_are_not(tmp_path):
    make_proc(tmp_path, {10: (1, "/dev/kfd"), 20: (1, "/dev/dri/renderD128"), 30: (1, "/dev/kfd"), 31: (30, "/dev/kfd"), 40: (1, "/tmp/x")})
    assert bench.other_device_jobs(str(tmp_path), own=30) == [10]
    assert bench.other_device_jobs(str(tmp_path), own=99) == [10, 30, 31]


def test_the_session_tables_are_what_tables_py_reads_and_publishing_replaces_a_session_of_the_same_label_and_machine(tmp_path, monkeypatch):
    report = {"hardware": {"cpu": "cpu", "gpu": None, "gpu_runtime": None}, "versions": {"numpy": "2"}, "command": "bench.py", "input": {"points": 40},
              "rows": [{"method": "triceratops-numba", "draws": 100, "threads": 4, "median": 1.5}, {"method": "velociraptor-cpu", "draws": 100, "threads": 4, "median": 0.5}],
              "skipped": [{"method": "velociraptor-cpu", "draws": 100, "reason": "it is slow"}, {"method": "velociraptor-cpu", "draws": 1000, "reason": "another size"}]}
    entries = bench.tables_of(report)
    assert [row["scenarios"] for row in entries[0]["rows"]] == [300, 300]
    assert "likelihoods" in entries[0]["label"], "the table says what it times: the likelihoods and not a whole calculation"
    monkeypatch.setattr(tables, "ROOT", tmp_path)
    (tmp_path / "benchmarks").mkdir()
    monkeypatch.setattr(bench, "ROOT", tmp_path)
    (tmp_path / "benchmarks" / "results.json").write_text(json.dumps({"results": []}))
    session = tmp_path / "session.json"
    session.write_text(json.dumps({"results": entries}))
    bench.publish(session)
    bench.publish(session)
    stored = json.loads((tmp_path / "benchmarks" / "results.json").read_text())["results"]
    assert len(stored) == 1, "the second publication of the same session replaces the first"
    assert "| TRICERATOPS 1.1.0, numba path, 4 threads | 300 | 1.5 | 1.00 |" in tables.blocks()["speed"]
    assert "Not run: velociraptor, torch on the CPU, 4 threads, because it is slow." in tables.blocks()["speed"] and "another size" not in tables.blocks()["speed"], "a skipped row is named under the table of its size and no other"


def test_a_small_session_on_the_cpu_writes_every_cpu_row_with_its_first_call_apart_and_no_device_row(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("triceratops")
    path = tmp_path / "session.json"
    report = bench.main(["--draws", "60", "--points", "30", "--repeats", "2", "--threads", "1", "--cpu-only", "--json", str(path)])
    assert [row["method"] for row in report["rows"]] == ["triceratops-numba", "velociraptor-cpu"]
    for row in report["rows"]:
        assert len(row["seconds"]) == 2 and row["first_call_seconds"] > 0
    assert report["rows"][1]["agreement"]["infinity_mismatches"] >= 0 and np.isfinite(report["rows"][1]["agreement"]["largest_difference"])
    assert report["scan"]["fastest_threads"] == 1 and json.loads(path.read_text())["results"] == report["results"]
    assert {"cpu", "logical_cpus", "cpus_allowed"} <= set(report["hardware"]) and "numpy" in report["versions"] and "triceratops" in report["versions"]


def test_the_cpus_for_a_thread_count_are_one_thread_to_a_core_up_to_the_cores_and_all_of_them_above():
    logical = list(range(32))
    assert bench.cpus_for(1, logical) == {0}
    assert bench.cpus_for(8, logical) == set(range(8))
    assert bench.cpus_for(16, logical) == set(range(16))
    assert bench.cpus_for(17, logical) == set(logical) and bench.cpus_for(32, logical) == set(logical)
    assert bench.cpus_for(1, [0]) == {0} and bench.cpus_for(2, [0, 1]) == {0, 1}


def test_a_skipped_row_is_not_run_and_the_session_says_it_was_skipped_and_why():
    pytest.importorskip("torch")
    pytest.importorskip("triceratops")
    report = bench.main(["--draws", "60", "--points", "30", "--repeats", "2", "--threads", "1", "--cpu-only", "--skip", "velociraptor-cpu:60", "--skip-reason", "too slow"])
    assert [row["method"] for row in report["rows"]] == ["triceratops-numba"]
    assert report["skipped"] == [{"method": "velociraptor-cpu", "draws": 60, "reason": "too slow"}]


def test_the_float32_row_sets_the_precision_inside_its_block_and_restores_it_afterwards_also_after_an_error(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("triton")
    import contextlib

    from velociraptor import triceratops, triton_analytic

    seen = []

    @contextlib.contextmanager
    def fake_install(*arguments, **options):
        seen.append(triton_analytic.PRECISION)
        yield

    monkeypatch.setattr(triceratops, "install", fake_install)
    assert triton_analytic.PRECISION == "float64"
    with bench.mixed_precision():
        assert triton_analytic.PRECISION == "mixed"
    assert triton_analytic.PRECISION == "float64" and seen == ["mixed"]
    with pytest.raises(RuntimeError):
        with bench.mixed_precision():
            raise RuntimeError
    assert triton_analytic.PRECISION == "float64"


def test_the_caption_command_leaves_out_the_skip_reason_and_the_json_file():
    command = "bench.py --skip velociraptor-cpu:100000 --skip-reason a call of this row took 44 s and the row has six calls --json out/session.json"
    assert bench.caption_command(command) == "bench.py --skip velociraptor-cpu:100000 --json ..."
    assert bench.caption_command("bench.py --repeats 3") == "bench.py --repeats 3"


@pytest.mark.default_precision
def test_the_float64_rows_set_float64_inside_their_block_whatever_the_shipped_precision_is_and_put_it_back(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("triton")
    import contextlib

    from velociraptor import triceratops, triton_analytic

    seen = []

    @contextlib.contextmanager
    def fake_install(*arguments, **options):
        seen.append((triton_analytic.PRECISION, options["exposure_nodes"]))
        yield

    monkeypatch.setattr(triceratops, "install", fake_install)
    shipped = triton_analytic.PRECISION
    assert shipped == "mixed"
    with bench.device_precision("float64", None):
        pass
    with bench.device_precision("float64", 3):
        pass
    with bench.mixed_precision():
        pass
    assert seen == [("float64", None), ("float64", 3), ("mixed", None)]
    assert triton_analytic.PRECISION == shipped
