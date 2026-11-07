"""The package's own content: nothing in it derives from PyTransit, which is GPL (README.md, LICENSE), the repository holds no planning or notebook file, and the tests import only what the test groups install."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_FILE = re.compile(r"^(outline|review|notes?|todo|plan|planning|notebook|scratch)(\..*)?$", re.IGNORECASE)


def test_no_file_of_the_repository_is_a_planning_or_notebook_file():
    """The files that a reader gets (tracked by git, or untracked and not ignored) include no outline, review, notes or plan."""
    listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    found = [name for name in listed if name and (ROOT / name).is_file() and NOTEBOOK_FILE.match(Path(name).name)]
    assert not found, found
    assert NOTEBOOK_FILE.match("outline.md") and NOTEBOOK_FILE.match("REVIEW.md") and not NOTEBOOK_FILE.match("README.md")


def test_the_package_does_not_import_pytransit_and_holds_no_file_named_after_the_code_that_translates_it():
    for path in sorted((ROOT / "src").rglob("*.py")):
        text = path.read_text()
        assert not re.search(r"^\s*(import|from)\s+pytransit", text, re.IGNORECASE | re.MULTILINE), path
    names = {path.name for path in ROOT.rglob("*.py") if ".oracle" not in path.parts and ".venv" not in path.parts}
    assert not names & {"triceratops_torch.py", "triceratops_triton.py", "triceratops_device.py"}, names


ALLOWED_IMPORTS = {"numpy", "pytest", "yaml", "mpmath", "velociraptor", "torch"}  # the cpu group and the test group; anything else has to be reached through pytest.importorskip, so that a clone with the cpu group alone skips the test and does not fail it


def test_a_test_imports_nothing_outside_the_cpu_and_test_groups_except_through_importorskip():
    import ast
    import sys

    standard = set(sys.stdlib_module_names)
    outside = []
    for path in sorted((ROOT / "tests").glob("*.py")):
        skipped = set(re.findall(r'importorskip\(\s*"([A-Za-z0-9_]+)"', path.read_text()))  # a module that the file reaches through importorskip is skipped, not failed, where it is missing
        for node in ast.walk(ast.parse(path.read_text())):
            names = [alias.name.split(".")[0] for alias in node.names] if isinstance(node, ast.Import) else [node.module.split(".")[0]] if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 else []
            outside += [f"{path.name}:{node.lineno} {name}" for name in names if name not in standard | ALLOWED_IMPORTS | skipped]
    assert not outside, outside


def test_the_readme_states_no_speed_until_the_benchmark_results_hold_a_measurement():
    """Every number of the README that reports a measurement comes from benchmarks/results.json; while that file is empty the README has no figure of speed."""
    import json

    results = json.loads((ROOT / "benchmarks" / "results.json").read_text())["results"]
    text = (ROOT / "README.md").read_text()
    speed = re.search(r"\d+(\.\d+)?\s*(x|times)\b|\bfaster\b|\bspeed-?up\b|\d+\s*(ms|milliseconds|seconds)\b", text, re.IGNORECASE)
    assert results or not speed, speed and speed.group(0)
