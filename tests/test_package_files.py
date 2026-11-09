"""Checks the licence, the TRICERATOPS notice and pin, and the package metadata."""
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_the_package_is_licensed_under_mit_and_the_readme_says_the_source_does_not_import_or_copy_the_gpl_pytransit_that_triceratops_requires_and_names_the_files_that_do():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert metadata["license"] == "MIT" and set(metadata["license-files"]) == {"LICENSE", "THIRD_PARTY/TRICERATOPS-LICENSE"}
    licence = (ROOT / "LICENSE").read_text()
    assert licence.startswith("MIT License") and "Copyright (c) 2026 Mauri de Souza Meneguzzo" in licence
    readme = (ROOT / "README.md").read_text()
    assert "velociraptor neither imports nor includes it" in readme and "THIRD_PARTY/TRICERATOPS-LICENSE" in readme
    def imports_pytransit(path):
        return any(re.match(r"\s*(import pytransit|from pytransit)(\s|\.|$)", line) or 'importorskip("pytransit")' in line for line in path.read_text().splitlines())

    tracked = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()  # the files of the repository, not the local environments beside it
    importers = {name for name in tracked if Path(name).name != "test_package_files.py" and imports_pytransit(ROOT / name)}
    assert importers == {"tests/test_quadratic.py", "validation/pytransit_grid.py"}, importers  # the files docs/internals.md names
    assert not any(imports_pytransit(path) for path in (ROOT / "src").rglob("*.py"))
    internals = (ROOT / "docs" / "internals.md").read_text()
    assert all(name in internals for name in importers)
    assert "license: MIT" in (ROOT / "CITATION.cff").read_text()


def test_the_notice_of_triceratops_is_kept_and_its_pin_is_in_the_metadata():
    assert "Copyright (c) 2020 stevengiacalone" in (ROOT / "THIRD_PARTY" / "TRICERATOPS-LICENSE").read_text()
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert metadata["project"]["optional-dependencies"]["triceratops"][0] == "triceratops==1.1.0"
    assert "THIRD_PARTY/TRICERATOPS-LICENSE" in metadata["project"]["license-files"]
    assert "1.1.0" in (ROOT / "README.md").read_text() and "GPL" in (ROOT / "README.md").read_text()


def test_the_model_needs_only_numpy_and_the_device_builds_of_torch_are_groups():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert metadata["project"]["dependencies"] == ["numpy>=2.2"], "SPEC 0's minimum at 2026-11-08, which triceratops 1.1.0 allows on 3.13 (its own metadata pins numpy>=2.1 there)"
    groups = metadata["dependency-groups"]
    assert {"cpu", "cu128", "rocm", "test", "dev"} <= set(groups)
    assert {"include-group": "test"} in groups["dev"] and any(str(entry).startswith("ruff") for entry in groups["dev"])


def test_the_citation_file_names_the_version_of_the_package():
    text = (ROOT / "CITATION.cff").read_text()
    assert f'version: "{tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]}"' in text


def test_the_benchmark_results_file_exists_and_is_json():
    import json

    assert json.loads((ROOT / "benchmarks" / "results.json").read_text()) == {"results": []} or json.loads((ROOT / "benchmarks" / "results.json").read_text())["results"]


def test_the_distribution_is_named_velociraptor_gpu_and_the_import_name_stays_velociraptor():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert metadata["project"]["name"] == "velociraptor-gpu"
    assert metadata["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["src/velociraptor"]
    assert (ROOT / "src" / "velociraptor" / "__init__.py").exists() and not (ROOT / "src" / "velociraptor_gpu").exists()


PAPER_TITLE = "velociraptor: exact light curves and GPU-accelerated false-positive probabilities for TESS planet candidates"


def test_the_readme_and_the_citation_file_give_the_title_of_the_paper_in_one_line_and_the_readme_links_the_pdf():
    readme = (ROOT / "README.md").read_text()
    assert f"[{PAPER_TITLE}](paper/main.pdf)" in readme
    citation = (ROOT / "CITATION.cff").read_text()
    assert f'title: "{PAPER_TITLE}"' in citation and "preferred-citation:" in citation


def test_the_triceratops_extra_caps_setuptools_below_82_because_pytransit_imports_pkg_resources_and_the_readme_says_so():
    """PyTransit 2.2.0 imports pkg_resources, which setuptools 82 and later do not ship; with 84 forced, `import triceratops.triceratops` raised ModuleNotFoundError, and with 81.0.0 it imported."""
    extra = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]["triceratops"]
    assert "setuptools>=60,<82" in extra, "a floor was added so `uv sync --resolution lowest-direct` installs a wheel rather than building setuptools's own pre-wheel era; the cap below 82 stands"
    assert "setuptools" in (ROOT / "README.md").read_text().split("## Getting started")[0]
