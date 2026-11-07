"""The tables of README.md, written from benchmarks/results.json and validation/results.json.

usage: uv run python benchmarks/tables.py          writes the blocks between `<!-- generated: NAME -->` and `<!-- end generated: NAME -->` in README.md
       uv run python benchmarks/tables.py --check  exits 1 and names the block that differs from what the files give

Every number in a table is read from a file. tests/test_tables.py fails when a block of the README differs from what this script writes.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
FILES = ((README, ("headline", "speed", "flagged", "references", "pytransit")),)  # the blocks of each file
NOTHING = "Nothing measured yet."


def load(path):
    """The JSON file at `path`, or {} when it does not exist."""
    return json.loads(path.read_text()) if path.exists() else {}


def speed(results):
    """benchmarks/results.json: {"results": [{"label", "machine", "command", "versions", "rows": [{"method", "scenarios", "seconds"}]}], "notes": {method: sentence}}, one table for each result and, under it, the sentence of each of its methods that has a note. The last column is the seconds of the numba row of the result over the seconds of the row."""
    blocks = []
    for result in results.get("results", []):
        base = next((row["seconds"] for row in result["rows"] if "numba" in row["method"]), None)
        lines = [f"{result['label']} ({result['machine']}; `{result['command']}`; {result['versions']})", "", "| Method | Scenarios | Seconds | Times the numba row |", "| --- | ---: | ---: | ---: |"]
        lines += [f"| {row['method']} | {row['scenarios']} | {row['seconds']:.3g} | {base / row['seconds']:.2f} |" if base else f"| {row['method']} | {row['scenarios']} | {row['seconds']:.3g} | - |" for row in result["rows"]]
        for row in result["rows"]:
            if row["method"] in results.get("notes", {}):
                lines += ["", f"Note on {row['method']}: {results['notes'][row['method']]}"]
        for skip in result.get("not_run", []):
            lines += ["", f"Not run: {skip['method']}, because {skip['reason']}."]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else NOTHING


def speed_short(results):
    """The table of the README: the numba row and the row of install("cuda") with no arguments of the last result at 100000 draws."""
    for result in reversed(results.get("results", [])):
        if "100000 draws" not in result["label"]:
            continue
        numba = next((row for row in result["rows"] if "numba" in row["method"]), None)
        default = next((row for row in result["rows"] if 'install("cuda") with no arguments' in row["method"]), None)
        if numba and default:
            lines = [f"The likelihood step of one star's calculation (three scenario likelihoods, 100000 draws and 1000 points) ({result['machine']}):", "", "| Method | Seconds | Speed-up |", "| --- | ---: | ---: |"]
            lines += [f"| {row['method'].split(':')[0]} | {row['seconds']:.3g} | {numba['seconds'] / row['seconds']:.2f} |" for row in (numba, default)]
            return "\n".join(lines)
    return NOTHING


def headline(results, whole):
    """The two sentences that open the Performance section of the README: the likelihoods at the exposure integral of TRICERATOPS (20 midpoints, float64) and at the default of `install("cuda")`, from the last result at 100000 draws of benchmarks/results.json,
    and the whole calculation on the stars of benchmarks/whole_calculation.json ({"source", "draws", "fpp_shift_3_nodes_vs_64", "rows": [{"tic", "reference", "f64_20_midpoints", "default"}]}, the seconds of the whole calculation: the published code with its host-side lookups removed, the device at 20 midpoints in
    float64 and the default). Each ratio is the seconds of the published code over the seconds of the row."""
    rows = whole.get("rows") if whole else None
    for result in reversed(results.get("results", [])):
        if "100000 draws" not in result["label"]:
            continue
        numba = next((row for row in result["rows"] if "numba" in row["method"]), None)
        midpoints = next((row for row in result["rows"] if row["method"].endswith("20 midpoints")), None)
        default = next((row for row in result["rows"] if 'install("cuda") with no arguments' in row["method"]), None)
        if numba and midpoints and default and rows:
            ratios = lambda key: [row["reference"] / row[key] for row in rows]
            low, high, low_default, high_default = min(ratios("f64_20_midpoints")), max(ratios("f64_20_midpoints")), min(ratios("default")), max(ratios("default"))
            return (f"At the exposure integral of TRICERATOPS (20 midpoints) in float64, the three likelihoods run {numba['seconds'] / midpoints['seconds']:.2f} times as fast as the published code, and the whole calculation {low:.2f} to {high:.2f} times as fast "
                    f"on {len(rows)} stars at {whole['draws']} draws. At the default of `install(\"cuda\")` (3 Gauss-Legendre nodes, first pass in float32) the likelihoods run {numba['seconds'] / default['seconds']:.2f} times as fast and the whole calculation {low_default:.2f} to {high_default:.2f} times as fast, "
                    f"with a shift of the false-positive probability of at most {whole['fpp_shift_3_nodes_vs_64']:.1e} against 64 nodes on {len(rows)} stars. The whole-calculation figures are {whole['source']}; their reference is {whole['reference_is']}.")
    return NOTHING


def flagged(data):
    """benchmarks/flagged.json: {"command", "draws", "points", "samples", "rows": [{"scenario", "pairs", "flagged"}]}, the pairs of each scenario of the benchmark input that the first pass leaves to the closed form."""
    if not data.get("rows"):
        return NOTHING
    lines = [f"`{data['command']}`, {data['draws']} draws and {data['points']} points, {data['samples']} midpoints", "", "| Scenario | Pairs | Left to the closed form | Share |", "| --- | ---: | ---: | ---: |"]
    lines += [f"| {row['scenario']} | {row['pairs']} | {row['flagged']} | {row['flagged'] / row['pairs']:.3f} |" for row in data["rows"]]
    return "\n".join(lines)


def largest(node):
    """(cases, largest difference) of a branch of validation/results.json: a leaf is a dict with `largest_difference` (and the `points` it covers) or a number (one case); the branch's cases are summed and its differences maximised."""
    if isinstance(node, bool):
        return 0, 0.0
    if isinstance(node, (int, float)):
        return 1, abs(float(node))
    if not isinstance(node, dict):
        return 0, 0.0
    if "largest_difference" in node:
        return node.get("points", 1), abs(node["largest_difference"])
    found = [largest(child) for child in node.values()]
    found = [each for each in found if each[0]]
    return (sum(each[0] for each in found), max(each[1] for each in found)) if found else (0, 0.0)


REFERENCES = (("lens_area", "Closed form against the lens area, 50 digits (no limb darkening)"), ("two_dimensional_integral", "Closed form against a two-dimensional integral"), ("convergence", "Closed form against 768 nodes of the quadrature"), ("orbit_50_digit", "Separation of an orbit, 50 digits (numpy 1.26.4)"))
PYTRANSIT = (("pytransit_flux", "PyTransit 2.2.0, flux at a separation"), ("pytransit_orbits", "PyTransit 2.2.0, light curve of an orbit"))


def difference(validation, rows):
    """The markdown table of the largest difference of each comparison of `rows` that `validation` holds, or NOTHING when it holds none."""
    lines = ["| Comparison | Cases | Largest difference |", "| --- | ---: | ---: |"]
    for key, label in rows:
        if key in validation:
            points, value = largest(validation[key])
            lines.append(f"| {label} | {points} | {value:.2e} |")
    return "\n".join(lines) if len(lines) > 2 else NOTHING


def blocks():
    """The text of every generated block of the README, from benchmarks/results.json, benchmarks/flagged.json and validation/results.json."""
    validation = load(ROOT / "validation" / "results.json")
    results = load(ROOT / "benchmarks" / "results.json")
    return {"speed_short": speed_short(results), "speed": speed(load(ROOT / "benchmarks" / "results.json")), "headline": headline(results, load(ROOT / "benchmarks" / "whole_calculation.json")), "flagged": flagged(load(ROOT / "benchmarks" / "flagged.json")), "references": difference(validation, REFERENCES), "pytransit": difference(validation, PYTRANSIT)}


def pattern(name):
    """The regular expression that finds the generated block `name` between its two comment markers."""
    return re.compile(rf"(<!-- generated: {name} -->\n)(.*?)(\n<!-- end generated: {name} -->)", re.DOTALL)


def rewrite(text, names):
    """`text` with each block of `names` replaced by the one that blocks() writes; exits when a block is missing."""
    written = blocks()
    for name in names:
        found = pattern(name).search(text)
        if found is None:
            raise SystemExit(f"no block {name}")
        text = text[:found.start(2)] + written[name] + text[found.end(2):]
    return text


def main():
    changed = []
    for path, names in FILES:
        text = path.read_text()
        new = rewrite(text, names)
        if new != text:
            changed.append(path.name)
            if "--check" not in sys.argv:
                path.write_text(new)
    if "--check" in sys.argv and changed:
        raise SystemExit(f"a table of {', '.join(changed)} is not what benchmarks/results.json and validation/results.json give: run benchmarks/tables.py")


if __name__ == "__main__":
    main()
