"""docs/make_figure.py needs the network, so these tests read the PNG it wrote and run its fold, binning and best-draw functions on small synthetic inputs."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("make_figure", ROOT / "docs" / "make_figure.py")
make_figure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(make_figure)


def test_the_figure_of_the_readme_is_a_png_of_920_by_600_pixels_under_300_kb():
    data = (ROOT / "docs" / "figure.png").read_bytes()
    width, height = int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and (width, height) == (920, 600) and len(data) < 300_000, (width, height, len(data))


def test_the_fold_puts_the_transit_in_the_central_bins_and_gives_its_depth_and_the_error_of_each_bin():
    period, mid = 1.5, 10.0
    time = np.arange(0.0, 60.0, 2.0 / 1440.0)
    offset = ((time - mid) / period + 0.5) % 1 - 0.5
    flux = np.ones_like(time) + 1e-4 * np.sin(time * 997.0)  # a deterministic scatter
    flux[np.abs(offset * period) < 0.03] -= 4e-4
    centres, means, errors, depth = make_figure.fold_and_bin(time, flux, period, mid)
    assert abs(depth - 4e-4) < 3e-5, depth
    assert abs(centres[np.argmin(means)]) < 0.03 and centres.min() > -make_figure.WINDOW and centres.max() < make_figure.WINDOW
    assert errors.min() > 0 and np.all(errors < 1e-4)


def test_a_bin_with_fewer_points_than_the_minimum_is_left_out():
    time = np.array([0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.2])
    centres, means, errors, depth = make_figure.fold_and_bin(time, np.ones_like(time), 1.5, 0.0, window=0.25, width=0.01, minimum=6)
    assert len(centres) == 1 and abs(centres[0]) < 0.01, centres


def test_the_best_draw_of_a_call_is_the_one_with_the_smallest_half_chi_squared_and_its_values_are_those_of_that_draw():
    result = np.array([5.0, np.inf, 1.5, 3.0])
    call = ("lnL_TP_p", (np.zeros(2), np.zeros(2), 1.0, np.array([1.0, 2.0, 3.0, 4.0]), 1.45, np.array([80.0, 81.0, 82.0, 83.0])), {"companion_fluxratio": np.array([0.1, 0.2, 0.3, 0.4]), "exptime": 0.00139}, result)
    index, value, scenario, keywords = make_figure.best_draw(call)
    assert (index, value) == (2, 1.5)
    assert scenario == [3.0, 1.45, 82.0] and keywords == {"companion_fluxratio": 0.3, "exptime": 0.00139}
