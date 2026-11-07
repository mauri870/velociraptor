# Environment for the PyTransit comparison

PyTransit 2.2.0 imports names that current numpy and scipy removed, so it runs in its own environment:

    uv venv .oracle --python 3.12
    uv pip install --python .oracle/bin/python "numpy<2" "scipy<1.14" numba "setuptools<81" pytest mpmath "astropy<7" emcee corner tqdm pandas xarray matplotlib semantic-version ldtk uncertainties lmfit tables seaborn pyyaml pytransit==2.2.0
    PYTHONPATH=src:validation .oracle/bin/python validation/pytransit_grid.py

`validation/pytransit_grid.py` restores the removed aliases `np.int`, `np.float` and `np.bool` before the import and changes nothing else.

The CI installs the CPU group alone, which has no PyTransit, so it runs none of the comparisons with PyTransit (the tests in `tests/test_quadratic.py` that import it are skipped there). In this environment they run:

    PYTHONPATH=src .oracle/bin/python -m pytest tests/test_quadratic.py -q

and the table of the README is checked by running the grid again; `validation/results.json` is unchanged when it reproduces:

    PYTHONPATH=src:validation .oracle/bin/python validation/pytransit_grid.py && git diff --exit-code validation/results.json
