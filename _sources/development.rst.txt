Development and release
=======================

.. code-block:: shell

   uv sync
   uv run pytest
   uv sync --group docs
   uv run sphinx-build -W -b html docs docs/_build/html
   uv build --no-config

A C++17 compiler is required for development. Documentation also needs Doxygen
(``apt install doxygen``, ``brew install doxygen``, or its Windows installer). ``uv.lock`` pins development
dependencies; consumers resolve their own environments. Editable builds use
the development environment so CasADi, SWIG and CMake remain available for
rebuilds. For pip development, install the ``dev`` dependency group with uv,
or install the build requirements before ``pip install --no-build-isolation -e .``.

Native tests:

.. code-block:: shell

   cmake -S . -B build/cpp -Dcasadi_DIR=/path/to/casadi/cmake \
     -DSPATIAL_CASADI_BUILD_TESTS=ON
   cmake --build build/cpp
   ctest --test-dir build/cpp --output-on-failure

Benchmarks:

.. code-block:: shell

   uv run python benchmarks/run.py

The report separates graph construction, graph size, value/derivative
evaluation and generated-code size. Timing is informational; graph-size
budgets are checked in tests. Re-run benchmarks when changing formulas.
The checked-in ``benchmarks/baseline.json`` compares the core with the previous
``master`` implementation: SX graph sizes are unchanged for the Euler-chain
workload; the 50-link MX Hessian graph is about 15% smaller. Small MX graphs
can be larger, so timing and workload still matter.

Release gate
------------

* Python and C++ tests pass, including derivative and generated-code checks.
* Source archives build, and pip/uv install wheels in clean environments.
* The CMake consumer builds against the installed package.
* Documentation builds without warnings; examples run.
* Wheel CI passes for each advertised platform and Python version.

CI builds and tests pull requests. Successful documentation builds on ``master``
publish the generated HTML to ``gh-pages``, including ``.nojekyll`` so Sphinx's
``_static`` assets are served. Pull requests and other branches only build.

In repository settings:

* **Pages:** select **Deploy from a branch**, ``gh-pages``, ``/(root)``.
* **Environments → github-pages:** allow ``gh-pages`` in deployment branch rules.
  The documentation workflow publishes a branch; GitHub's Pages workflow handles
  deployment. ``master`` does not need access to that environment.

After merging a workflow change, use **Actions → Documentation → Run workflow**
on ``master`` to publish again. Re-running an old failed run uses its old workflow.
The publisher uses the automatic ``GITHUB_TOKEN`` with ``contents: write``;
repository or organization rules must permit Actions to update ``gh-pages``.

Publishing to PyPI
------------------

The **Release wheels** workflow builds and tests Linux, Windows and macOS wheels,
checks package metadata, and collects the wheels and source archive in the
``pypi-distributions`` artifact. Publishing is off by default.

One-time setup:

* In GitHub, create the ``pypi`` environment and restrict deployments to ``master``.
* In PyPI's project settings, open **Publishing** and add a GitHub trusted
  publisher: owner ``cmower``, repository ``spatial-casadi``, workflow
  ``wheels.yaml``, environment ``pypi``. No API token is needed.

For each release, update the version and README release notes, merge into
``master``, and ensure CI passes. Run **Actions → Release wheels → Run workflow**
on ``master`` with **Publish to PyPI** selected. Publication runs only after all
wheel tests and metadata checks pass. Tag the built commit after publication.
PyPI does not allow replacing uploaded files under the same version.

For a manual upload, download and extract ``pypi-distributions``, then run
``uvx twine upload --username __token__ /path/to/extracted/*`` and enter a
project-scoped PyPI API token at the password prompt.

The API is feature-complete for the scope in :doc:`compatibility`. New SciPy
features are optional. Dependency upgrades need explicit ABI and derivative
checks. See ``THIRD_PARTY_NOTICES`` for SciPy attribution.
