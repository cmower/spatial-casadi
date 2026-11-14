Installation
============

Python
------

.. code-block:: shell

   pip install spatial-casadi
   # or, in a uv project:
   uv add spatial-casadi
   # or, in an existing environment:
   uv pip install spatial-casadi

Version 2 targets CPython 3.10–3.14 and CasADi 3.8.2. Wheels are configured for
Linux x86-64, Windows x86-64, and macOS Intel/Apple Silicon. Supported wheels
need no compiler. Other platforms require a source build with a C++17 compiler.
The CasADi version is pinned because the extension shares its binary runtime
and SWIG types; upgrade them together.

C++
---

The C++ library is header-only and links to CasADi 3.8.2. With its SDK installed:

.. code-block:: shell

   cmake -S . -B build/cpp -Dcasadi_DIR=/path/to/casadi/cmake \
     -DCMAKE_INSTALL_PREFIX=/path/to/install
   cmake --install build/cpp

The Python CasADi wheel also includes the SDK. Find it with:

.. code-block:: shell

   python -c "import casadi, pathlib; print(pathlib.Path(casadi.__file__).parent / 'cmake')"

In your project:

.. code-block:: cmake

   find_package(spatial_casadi CONFIG REQUIRED)
   target_link_libraries(app PRIVATE spatial_casadi::spatial_casadi)

Set ``CMAKE_PREFIX_PATH`` to the installation prefix and ``casadi_DIR`` to
the CasADi SDK. A complete consumer project is in ``examples/cpp``.
