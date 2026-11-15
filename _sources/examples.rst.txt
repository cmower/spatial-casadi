Examples
========

Optimization
------------

This fits a rotation to transformed points. Matrix/point residuals avoid
Euler-angle singularities. Run with ``uv run python examples/fit_rotation.py``.

.. literalinclude:: ../examples/fit_rotation.py
   :language: python

Generated code
--------------

Run ``uv run python examples/codegen.py`` to produce C code containing values
and a Jacobian. Compile it with your C toolchain; evaluation does not require
Python. The test suite also compiles and compares generated values, Jacobians
and Hessians against CasADi.

.. literalinclude:: ../examples/codegen.py
   :language: python
