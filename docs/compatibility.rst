SciPy compatibility
===================

The interface targets SciPy 1.18's ``spatial.transform`` conventions.
SciPy is a test dependency, not a runtime dependency.

================== ======================================================
Class              Supported operations
================== ======================================================
Rotation           Quaternion, matrix, rotation-vector, MRP and Euler
                   conversions; composition, inverse, powers, application,
                   magnitude, approximate equality, identity, random,
                   concatenation, integer indexing and slicing
RigidTransform     Matrix/component, exponential-coordinate and dual-
                   quaternion conversions; composition, inverse, powers,
                   application, identity, concatenation, indexing and slicing
Slerp              Fixed knot times; numeric or symbolic query times
================== ======================================================

Deliberate differences
----------------------

* Results are CasADi matrices, with the shapes described in :doc:`conventions`.
* One batch dimension is supported. Arbitrary tensor broadcasting, empty
  batches, advanced indexing and in-place item assignment are excluded.
* Matrix constructors require proper rotations; they do not project noisy
  matrices onto SO(3). Clean numerical measurements before constructing graphs.
* ``approx_equal`` returns a CasADi comparison, including symbolic comparisons.
* Powers use a numeric double exponent. NumPy random seeds use ``rng``;
  ``random_state`` remains accepted. C++ sampling uses its standard RNG.
* Copy-control and array-backend options are excluded.

Davenport angles, means, vector alignment, symmetry groups/reduction, and
``RotationSpline`` are outside this release's scope.

Migration from 1.x
------------------

Spatial classes remain available from the top level and ``spatial_casadi.spatial``.
``Translation`` and ``Transformation`` delegate to the shared core.
``Transformation(rotation, translation)`` retains its constructor order and
``rotation()`` / ``translation()`` accessors.

New code can use ``RigidTransform.from_components(translation, rotation)``
and its ``rotation`` / ``translation`` properties. Legacy quaternion-order
strings, ``Rotation.from_quat(q, "wxyz")`` and ``r.as_quat("wxyz")``, still work.
Prefer ``scalar_first=True`` for SciPy-style calls.

Zero-angle conversions and derivatives now work. Symbolic matrix conversion
now normalizes its quaternion. Invalid numerical inputs raise ``ValueError``
instead of producing undefined rotations. A single rotation's ``len`` raises
``TypeError``, following SciPy.

The standalone ``deg2rad`` and ``rad2deg`` helpers have been removed.
Use the rotation methods' ``degrees=True`` option, NumPy for numeric angle
conversion, or ordinary CasADi expressions for symbolic conversion.
