spatial-casadi
==============

Rotations and rigid transforms for CasADi, with a shared C++ implementation
and a SciPy-style Python interface. Expressions support automatic
differentiation, serialization, and CasADi code generation.

.. code-block:: python

   import casadi as cs
   from spatial_casadi import Rotation, RigidTransform

   x = cs.SX.sym("x", 6)
   pose = RigidTransform.from_exp_coords(x)
   point = pose.apply([1, 0, 0])
   f = cs.Function("point", [x], [point, cs.jacobian(point, x)])

.. toctree::
   :maxdepth: 1

   installation
   conventions
   compatibility
   examples
   python
   cpp
   development
