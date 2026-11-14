C++ API
=======

Include ``<spatial_casadi/spatial.hpp>`` and instantiate the classes with
``casadi::DM``, ``casadi::SX`` or ``casadi::MX``. C++ uses the same column/block
layouts as Python's CasADi inputs. Use ``compose`` or ``operator*``, ``power``
and ``at`` for composition, powers and indexing. ``size()`` reports batch size.

.. code-block:: cpp

   using casadi::SX;
   using spatial_casadi::RigidTransform;
   SX x = SX::sym("x", 6);
   auto pose = RigidTransform<SX>::from_exp_coords(x);
   SX point = pose.apply(SX(std::vector<double>{1, 0, 0}));
   casadi::Function f("point", {x}, {point, SX::jacobian(point, x)});

.. doxygenclass:: spatial_casadi::Rotation
   :members:

.. doxygenclass:: spatial_casadi::RigidTransform
   :members:

.. doxygenclass:: spatial_casadi::Slerp
   :members:

.. doxygenclass:: spatial_casadi::Translation
   :members:
