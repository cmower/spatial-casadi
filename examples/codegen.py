"""Generate a C function for a transformed point and its Jacobian."""

import casadi as cs

from spatial_casadi import RigidTransform

x = cs.SX.sym("x", 6)
point = RigidTransform.from_exp_coords(x).apply([1, 0, 0])
f = cs.Function("point", [x], [point, cs.jacobian(point, x)])
f.generate("point.c", {"with_header": True})
