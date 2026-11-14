"""Fit a rotation to three point correspondences."""

import casadi as cs
import numpy as np

from spatial_casadi import Rotation

opti = cs.Opti()
rotvec = opti.variable(3)
points = cs.DM.eye(3)
target = Rotation.from_rotvec([0.2, -0.1, 0.3]).apply(points)
residual = Rotation.from_rotvec(rotvec).apply(points) - target
opti.minimize(cs.sumsqr(residual))
opti.solver(
    "sqpmethod",
    {
        "qpsol": "qrqp",
        "print_header": False,
        "print_iteration": False,
        "print_status": False,
        "print_time": False,
        "qpsol_options": {"print_header": False, "print_iter": False},
    },
)
solution = opti.solve()
result = np.asarray(solution.value(rotvec))
np.testing.assert_allclose(result, [0.2, -0.1, 0.3], atol=1e-6)
print(result)
