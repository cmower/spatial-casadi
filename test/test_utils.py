import numpy as np
import casadi as cs
from spatial_casadi import deg2rad, rad2deg

NUM_RANDOM = 100


def test_deg2rad_symbolic():
    x = cs.SX.sym("x", 10)
    fun = cs.Function("deg2rad", [x], [deg2rad(x)])

    for _ in range(NUM_RANDOM):
        xr = np.random.random(size=(10,))

        assert np.allclose(fun(xr).toarray().flatten(), np.deg2rad(xr))
