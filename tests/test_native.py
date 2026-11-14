"""The binding must share CasADi's types and original symbolic variables."""

import casadi as cs
import numpy as np
import pytest

from spatial_casadi import native


@pytest.mark.parametrize("kind", ["SX", "MX"])
def test_symbol_identity_and_derivatives(kind):
    typ = getattr(cs, kind)
    cls = getattr(native, "Rotation" + kind)
    x = typ.sym("x", 3)
    q = cls.from_rotvec(x).as_quat()
    f = cs.Function("f", [x], [q, cs.jacobian(q, x), cs.hessian(q[3], x)[0]])
    value, jac, hess = f([0, 0, 0])
    np.testing.assert_allclose(value.full().ravel(), [0, 0, 0, 1])
    np.testing.assert_allclose(
        jac.full(), np.vstack([0.5 * np.eye(3), np.zeros((1, 3))])
    )
    np.testing.assert_allclose(hess.full(), -0.25 * np.eye(3))
    identity = cls.from_matrix(typ.eye(3)).as_matrix()
    np.testing.assert_allclose(
        cs.Function("identity", [], [identity])()["o0"], np.eye(3)
    )


def test_dm_roundtrip_and_lifetime():
    q = cs.DM([1, 2, 3, 4])
    rotation = native.RotationDM(q)
    del q
    result = rotation.as_quat()
    del rotation
    assert isinstance(result, cs.DM)
    np.testing.assert_allclose(cs.norm_2(result), 1)
