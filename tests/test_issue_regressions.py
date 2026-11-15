"""Regressions for GitHub issues #3 (normalization) and #5 (identity rotvec)."""

import casadi as cs
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation as SciPyRotation

from spatial_casadi import Rotation


def evaluate(kind, values, operation):
    if kind is cs.DM:
        return operation(cs.DM(values)).full()
    x = kind.sym("x", *values.shape)
    return cs.Function("evaluate", [x], [operation(x)])(values).full()


@pytest.mark.parametrize("kind", [cs.DM, cs.SX, cs.MX])
def test_issue_5_identity_rotvec(kind):
    for sign in [1, -1]:
        values = np.array([[0], [0], [0], [sign]])
        actual = evaluate(kind, values, lambda q: Rotation.from_quat(q).as_rotvec())
        assert_allclose(actual, np.zeros((3, 1)), atol=0)


@pytest.mark.parametrize("kind", [cs.SX, cs.MX])
def test_issue_5_identity_quaternion_derivatives(kind):
    q = kind.sym("q", 4)
    v = Rotation.from_quat(q).as_rotvec()
    f = cs.Function(
        "derivatives", [q], [cs.jacobian(v, q), cs.hessian(cs.sumsqr(v), q)[0]]
    )
    for sign in [1, -1]:
        jacobian, hessian = f([0, 0, 0, sign])
        assert_allclose(
            jacobian.full(), np.column_stack([2 * sign * np.eye(3), np.zeros(3)])
        )
        assert_allclose(hessian.full(), np.diag([8, 8, 8, 0]))


@pytest.mark.parametrize("kind", [cs.DM, cs.SX, cs.MX])
@pytest.mark.parametrize("scalar_first", [False, True])
def test_issue_3_quaternion_input_normalization(kind, scalar_first):
    values = np.array([[1, 2, 3, 4], [0, 0, 0, 3], [2, 0, 0, 0]], dtype=float).T
    expected = values / np.linalg.norm(values, axis=0)
    supplied = values[[3, 0, 1, 2]] if scalar_first else values
    actual = evaluate(
        kind,
        supplied,
        lambda q: Rotation.from_quat(q, scalar_first=scalar_first).as_quat(),
    )
    assert_allclose(actual, expected, atol=1e-15)
    assert_allclose(np.linalg.norm(actual, axis=0), 1, atol=1e-15)


@pytest.mark.parametrize("kind", [cs.DM, cs.SX, cs.MX])
def test_issue_3_matrix_quaternion_normalization(kind):
    # Exercise the trace branch and each largest-diagonal extraction branch.
    vectors = [[0, 0, 0], [np.pi, 0, 0], [0, np.pi, 0], [0, 0, np.pi], [0.2, -0.3, 0.4]]
    matrices = SciPyRotation.from_rotvec(vectors).as_matrix()
    quaternions = evaluate(
        kind, np.hstack(matrices), lambda m: Rotation.from_matrix(m).as_quat()
    )
    assert_allclose(np.linalg.norm(quaternions, axis=0), 1, atol=1e-15)
    assert_allclose(
        SciPyRotation.from_quat(quaternions.T).as_matrix(), matrices, atol=1e-14
    )
