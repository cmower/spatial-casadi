import shutil
import subprocess

import casadi as cs
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation as SR

from spatial_casadi import RigidTransform as T
from spatial_casadi import Rotation as R
from spatial_casadi import Slerp


def evaluate_function(kind, operation, dimension):
    x = kind.sym("x", dimension)
    y = cs.vec(operation(x))
    weights = cs.DM(np.arange(1, y.numel() + 1))
    cost = cs.dot(y, weights)
    return cs.Function("f", [x], [y, cs.jacobian(y, x), cs.hessian(cost, x)[0]])


def check_derivatives(f, point, step=2e-6, atol=2e-6):
    point = np.asarray(point, dtype=float)
    value, jac, hess = [a.full() for a in f(point)]
    assert all(np.isfinite(a).all() for a in (value, jac, hess))
    identity = np.eye(len(point))
    pairs = [[a.full() for a in f(point + step * d)] for d in identity]
    minus = [[a.full() for a in f(point - step * d)] for d in identity]
    fd_jac = np.column_stack(
        [(p[0] - m[0]).ravel() / (2 * step) for p, m in zip(pairs, minus)]
    )
    weights = np.arange(1, value.size + 1)
    fd_hess = np.column_stack(
        [weights @ (p[1] - m[1]) / (2 * step) for p, m in zip(pairs, minus)]
    )
    assert_allclose(jac, fd_jac, atol=atol, rtol=2e-6)
    assert_allclose(hess, fd_hess, atol=atol, rtol=2e-6)


@pytest.mark.parametrize("kind", [cs.SX, cs.MX])
@pytest.mark.parametrize(
    "operation,dimension",
    [
        (lambda x: R.from_rotvec(x).as_quat(), 3),
        (lambda x: R.from_rotvec(x).as_matrix(), 3),
        (lambda x: R.from_rotvec(x).as_rotvec(), 3),
        (lambda x: R.from_matrix(R.from_rotvec(x).as_matrix()).as_matrix(), 3),
        (lambda x: R.from_rotvec(x).apply([0.2, 1, -0.7]), 3),
        (lambda x: T.from_exp_coords(x).as_matrix(), 6),
        (lambda x: T.from_exp_coords(x).as_exp_coords(), 6),
        (
            lambda x: T.from_matrix(T.from_exp_coords(x).as_matrix()).inv().as_matrix(),
            6,
        ),
        (
            lambda x: T.from_dual_quat(T.from_exp_coords(x).as_dual_quat()).as_matrix(),
            6,
        ),
    ],
)
def test_jacobians_and_hessians(kind, operation, dimension):
    f = evaluate_function(kind, operation, dimension)
    for scale in [0, 1e-10, 1e-4, 0.2]:
        point = np.linspace(0.2, 0.7, dimension) * scale
        check_derivatives(f, point)


@pytest.mark.parametrize("kind", [cs.SX, cs.MX])
def test_zero_translation_and_rotation_limits(kind):
    x = kind.sym("x", 6)
    coordinates = T.from_exp_coords(x).as_exp_coords()
    f = cs.Function("f", [x], [coordinates, cs.jacobian(coordinates, x)])
    for value in [np.zeros(6), [0, 0, 0, 1, 2, 3]]:
        output, jac = f(value)
        assert_allclose(output.full().ravel(), value, atol=1e-14)
        assert_allclose(jac.full(), np.eye(6), atol=1e-14)
    half = R.from_rotvec(kind([np.pi, 0, 0])).as_rotvec()
    assert_allclose(
        cs.Function("half", [], [half])()["o0"].full().ravel(),
        [np.pi, 0, 0],
        atol=1e-14,
    )


@pytest.mark.parametrize("kind", [cs.SX, cs.MX])
def test_euler_quaternion_and_slerp_derivatives(kind):
    check_derivatives(
        evaluate_function(kind, lambda x: R.from_euler("xyz", x).as_euler("xyz"), 3),
        [0.2, 0.3, -0.4],
    )
    check_derivatives(
        evaluate_function(kind, lambda x: R.from_quat(x).as_matrix(), 4),
        [0.1, -0.2, 0.3, 0.9],
    )
    interpolation = Slerp(
        [0, 1, 2], R.from_rotvec([[0, 0, 0], [0.2, 0.3, 0.1], [0.5, 0.2, 0.3]])
    )
    check_derivatives(
        evaluate_function(kind, lambda x: interpolation(x).as_matrix(), 1), [0.4]
    )
    constant = Slerp([0, 1], R.identity(2))
    check_derivatives(
        evaluate_function(kind, lambda x: constant(x).as_matrix(), 1), [0.4]
    )


@pytest.mark.parametrize("kind", [cs.SX, cs.MX])
def test_near_pi_and_branch_switches(kind):
    f = evaluate_function(
        kind, lambda x: R.from_matrix(R.from_rotvec(x).as_matrix()).as_matrix(), 3
    )
    # Compare rotations across quaternion extraction branches, not quaternion signs.
    for point in [[np.pi - 1e-6, 0, 0], [0, np.pi + 1e-6, 0], [1.21, 1.21, 1.21]]:
        check_derivatives(f, point, atol=2e-5)
        assert_allclose(
            f(point)[0].full().ravel().reshape(3, 3, order="F"),
            SR.from_rotvec(point).as_matrix(),
            atol=1e-14,
        )


@pytest.mark.parametrize("kind", [cs.SX, cs.MX])
def test_generated_code(kind, tmp_path, monkeypatch):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler unavailable")
    monkeypatch.chdir(tmp_path)
    f = evaluate_function(kind, lambda x: T.from_exp_coords(x).as_matrix(), 6)
    f.generate("generated.c")
    subprocess.run(
        [
            compiler,
            "-O2",
            "-shared",
            "-fPIC",
            "generated.c",
            "-lm",
            "-o",
            "generated.so",
        ],
        check=True,
    )
    compiled = cs.external("f", str(tmp_path / "generated.so"))
    for point in [np.zeros(6), np.arange(6) * 0.1]:
        for actual, expected in zip(compiled(point), f(point)):
            assert_allclose(actual, expected, atol=1e-13)
    f.save(str(tmp_path / "function.casadi"))
    restored = cs.Function.load(str(tmp_path / "function.casadi"))
    for actual, expected in zip(restored(np.zeros(6)), f(np.zeros(6))):
        assert_allclose(actual, expected, atol=1e-13)


@pytest.mark.parametrize("kind", [cs.SX, cs.MX])
def test_small_angle_transition(kind):
    for operation in [
        lambda x: T.from_exp_coords(x).as_matrix(),
        lambda x: T.from_exp_coords(x).as_exp_coords(),
    ]:
        f = evaluate_function(kind, operation, 6)
        for angle in [1.001e-4, 1e-3, 0.01, 0.099999, 0.100001]:
            check_derivatives(f, [angle, 0, 0, 1, 2, 3], atol=2e-5)
