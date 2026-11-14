"""Structural budgets catch accidental expression growth, independent of CPU speed."""

import casadi as cs
import pytest

from spatial_casadi import RigidTransform, Rotation


@pytest.mark.parametrize("kind", [cs.SX, cs.MX])
def test_rotation_chain_growth(kind):
    counts = []
    for length in [10, 20]:
        x = kind.sym("x", 3, length)
        r = Rotation.identity()
        for i in range(length):
            r = r * Rotation.from_euler("xyz", x[:, i])
        f = cs.Function("chain", [x], [r.as_matrix()])
        counts.append(f.n_nodes())
    assert counts[1] < 2.2 * counts[0]
    assert counts[1] < 1350


@pytest.mark.parametrize("kind", [cs.SX, cs.MX])
def test_transform_inverse_avoids_matrix_conversion(kind):
    q, t = kind.sym("q", 4), kind.sym("t", 3)
    transform = RigidTransform.from_components(t, Rotation(q, normalize=False))
    inverse = transform.inv()
    f = cs.Function(
        "inverse", [q, t], [inverse.rotation.as_quat(), inverse.translation]
    )
    assert f.n_nodes() < 120
