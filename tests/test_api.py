import itertools
import pickle
import casadi as cs
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation as SR, Slerp as SS
import scipy.spatial.transform as scipy_transform
ST = getattr(scipy_transform, "RigidTransform", None)
from spatial_casadi import Rotation as R, RigidTransform as T, Slerp, Translation, Transformation


def blocks(array):
    return np.hstack(array) if array.ndim == 3 else array


@pytest.mark.parametrize('kind', [cs.DM, cs.SX, cs.MX])
def test_conversions_and_batches(kind):
    reference = SR.random(7, rng=np.random.default_rng(41))
    for ctor, values in [(R.from_quat, reference.as_quat()), (R.from_rotvec, reference.as_rotvec()),
                         (R.from_mrp, reference.as_mrp())]:
        actual = ctor(kind(values.T))
        output = actual.as_matrix()
        if kind != cs.DM:
            output = cs.Function('constant', [], [output])()['o0']
        assert_allclose(output.full(), blocks(reference.as_matrix()), atol=2e-14)
    actual = R.from_matrix(kind(blocks(reference.as_matrix())))
    output = actual.as_matrix()
    if kind != cs.DM:
        output = cs.Function('constant', [], [output])()['o0']
    assert_allclose(output.full(), blocks(reference.as_matrix()), atol=2e-14)


SEQUENCES = [''.join(s) for s in itertools.product('xyz', repeat=3) if s[0] != s[1] and s[1] != s[2]]


@pytest.mark.parametrize('seq', SEQUENCES + [s.upper() for s in SEQUENCES])
def test_all_euler_sequences(seq):
    reference = SR.random(11, rng=np.random.default_rng(17))
    r = R.from_euler(seq, reference.as_euler(seq))
    assert_allclose(r.as_matrix().full(), blocks(reference.as_matrix()), atol=2e-14)
    assert_allclose(r.as_euler(seq).full().T, reference.as_euler(seq), atol=2e-14)
    singular = [0, np.pi] if seq[0] == seq[2] else [-np.pi/2, np.pi/2]
    for middle in singular:
        r = R.from_euler(seq, [0.3, middle, -0.8])
        with pytest.warns(UserWarning, match='Gimbal lock'):
            angles = r.as_euler(seq)
        assert_allclose(R.from_euler(seq, angles).as_matrix(), r.as_matrix(), atol=1e-14)


def test_collection_and_orders():
    r = R.from_quat([[1, 0, 0, 0], [0, 0, 0, 2]], scalar_first=True)
    assert not r.single and len(r) == 2 and r[0].single
    assert not r[:1].single and len(r[:1]) == 1
    assert_allclose(r[0].as_matrix(), np.eye(3))
    assert_allclose(r.as_quat(scalar_first=True), [[1,0],[0,0],[0,0],[0,1]])
    assert_allclose(R.from_quat([-1,0,0,0]).as_quat(canonical=True), [[1],[0],[0],[0]])
    assert len(R.concatenate([r[0], r[1]])) == 2
    assert R.identity().single and not R.identity(1).single
    assert R.from_euler('x', 0).single and not R.from_euler('x', [0]).single
    assert_allclose(pickle.loads(pickle.dumps(r)).as_matrix(), r.as_matrix())
    assert_allclose(R.random(3, rng=42).as_quat(), R.random(3, rng=42).as_quat())
    assert_allclose((r**0).as_matrix(), np.hstack([np.eye(3)]*2))
    with pytest.raises(ValueError): r[0:0]
    with pytest.raises(TypeError): len(r[0])


@pytest.mark.skipif(ST is None, reason='RigidTransform requires SciPy 1.16')
@pytest.mark.parametrize('count', [1, 8])
def test_rigid_transform_against_scipy(count):
    rng = np.random.default_rng(18)
    exp = rng.normal(size=(count, 6))
    reference = ST.from_exp_coords(exp)
    actual = T.from_exp_coords(exp)
    assert_allclose(actual.as_matrix().full(), blocks(reference.as_matrix()), atol=2e-14)
    assert_allclose(actual.as_exp_coords().full().T, reference.as_exp_coords(), atol=2e-14)
    for scalar_first in [False, True]:
        dq = actual.as_dual_quat(scalar_first=scalar_first)
        expected = reference.as_dual_quat(scalar_first=scalar_first)
        # Quaternion sign is free, so compare reconstructed transforms too.
        assert_allclose(dq.full().T, expected, atol=2e-14)
        assert_allclose(T.from_dual_quat(dq, scalar_first=scalar_first).as_matrix(), actual.as_matrix(), atol=2e-14)
        assert_allclose(T.from_dual_quat(2*dq, scalar_first=scalar_first).as_matrix(), actual.as_matrix(), atol=2e-14)
    assert_allclose(T.from_matrix(reference.as_matrix()).as_matrix(), actual.as_matrix(), atol=2e-14)
    assert_allclose(actual.inv().as_matrix().full(), blocks(reference.inv().as_matrix()), atol=2e-14)
    assert_allclose((actual**0.4).as_matrix().full(), blocks(ST.from_exp_coords(reference.as_exp_coords()*0.4).as_matrix()), atol=2e-14)
    vectors = rng.normal(size=(count, 3))
    for inverse in [False, True]:
        assert_allclose(actual.apply(vectors, inverse).full().T, reference.apply(vectors, inverse=inverse), atol=2e-14)
    assert_allclose((actual*actual.inv()).as_matrix(), np.hstack([np.eye(4)]*count), atol=2e-14)
    assert_allclose(T.concatenate([actual[i] for i in range(count)]).as_matrix(), actual.as_matrix())


@pytest.mark.skipif(ST is None, reason='RigidTransform requires SciPy 1.16')
def test_broadcasting_and_legacy():
    r = R.from_rotvec([[0.2, 0, 0], [0, 0.4, 0]])
    single = R.from_rotvec([0, 0, 0.5])
    reference = SR.from_quat(r.as_quat().full().T)
    sr = SR.from_quat(single.as_quat().full().ravel())
    assert_allclose((r*single).as_matrix().full(), blocks((reference*sr).as_matrix()), atol=1e-14)
    tf = T.from_components([1, 2, 3], r)
    assert len(tf) == 2
    assert_allclose(tf.apply([1, 0, 0]).full().T, reference.apply([1,0,0])+[1,2,3], atol=1e-14)
    single_tf = T.from_components([0, 2, 1], single)
    for a, b in [(tf, single_tf), (single_tf, tf)]:
        expected = ST.from_matrix(np.array([a.as_matrix().full()[:,i:i+4] for i in range(0,a.as_matrix().size2(),4)]))*ST.from_matrix(np.array([b.as_matrix().full()[:,i:i+4] for i in range(0,b.as_matrix().size2(),4)]))
        assert_allclose((a*b).as_matrix().full(), blocks(expected.as_matrix()), atol=1e-14)
    old = Transformation(single, Translation([1,2,3]))
    assert_allclose(old.flatten(), cs.vertcat(single.as_quat(), cs.DM([1,2,3])))
    assert_allclose(old.inv().as_matrix(), np.linalg.inv(old.as_matrix().full()))


def test_slerp():
    times = [0, 0.4, 2, 3]
    rotations = SR.random(4, rng=np.random.default_rng(11))
    queries = np.linspace(0,3,15)
    ours = Slerp(times, R.from_quat(rotations.as_quat()))
    assert_allclose(ours(queries).as_matrix().full(), blocks(SS(times, rotations)(queries).as_matrix()), atol=2e-14)
    assert ours(1).single and not ours([1]).single
    identity = Slerp([0, 1], R.identity(2))
    assert_allclose(identity(0.5).as_matrix(), np.eye(3))
    with pytest.raises(ValueError): ours(-1)
    with pytest.raises(ValueError): Slerp([0,0], R.identity(2))


@pytest.mark.parametrize('make', [lambda: R.from_quat([0,0,0,0]),
    lambda: R.from_quat([0,0,0,np.inf]), lambda: R.from_matrix(np.diag([1,1,-1])),
    lambda: R.from_matrix(np.ones((3,3))), lambda: T.from_matrix(np.zeros((4,4))),
    lambda: R.from_euler('xx', [0,0]), lambda: R.from_euler('xY', [0,0]),
    lambda: R.identity(2).apply(np.ones((3,3))), lambda: T.from_dual_quat(np.zeros(8)),
    lambda: R.from_quat(cs.SX.sym('q',4))*R.from_quat(cs.MX.sym('q',4))])
def test_invalid_inputs(make):
    with pytest.raises((ValueError, TypeError)):
        make()
