import numpy as np
from spatial_casadi import Transformation
from scipy.spatial.transform import Rotation as Rot

NUM_RANDOM = 100


def test_Transformation_inv():
    for i in range(NUM_RANDOM):
        t = Transformation.random()
        invt = t.inv().as_matrix().toarray()
        INVT = np.linalg.inv(t.as_matrix().toarray())

        print("----- Test", i + 1, "/", NUM_RANDOM)
        print("Scipy:\n", INVT)
        print("Lib:\n", invt)

        assert np.allclose(invt, INVT)


def test_Transformation_identity():
    assert np.allclose(Transformation.identity().as_matrix().toarray(), np.eye(4))


def test_Transformation_from_matrix():
    for i in range(NUM_RANDOM):
        T = np.eye(4)
        T[:3, :3] = Rot.random().as_matrix()
        T[:3, 3] = np.random.normal(size=(3,))
        t = Transformation.from_matrix(T).as_matrix().toarray()

        print("----- Test", i + 1, "/", NUM_RANDOM)
        print("Scipy:\n", T)
        print("Lib:\n", t)

        assert np.allclose(t, T)


def test_Transformation_mul():
    for i in range(NUM_RANDOM):
        ta = Transformation.random()
        tb = Transformation.random()

        Ta = ta.as_matrix().toarray()
        Tb = tb.as_matrix().toarray()

        t = (ta * tb).as_matrix().toarray()
        T = Ta @ Tb

        assert np.allclose(t, T)
