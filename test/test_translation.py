import numpy as np
from spatial_casadi import Translation, Transformation, Rotation

NUM_RANDOM = 100


def test_Translation_add():
    for _ in range(NUM_RANDOM):
        ta = Translation.random()
        tb = Translation.random()

        Ta = ta.as_vector().toarray().flatten()
        Tb = tb.as_vector().toarray().flatten()

        t = (ta + tb).as_vector().toarray().flatten()
        T = Ta + Tb

        assert np.allclose(t, T)


def test_Translation_neg():
    for _ in range(NUM_RANDOM):
        t = Translation.random()
        T = t.as_vector().toarray().flatten()

        nt = (-t).as_vector().toarray().flatten()
        nT = (-T).copy()

        assert np.allclose(nt, nT)


def test_Translation_sub():
    for _ in range(NUM_RANDOM):
        ta = Translation.random()
        tb = Translation.random()

        Ta = ta.as_vector().toarray().flatten()
        Tb = tb.as_vector().toarray().flatten()

        t = (ta - tb).as_vector().toarray().flatten()
        T = Ta - Tb

        assert np.allclose(t, T)


def test_Translation_identity():
    assert np.allclose(
        Translation.identity().as_vector().toarray().flatten(), np.zeros(3)
    )


def test_Translation_from_vector():
    for _ in range(NUM_RANDOM):
        T = np.random.normal(size=(3,))
        t = Translation.from_vector(T)
        assert np.allclose(t.as_vector().toarray().flatten(), T)


def test_Translation_from_matrix():
    for _ in range(NUM_RANDOM):
        T = Transformation.random().as_matrix().toarray()
        t = Translation.from_matrix(T)
        assert np.allclose(t.as_vector().toarray().flatten(), T[:3, 3])


def test_Translation_as_matrix():
    for _ in range(NUM_RANDOM):
        T = Transformation.random().as_matrix().toarray()
        T[:3, :3] = Rotation.identity().as_matrix().toarray()
        t = Translation.from_matrix(T)
        assert np.allclose(t.as_matrix().toarray(), T)
