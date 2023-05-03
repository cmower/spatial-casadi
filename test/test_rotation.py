import numpy as np
import casadi as cs
from spatial_casadi import Rotation
from scipy.spatial.transform import Rotation as Rot


NUM_RANDOM = 100

EULER_SEQS = (
    "xyz",
    "zyx",
    "xyx",
    "zyz",
    "xzy",
)


def test_symbolic():
    quat = cs.SX.sym("quat", 4)
    rot = Rotation(quat)

    assert isinstance(rot.as_quat(), cs.SX)
    assert rot.as_quat().shape == (4, 1)

    assert isinstance(rot.as_matrix(), cs.SX)
    assert rot.as_matrix().shape == (3, 3)

    assert isinstance(rot.as_rotvec(), cs.SX)
    assert rot.as_rotvec().shape == (3, 1)

    assert isinstance(rot.as_mrp(), cs.SX)
    assert rot.as_mrp().shape == (3, 1)

    assert isinstance(rot.as_euler("xyz"), cs.SX)
    assert rot.as_euler("xyz").shape == (3, 1)


def test_Rotation_mul():
    for _ in range(NUM_RANDOM):
        r1 = Rotation.random()
        r2 = Rotation.random()

        result = r1 * r2

        R1 = Rot.from_quat(r1.as_quat().toarray().flatten())
        R2 = Rot.from_quat(r2.as_quat().toarray().flatten())

        result_test = R1 * R2

        test1 = np.allclose(result.as_quat().toarray().flatten(), result_test.as_quat())
        test2 = np.allclose(
            -result.as_quat().toarray().flatten(), result_test.as_quat()
        )
        assert test1 or test2


def test_Rotation_from_quat_seq():
    for _ in range(NUM_RANDOM):
        r1 = Rotation.random()
        r2 = Rotation.random()

        q1_sf = r1.as_quat("wxyz")
        q2_sf = r2.as_quat("wxyz")

        r1_sf = Rotation.from_quat(q1_sf, "wxyz")
        r2_sf = Rotation.from_quat(q2_sf, "wxyz")

        q1_sl = r1.as_quat("xyzw")
        q2_sl = r2.as_quat("xyzw")

        r1_sl = Rotation.from_quat(q1_sl)  # scalar-last is default seq
        r2_sl = Rotation.from_quat(q2_sl)  # scalar-last is default seq

        prod_sf = r1_sf * r2_sf
        prod_sl = r1_sl * r2_sl

        assert np.allclose(prod_sf.as_quat().toarray(), prod_sl.as_quat().toarray())


def test_Rotation_identity():
    assert np.allclose(
        Rotation.identity().as_quat().toarray().flatten(), np.array([0, 0, 0, 1])
    )


def test_Rotation_inv():
    for _ in range(NUM_RANDOM):
        r = Rotation.random()
        R = Rot.from_quat(r.as_quat().toarray().flatten())
        assert np.allclose(r.inv().as_quat().toarray().flatten(), R.inv().as_quat())


def test_Rotation_from_matrix():
    for i in range(NUM_RANDOM):
        R = Rot.random()
        r = Rotation.from_matrix(R.as_matrix())

        Q = R.as_quat()
        q = r.as_quat().toarray().flatten()

        print("Test:", i + 1, "/", NUM_RANDOM)
        print("  Scipy:", Q)
        print("  Lib:  ", q)

        test1 = np.allclose(q, Q)
        test2 = np.allclose(-q, Q)

        assert test1 or test2


def test_Rotation_from_rotvec():
    for i in range(NUM_RANDOM):
        R = Rot.random()
        r = Rotation.from_rotvec(R.as_rotvec())

        Q = R.as_quat()
        q = r.as_quat().toarray().flatten()

        print("Test:", i + 1, "/", NUM_RANDOM)
        print("  Scipy:", Q)
        print("  Lib:  ", q)

        test1 = np.allclose(q, Q)
        test2 = np.allclose(-q, Q)

        assert test1 or test2


def test_Rotation_from_mrp():
    for i in range(NUM_RANDOM):
        R = Rot.random()
        r = Rotation.from_mrp(R.as_mrp())

        Q = R.as_quat()
        q = r.as_quat().toarray().flatten()

        print("Test:", i + 1, "/", NUM_RANDOM)
        print("  Scipy:", Q)
        print("  Lib:  ", q)

        test1 = np.allclose(q, Q)
        test2 = np.allclose(-q, Q)

        assert test1 or test2


def test_Rotation_from_euler():
    for seq in EULER_SEQS:
        for i in range(NUM_RANDOM):
            R = Rot.random()
            r = Rotation.from_euler(seq, R.as_euler(seq))

            Q = R.as_quat()
            q = r.as_quat().toarray().flatten()

            print(f"Test ({seq}):", i + 1, "/", NUM_RANDOM)
            print("  Scipy:", Q)
            print("  Lib:  ", q)

            test1 = np.allclose(q, Q)
            test2 = np.allclose(-q, Q)
            assert test1 or test2

        seq = seq.upper()
        for i in range(NUM_RANDOM):
            R = Rot.random()
            r = Rotation.from_euler(seq, R.as_euler(seq))

            Q = R.as_quat()
            q = r.as_quat().toarray().flatten()

            print(f"Test ({seq}):", i + 1, "/", NUM_RANDOM)
            print("  Scipy:", Q)
            print("  Lib:  ", q)

            test1 = np.allclose(q, Q)
            test2 = np.allclose(-q, Q)
            assert test1 or test2


def test_Rotation_as_quat():
    for i in range(NUM_RANDOM):
        R = Rot.random()
        r = Rotation(R.as_quat())

        Q = R.as_quat()
        q = r.as_quat().toarray().flatten()

        print("Test:", i + 1, "/", NUM_RANDOM)
        print("  Scipy:", Q)
        print("  Lib:  ", q)

        assert np.allclose(q, Q)


def test_Rotation_as_matrix():
    for i in range(NUM_RANDOM):
        R = Rot.random()
        r = Rotation(R.as_quat())

        M = R.as_matrix()
        m = r.as_matrix().toarray()

        print("Test:", i + 1, "/", NUM_RANDOM)
        print("  Scipy:", M)
        print("  Lib:  ", m)

        assert np.allclose(m, M)


def test_Rotation_as_rotvec():
    for i in range(NUM_RANDOM):
        R = Rot.random()
        r = Rotation(R.as_quat())

        RV = R.as_rotvec()
        rv = r.as_rotvec().toarray().flatten()

        print("Test:", i + 1, "/", NUM_RANDOM)
        print("  Scipy:", RV)
        print("  Lib:  ", rv)

        assert np.allclose(rv, RV)


def test_Rotation_as_mrp():
    for i in range(NUM_RANDOM):
        R = Rot.random()
        r = Rotation(R.as_quat())

        MRP = R.as_mrp()
        mrp = r.as_mrp().toarray().flatten()

        print("Test:", i + 1, "/", NUM_RANDOM)
        print("  Scipy:", MRP)
        print("  Lib:  ", mrp)

        assert np.allclose(mrp, MRP)


def test_Rotation_as_euler():
    for seq in EULER_SEQS:
        for i in range(NUM_RANDOM):
            R = Rot.random()
            r = Rotation(R.as_quat())

            EULER = R.as_euler(seq)
            euler = r.as_euler(seq).toarray().flatten()

            print(f"Test({seq}):", i + 1, "/", NUM_RANDOM)
            print("  Scipy:", EULER)
            print("  Lib:  ", euler)

            assert np.allclose(euler, EULER)

        seq = seq.upper()
        for i in range(NUM_RANDOM):
            R = Rot.random()
            r = Rotation(R.as_quat())

            EULER = R.as_euler(seq)
            euler = r.as_euler(seq).toarray().flatten()

            print(f"Test({seq}):", i + 1, "/", NUM_RANDOM)
            print("  Scipy:", EULER)
            print("  Lib:  ", euler)

            assert np.allclose(euler, EULER)
