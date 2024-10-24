import re
import numpy as np
import casadi as cs
from spatial_casadi.angles import deg2rad


def _make_elementary_quat(axis, angles):
    """
    Create an elementary quaternion representing a rotation around a specified axis.

    Args
        axis (str): The axis to rotate around. Must be 'x', 'y', or 'z'.
        angles (array-like): The rotation angle(s) in radians.

    Returns
        casadi-array: A quaternion representing the rotation.
    """
    quat = [0.0, 0.0, 0.0, 0.0]

    if axis == "x":
        axis_ind = 0
    elif axis == "y":
        axis_ind = 1
    elif axis == "z":
        axis_ind = 2

    quat[3] = cs.cos(angles / 2.0)
    quat[axis_ind] = cs.sin(angles / 2.0)

    return cs.vertcat(*quat)


def _compose_quat(p, q):
    """
    Compose two quaternions by performing quaternion multiplication.

    Args
        p (array-like): The first quaternion as an array-like object.
        q (array-like): The second quaternion as an array-like object.

    Returns
        casadi-array: The resulting quaternion from the multiplication of p and q.
    """
    cross = cs.cross(p[:3], q[:3])
    return cs.vertcat(
        p[3] * q[0] + q[3] * p[0] + cross[0],
        p[3] * q[1] + q[3] * p[1] + cross[1],
        p[3] * q[2] + q[3] * p[2] + cross[2],
        p[3] * q[3] - p[0] * q[0] - p[1] * q[1] - p[2] * q[2],
    )


def _elementary_quat_compose(seq, angles, intrinsic):
    """
    Compose a sequence of elementary quaternions into a single quaternion.

    Args
        seq (array-like): A sequence of axes ('x', 'y', 'z') around which to rotate.
        angles (array-like): A sequence of rotation angles in radians corresponding to each axis in `seq`.
        intrinsic (bool): If `True`, applies intrinsic rotations; otherwise, applies extrinsic rotations.

    Returns
        casadi-array: The resulting quaternion from composing the sequence of elementary quaternions.
    """
    result = _make_elementary_quat(seq[0], angles[0])
    seq_len = len(seq)
    for idx in range(1, seq_len):
        if intrinsic:
            result = _compose_quat(result, _make_elementary_quat(seq[idx], angles[idx]))
        else:
            result = _compose_quat(_make_elementary_quat(seq[idx], angles[idx]), result)
    return result


def _single_matrix_to_quat(matrix):
    matrix = cs.horzcat(matrix)
    shape = matrix.shape
    if shape != (3, 3):
        raise ValueError(f"expected matrix with shape (3, 3), instead got {shape}")

    decision = cs.vertcat(
        matrix[0, 0],
        matrix[1, 1],
        matrix[2, 2],
        matrix[0, 0] + matrix[1, 1] + matrix[2, 2],
    )

    true_case_3 = cs.vertcat(
        matrix[2, 1] - matrix[1, 2],
        matrix[0, 2] - matrix[2, 0],
        matrix[1, 0] - matrix[0, 1],
        1.0 + decision[3],
    )

    def alt_true_case(i, j, k):
        quat = [None, None, None, None]
        quat[i] = 1.0 - decision[3] + 2 * matrix[i, i]
        quat[j] = matrix[j, i] + matrix[i, j]
        quat[k] = matrix[k, i] + matrix[i, k]
        quat[3] = matrix[k, j] - matrix[j, k]
        return cs.vertcat(*quat)

    max_decision = cs.fmax(
        decision[0], cs.fmax(decision[1], cs.fmax(decision[2], decision[3]))
    )

    quat = cs.if_else(
        max_decision == decision[3],
        true_case_3,
        cs.if_else(
            max_decision == decision[2],
            alt_true_case(2, 0, 1),
            cs.if_else(
                max_decision == decision[1],
                alt_true_case(1, 2, 0),
                alt_true_case(0, 1, 2),
            ),
        ),
    )

    return quat


class Rotation:

    def __init__(self, quat, normalize=True, scalar_first=False):
        self._single = False
        quat = cs.hozcat(quat)

        if quat.shape[0] != 4:
            msg = f"Expected `quat` to have shape (4,) or (4, N), got {quat.shape}."
            raise ValueError(msg)

        # If a single quaternion is given set self._single to True so
        # that we can return appropriate objects in the `to_...`
        # methods
        if quat.shape[1] == 1:
            self._single = True

        if scalar_first:
            w = quat[0, :]
            x = quat[1, :]
            y = quat[2, :]
            z = quat[3, :]
            quat_ = cs.vertcat(x, y, z, w)
        else:
            quat_ = quat

        if normalize:
            num_rotations = quat_.shape[1]
            for i in range(num_rotations):
                norm = cs.sqrt(cs.sumsqr(quat_[:, i]))
                quat_[:, i] = quat_[:, i] / norm

        self._quat = quat

    def __getstate__(self):
        if isinstance(self._quat, np.ndarray):
            return self._quat, self._single
        quat = None
        try:
            quat = self._quat.toarray().astype(float)
        except:
            pass
        if quat is None:
            raise ValueError("unable to get state")
        return quat, self._single

    def __setstate__(self, state):
        quat, single = state
        self._quat = quat
        self._single = single

    @property
    def single(self):
        return self._single

    def __bool__(self):
        """Comply with Python convention for objects to be True.

        Required because Rotation.__len__() is defined and not always truthy.
        """
        return True

    def __len__(self):
        if self._single:
            raise ValueError("Single rotation has no len().")
        return self._quat.shape[1]

    @classmethod
    def from_quat(cls, quat, *, scalar_first=False):
        """Initialize from quaternion."""
        return cls(quat, normalize=not isinstance(quat, (cs.SX, cs.MX)))

    @classmethod
    def from_matrix(cls, matrix):
        if isinstance(matrix, list):
            quat_list = [_single_matrix_to_quat(m) for m in matrix]
        else:
            quat_list = [_single_matrix_to_quat(matrix)]
        quat = cs.horzcat(*quat_list)
        return cls(quat, normalize=not isinstance(quat, (cs.SX, cs.MX)))

    @classmethod
    def from_rotvec(cls, rotvec, degrees=False):

        # TODO

        rotvec = cs.vec(rotvec)
        n = rotvec.shape[0]
        assert n == 3, f"expected rotvec to be 3-dimensional, got {n}"

        if degrees:
            rotvec = deg2rad(rotvec)

        angle = cs.norm_fro(rotvec)

        scale = cs.if_else(
            angle <= 1e-3,
            0.5 - angle**2 / 48.0 + angle**2 * angle**2 / 3840.0,
            cs.sin(0.5 * angle) / angle,
        )

        quat = cs.vertcat(
            scale * rotvec[0],
            scale * rotvec[1],
            scale * rotvec[2],
            cs.cos(angle * 0.5),
        )

        return Rotation(quat, normalize=not isinstance(quat, (cs.SX, cs.MX)))

    def __mul__(self, other):
        """! Compose this rotation with the other.

        @param other Object containing the rotation or translation to be composed with this one. Note that compositions are not commutative, so p * q is different from q * p. In the case of translations q * p is undefined.
        @return The product A * B, if other is a rotation the output will be a rotation. However, if other is a translation then the output will also be a translation.
        """
        # DEV NOTE: this computes self * other
        if isinstance(other, Rotation):
            p = self.as_quat()
            q = other.as_quat()
            r = _compose_quat(p, q)
            return Rotation(r, normalize=not isinstance(r, (cs.SX, cs.MX)))

        elif isinstance(other, Translation):
            return Translation(self.as_matrix() @ other.as_vector())

        else:
            raise TypeError(
                f"The input type for the other object is not recognized, expected either 'Rotation' or 'Translation', got '{type(other)}'."
            )

    @staticmethod
    def identity():
        """! Get the identity rotation."""
        return Rotation([0.0, 0.0, 0.0, 1.0])

    @staticmethod
    def random():
        """! Generate uniformly distributed rotations.

        @return Random rotation.
        """
        return Rotation(cs.np.random.normal(size=(4,)))

    @staticmethod
    def symbolic():
        """! Symbolic representation.

        @return Symbolic rotation.
        """
        quat = cs.SX.sym("quat", 4)
        return Rotation(quat, normalize=False)

    def inv(self):
        """! Invert this rotation."""
        return Rotation(
            cs.vertcat(self._quat[:-1], -self._quat[-1]),
            normalize=not isinstance(self._quat, (cs.SX, cs.MX)),
        )

    def magnitude(self):
        """! Get the magnitude of the rotation."""
        quat = self._quat
        return 2.0 * cs.arctan2(cs.norm_fro(quat[:3]), cs.fabs(quat[3]))

    #
    # From methods
    #

    @staticmethod
    def from_matrix(matrix):
        """! Initialize from rotation matrix.

        @param matrix A 3-by-3 rotation matrix or 4-by-4 homogeneous transformation matrix.
        @return Object containing the rotation represented by the rotation matrix.
        """
        matrix = cs.horzcat(matrix)[
            :3, :3
        ]  # ensure matrix is 3-by-3 and in casadi format

        decision = cs.vertcat(
            matrix[0, 0],
            matrix[1, 1],
            matrix[2, 2],
            matrix[0, 0] + matrix[1, 1] + matrix[2, 2],
        )

        true_case_3 = cs.vertcat(
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
            1.0 + decision[3],
        )

        def alt_true_case(i, j, k):
            quat = [None, None, None, None]
            quat[i] = 1.0 - decision[3] + 2 * matrix[i, i]
            quat[j] = matrix[j, i] + matrix[i, j]
            quat[k] = matrix[k, i] + matrix[i, k]
            quat[3] = matrix[k, j] - matrix[j, k]
            return cs.vertcat(*quat)

        max_decision = cs.fmax(
            decision[0], cs.fmax(decision[1], cs.fmax(decision[2], decision[3]))
        )

        quat = cs.if_else(
            max_decision == decision[3],
            true_case_3,
            cs.if_else(
                max_decision == decision[2],
                alt_true_case(2, 0, 1),
                cs.if_else(
                    max_decision == decision[1],
                    alt_true_case(1, 2, 0),
                    alt_true_case(0, 1, 2),
                ),
            ),
        )

        return Rotation(quat, normalize=not isinstance(quat, (cs.SX, cs.MX)))

    @staticmethod
    def from_mrp(mrp):
        """! Initialize from Modified Rodrigues Parameters (MRPs).

        @param mrp A vector giving the MRP, a 3 dimensional vector co-directional to the axis of rotation and whose magnitude is equal to tan(theta / 4), where theta is the angle of rotation (in radians).
        """

        mrp = cs.vec(mrp)

        mrp_squared_plus_1 = 1.0 + cs.sumsqr(mrp)

        quat = cs.vertcat(
            2.0 * mrp[0] / mrp_squared_plus_1,
            2.0 * mrp[1] / mrp_squared_plus_1,
            2.0 * mrp[2] / mrp_squared_plus_1,
            (2.0 - mrp_squared_plus_1) / mrp_squared_plus_1,
        )

        return Rotation(quat, normalize=not isinstance(mrp, (cs.SX, cs.MX)))

    @staticmethod
    def from_euler(seq, angles, degrees=False):
        """! Initialize from Euler angles.

        @param seq Specifies sequence of axes for rotations. Up to 3 characters belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations. Extrinsic and intrinsic rotations cannot be mixed in one function call.
        @param angles Euler angles specified in radians (degrees is False) or degrees (degrees is True). For a single character seq, angles can be:
                            - a single value.
                            - array_like with shape (N,), where each angle[i] corresponds to a single rotation.
        @param degrees If True, then the given angles are assumed to be in degrees. Default is False.
        @return Object containing the rotation represented by the rotation around given axes with given angles.
        """

        angles = cs.vec(angles)

        num_axes = len(seq)
        if num_axes < 1 or num_axes > 3:
            raise ValueError(
                "Expected axis specification to be a non-empty "
                "string of upto 3 characters, got {}".format(seq)
            )

        intrinsic = re.match(r"^[XYZ]{1,3}$", seq) is not None
        extrinsic = re.match(r"^[xyz]{1,3}$", seq) is not None
        if not (intrinsic or extrinsic):
            raise ValueError(
                "Expected axes from `seq` to be from ['x', 'y', "
                "'z'] or ['X', 'Y', 'Z'], got {}".format(seq)
            )

        if any(seq[i] == seq[i + 1] for i in range(num_axes - 1)):
            raise ValueError(
                "Expected consecutive axes to be different, " "got {}".format(seq)
            )

        seq = seq.lower()

        if degrees:
            angles = deg2rad(angles)

        quat = _elementary_quat_compose(seq, angles, intrinsic)

        return Rotation(quat, normalize=not isinstance(quat, (cs.SX, cs.MX)))

    #
    # As methods
    #

    def as_quat(self, seq: str = "xyzw"):
        """! Represent as quaternions.

        @param seq Specifies the ordering of the quaternion. Available options are 'wxyz' (i.e. scalar-first) and 'xyzw' (i.e. scalar-last). The default is the scalar-last format given by 'xyzw'.
        @return A quaternion vector.
        """

        if seq == "xyzw":
            return self._quat
        elif seq == "wxyz":
            x = self._quat[0]
            y = self._quat[1]
            z = self._quat[2]
            w = self._quat[3]
            return cs.vertcat(w, x, y, z)
        else:
            raise ValueError(f"Sequence '{seq}' is not supported.")

    def as_matrix(self):
        """! Represent as rotation matrix.

        @return A 3-by-3 rotation matrix.
        """

        x = self._quat[0]
        y = self._quat[1]
        z = self._quat[2]
        w = self._quat[3]

        x2 = x * x
        y2 = y * y
        z2 = z * z
        w2 = w * w

        xy = x * y
        zw = z * w
        xz = x * z
        yw = y * w
        yz = y * z
        xw = x * w

        matrix = cs.horzcat(
            cs.vertcat(
                x2 - y2 - z2 + w2,
                2.0 * (xy + zw),
                2.0 * (xz - yw),
            ),
            cs.vertcat(
                2.0 * (xy - zw),
                -x2 + y2 - z2 + w2,
                2.0 * (yz + xw),
            ),
            cs.vertcat(
                2.0 * (xz + yw),
                2.0 * (yz - xw),
                -x2 - y2 + z2 + w2,
            ),
        )

        return matrix

    def as_rotvec(self, degrees: bool = False):
        """! Represent as rotation vector.

        @param degrees If True, then the given magnitudes are assumed to be in degrees. Default is False.
        @return A 3-dimensional rotation vector
        """

        # w > 0 to ensure 0 <= angle <= pi
        quat = cs.if_else(self._quat[3] < 0, -self._quat, self._quat)

        # Use formula: https://uk.mathworks.com/help/fusion/ref/quaternion.rotvec.html
        theta = 2.0 * cs.acos(quat[3])
        scale = theta / cs.sin(theta * 0.5)

        rotvec = cs.vertcat(
            scale * quat[0],
            scale * quat[1],
            scale * quat[2],
        )

        if degrees:
            rotvec = rad2deg(rotvec)

        return rotvec

    def as_mrp(self):
        """! Represent as Modified Rodrigues Parameters (MRPs).

        @return A vector giving the MRP, a 3 dimensional vector co-directional to the axis of rotation and whose magnitude is equal to tan(theta / 4), where theta is the angle of rotation (in radians).
        """
        sign = cs.if_else(self._quat[3] < 0.0, -1.0, 1.0)
        denominator = 1.0 + sign * self._quat[3]
        mrp = sign * self._quat[:3] / denominator
        return mrp

    def as_euler(self, seq: str, degrees: bool = False):
        """! Represent as Euler angles.

        @param seq Specifies sequence of axes for rotations. Up to 3 characters belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations. Extrinsic and intrinsic rotations cannot be mixed in one function call.
        @param degrees Returned angles are in degrees if this flag is True, else they are in radians. Default is False.
        @return Euler angles specified in radians (degrees is False) or degrees (degrees is True).
        """
        if len(seq) != 3:
            raise ValueError(f"Expected 3 axes, got {len(seq)}.")

        intrinsic = re.match(r"^[XYZ]{1,3}$", seq) is not None
        extrinsic = re.match(r"^[xyz]{1,3}$", seq) is not None

        if not (intrinsic or extrinsic):
            raise ValueError(
                "Expected axes from `seq` to be from "
                "['x', 'y', 'z'] or ['X', 'Y', 'Z'], "
                "got {}".format(seq)
            )

        if any(seq[i] == seq[i + 1] for i in range(2)):
            raise ValueError(
                "Expected consecutive axes to be different, " "got {}".format(seq)
            )

        seq = seq.lower()

        # Compute euler from quat
        if extrinsic:
            angle_first = 0
            angle_third = 2
        else:
            seq = seq[::-1]
            angle_first = 2
            angle_third = 0

        def elementary_basis_index(axis):
            if axis == "x":
                return 0
            elif axis == "y":
                return 1
            elif axis == "z":
                return 2

        i = elementary_basis_index(seq[0])
        j = elementary_basis_index(seq[1])
        k = elementary_basis_index(seq[2])

        symmetric = i == k

        if symmetric:
            k = 3 - i - j  # get third axis

        # Check if permutation is even (+1) or odd (-1)
        sign = (i - j) * (j - k) * (k - i) // 2

        eps = 1e-7

        if symmetric:
            a = self._quat[3]
            b = self._quat[i]
            c = self._quat[j]
            d = self._quat[k] * sign
        else:
            a = self._quat[3] - self._quat[j]
            b = self._quat[i] + self._quat[k] * sign
            c = self._quat[j] + self._quat[3]
            d = self._quat[k] * sign - self._quat[i]

        angles1 = 2.0 * cs.arctan2(cs.sqrt(c**2 + d**2), cs.sqrt(a**2 + b**2))

        case = cs.if_else(
            cs.fabs(angles1) <= eps,
            1,
            cs.if_else(
                cs.fabs(angles1 - pi) <= eps,
                2,
                0,
            ),
        )

        half_sum = cs.arctan2(b, a)
        half_diff = cs.arctan2(d, c)

        angles_case_0_ = [None, angles1, None]
        angles_case_0_[angle_first] = half_sum - half_diff
        angles_case_0_[angle_third] = half_sum + half_diff
        angles_case_0 = cs.vertcat(*angles_case_0_)

        angles_case_else_ = [None, angles1, 0.0]
        angles_case_else_[0] = cs.if_else(
            case == 1, 2.0 * half_sum, 2.0 * half_diff * (-1.0 if extrinsic else 1.0)
        )
        angles_case_else = cs.vertcat(*angles_case_else_)

        angles = cs.if_else(case == 0, angles_case_0, angles_case_else)

        if not symmetric:
            angles[angle_third] *= sign
            angles[1] -= pi * 0.5

        for i in range(3):
            angles[i] += cs.if_else(
                angles[i] < -pi,
                2.0 * pi,
                cs.if_else(angles[i] > pi, -2.0 * pi, 0.0),
            )

        if degrees:
            angles = rad2deg(angles)

        return angles

    def rotation_angle(self, other):
        """Angle between two rotations."""
        return (self.inv() * other).magnitude()

    @property
    def x(self):
        return self._quat[0]

    @property
    def y(self):
        return self._quat[1]

    @property
    def z(self):
        return self._quat[2]

    @property
    def w(self):
        return self._quat[3]


class Translation:
    """! A class defining a translation vector."""

    def __init__(self, t):
        """! Initializer for the Translation class.

        @param t A 3-dimensional translation vector.
        @return An instance of the Translation class.
        """
        self._t = cs.vec(t)
        assert (
            self._t.shape[0] == 3
        ), f"expected translation vector to have length 3, got {self._t.shape[0]}."

    @property
    def x(self):
        return self._t[0]

    @property
    def y(self):
        return self._t[1]

    @property
    def z(self):
        return self._t[2]

    def __add__(self, other):
        """! Compose this translation with the other via vector addition.

        @param other Object containing the translation to be composed with this one.
        @return The translation that is the result of A + B.
        """
        # DEV NOTE: this computes self + other
        return Translation(self.as_vector() + other.as_vector())

    def __neg__(self):
        """! Negated translation.

        @return The translation that is the negation of this translation, i.e. -t.
        """
        return Translation(-self.as_vector())

    def __sub__(self, other):
        """! Compose this translation with the other via vector subtraction.

        @param other Object containing the translation to be composed with this one via subtraction.
        @return The translation that is the result of A - B.
        """
        # DEV NOTE: this computes self - other
        return Translation(self.as_vector() - other.as_vector())

    @staticmethod
    def identity():
        """! Get the identity translation."""
        return Translation([0.0, 0.0, 0.0])

    @staticmethod
    def random():
        """! Generate uniformly distributed translations.

        @return Random translation.
        """
        return Translation(cs.np.random.normal(size=(3,)))

    @staticmethod
    def symbolic():
        """! Symbolic representation.

        @return Symbolic translation.
        """
        t = cs.SX.sym("t", 3)
        return Translation(t)

    def magnitude(self):
        """! Get the magnitude of the translation."""
        return cs.norm_fro(self._t)

    @staticmethod
    def from_vector(t):
        """! Initialize from translation vector.

        @param t A 3-dimensional translation vector.
        @return Object containing the translation represented by the input vector.
        """
        return Translation(t)

    @staticmethod
    def from_matrix(T):
        """! Initialize from a homogenous transformation matrix.

        @param T A 4-by-4 homogenous transformation matrix.
        @return Object containing the translation represented by the input matrix.
        """
        return Translation(cs.horzcat(T[:3, 3]))

    def as_vector(self):
        """! Represent as a translation vector.

        @return A 3-dimensional translation vector.
        """
        return self._t

    def as_matrix(self):
        """! Represent as homogenous transformation matrix.

        @return A 4-by-4 homogenous transformation matrix.
        """
        return self.as_transformation().as_matrix()

    def as_transformation(self):
        """! Represent as homogenous transformation.

        @return A instance of the Transformation class with identity rotation.
        """
        return Transformation(Rotation.identity(), Translation(self._t))


class Transformation:
    """! A class for representing homogenous transformations."""

    def __init__(self, rotation: Rotation, translation: Translation):
        """! Initializer for the Transformation class.

        @param rotation The rotation part of the homogenous transformation.
        @param translation The translation part of the homogenous transformation.
        @return An instance of the Transformation class.
        """

        ## Rotation object.
        self._rotation = rotation

        ## Translation object.
        self._translation = translation

    def inv(self):
        """! Invert this homogeneous transformation."""
        R = self._rotation.as_matrix()
        t = self._translation.as_vector()
        return Transformation(Rotation.from_matrix(R.T), Translation(-R.T @ t))

    def translation(self) -> Translation:
        """! Return the translation part of the homogeneous transformation.

        @return The translation part of the homogeneous transformation.
        """
        return self._translation

    def rotation(self) -> Rotation:
        """! Return the rotation part of the homogeneous transformation.

        @return The rotation part of the homogeneous transformation.
        """
        return self._rotation

    @staticmethod
    def identity():
        """! Get the identity homogenous transform."""
        return Transformation(Rotation.identity(), Translation.identity())

    @staticmethod
    def random():
        """! Generate uniformly distributed homogeneous transforms.

        @return Random homogeneous transform.
        """
        return Transformation(Rotation.random(), Translation.random())

    @staticmethod
    def symbolic():
        """! Symbolic representation.

        @return Symbolic homogenous transform.
        """
        return Transformation(Rotation.symbolic(), Translation.symbolic())

    @staticmethod
    def from_matrix(T):
        """! Initialize from homogenous transformation matrix.

        @param matrix A 4-by-4 homogeneous transformation matrix.
        @return Object containing the homogeneous transformation represented by the matrix.
        """
        T = cs.horzcat(T)
        return Transformation(
            Rotation.from_matrix(T[:3, :3]), Translation.from_matrix(T)
        )

    def as_matrix(self):
        """! Represent as homogenous transformation matrix.

        @return A 4-by-4 homogenous transformation matrix.
        """
        return cs.vertcat(
            cs.horzcat(self._rotation.as_matrix(), self._translation.as_vector()),
            cs.DM([[0.0, 0.0, 0.0, 1.0]]),
        )

    def __mul__(self, other):
        """! Compose this transformation with the other.

        @param other Object containing the transformation to be composed with this one. Note that transformation compositions are not commutative, so p * q is different from q * p.
        @return The homgeonous transformation that is the product A * B.
        """
        # DEV NOTE: this computes the product self * other
        rotation = self._rotation * other.rotation()
        translation = self._rotation * other.translation() + self._translation
        return Transformation(rotation, translation)

    def flatten(self):
        """! Returns the homogenous transform as a vector representation [quat, t] where quat is a unit-quaternion for the rotation and t is the translation.

        @return Vector representation for the homogeneous transform.
        """
        return cs.vertcat(
            self._rotation.as_quat(),
            self._translation.as_vector(),
        )
