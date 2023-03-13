import casadi as cs
from typing import Union


## CasADi array types.
ArrayType = Union[cs.DM, cs.SX]


def deg2rad(x: ArrayType) -> ArrayType:
    """! Convert degrees to radians.

    @param x An array containing angles in degrees.
    @return An array containing angles in radians.
    """
    return (pi / 180.0) * x


def rad2deg(x: ArrayType) -> ArrayType:
    """! Convert radians to degrees.

    @param x An array containing angles in radians.
    @return An array containing angles in degrees.
    """
    return (180.0 / pi) * x


class Rotation:
    def __init__(self, quat: ArrayType, normalize: bool = True):
        """! Initializer for the Rotation class.

        @param quat Quaternion representing the rotation.
        @param normalize When true, the quaternion is normalized.
        @return An instance of the Rotation class.
        """
        quat = cs.vec(quat)

        assert (
            quat.shape[0] == 4
        ), f"Incorrect length for input quaternion. Got {quat.shape[0]}, expected 4!"

        if normalize:
            quat = quat / cs.norm_fro(quat)

        self._quat = quat

    @staticmethod
    def identity():
        """! Get the identity rotation."""
        return Rotation([0.0, 0.0, 0.0, 1.0])

    def inv(self):
        """! Invert this rotation."""
        return Rotation(cs.vertcat(self._quat[:-1], -self._quat[-1]))

    #
    # From methods
    #

    @staticmethod
    def from_quat(quat: ArrayType):
        """! Initialize from quaternion.

        @param quat Quaternion in scalar-last (x, y, z, w) format. The quaternion will be normalized to unit norm.
        @return Object containing the rotation represented by the input quaternion.
        """
        return Rotation(quat)

    @staticmethod
    def from_matrix(matrix: ArrayType):
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
            return cs.vertcat(
                1.0 - decision[3] + 2 * matrix[i, i],
                matrix[j, i] + matrix[i, j],
                matrix[k, i] + matrix[i, k],
                matrix[k, j] - matrix[j, k],
            )

        max_decision = cs.fmax(decision)
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

        return Rotation(quat)

    @staticmethod
    def from_rotvec(rotvec: ArrayType, degrees: bool = False):
        """! Initialize from rotation vectors.

        @param rotvec A 3-dimensional rotation vector
        @param degrees If True, then the given magnitudes are assumed to be in degrees. Default is False.
        """

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

        return Rotation(quat)

    @staticmethod
    def from_mrp(mrp: ArrayType):
        """! Initialize from Modified Rodrigues Parameters (MRPs).

        @param mrp A vector giving the MRP, a 3 dimensional vector co-directional to the axis of rotation and whose magnitude is equal to tan(theta / 4), where theta is the angle of rotation (in radians).
        """
        mrp_squared_plus_1 = 1.0 + cs.sumsqr(mrp)

        quat = cs.vertcat(
            2.0 * mrp[0] / mrp_squared_plus_1,
            2.0 * mrp[1] / mrp_squared_plus_1,
            2.0 * mrp[2] / mrp_squared_plus_1,
            (2.0 - mrp_squared_plus_1) / mrp_squared_plus_1,
        )

        return Rotation(quat)

    @staticmethod
    def from_euler(seq, angles, degrees=False):
        pass

    #
    # As methods
    #

    def as_quat(self):
        return self._quat

    def as_matrix(self):
        pass

    def as_rotvec(self, degrees=False):
        pass

    def as_mrp(self):
        pass

    def as_euler(self, seq, degrees=False):
        pass


class Translation:
    def __init__(self, t):
        self._t = cs.vec(t)
        assert (
            self._t.shape[0] == 3
        ), f"expected translation vector to have length 3, got {self._t.shape[0]}."

    @staticmethod
    def from_vector(self, t):
        return Translation(t)

    @staticmethod
    def from_matrix(T):
        return Translation(T[:3, 3])

    def as_vector(self):
        return self._t

    def as_matrix(self):
        return cs.vertcat(
            cs.horzcat(Rotation.identity().as_matrix(), self._t),
            cs.DM([[0.0, 0.0, 0.0, 1.0]]),
        )


class Transformation:
    def __init__(self, rotation: Rotation, translation: Translation):
        """! Initializer for the Transformation class.

        @param rotation The rotation part of the homogenous transformation.
        @param translation The translation part of the homogenous transformation.
        @return An instance of the Transformation class.
        """
        self._rotation = rotation
        self._translation = translation

    def inv(self):
        """! Invert this homogeneous transformation."""
        R = self._rotation.inv().as_matrix()
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
    def from_matrix(T: ArrayType):
        """! Initialize from homogenous transformation matrix.

        @param matrix A 4-by-4 homogeneous transformation matrix.
        @return Object containing the homogeneous transformation represented by the matrix.
        """
        return Transformation(Rotation.from_matrix(T), Translation.from_matrix(T))

    def as_matrix(self) -> ArrayType:
        """! Represent as homogenous transformation matrix.

        @return A 4-by-4 homogenous transformation matrix.
        """
        return cs.vertcat(
            cs.horzcat(self._rotation.as_matrix(), self._translation.as_vector()),
            cs.DM([[0.0, 0.0, 0.0, 1.0]]),
        )
