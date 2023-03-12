import casadi as cs
from typing import Union


def deg2rad(x):
    """! Convert degrees to radians.

    @param x An array containing angles in degrees.
    @return An array containing angles in radians.
    """
    return (pi / 180.0) * x


def rad2deg(x):
    """! Convert radians to degrees.

    @param x An array containing angles in radians.
    @return An array containing angles in degrees.
    """
    return (180.0 / pi) * x


class Rotation:
    def __init__(self, quat, normalize=True):
        quat = cs.vec(quat)

        assert (
            quat.shape[0] == 4
        ), f"Incorrect length for input quaternion. Got {quat.shape[0]}, expected 4!"

        if normalize:
            quat = quat / cs.norm_fro(quat)

        self._quat = quat

    @staticmethod
    def identity():
        return Rotation([0.0, 0.0, 0.0, 1.0])

    def inv(self):
        return Rotation(cs.vertcat(self._quat[:-1], -self._quat[-1]))

    #
    # From methods
    #

    @staticmethod
    def from_quat(quat):
        return Rotation(quat)

    @staticmethod
    def from_matrix(matrix):
        pass

    @staticmethod
    def from_rotvec(rotvec, degrees=False):
        pass

    @staticmethod
    def from_mrp(mrp):
        pass

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
    def from_matrix(T):
        return Translation(T[:3, 3])


class Transformation:
    def __init__(self, rotation, translation):
        self.rotation = rotation
        self.translation = translation

    def inv(self):
        pass
