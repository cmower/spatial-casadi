"""Original Transformation API, implemented through RigidTransform."""

import casadi as cs

from .rigid_transform import RigidTransform
from .rotation import Rotation
from .translation import Translation


class Transformation:
    """Compatibility adapter. New code can use RigidTransform."""

    def __init__(self, rotation, translation):
        self._rigid = RigidTransform.from_components(translation.as_vector(), rotation)

    @classmethod
    def _from_rigid(cls, rigid):
        result = object.__new__(cls)
        result._rigid = rigid
        return result

    def rotation(self):
        return self._rigid.rotation

    def translation(self):
        return Translation(self._rigid.translation)

    def inv(self):
        return self._from_rigid(self._rigid.inv())

    def __mul__(self, other):
        if not isinstance(other, Transformation):
            return NotImplemented
        return self._from_rigid(self._rigid * other._rigid)

    @classmethod
    def identity(cls):
        return cls._from_rigid(RigidTransform.identity())

    @classmethod
    def random(cls):
        return cls(Rotation.random(), Translation.random())

    @classmethod
    def symbolic(cls, kind="SX"):
        return cls(Rotation.symbolic(kind=kind), Translation.symbolic(kind=kind))

    @classmethod
    def from_matrix(cls, matrix):
        return cls._from_rigid(RigidTransform.from_matrix(matrix))

    def as_matrix(self):
        return self._rigid.as_matrix()

    def flatten(self):
        return cs.vertcat(self.rotation().as_quat(), self.translation().as_vector())
