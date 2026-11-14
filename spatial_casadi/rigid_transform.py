"""Rigid transforms with SciPy names and CasADi expressions."""

import casadi as cs

from ._arrays import columns, implementation, indices, matrices, promote, wrap
from .rotation import Rotation


class RigidTransform:
    """A rotation followed by a translation; use the from_* constructors."""

    def __init__(self, matrix):
        value, self._single = matrices(matrix, 4)
        self._impl = implementation("RigidTransform", value).from_matrix(value)

    @classmethod
    def from_matrix(cls, matrix):
        """Construct from proper homogeneous matrices."""
        return cls(matrix)

    @classmethod
    def from_components(cls, translation, rotation):
        """Construct from translation and Rotation, in that order."""
        t, single = columns(translation, 3)
        t, q = promote(t, rotation.as_quat())
        r = implementation("Rotation", q)(q, False)
        return wrap(
            cls, implementation("RigidTransform", t)(t, r), single and rotation.single
        )

    @classmethod
    def from_rotation(cls, rotation):
        return cls.from_components([0, 0, 0], rotation)

    @classmethod
    def from_translation(cls, translation):
        return cls.from_components(translation, Rotation.identity())

    @classmethod
    def from_exp_coords(cls, exp_coords):
        """Construct from columns [rotation vector; translational velocity]."""
        value, single = columns(exp_coords, 6)
        return wrap(
            cls, implementation("RigidTransform", value).from_exp_coords(value), single
        )

    @classmethod
    def from_dual_quat(cls, dual_quat, *, scalar_first=False):
        """Construct from real and dual quaternion pairs; normalize the pair."""
        value, single = columns(dual_quat, 8)
        return wrap(
            cls,
            implementation("RigidTransform", value).from_dual_quat(value, scalar_first),
            single,
        )

    @classmethod
    def identity(cls, num=None):
        impl = implementation("RigidTransform", cs.DM()).identity(
            1 if num is None else num
        )
        return wrap(cls, impl, num is None)

    @classmethod
    def concatenate(cls, transforms):
        transforms = list(transforms)
        if not transforms:
            raise ValueError("Provide at least one transform.")
        t = cs.horzcat(*promote(*[x.translation for x in transforms]))
        result = cls.from_components(
            t, Rotation.concatenate([x.rotation for x in transforms])
        )
        result._single = False
        return result

    @property
    def rotation(self):
        return wrap(Rotation, self._impl.rotation(), self.single)

    @property
    def translation(self):
        return self._impl.translation()

    @property
    def single(self):
        return self._single

    def as_components(self):
        return self.translation, self.rotation

    def as_matrix(self):
        """Return a 4 x (4*N) matrix of horizontal transform blocks."""
        return self._impl.as_matrix()

    def as_exp_coords(self):
        """Return principal exponential coordinates as 6 x N columns."""
        return self._impl.as_exp_coords()

    def as_dual_quat(self, *, scalar_first=False):
        return self._impl.as_dual_quat(scalar_first)

    def _cast(self, typ):
        r = implementation("Rotation", typ())(typ(self.rotation.as_quat()), False)
        return implementation("RigidTransform", typ())(typ(self.translation), r)

    def __mul__(self, other):
        if not isinstance(other, RigidTransform):
            return NotImplemented
        a, _ = promote(self.rotation.as_quat(), other.rotation.as_quat())
        return wrap(
            RigidTransform,
            self._cast(type(a)).compose(other._cast(type(a))),
            self.single and other.single,
        )

    def inv(self):
        return wrap(RigidTransform, self._impl.inv(), self.single)

    def __pow__(self, n, modulus=None):
        if modulus is not None:
            raise TypeError("Modular powers are not supported.")
        return wrap(RigidTransform, self._impl.power(float(n)), self.single)

    def apply(self, vector, inverse=False):
        v, _ = columns(vector, 3)
        q, v = promote(self.rotation.as_quat(), v)
        return self._cast(type(q)).apply(v, inverse)

    def __getitem__(self, index):
        if self.single:
            raise TypeError("A single transform is not indexable.")
        selected, single = indices(index, self._impl.size())
        r = Rotation(self.rotation.as_quat()[:, selected], normalize=False)
        result = RigidTransform.from_components(self.translation[:, selected], r)
        result._single = single
        return result

    def __len__(self):
        if self.single:
            raise TypeError("A single transform has no length.")
        return self._impl.size()

    def __bool__(self):
        return True
