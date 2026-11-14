"""Compatibility translation wrapper."""
import casadi as cs
import numpy as np
from ._arrays import columns, promote, implementation


class Translation:
    """A translation vector, retained for the original API."""

    def __init__(self, t):
        value, _ = columns(t, 3)
        self._impl = implementation('Translation', value)(value)

    def as_vector(self):
        return self._impl.as_vector()

    def magnitude(self):
        return self._impl.magnitude()

    def __add__(self, other):
        if not isinstance(other, Translation):
            return NotImplemented
        a, b = promote(self.as_vector(), other.as_vector())
        cls = implementation('Translation', a)
        return Translation(cls(a).add(cls(b)).as_vector())

    def __neg__(self):
        return Translation(self._impl.neg().as_vector())

    def __sub__(self, other):
        if not isinstance(other, Translation):
            return NotImplemented
        return self + (-other)

    @classmethod
    def identity(cls):
        return cls([0, 0, 0])

    @classmethod
    def random(cls, rng=None):
        generator = np.random if rng is None else (rng if hasattr(rng, 'normal') else np.random.default_rng(rng))
        return cls(generator.normal(size=3))

    @classmethod
    def symbolic(cls, name='t', kind='SX'):
        if kind not in ('SX', 'MX'):
            raise ValueError('kind must be SX or MX.')
        return cls(getattr(cs, kind).sym(name, 3))

    @classmethod
    def from_vector(cls, t):
        return cls(t)

    @classmethod
    def from_matrix(cls, matrix):
        from .rigid_transform import RigidTransform
        return cls(RigidTransform.from_matrix(matrix).translation)

    def as_transformation(self):
        from .transformation import Transformation
        from .rotation import Rotation
        return Transformation(Rotation.identity(), self)

    def as_matrix(self):
        return self._impl.as_transform().as_matrix()

    @property
    def x(self): return self.as_vector()[0, :]
    @property
    def y(self): return self.as_vector()[1, :]
    @property
    def z(self): return self.as_vector()[2, :]
