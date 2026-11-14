"""Spherical interpolation of rotations."""
import casadi as cs
import numpy as np
from ._arrays import promote, implementation, wrap
from .rotation import Rotation


class Slerp:
    """Shortest-path interpolation at fixed, strictly increasing knot times."""

    def __init__(self, times, rotations):
        self._times = cs.vec(cs.DM(times))
        self._rotations = rotations
        q = rotations.as_quat()
        implementation('Slerp', q)(self._times, rotations._impl)

    def __call__(self, times):
        if isinstance(times, (cs.DM, cs.SX, cs.MX)):
            single = times.numel() == 1
            value = cs.vec(times).T
        else:
            single = np.ndim(times) == 0
            value = cs.vec(cs.DM(times)).T
        q, value = promote(self._rotations.as_quat(), value)
        rotation = implementation('Rotation', q)(q, False)
        interpolator = implementation('Slerp', q)(self._times, rotation)
        return wrap(Rotation, interpolator.evaluate(value), single)
