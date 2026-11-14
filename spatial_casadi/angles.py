"""Angle conversions preserving CasADi types."""
import casadi as cs
from . import native


def _convert(name, x):
    if not isinstance(x, (cs.DM, cs.SX, cs.MX)):
        x = cs.DM(x)
    return getattr(native, name + type(x).__name__)(x)


def deg2rad(x):
    """Convert degrees to radians."""
    return _convert('deg2rad', x)


def rad2deg(x):
    """Convert radians to degrees."""
    return _convert('rad2deg', x)
