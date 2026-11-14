"""Compatibility imports for spatial_casadi.spatial."""
from .angles import deg2rad, rad2deg
from .rotation import Rotation
from .rigid_transform import RigidTransform
from .translation import Translation
from .transformation import Transformation
from .slerp import Slerp

__all__ = ['Rotation', 'RigidTransform', 'Translation', 'Transformation', 'Slerp', 'deg2rad', 'rad2deg']
