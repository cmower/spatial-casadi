"""Differentiable spatial transforms backed by CasADi."""
import os
from pathlib import Path
import casadi

if os.name == "nt":
    _dll_directory = os.add_dll_directory(str(Path(casadi.__file__).parent))

from .angles import deg2rad, rad2deg
from .spatial import Rotation, RigidTransform, Translation, Transformation, Slerp

__version__ = "2.0.0"
__all__ = ["Rotation", "RigidTransform", "Translation", "Transformation", "Slerp", "deg2rad", "rad2deg"]
