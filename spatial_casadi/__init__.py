"""Differentiable spatial transforms backed by CasADi."""

import os
from pathlib import Path

import casadi

if os.name == "nt":
    _dll_directory = os.add_dll_directory(str(Path(casadi.__file__).parent))

from .spatial import RigidTransform, Rotation, Slerp, Transformation, Translation

__version__ = "2.0.0"
__all__ = [
    "RigidTransform",
    "Rotation",
    "Slerp",
    "Transformation",
    "Translation",
]
