"""Shape and type adaptation; numerical work lives in C++."""

import casadi as cs
import numpy as np

from . import native

CASADI = (cs.DM, cs.SX, cs.MX)


def columns(value, rows):
    if isinstance(value, CASADI):
        if value.shape == (1, rows) and rows != 1:
            value = value.T
        if value.size1() != rows or value.size2() == 0:
            raise ValueError(f"Expected {rows} x N CasADi matrix.")
        return value, value.size2() == 1
    array = np.asarray(value, dtype=float)
    single = array.ndim == 1 or array.ndim == 0
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim == 1:
        if rows == 1:
            return cs.DM(array.reshape(1, -1)), single and array.size == 1
        if array.size != rows:
            raise ValueError(f"Expected a vector of length {rows}.")
        return cs.DM(array), True
    if array.ndim == 2 and array.shape == (rows, 1) and rows != 1:
        return cs.DM(array), True
    if array.ndim != 2 or array.shape[1] != rows or array.shape[0] == 0:
        raise ValueError(f"Expected NumPy shape ({rows},) or (N, {rows}).")
    return cs.DM(array.T), False


def matrices(value, dim):
    if isinstance(value, CASADI):
        if value.size1() != dim or value.size2() == 0 or value.size2() % dim:
            raise ValueError(f"Expected {dim} x ({dim}*N) CasADi matrix.")
        return value, value.size2() == dim
    array = np.asarray(value, dtype=float)
    if array.shape == (dim, dim):
        return cs.DM(array), True
    if array.ndim == 3 and array.shape[1:] == (dim, dim) and len(array):
        return cs.horzcat(*[cs.DM(m) for m in array]), False
    raise ValueError(f"Expected ({dim}, {dim}) or (N, {dim}, {dim}) matrices.")


def promote(*values):
    symbolic = {type(x) for x in values if isinstance(x, (cs.SX, cs.MX))}
    if len(symbolic) > 1:
        raise TypeError(
            "SX and MX cannot be mixed; use a CasADi Function to bridge them."
        )
    typ = next(iter(symbolic), cs.DM)
    return tuple(typ(x) for x in values)


def implementation(name, value):
    return getattr(native, name + type(value).__name__)


def wrap(cls, impl, single):
    result = object.__new__(cls)
    result._impl, result._single = impl, single
    return result


def indices(index, size):
    if not isinstance(index, (int, np.integer, slice)):
        raise TypeError("Use an integer or slice.")
    selected = np.arange(size)[index]
    scalar = np.ndim(selected) == 0
    selected = np.atleast_1d(selected).tolist()
    if not selected:
        raise ValueError("Empty batches are not supported.")
    return selected, scalar
