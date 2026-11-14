"""SciPy-style rotations with CasADi outputs."""
import warnings
import casadi as cs
import numpy as np
from ._arrays import columns, matrices, promote, implementation, wrap, indices


class Rotation:
    """3D rotations. NumPy batches use rows; CasADi batches use columns."""

    def __init__(self, quat, normalize=True, scalar_first=False):
        q, self._single = columns(quat, 4)
        self._impl = implementation('Rotation', q)(q, normalize, scalar_first)

    @classmethod
    def _from(cls, method, value, single, *args):
        return wrap(cls, getattr(implementation('Rotation', value), method)(value, *args), single)

    @classmethod
    def from_quat(cls, quat, seq=None, *, scalar_first=False):
        """Construct from nonzero quaternions; normalize each quaternion."""
        if seq is not None:
            if seq not in ('xyzw', 'wxyz'):
                raise ValueError('Quaternion order must be xyzw or wxyz.')
            scalar_first = seq == 'wxyz'
        return cls(quat, scalar_first=scalar_first)

    @classmethod
    def from_matrix(cls, matrix):
        """Construct from proper rotation matrices (no orthogonal projection)."""
        # Preserve the original single homogeneous-matrix shortcut.
        if getattr(matrix, 'shape', None) == (4, 4):
            matrix = matrix[:3, :3]
        value, single = matrices(matrix, 3)
        return cls._from('from_matrix', value, single)

    @classmethod
    def from_rotvec(cls, rotvec, degrees=False):
        """Construct from axis-angle vectors, including zero."""
        value, single = columns(rotvec, 3)
        return cls._from('from_rotvec', value, single, degrees)

    @classmethod
    def from_mrp(cls, mrp):
        """Construct from modified Rodrigues parameters."""
        value, single = columns(mrp, 3)
        return cls._from('from_mrp', value, single)

    @classmethod
    def from_euler(cls, seq, angles, degrees=False):
        """Construct from intrinsic XYZ or extrinsic xyz sequences."""
        value, single = columns(angles, len(seq))
        if len(seq) == 1 and not isinstance(angles, (cs.DM, cs.SX, cs.MX)):
            single = np.ndim(angles) == 0
        impl = implementation('Rotation', value).from_euler(seq, value, degrees)
        return wrap(cls, impl, single)

    def as_quat(self, seq=None, *, canonical=False, scalar_first=False):
        """Return quaternion columns; the default order is xyzw."""
        if isinstance(seq, (bool, np.bool_)):
            canonical = bool(seq)
        elif seq is not None:
            if seq not in ('xyzw', 'wxyz'):
                raise ValueError('Quaternion order must be xyzw or wxyz.')
            scalar_first = seq == 'wxyz'
        return self._impl.as_quat(canonical, scalar_first)

    def as_matrix(self):
        """Return a 3 x (3*N) matrix of horizontal rotation blocks."""
        return self._impl.as_matrix()

    def as_rotvec(self, degrees=False):
        """Return principal rotation vectors as 3 x N columns."""
        return self._impl.as_rotvec(degrees)

    def as_mrp(self):
        """Return modified Rodrigues parameters as 3 x N columns."""
        return self._impl.as_mrp()

    def as_euler(self, seq, degrees=False, *, suppress_warnings=False):
        """Return three Euler angles per column; third angle is zero at gimbal lock."""
        result = self._impl.as_euler(seq, degrees)
        if isinstance(result, cs.DM) and not suppress_warnings:
            middle = result.full()[1]
            limit = 180 if degrees else np.pi
            symmetric = seq[0].lower() == seq[2].lower()
            locked = np.minimum(np.abs(middle), np.abs(middle-limit)) if symmetric else np.abs(np.abs(middle)-limit/2)
            if np.any(locked <= (1e-7 * (180/np.pi if degrees else 1))):
                warnings.warn('Gimbal lock: third Euler angle set to zero.', UserWarning, stacklevel=2)
        return result

    def _pair(self, other):
        a, b = promote(self.as_quat(), other.as_quat())
        cls = implementation('Rotation', a)
        return cls(a, False), cls(b, False)

    def __mul__(self, other):
        from .translation import Translation
        if isinstance(other, Translation):
            return Translation(self.apply(other.as_vector()))
        if not isinstance(other, Rotation):
            return NotImplemented
        a, b = self._pair(other)
        return wrap(Rotation, a.compose(b), self.single and other.single)

    def inv(self):
        """Return the inverse rotation."""
        return wrap(Rotation, self._impl.inv(), self.single)

    def __pow__(self, n, modulus=None):
        if modulus is not None:
            raise TypeError('Modular powers are not supported.')
        return wrap(Rotation, self._impl.power(float(n)), self.single)

    def apply(self, vectors, inverse=False):
        """Rotate vectors, broadcasting singleton batches."""
        v, _ = columns(vectors, 3)
        q, v = promote(self.as_quat(), v)
        return implementation('Rotation', q)(q, False).apply(v, inverse)

    def magnitude(self):
        """Return principal angles in radians as a 1 x N row."""
        return self._impl.magnitude()

    def approx_equal(self, other, atol=1e-8, degrees=False):
        """Return CasADi comparisons of relative rotation angles."""
        a, b = self._pair(other)
        return a.approx_equal(b, atol, degrees)

    def rotation_angle(self, other):
        return (self.inv() * other).magnitude()

    @classmethod
    def identity(cls, num=None):
        return wrap(cls, implementation('Rotation', cs.DM()).identity(1 if num is None else num), num is None)

    @classmethod
    def random(cls, num=None, rng=None, *, random_state=None):
        """Sample uniform rotations using a NumPy generator or seed."""
        if rng is not None and random_state is not None:
            raise ValueError('Pass rng or random_state, not both.')
        rng = random_state if random_state is not None else rng
        generator = np.random if rng is None else (rng if hasattr(rng, 'normal') else np.random.default_rng(rng))
        result = cls(generator.normal(size=(1 if num is None else num, 4)))
        result._single = num is None
        return result

    @classmethod
    def symbolic(cls, name='quat', kind='SX'):
        """Create a raw unit-quaternion variable; constrain its norm to one."""
        if kind not in ('SX', 'MX'):
            raise ValueError('kind must be SX or MX.')
        return cls(getattr(cs, kind).sym(name, 4), normalize=False)

    @classmethod
    def concatenate(cls, rotations):
        rotations = list(rotations)
        if not rotations:
            raise ValueError('Provide at least one rotation.')
        q = cs.horzcat(*promote(*[r.as_quat() for r in rotations]))
        result = cls(q, normalize=False)
        result._single = False
        return result

    def __getitem__(self, index):
        if self.single:
            raise TypeError('A single rotation is not indexable.')
        selected, single = indices(index, self._impl.size())
        result = Rotation(self.as_quat()[:, selected], normalize=False)
        result._single = single
        return result

    def __len__(self):
        if self.single:
            raise TypeError('A single rotation has no length.')
        return self._impl.size()

    def __bool__(self):
        return True

    @property
    def single(self):
        return self._single

    @property
    def x(self): return self.as_quat()[0, :]
    @property
    def y(self): return self.as_quat()[1, :]
    @property
    def z(self): return self.as_quat()[2, :]
    @property
    def w(self): return self.as_quat()[3, :]

    def __getstate__(self):
        if not isinstance(self.as_quat(), cs.DM):
            raise TypeError('Serialize symbolic expressions with casadi.Function.save instead.')
        return self.as_quat().full(), self.single

    def __setstate__(self, state):
        q, self._single = state
        self._impl = implementation('Rotation', cs.DM())(cs.DM(q), False)
