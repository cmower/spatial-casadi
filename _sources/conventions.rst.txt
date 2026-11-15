Conventions and derivatives
===========================

* Right-handed frames, radians by default, scalar-last quaternions ``[x,y,z,w]``.
* ``a * b`` applies ``b`` first, then ``a``.
* ``T_A_B.apply(p_B)`` produces coordinates in frame A.
* A rigid transform applies rotation first, then translation.
* Exponential coordinates are ``[rx,ry,rz,vx,vy,vz]``; the last three values
  equal translation only for a pure translation.

Shapes
------

================== ================== ========================
Representation     NumPy input batch  CasADi input/output batch
================== ================== ========================
Quaternion         ``(N, 4)``         ``4 x N``
Vector/Euler       ``(N, 3)``         ``3 x N``
Rotation matrix    ``(N, 3, 3)``      ``3 x (3*N)``
Transform matrix   ``(N, 4, 4)``      ``4 x (4*N)``
Exponential coords ``(N, 6)``         ``6 x N``
Dual quaternion    ``(N, 8)``         ``8 x N``
================== ================== ========================

Single vectors are columns. Matrix batches contain horizontal blocks.
Operations broadcast a singleton batch against a larger batch.
An explicit NumPy batch of length one remains indexable; a single CasADi
column is treated as a single object. Empty batches are rejected.

Symbolic inputs
---------------

``DM`` is numeric; ``SX`` and ``MX`` build symbolic graphs. Numeric values
promote to either symbolic type. Mixing ``SX`` with ``MX`` requires a CasADi
``Function`` boundary, as in CasADi itself.

Quaternion constructors normalize nonzero inputs, including symbolic inputs.
``Rotation.symbolic()`` instead creates a raw quaternion variable: constrain
its squared norm to one, or use rotation-vector variables. Internal operations
preserve unit quaternions without redundant normalization.

Numeric constructors reject invalid shapes, nonfinite values, zero
quaternions, and improper matrices. Symbolic matrices must represent valid
rigid transforms, and symbolic quaternions must be nonzero. These
value-dependent conditions cannot be checked when constructing a graph.

Smoothness
----------

Rotation and rigid-transform exponential/logarithmic maps have stable
small-angle expansions. Their values, Jacobians and Hessians are tested at
identity and near branch thresholds for both symbolic types.

Euler angles are singular at gimbal lock. Principal rotation vectors and
canonical quaternions have a branch cut at 180 degrees. Rotation magnitude
is not differentiable at identity. Avoid these representations as globally
smooth optimization residuals; matrix or transformed-point residuals are
usually simpler. Symbolic Euler conversions cannot issue runtime gimbal-lock
warnings. Slerp derivatives are local to a segment; they may jump at knots,
and symbolic query times must stay inside the knot interval.
