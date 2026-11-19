# Mathrobo

Mathrobo is a Python library for robotics-oriented mathematical computation, including Lie group operations, spatial transformations, numerical differentiation/integration, and B-spline utilities.

## Installation

Python 3.9 or later is required; Python 3.11 or later is recommended.
On Apple M4 Macs, use Python 3.11+ with NumPy 2.3.1+ to avoid the
[known Accelerate runtime warnings](https://numpy.org/doc/stable/release/2.3.1-notes.html).

Install the latest release from PyPI:

```bash
pip install mathrobo
```

If you use uv, you can add Mathrobo to your project with:

```bash
uv add mathrobo
```

### Development install

For local development, clone the repository:

```bash
git clone https://github.com/MathRobotics/MathRobo.git
cd MathRobo
```

Sync the development environment from `pyproject.toml` and `uv.lock`:

```bash
uv sync --extra dev
```

Install the package in editable mode:

```bash
uv pip install -e .
```

## Examples
Refer to the examples in the `example` folder, where you can find Jupyter notebooks and scripts demonstrating various use cases of the library.

## Usage

You can also work with spatial transformations using the `SE3` class. The
following snippet creates a 90 degree rotation around the Z axis with a
translation, applies the transformation to a point and then inverts it:

```python
import numpy as np
import mathrobo as mr

# Rotation of 90 deg about Z and translation of 1 m along X
rot = mr.SO3.exp(np.array([0.0, 0.0, 1.0]), np.pi / 2)
T = mr.SE3(rot, np.array([1.0, 0.0, 0.0]))

point = np.array([0.0, 1.0, 0.0])
transformed = T @ point
recovered = T.inv() @ transformed

print(transformed)
print(recovered)
```

Here is a quick example that computes the numerical gradient of a simple function:

```python
import numpy as np
import mathrobo as mr

f = lambda x: np.sum(x**2)
x = np.array([1.0, 2.0, -3.0])
grad = mr.numerical_grad(x, f)
print(grad)
```

You can also perform numerical integration using the Gaussian quadrature helper
`gq_integrate`:

```python
import numpy as np
import mathrobo as mr

f = lambda s: np.array([np.sin(s)])
val = mr.gq_integrate(f, 0.0, np.pi, digit=5)
print(val)  # ~ 2.0
```

For spline trajectories, SciPy's `BSpline` can be used to create and evaluate a
curve:

```python
import numpy as np
from scipy.interpolate import make_interp_spline

t = np.array([0, 1, 2, 3])
points = np.array([0.0, 1.0, 0.0, 1.0])
spl = make_interp_spline(t, points, k=3)

ts = np.linspace(0, 3, 20)
ys = spl(ts)
print(ys)
```

### SE3 pose increments

`pose.integrate(delta)` returns a new pose using the right increment
`pose @ Exp(delta)`. `pose.difference(other)` returns `Log(pose.inv() @ other)`
in the body frame of `pose`. `pose.log()` returns the exponential coordinates
of the pose itself. For example:

```python
pose = mr.SE3.eye()
delta = np.array([0.2, 0.0, 0.0, 0.0, 0.3, 0.0])
next_pose = pose.integrate(delta)
recovered = pose.difference(next_pose)
```

The tangent layout is `(..., 6)` with rotation vector `phi` in radians followed
by linear exponential coordinates `rho`. The translation of `Exp(delta)` is
`SO3.exp_integ(phi) @ rho`. `next_pose.pos()` is the position of the resulting
pose in world coordinates.

These methods broadcast leading batch axes and support NumPy and JAX,
including JIT when poses are constructed inside the traced function. `log`
and `difference` operate on real poses. They choose a principal rotation with
norm at most π. At π either axis sign represents the rotation, and the
logarithm is discontinuous across that branch. For increments outside the
principal branch, `difference` returns an equivalent principal increment
that reproduces the resulting pose.

### SE3 right Jacobian

The exponential Jacobian maps the time derivative of exponential coordinates
`xi` to body velocity `Jr(xi) @ xi_dot` for `T = SE3.exp(xi)`.
The following methods return `(..., 6, 6)` matrices:

```python
jr = mr.SE3.right_jacobian(xi)
dj = mr.SE3.right_jacobian_directional(xi, u)
d2j = mr.SE3.right_jacobian_second_directional(xi, u, v)
```

`dj` and `d2j` are `DJr(xi)[u]` and `D²Jr(xi)[u, v]`. The input coordinates
and directions have shape `(..., 6)` in angular-first order; leading axes
broadcast together. The same analytic kernel computes `SE3.exp_integ_adj`.
It evaluates a Taylor series near zero rotation and trigonometric coefficients
at larger angles. NumPy and JAX (including JIT) use the same coefficients and
analytic product rules. `Jr` and its directional derivatives are smooth at π;
the principal logarithm used for pose differences has a branch there.

### CMTM from SE3 exponential coordinates

Given `coordinates[..., k, :] = xi^(k)(0) / k!` in angular-first order, use:

```python
motion, jr = mr.CMTM.from_se3_exponential_coordinates(
    coordinates, return_jacobian=True,
)
velocity = motion.cmvecs().cm_vecs()
```

For `order` coordinate rows, `motion` stores the pose `Exp(xi(0))` and
`order - 1` normalized coefficients of body velocity `Jr(xi(t)) @ xi_dot(t)`.
`jr` has shape `(..., order, 6, 6)` and stores normalized coefficients of
`Jr(xi(t))`. To compute the Jacobian series alone, use
`CMTM.se3_right_jacobian_coefficients(coordinates)`. Omit `return_jacobian`
to receive just the motion. A single coordinate row creates a pose-only motion.

Both methods also accept a `CMVector`. Wrap ordinary derivatives with
`CMVector(derivatives)` or pass normalized rows directly. The methods preserve
batch axes and support NumPy and JAX. For JAX JIT, pass `LIB="jax"`, keep the
coefficient count and `return_jacobian` static, and return arrays from the
traced function. The coefficient recurrences use analytic series products.

### CMTM coefficient operators

CMTM block operators use `CMVector`, whose coefficients are ordinary
kth derivatives divided by `k!`:

```python
series = mr.CMVector(np.ones((5, 6)))  # Ordinary derivatives as input
matrix = mr.CMTM.hat(mr.SE3, series)
recovered = mr.CMTM.vee(mr.SE3, matrix)  # Returns CMVector
ordinary = recovered.vecs()
coefficients = recovered.cm_vecs()
```

Use `CMVector.set_cmvecs(coefficients)` when the input is already normalized.
`hat_adj`, `hat_commute`, and `hat_commute_adj` follow the same input convention;
`vee_adj` also returns a `CMVector`.

The older `hat_cm`, `hat_cm_adj`, `hat_cm_commute`, `hat_cm_commute_adj`,
`vee_cm`, and `vee_cm_adj` names remain compatibility wrappers. Legacy ndarray
inputs to the hat operators are still used directly, without factorial scaling.
For the former ndarray output of `vee` or `vee_adj`, pass `as_array=True`;
this returns CM coefficients. Use `.vecs()` to obtain ordinary derivatives.
The single-vector `SO3.hat` and `SE3.hat` APIs are unchanged.

CMTM also stores its motion series exclusively as CM coefficients. Its constructor
accepts either ordinary derivative arrays or a `CMVector` directly; passing a
`CMVector` preserves its coefficient storage without a conversion round trip.
`cmvecs()` exposes that canonical series. `vecs()`, `vecs_flatten()`, and
`elem_vecs()` reconstruct ordinary values on request; the returned NumPy arrays
are detached from the coefficient storage and may differ by floating-point
rounding. Composition, inversion, matrix recovery, and element-class changes
keep coefficients normalized internally.

Numeric inputs to SO3, SE3 (including wrench/inertia operators), CMVector, and
CMTM matrix recovery/hat/vee paths are normalized at entry. NumPy integer and
boolean arrays become float64; JAX integer/boolean arrays become float32, or
float64 with `jax_enable_x64=True`. Floating-point and complex precision is
preserved subject to JAX's configured precision. This applies to constructors,
`set_*`, `update_*`, and direct Lie operations such as `exp` and its integrals.
SE3 promotes rotation and translation to a common dtype at construction/update,
so fractional positions and imaginary components are not discarded. At construction,
rotation `(..., 3, 3)` and translation `(..., 3)` batch axes are broadcast to a
common shape; invalid shapes are rejected immediately. Updates preserve these
stored shapes (supply the full batch shape to `update_*`).
Legacy `FactorialVector` keeps its existing conversion rules.

SO3, SE3, and CMVector capture input arrays independently. NumPy internal
arrays returned by `mat()` (SO3), `rot()`, `pos()`, `cm_vecs()`, and `cm_vec()`
are read-only. Detached calculated outputs, including SE3/CMTM matrices and
ordinary derivatives, remain writable without affecting the object.

Use explicit updates (all preserve existing shapes; construct a new object to
change order or batch dimensions):

```python
pose.update_pos(new_pos)
pose.update_rot(new_rot)
pose.update_pose(new_rot, new_pos)  # validates both before replacing either
series.update_vecs(new_derivatives)
series.update_cmvecs(new_coefficients)
cm.update_pose(pose)
cm.update_vecs(new_derivatives)
cm.update_cmvecs(new_coefficients)
```

SO3 provides `update_rot`. Existing `set_*` methods remain constructors.
CMTM captures its pose and series; updating their original objects does not
update CMTM. `cm.cmvecs()` returns an independent wrapper, so update `cm`
explicitly. Immutable internal storage may be shared, including by `truncate`,
but updates replace storage and leave earlier snapshots unchanged.
CMTM updates invalidate all derived caches; repeated reads do not scan inputs.
Shallow/deep copies and pickle round trips preserve read-only storage and
independent updates. CMTM copies/restored objects rebuild derived caches on
first use instead of copying or serializing cached matrices.
For JAX, construct objects inside traced functions; do not update persistent
Python objects from inside JIT/grad transformations.

Matrix recurrences promote the pose and coefficient dtypes together and use
floating-point storage when both inputs are integral.

With JAX-backed poses and `LIB="jax"`, `mat`, `mat_adj`, `mat_inv`,
`mat_inv_adj`, and their `apply_` methods support JIT, batching, and automatic
differentiation. The series shape and `output_order` must be static during
tracing. These methods do not retain JAX tracers in NumPy caches; this does not
imply JIT support for every CMTM API.
SO3/SE3 exponential and integral paths use Taylor coefficients near zero angle,
so forward/reverse differentiation also works at zero and pure translation.

`CMVector` stores only CM coefficients. Ordinary derivative arrays returned
by `.vecs()` and `.vec()` are reconstructed on each call; modifying them does
not change the stored coefficients. Reconstruction preserves the coefficient
dtype, including float32 and complex64.

For NumPy inputs, both constructors capture independent coefficients;
`set_cmvecs(coefficients)` avoids factorial conversion. Floating and complex
inputs retain their dtype; integer and boolean inputs become float64.
Coefficient accessors are read-only. Use `update_cmvecs` to replace coefficients
or `update_vecs` to supply ordinary derivatives. Truncated series share immutable
storage until updated, and updates do not propagate to other objects.

| NumPy input dtype | `CMVector(values)` coefficients | `set_cmvecs(values)` coefficients | `.vecs()` / `.vec()` |
| --- | --- | --- | --- |
| float32 / float64 | Input dtype | Input dtype | Input dtype |
| Integers / bool | float64 | float64 | float64 |
| complex64 / complex128 | Input dtype | Input dtype | Input dtype |

`vee` and `vee_adj` preserve floating-point and complex matrix dtypes in their
coefficient outputs, including `as_array=True` and the compatibility names.
Integer matrices are promoted to a floating-point dtype (int32/int64 to float64)
so subdiagonal averages retain fractional values. Recovered coefficients are
independent of the input matrix. JAX inputs follow JAX dtype promotion and
immutable array semantics instead of NumPy's float64 promotion.

Series can be truncated, added, subtracted, or negated directly in CM form:

```python
short = series.truncate(3)  # Keep three rows: derivatives 0 through 2
combined = short + recovered.truncate(3)
difference = short - recovered.truncate(3)
negative = -short
```

Addition and subtraction require equal row counts and dimensions; only batch
axes broadcast. Mixed one-dimensional and column series return column layouts.
`truncate(order)` accepts integer row counts from zero to the source row count
and shares coefficient storage. Arithmetic creates new coefficient arrays.

## Running Tests

Run the test suite with uv:

```bash
uv run --extra dev pytest
```

Test Python 3.9–3.14 with separate environments (requires Bash and uv):

```bash
./scripts/test-python.sh
./scripts/test-python.sh 3.9 3.13  # Selected versions only
```

The script uses `uv.lock` and may download Python and dependencies. It continues
after failures, prints a summary, and exits nonzero if any version fails.

## Planar rotations (SO2)

`SO2` has one angular degree of freedom, a 2x2 rotation matrix and a 1x1
identity adjoint. Use `SO2.set_angle(angle, LIB="numpy")` or `set_mat(matrix)`;
angles can have batch axes and tangent vectors have shape `(..., 1)`.
`angle()` returns a principal angle in `[-pi, pi]`.

`inv()` returns an SO2 object; `mat_inv()` returns its matrix. `update_rot()`
replaces owned read-only rotation storage, with the same copy/pickle semantics
as SO3. Composition preserves the backend and rejects mixed backends.
`exp`, `exp_integ`, `exp_integ2nd`, `hat`/`vee`, and the adjoint action support
NumPy/JAX, batching and JIT. Zero-angle derivatives are supported.

`CMTM(SO2.set_angle(angle), derivatives)` accepts derivatives of shape
`(..., order - 1, 1)`. Matrix/inverse/apply paths work with JAX; composition
and matrix recovery are tested with NumPy, as for the existing CMTM API.
The adjoint contains no angle or angular-velocity information: `set_mat_adj`
and `vee_adj` raise `ValueError`. Use `set_mat`/`set_angle` and ordinary `vee`.
The misleading legacy `set_quaternion(angle)` remains a deprecated alias for
`set_angle`; it does not accept a quaternion.

## User-defined matrix Lie groups

Subclass the public `mathrobo.MatrixLieGroup` in your own module. No registry,
package fork, or edits to MathRobo are needed. Implement these four methods:

| Method | Contract |
| --- | --- |
| `dof()` | Positive integer: Lie algebra dimension |
| `mat_size()` | Positive integer: square representation size |
| `hat(vec, LIB="numpy")` | Linear map `(..., dof)` to `(..., size, size)` |
| `vee(matrix, LIB="numpy")` | Inverse of `hat` on the algebra; returns `(..., dof)` |

A runnable, noncommutative example is provided in
[examples/custom_lie_group.py](examples/custom_lie_group.py): `Affine1D`
represents `x -> scale*x + offset` using a 2x2 matrix.
Copy this class into your own module and use it directly:

```python
from my_groups import Affine1D  # your own module containing the example class
from mathrobo import CMTM
import numpy as np

pose = Affine1D.set_mat(np.array([[2., .3], [0., 1.]]))
series = CMTM(pose, np.array([[.2, .4], [.1, -.1]]))
matrix = series.mat()
pose.update_mat(np.array([[3., .2], [0., 1.]]))
series.update_pose(pose)
```

Defaults provide identity, composition, inverse, adjoint, Lie brackets,
commute operators, exponential/integral matrices, owned read-only storage,
`update_mat`, and copy/pickle support. Adjoint columns are computed as
`vee(g @ hat(e_i) @ inv(g))`; algebra adjoints use the corresponding matrix
commutator. You may override these defaults with specialized faster methods.
The representation must be faithful on the Lie algebra, with
`vee(hat(v)) == v` and closure under commutators. Matrices supplied to the class
must belong to the group: the base checks dimensions, not group membership.
If needed, override construction and `update_mat` to validate membership.

For JAX, implement `hat`/`vee` using `jax.numpy` when `LIB="jax"`, without
converting tracers to NumPy or branching on their values. Batch axes must be
preserved. Generic exponentials use SciPy or JAX matrix exponentials and are
not intended to match hand-optimized group-specific performance.
CMTM NumPy matrix/recovery/composition paths and JAX matrix/apply paths support
these classes; this does not extend JIT support to all CMTM APIs. The optional
`CMTM.rand` helper requires a user-defined `rand(LIB=...)` factory.
Adjoint recovery (`set_mat_adj`, `vee_adj`) raises `NotImplementedError` by
default because these maps need not be injective. Override only when justified.
For pickle, define the class at module scope in an importable module.

Test a new group against independent formulas: identity/inverse/composition,
`vee(hat(v))`, conjugation and bracket identities, and finite-difference
exponential derivatives. Check batch shapes and each backend you support.

## Changelog

### Unreleased

- Added public `MatrixLieGroup` extension base and an external affine-group example; custom groups can be used with CMTM without registration. CMTM specialized paths now select built-in classes by identity, not class name.

- Breaking: complete SO2 with one degree of freedom, 1x1 adjoints, SO2-valued inverse, owned storage/update APIs, batch/JAX operations and CMTM integration. Non-invertible adjoint recovery now raises explicitly; use `set_angle` instead of the deprecated `set_quaternion` alias.

- Fixed SO2 principal-angle extraction for negative angles and all quadrants (`[-pi, pi]`).
- Breaking: SE3wrench exponential/integral methods now use the same sign convention as its pose adjoint and `hat_adj`; `exp_adj`/`exp_integ_adj` also use the wrench action.
- Breaking: `SE3.exp_integ(v, a)` now returns the full matrix integral, including bottom-right entry `a` instead of `1`. For the former homogeneous-coordinate convention, explicitly set that entry to `1` on a detached result.
- SE3 now broadcasts rotation/translation batch axes at construction and rejects invalid layouts immediately.

- Preserve immutable pose/coefficient storage after deepcopy and pickle restoration; rebuild CMTM caches from restored values.
- Fix non-finite reverse derivatives of SO3/SE3 exponentials at zero and lost zero-angle derivatives in integral paths.

- Breaking: integer/bool inputs are normalized to floating point at numeric entry points; SE3 rotation/position share a common dtype. This fixes truncation in integer poses and exponentials.
- Breaking: CMVector now preserves float32/complex64 for normalization and ordinary outputs instead of promoting them to float64/complex128.

- Fixed CMTM matrix recurrences discarding fractional or complex coefficients when the pose has an incompatible dtype.
- Breaking: pose/CM coefficient inputs are snapshots; use `update_*` instead of direct array mutation. CMTM updates invalidate caches.
- Fixed JAX/JIT transformation and inverse-adjoint matrix/apply paths, including batched inputs and differentiation.

### 0.0.6

- CMTM now stores and computes motion series in CM coefficients, reducing conversions to ordinary derivatives.
- **Compatibility:** `CMTM.vee()` and `vee_adj()` now return `CMVector`. Use `.vecs()` for ordinary derivatives or `.cm_vecs()` for CM coefficients; `as_array=True` returns CM coefficients directly.
- **Compatibility:** `vecs()` reconstructs ordinary values without sharing coefficient storage. NumPy results use at least float64 precision (complex128 for complex inputs).
- Added `CMVector` addition, subtraction, negation, and `truncate(order)` to retain the first `order` coefficient rows.

### Earlier changes

- Removed SymPy as a runtime dependency. Mathrobo now supports the NumPy and JAX code paths only, and `uv.lock` no longer includes SymPy or mpmath.

## Contributing

Contributions are welcome! Feel free to report issues, suggest features, or submit pull requests.

## License

This project is licensed under the [MIT License](LICENSE).
