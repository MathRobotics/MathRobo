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
Refer to the examples in the `examples` folder, where you can find Jupyter notebooks and scripts demonstrating various use cases of the library.

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

`CMVector` stores only CM coefficients. Ordinary derivative arrays returned
by `.vecs()` and `.vec()` are reconstructed on each call; modifying them does
not change the stored coefficients. NumPy reconstruction uses at least float64
precision, so the dtype may differ from the original input.

For NumPy inputs, `CMVector(values)` creates independent coefficients, whereas
`set_cmvecs(coefficients)` retains the supplied array and its dtype. Mutating
that input or the array returned by `.cm_vecs()` updates the stored coefficients.
Read-only inputs remain read-only with `set_cmvecs`. `.cm_vec()` reshapes the
coefficient array and may copy non-contiguous data.

| NumPy input dtype | `CMVector(values)` coefficients | `set_cmvecs(values)` coefficients | `.vecs()` / `.vec()` |
| --- | --- | --- | --- |
| float32 / float64 | float64 | Input dtype | float64 |
| int32 / int64 | float64 | Input dtype | float64 |
| complex64 / complex128 | complex128 | Input dtype | complex128 |

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

## Changelog

### Unreleased

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
