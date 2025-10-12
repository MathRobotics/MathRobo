"""Generate a MathJax rendering of common SE(3) relationships.

Why ship actual symbolic equations here?  The tests under ``tests/basic`` and
the documentation snippets need a deterministic way of verifying that
``render_mj`` keeps producing consistent MathJax.  By expressing a handful of
canonical rigid-body identities (rotation matrix, position, velocity and
angular velocity) with SymPy, we can lock the formatting behaviour down to a
realistic robotics use case instead of ad-hoc strings.  This keeps the
regression test close to how downstream notebooks display formulas while still
remaining lightweight enough to run during continuous integration.
"""

from __future__ import annotations

import sympy as sp

from ei_vo.render import render_mj


def make_equations() -> dict[str, sp.Expr]:
    """Return a small collection of symbolic expressions for demonstration."""

    t = sp.symbols("t")
    theta = sp.Function("theta")(t)
    omega = sp.diff(theta, t)

    rot = sp.Matrix([
        [sp.cos(theta), -sp.sin(theta), 0],
        [sp.sin(theta), sp.cos(theta), 0],
        [0, 0, 1],
    ])
    pos = sp.Matrix([sp.cos(theta), sp.sin(theta), t])
    vel = sp.diff(pos, t)

    return {
        "R(t)": rot,
        "p(t)": pos,
        "v(t)": vel,
        "\u03c9(t)": omega,
    }


def demo_mathjax() -> str:
    """Create a MathJax block containing the demonstration equations."""

    equations = make_equations()
    return render_mj(equations, title="Planar rigid-body motion")


if __name__ == "__main__":  # pragma: no cover - convenience script entry point
    print(demo_mathjax())
