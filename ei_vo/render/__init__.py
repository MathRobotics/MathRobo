"""MathJax rendering helpers for symbolic expressions.

The :func:`render_mj` function accepts individual SymPy expressions or
collections of named expressions and produces a MathJax snippet that can be
embedded into static documentation or notebooks.  The helper provides a
single place for formatting so that downstream examples can focus on the
mathematics instead of display details.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, List, Tuple

import sympy as sp

__all__ = ["render_mj"]

EquationInput = Any


def _is_equation_sequence(obj: Sequence[Any]) -> bool:
    """Return ``True`` if *obj* looks like a sequence of ``(name, expr)`` pairs."""

    if isinstance(obj, (str, bytes)):
        return False
    for item in obj:
        if not (isinstance(item, Sequence) and len(item) == 2):
            return False
    return True


def _convert_to_items(content: EquationInput) -> List[Tuple[str | None, Any]]:
    """Normalise supported inputs into ``(name, expr)`` pairs."""

    if isinstance(content, Mapping):
        return [(str(name), expr) for name, expr in content.items()]
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        if _is_equation_sequence(content):
            return [(str(name) if name is not None else None, expr) for name, expr in content]
        return [(None, expr) for expr in content]
    return [(None, content)]


def _to_mathjax(expr: Any) -> str:
    """Convert *expr* into a MathJax string, falling back to ``str``."""

    if isinstance(expr, str):
        return expr
    try:
        return sp.latex(expr)  # type: ignore[arg-type]
    except Exception:
        return str(expr)


def render_mj(content: EquationInput, *, inline: bool = False, title: str | None = None) -> str:
    """Render *content* as a MathJax string.

    Parameters
    ----------
    content:
        Either a single SymPy expression, a sequence of expressions, or a
        mapping/sequence of ``(name, expression)`` pairs.  Strings are
        considered pre-rendered snippets and are used verbatim.
    inline:
        If ``True`` the expression is wrapped in ``\( … \)`` instead of
        ``\[ … \]``.
    title:
        Optional plain text title prepended to the rendered output.

    Returns
    -------
    str
        A MathJax snippet suitable for embedding in HTML documents.
    """

    items = _convert_to_items(content)
    if not items:
        raise ValueError("render_mj requires at least one expression to render")

    lines = []
    for name, expr in items:
        rhs = _to_mathjax(expr)
        if name:
            # Ensure the name is interpreted verbatim.  Users can pass pre-built
            # MathJax names if they need more control.
            lines.append(f"{name} = {rhs}")
        else:
            lines.append(rhs)

    body = r" \\ ".join(lines)
    wrapper = r"\( {} \)" if inline else r"\[ {} \]"
    rendered = wrapper.format(body)

    if title:
        return f"<div class=\"mathjax-block\"><strong>{title}</strong>: {rendered}</div>"
    return rendered
