"""Utilities for presenting symbolic robotics equations.

This module exposes helpers for rendering SymPy expressions to MathJax so
that tutorials and documentation can share the same rendering pipeline.
"""

from .render import render_mj

__all__ = ["render_mj"]
