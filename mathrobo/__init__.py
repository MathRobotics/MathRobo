"""Mathematical tools for robotics: Lie groups, CMTM, calculus and rotations."""
from .basic import *
from .lie import *
from .calculus import *
from .transformation import *
from . import basic as _basic, lie as _lie, calculus as _calculus, transformation as _transformation

__all__ = [
    *_basic.__all__,
    *_lie.__all__,
    *_calculus.__all__,
    *_transformation.__all__,
]


def __getattr__(name):
    # Kept for code written against the old star-import namespace.
    if name == "cmvec":
        import warnings
        warnings.warn("mathrobo.cmvec is deprecated; use mathrobo.CMVector",
                      DeprecationWarning, stacklevel=2)
        from .basic import cm_vec
        return cm_vec
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
