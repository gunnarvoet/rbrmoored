"""
Library for RBR data processing.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

__all__ = ["solo", "utils"]

try:
    __version__ = _version("rbrmoored")
except PackageNotFoundError:  # not installed, e.g. run from a source tree
    __version__ = "unknown"

from . import solo, utils
