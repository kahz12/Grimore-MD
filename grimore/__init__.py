"""Grimore — an automated knowledge engine for your document vault."""
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("grimore")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "3.2.0"

__all__ = ["__version__"]
