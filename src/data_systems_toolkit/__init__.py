"""
Data Systems Toolkit - Simulate and understand enterprise data architectures.
"""

from importlib.metadata import version, PackageNotFoundError

try:
 __version__ = version("data-systems-toolkit")
except PackageNotFoundError:
 __version__ = "0.1.0.dev0"

__all__ = ["__version__"]
