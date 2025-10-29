from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("sentinel1to2")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .train_model import train_model 

__all__ = [
    "__version__",
    "train_model",
]
