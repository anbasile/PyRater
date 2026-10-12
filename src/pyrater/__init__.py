"""PyRater: a toolkit for annotation analysis with Bayesian models."""

from pyrater.data import AnnotationData, load_dataset
from pyrater.models import PyRaterModel

__version__ = "0.2.0"
__all__ = ["AnnotationData", "PyRaterModel", "__version__", "load_dataset"]
