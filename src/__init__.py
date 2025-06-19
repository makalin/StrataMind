"""
StrataMind - AI for Detecting Rare Minerals in Geological Data
"""

__version__ = "0.1.0"
__author__ = "makalin"
__email__ = "contact@stratamind.ai"

from .model import MineralDetector, load_model, predict_mineral
from .utils import load_image, preprocess_image, visualize_results
from .data import MineralDataset, get_transforms

__all__ = [
    "MineralDetector",
    "load_model", 
    "predict_mineral",
    "load_image",
    "preprocess_image", 
    "visualize_results",
    "MineralDataset",
    "get_transforms"
] 