"""
data subpackage:
fetching, funding, and preprocessing for crypto datasets
"""
# src/data_pipeline/data/__init__.py
"""
Initialization for data subpackage.
Handles preprocessing, normalization, standardization, and postprocessing.
"""

from .preprocessing import preprocess
from .normalize import normalize_features
from .data_postprocess import prepare_dataframe_for_model
from src.data_pipeline.features.technical import add_technical_indicators 

__all__ = [
    "preprocess",
    "normalize_features",
    "prepare_dataframe_for_model",
    "add_technical_indicators",
]
