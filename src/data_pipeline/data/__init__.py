"""
data subpackage:
fetching, funding, and preprocessing for crypto datasets
"""

from .data_preprocess import main as fetch_and_merge
from .preprocessing import preprocess

__all__ = ["fetch_and_merge", "preprocess"]
