"""utils: помощни функции за упражнение 1 по Машинно самообучение.

Пакетът съществува, за да илюстрира как се организира код извън тетрадката.
"""

from .preprocessing import standardize, minmax_scale
from .distances import euclidean, pairwise_euclidean, cosine_similarity

__all__ = [
    "standardize",
    "minmax_scale",
    "euclidean",
    "pairwise_euclidean",
    "cosine_similarity",
]

__version__ = "0.1.0"
