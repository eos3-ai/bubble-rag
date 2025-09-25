"""
Specialized trainers for different model types.

This module contains trainers specialized for embedding and reranker models,
implementing the common BaseTrainer interface.
"""

from .embedding_trainer import EmbeddingTrainer
from .reranker_trainer import RerankerTrainer

__all__ = [
    'EmbeddingTrainer',
    'RerankerTrainer'
]