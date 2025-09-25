"""
Core training infrastructure for bubble_rag.

This module contains common training components shared between 
embedding and reranker training.
"""

from .base_trainer import BaseTrainer
from .training_coordinator import TrainingCoordinator
from .device_manager import DeviceManager
from .config_builder import ConfigBuilder

__all__ = [
    'BaseTrainer',
    'TrainingCoordinator', 
    'DeviceManager',
    'ConfigBuilder'
]