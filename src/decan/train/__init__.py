"""
DeCAN training package.

Contains modules to train a DeCAN module using datasets from the `data` package,
and utilities to checkpoint and configure the training pipeline.
"""

from .trainer import Trainer
from .configuration_trainer import TrainerConfig
