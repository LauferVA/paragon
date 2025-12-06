"""
PARAGON INFRASTRUCTURE - System-Level Modules

This package contains infrastructure components:
- data_loader: Polars-based data loading and schema validation
- environment: Auto-detection of system capabilities
- logger: Mutation event logging with Rerun integration
- metrics: Graph metrics computation using Pygmtools
- rerun_logger: Visual flight recorder for temporal debugging
- training_store: SQLite persistence for learning system
"""

from infrastructure.rerun_logger import RerunLogger, create_logger
from infrastructure.environment import (
    EnvironmentDetector,
    EnvironmentReport,
    detect_environment,
)
from infrastructure.training_store import TrainingStore

__all__ = [
    "RerunLogger",
    "create_logger",
    "EnvironmentDetector",
    "EnvironmentReport",
    "detect_environment",
    "TrainingStore",
]
