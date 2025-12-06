"""
PARAGON INFRASTRUCTURE - System-Level Modules

This package contains infrastructure components:
- data_loader: Polars-based data loading and schema validation
- logger: Mutation event logging with Rerun integration
- metrics: Graph metrics computation using Pygmtools
- rerun_logger: Visual flight recorder for temporal debugging
"""

from infrastructure.rerun_logger import RerunLogger, create_logger

__all__ = [
    "RerunLogger",
    "create_logger",
]
