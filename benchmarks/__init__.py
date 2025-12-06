"""
Paragon Verification Benchmarks

Protocol Alpha: Speed verification (performance targets)
Protocol Beta: Integrity verification (correctness)

Run with:
    python -m benchmarks.protocol_alpha
    python -m benchmarks.protocol_beta
"""

from .protocol_alpha import run_protocol_alpha
from .protocol_beta import run_protocol_beta

__all__ = ["run_protocol_alpha", "run_protocol_beta"]
