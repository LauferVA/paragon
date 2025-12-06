"""
PARAGON CORE - Central exports for core functionality.

This module provides access to:
- LLM interfaces (StructuredLLM, ModelRouter)
- Resource monitoring (ResourceGuard)
- System optimization utilities
"""

# LLM and Intelligence
from core.llm import (
    StructuredLLM,
    ModelRouter,
    TaskType,
    LLMError,
    ValidationError,
    RateLimitError,
    get_llm,
    set_llm,
    reset_llm,
)

# Resource Management (optional import - requires psutil)
try:
    from core.resource_guard import (
        ResourceGuard,
        ResourceSignal,
        get_resource_guard,
        init_resource_guard,
        shutdown_resource_guard,
    )
    _RESOURCE_GUARD_AVAILABLE = True
except ImportError:
    # psutil not installed - resource guard unavailable
    ResourceGuard = None
    ResourceSignal = None
    get_resource_guard = None
    init_resource_guard = None
    shutdown_resource_guard = None
    _RESOURCE_GUARD_AVAILABLE = False

__all__ = [
    # LLM
    "StructuredLLM",
    "ModelRouter",
    "TaskType",
    "LLMError",
    "ValidationError",
    "RateLimitError",
    "get_llm",
    "set_llm",
    "reset_llm",
    # Resource Guard (may be None if psutil not available)
    "ResourceGuard",
    "ResourceSignal",
    "get_resource_guard",
    "init_resource_guard",
    "shutdown_resource_guard",
]
