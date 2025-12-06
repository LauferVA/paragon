"""
PARAGON TOKEN COUNTER - The Context Budget Manager

Model-aware token counting and context budget management.
Uses tiktoken with character-based fallback for unsupported models.

Architecture:
- TokenCounter: Model-specific token counting
- ContextBudget: Budget management with truncation
- Caching: LRU cache for repeated content

Design Principles:
1. MODEL AWARENESS: Different models have different tokenizers
2. BUDGET ENFORCEMENT: Hard limits prevent context overflow
3. GRACEFUL DEGRADATION: Character fallback when tiktoken unavailable
4. TRUNCATION STRATEGY: Binary search for efficient truncation to budget
"""
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from functools import lru_cache
import re

# Try to import tiktoken, fall back to character-based counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None
    TIKTOKEN_AVAILABLE = False


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a specific model's tokenization."""
    name: str                           # Model name/ID
    context_limit: int                  # Maximum context tokens
    encoding_name: str = "cl100k_base"  # tiktoken encoding
    chars_per_token: float = 4.0        # Fallback estimate


# Common model configurations
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    # OpenAI GPT-4 family
    "gpt-4": ModelConfig("gpt-4", 8192, "cl100k_base"),
    "gpt-4-32k": ModelConfig("gpt-4-32k", 32768, "cl100k_base"),
    "gpt-4-turbo": ModelConfig("gpt-4-turbo", 128000, "cl100k_base"),
    "gpt-4o": ModelConfig("gpt-4o", 128000, "o200k_base"),
    "gpt-4o-mini": ModelConfig("gpt-4o-mini", 128000, "o200k_base"),

    # OpenAI GPT-3.5
    "gpt-3.5-turbo": ModelConfig("gpt-3.5-turbo", 16385, "cl100k_base"),

    # Anthropic Claude family
    "claude-3-opus": ModelConfig("claude-3-opus", 200000, "cl100k_base", 3.5),
    "claude-3-sonnet": ModelConfig("claude-3-sonnet", 200000, "cl100k_base", 3.5),
    "claude-3-haiku": ModelConfig("claude-3-haiku", 200000, "cl100k_base", 3.5),
    "claude-3.5-sonnet": ModelConfig("claude-3.5-sonnet", 200000, "cl100k_base", 3.5),
    "claude-opus-4-5": ModelConfig("claude-opus-4-5", 200000, "cl100k_base", 3.5),

    # Default fallback
    "default": ModelConfig("default", 8192, "cl100k_base", 4.0),
}


def get_model_config(model: str) -> ModelConfig:
    """
    Get configuration for a model.

    Handles model name variations (e.g., "gpt-4-0613" -> "gpt-4").
    """
    # Direct match
    if model in MODEL_CONFIGS:
        return MODEL_CONFIGS[model]

    # Try prefix matching (e.g., "gpt-4-0613" -> "gpt-4")
    for prefix in ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4-32k", "gpt-4",
                   "gpt-3.5-turbo", "claude-3.5-sonnet", "claude-3-opus",
                   "claude-3-sonnet", "claude-3-haiku", "claude-opus-4"]:
        if model.startswith(prefix):
            return MODEL_CONFIGS.get(prefix, MODEL_CONFIGS["default"])

    return MODEL_CONFIGS["default"]


# =============================================================================
# TOKEN COUNTER
# =============================================================================

class TokenCounter:
    """
    Model-aware token counter with caching.

    Uses tiktoken for accurate counting when available,
    falls back to character estimation otherwise.

    Usage:
        counter = TokenCounter("gpt-4")
        tokens = counter.count("Hello, world!")
        truncated = counter.truncate_to_budget("Long text...", 100)
    """

    def __init__(self, model: str = "gpt-4"):
        """
        Initialize the counter for a specific model.

        Args:
            model: Model name (e.g., "gpt-4", "claude-3-opus")
        """
        self.model = model
        self.config = get_model_config(model)
        self._encoding = None
        self._use_tiktoken = TIKTOKEN_AVAILABLE

        if self._use_tiktoken:
            try:
                self._encoding = tiktoken.get_encoding(self.config.encoding_name)
            except Exception:
                self._use_tiktoken = False

    @property
    def context_limit(self) -> int:
        """Get the context limit for this model."""
        return self.config.context_limit

    def count(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: The text to count tokens for

        Returns:
            Number of tokens
        """
        if not text:
            return 0

        if self._use_tiktoken and self._encoding:
            try:
                return len(self._encoding.encode(text))
            except Exception:
                pass

        # Character-based fallback
        return int(len(text) / self.config.chars_per_token)

    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens in a list of chat messages.

        Accounts for message overhead (role tokens, separators).

        Args:
            messages: List of {"role": "...", "content": "..."} dicts

        Returns:
            Total tokens including overhead
        """
        total = 0
        # Overhead per message (roughly 4 tokens for role, separators)
        message_overhead = 4

        for msg in messages:
            total += message_overhead
            if "content" in msg:
                total += self.count(msg["content"])
            if "name" in msg:
                total += self.count(msg["name"])

        # Final overhead for assistant response priming
        total += 3

        return total

    def fits_in_context(self, text: str, buffer: int = 0) -> bool:
        """
        Check if text fits in context with optional buffer.

        Args:
            text: Text to check
            buffer: Reserve tokens for output

        Returns:
            True if fits, False otherwise
        """
        return self.count(text) <= (self.context_limit - buffer)

    def truncate_to_budget(
        self,
        text: str,
        budget: int,
        truncate_from: str = "end",
        ellipsis: str = "...",
    ) -> Tuple[str, int]:
        """
        Truncate text to fit within a token budget.

        Uses binary search for efficient truncation.

        Args:
            text: Text to truncate
            budget: Maximum tokens allowed
            truncate_from: "end" or "start"
            ellipsis: Marker to indicate truncation

        Returns:
            Tuple of (truncated_text, actual_tokens)
        """
        current_tokens = self.count(text)

        if current_tokens <= budget:
            return text, current_tokens

        # Binary search for optimal truncation point
        ellipsis_tokens = self.count(ellipsis)
        target = budget - ellipsis_tokens

        if target <= 0:
            return "", 0

        # Estimate characters needed
        low = 0
        high = len(text)

        while low < high:
            mid = (low + high + 1) // 2
            if truncate_from == "end":
                sample = text[:mid]
            else:
                sample = text[-mid:]

            if self.count(sample) <= target:
                low = mid
            else:
                high = mid - 1

        if truncate_from == "end":
            result = text[:low] + ellipsis
        else:
            result = ellipsis + text[-low:]

        return result, self.count(result)

    def split_to_chunks(
        self,
        text: str,
        chunk_size: int,
        overlap: int = 0,
    ) -> List[str]:
        """
        Split text into chunks of approximately chunk_size tokens.

        Args:
            text: Text to split
            chunk_size: Target tokens per chunk
            overlap: Tokens to overlap between chunks

        Returns:
            List of text chunks
        """
        if not text:
            return []

        total_tokens = self.count(text)
        if total_tokens <= chunk_size:
            return [text]

        chunks = []

        # Estimate characters per token
        chars_per_token = len(text) / total_tokens

        # Target characters per chunk
        chunk_chars = int(chunk_size * chars_per_token)
        overlap_chars = int(overlap * chars_per_token)

        start = 0
        while start < len(text):
            end = min(start + chunk_chars, len(text))

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end in last 20% of chunk
                search_start = start + int(chunk_chars * 0.8)
                for pattern in [r'\.\s', r'\n\n', r'\n', r'\s']:
                    matches = list(re.finditer(pattern, text[search_start:end]))
                    if matches:
                        end = search_start + matches[-1].end()
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap_chars if overlap_chars > 0 else end

        return chunks


# =============================================================================
# CONTEXT BUDGET MANAGER
# =============================================================================

@dataclass
class ContextBudget:
    """
    Manages a context budget with allocation tracking.

    Allows allocating portions of context to different uses
    (system prompt, history, current request, output reserve).

    Usage:
        budget = ContextBudget("gpt-4", output_reserve=1000)
        budget.allocate("system", system_prompt)
        budget.allocate("history", chat_history)
        remaining = budget.remaining()
        truncated = budget.fit_remaining("long content")
    """
    model: str
    output_reserve: int = 1000          # Reserve for model output
    allocations: Dict[str, int] = field(default_factory=dict)
    contents: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self._counter = TokenCounter(self.model)

    @property
    def limit(self) -> int:
        """Total context limit for model."""
        return self._counter.context_limit

    @property
    def available(self) -> int:
        """Available tokens (limit minus reserve)."""
        return self.limit - self.output_reserve

    @property
    def used(self) -> int:
        """Tokens currently allocated."""
        return sum(self.allocations.values())

    def remaining(self) -> int:
        """Tokens remaining in budget."""
        return self.available - self.used

    def allocate(self, name: str, content: str) -> int:
        """
        Allocate content to a named slot.

        Args:
            name: Slot name (e.g., "system", "history")
            content: Content to allocate

        Returns:
            Tokens allocated
        """
        tokens = self._counter.count(content)
        self.allocations[name] = tokens
        self.contents[name] = content
        return tokens

    def deallocate(self, name: str) -> None:
        """Remove an allocation."""
        self.allocations.pop(name, None)
        self.contents.pop(name, None)

    def fits(self, content: str) -> bool:
        """Check if content fits in remaining budget."""
        return self._counter.count(content) <= self.remaining()

    def fit_remaining(
        self,
        content: str,
        name: Optional[str] = None,
    ) -> Tuple[str, int]:
        """
        Truncate content to fit remaining budget and optionally allocate.

        Args:
            content: Content to fit
            name: If provided, allocate the result to this slot

        Returns:
            Tuple of (possibly truncated content, tokens used)
        """
        remaining = self.remaining()
        truncated, tokens = self._counter.truncate_to_budget(content, remaining)

        if name:
            self.allocations[name] = tokens
            self.contents[name] = truncated

        return truncated, tokens

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of budget allocations."""
        return {
            "model": self.model,
            "limit": self.limit,
            "output_reserve": self.output_reserve,
            "available": self.available,
            "used": self.used,
            "remaining": self.remaining(),
            "allocations": dict(self.allocations),
            "utilization": self.used / self.available if self.available > 0 else 0.0,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

@lru_cache(maxsize=128)
def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens with caching.

    Cached by (text, model) for repeated counting of same content.
    """
    return TokenCounter(model).count(text)


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    model: str = "gpt-4",
) -> str:
    """Truncate text to fit within token limit."""
    truncated, _ = TokenCounter(model).truncate_to_budget(text, max_tokens)
    return truncated


def split_into_chunks(
    text: str,
    chunk_size: int,
    model: str = "gpt-4",
    overlap: int = 0,
) -> List[str]:
    """Split text into chunks of approximately chunk_size tokens."""
    return TokenCounter(model).split_to_chunks(text, chunk_size, overlap)


def get_context_limit(model: str) -> int:
    """Get the context limit for a model."""
    return get_model_config(model).context_limit


def estimate_tokens(text: str) -> int:
    """
    Quick token estimation without model specificity.

    Uses ~4 chars per token as a rough estimate.
    Useful for pre-filtering before accurate counting.
    """
    return len(text) // 4


# =============================================================================
# PRUNE HELPER (For Context Pruning)
# =============================================================================

def prune_to_budget(
    items: List[Tuple[str, float]],
    budget: int,
    model: str = "gpt-4",
) -> Tuple[List[str], int, float]:
    """
    Prune a list of items to fit within a token budget.

    Items are sorted by priority (higher = more important) and
    greedily selected until budget is exhausted.

    Args:
        items: List of (content, priority) tuples
        budget: Maximum tokens
        model: Model for token counting

    Returns:
        Tuple of (selected_contents, total_tokens, pruning_ratio)
    """
    if not items:
        return [], 0, 1.0

    counter = TokenCounter(model)

    # Calculate tokens for each item
    with_tokens = [
        (content, priority, counter.count(content))
        for content, priority in items
    ]

    # Sort by priority (descending)
    with_tokens.sort(key=lambda x: x[1], reverse=True)

    # Greedily select
    selected = []
    total_tokens = 0

    for content, priority, tokens in with_tokens:
        if total_tokens + tokens <= budget:
            selected.append(content)
            total_tokens += tokens

    # Calculate pruning ratio
    original_count = len(items)
    selected_count = len(selected)
    pruning_ratio = 1.0 - (selected_count / original_count) if original_count > 0 else 0.0

    return selected, total_tokens, pruning_ratio
