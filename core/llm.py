"""
PARAGON INTELLIGENCE - Structured LLM Interface

Provides a strictly typed interface for LLM generation.
Enforces msgspec schema compliance via JSON Mode + Validation.

Design:
- "Option 1" Strategy: JSON Mode (Prompt) -> Output -> msgspec.decode
- Uses msgspec.json.schema() for ground-truth prompt generation.
- Agnostic to underlying provider (OpenAI, Anthropic, etc.) via LiteLLM.

Architecture:
    User Code
        |
        v
    StructuredLLM.generate(prompt, schema=T)
        |
        v
    [Inject JSON Schema into System Prompt]
        |
        v
    [RateLimitGuard - Token bucket throttling]
        |
        v
    LiteLLM.completion(response_format=json_object)
        |
        v
    [msgspec.json.decode() - Strict Validation]
        |
        v
    Return T (or retry on ValidationError)

Model Routing (Cost Arbitrage):
    ModelRouter routes tasks to appropriate models based on task complexity:
    - HIGH_REASONING tasks (Architect, Builder, Coder) -> Claude Sonnet
    - MUNDANE tasks (Documenter, LogScrubber, Formatter) -> Ollama/Llama3

Rate Limit Protection:
    RateLimitGuard implements proactive throttling to prevent 429 errors:
    - Tracks requests per minute (RPM) and tokens per minute (TPM)
    - Pre-emptively waits when approaching limits
    - Respects retry-after headers when limits are hit
    - Configurable per-tier limits (default: Tier 1)
"""
import os
import json
import time
import logging
import msgspec
from typing import Type, TypeVar, Optional, Any, Dict, List, Tuple
from enum import Enum
from threading import Lock
from collections import deque
import litellm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
    retry_if_exception_type,
    RetryError,
)

logger = logging.getLogger(__name__)

# Generic type for return values
T = TypeVar("T", bound=msgspec.Struct)


# =============================================================================
# TASK TYPES (for Model Routing)
# =============================================================================

class TaskType(Enum):
    """
    Task complexity classification for model routing.

    HIGH_REASONING: Complex tasks requiring deep reasoning
        - Examples: Architect (design decisions), Builder (code generation),
                   Coder (complex refactoring), Auditor (verification)

    MUNDANE: Simple tasks with deterministic outputs
        - Examples: Documenter (formatting), LogScrubber (cleanup),
                   Formatter (style enforcement)

    SENSITIVE: Tasks involving PII, secrets, or internal logs
        - HARD ROUTED to local models only (no network requests)
        - Examples: Credential handling, internal log scrubbing
    """
    HIGH_REASONING = "high_reasoning"
    MUNDANE = "mundane"
    SENSITIVE = "sensitive"


# =============================================================================
# RATE LIMIT GUARD (Proactive Throttling)
# =============================================================================

class RateLimitGuard:
    """
    Proactive rate limit protection using sliding window.

    Prevents 429 errors by tracking usage and waiting proactively.
    Thread-safe for concurrent usage.

    Tier Limits (Anthropic, as of Dec 2025):
        Tier 1: 50 RPM, 30K ITPM, 8K OTPM (Sonnet 4.x)
        Tier 2: 1000 RPM, 450K ITPM, 90K OTPM
        Tier 3: 2000 RPM, 800K ITPM, 160K OTPM
        Tier 4: 4000 RPM, 2M ITPM, 400K OTPM

    Usage:
        guard = RateLimitGuard(rpm_limit=50, tpm_limit=30000)
        guard.wait_if_needed(estimated_tokens=3000)
        # ... make API call ...
        guard.record_usage(actual_tokens=2500)
    """

    def __init__(
        self,
        rpm_limit: int = 50,
        tpm_limit: int = 30000,
        safety_margin: float = 0.8,  # Use only 80% of limit
    ):
        """
        Initialize rate limit guard.

        Args:
            rpm_limit: Maximum requests per minute
            tpm_limit: Maximum tokens per minute (input tokens for Anthropic)
            safety_margin: Fraction of limit to use (0.8 = 80%)
        """
        self.rpm_limit = int(rpm_limit * safety_margin)
        self.tpm_limit = int(tpm_limit * safety_margin)

        # Sliding window tracking (timestamp, tokens)
        self._requests: deque = deque()  # timestamps of requests
        self._tokens: deque = deque()    # (timestamp, token_count) tuples
        self._lock = Lock()

        # Retry-after tracking
        self._retry_after_until: float = 0.0

    def _cleanup_old_entries(self, now: float) -> None:
        """Remove entries older than 60 seconds."""
        cutoff = now - 60.0

        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()

        while self._tokens and self._tokens[0][0] < cutoff:
            self._tokens.popleft()

    def _current_rpm(self) -> int:
        """Get current requests in the last minute."""
        return len(self._requests)

    def _current_tpm(self) -> int:
        """Get current tokens in the last minute."""
        return sum(t[1] for t in self._tokens)

    def wait_if_needed(self, estimated_tokens: int = 3000) -> float:
        """
        Wait if we're approaching rate limits.

        Args:
            estimated_tokens: Estimated input tokens for the request

        Returns:
            Seconds waited (0 if no wait needed)
        """
        with self._lock:
            now = time.time()
            self._cleanup_old_entries(now)

            # Check retry-after first
            if now < self._retry_after_until:
                wait_time = self._retry_after_until - now
                logger.info(f"Rate limit: waiting {wait_time:.1f}s (retry-after)")
                time.sleep(wait_time)
                return wait_time

            total_wait = 0.0

            # Check RPM
            if self._current_rpm() >= self.rpm_limit:
                # Wait until oldest request expires
                oldest = self._requests[0]
                wait_time = 60.0 - (now - oldest) + 0.1  # +0.1s buffer
                if wait_time > 0:
                    logger.info(f"Rate limit: waiting {wait_time:.1f}s (RPM: {self._current_rpm()}/{self.rpm_limit})")
                    time.sleep(wait_time)
                    total_wait += wait_time
                    now = time.time()
                    self._cleanup_old_entries(now)

            # Check TPM
            projected_tpm = self._current_tpm() + estimated_tokens
            if projected_tpm >= self.tpm_limit:
                # Wait until enough tokens expire
                tokens_to_free = projected_tpm - self.tpm_limit + 1000  # +1000 buffer
                freed = 0
                wait_until = now

                for ts, count in self._tokens:
                    freed += count
                    wait_until = ts + 60.0
                    if freed >= tokens_to_free:
                        break

                wait_time = wait_until - now + 0.1
                if wait_time > 0:
                    logger.info(f"Rate limit: waiting {wait_time:.1f}s (TPM: {self._current_tpm()}/{self.tpm_limit})")
                    time.sleep(wait_time)
                    total_wait += wait_time

            return total_wait

    def record_usage(self, tokens: int) -> None:
        """Record a completed request and its token usage."""
        with self._lock:
            now = time.time()
            self._requests.append(now)
            self._tokens.append((now, tokens))

    def set_retry_after(self, seconds: float) -> None:
        """Set retry-after from a 429 response."""
        with self._lock:
            self._retry_after_until = time.time() + seconds
            logger.warning(f"Rate limit hit, retry-after: {seconds}s")

    def get_status(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        with self._lock:
            now = time.time()
            self._cleanup_old_entries(now)
            return {
                "rpm_used": self._current_rpm(),
                "rpm_limit": self.rpm_limit,
                "tpm_used": self._current_tpm(),
                "tpm_limit": self.tpm_limit,
                "retry_after_remaining": max(0, self._retry_after_until - now),
            }


# Global rate limit guard (shared across all LLM instances)
_rate_limit_guard: Optional[RateLimitGuard] = None
_rate_limit_lock = Lock()


def get_rate_limit_guard() -> RateLimitGuard:
    """Get or create the global rate limit guard."""
    global _rate_limit_guard
    with _rate_limit_lock:
        if _rate_limit_guard is None:
            # Default to Tier 1 limits, can be configured via env
            rpm = int(os.getenv("PARAGON_RATE_LIMIT_RPM", "50"))
            tpm = int(os.getenv("PARAGON_RATE_LIMIT_TPM", "30000"))
            _rate_limit_guard = RateLimitGuard(rpm_limit=rpm, tpm_limit=tpm)
        return _rate_limit_guard


def reset_rate_limit_guard() -> None:
    """Reset the global rate limit guard (for testing)."""
    global _rate_limit_guard
    with _rate_limit_lock:
        _rate_limit_guard = None


# =============================================================================
# MODEL ROUTER (Cost Arbitrage)
# =============================================================================

class ModelRouter:
    """
    Deterministic router for cost-optimized model selection.

    Routes tasks to appropriate models based on complexity:
    - HIGH_REASONING -> Expensive models (Claude Sonnet 4.5)
    - MUNDANE -> Cheap/local models (Claude Haiku 4.5 / Ollama/Llama3.3)

    Model Tiers (Updated December 2025):

    A. High Reasoning (Architect, Coder, Auditor):
        - Primary: claude-sonnet-4-5-20250929 (recursive self-correction)
        - Fast Reasoner: claude-haiku-4-5-20251001 (matches old Sonnet perf)
        - Legacy Stable: claude-sonnet-4-20250514
        - OpenAI Fallback: gpt-4o (superior one-shot coding)
        - Google Fallback: gemini-3-pro-preview

    B. Mundane / High Volume (Log Scrubbing, Formatting, Docs):
        - Primary: claude-haiku-4-5-20251001
        - Secondary: claude-3-5-haiku-20241022
        - Local: ollama/llama3.3 (70B, 128K context)
        - OpenAI: gpt-4o-mini

    Usage:
        config = load_config()
        router = ModelRouter(config["llm"]["routing"])

        provider, model = router.route(TaskType.HIGH_REASONING)
        # Returns: ("anthropic", "claude-sonnet-4-5-20250929")

        provider, model = router.route(TaskType.MUNDANE)
        # Returns: ("anthropic", "claude-haiku-4-5-20251001")

    Configuration (from paragon.toml):
        [llm.routing]
        high_reasoning_provider = "anthropic"
        high_reasoning_model = "claude-sonnet-4-5-20250929"
        mundane_provider = "anthropic"
        mundane_model = "claude-haiku-4-5-20251001"
        mundane_fallback_provider = "anthropic"
        mundane_fallback_model = "claude-3-5-haiku-20241022"
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model router.

        Args:
            config: Configuration dict from [llm.routing] section

        Note: Model names must include provider prefix for LiteLLM
        (e.g., "anthropic/claude-sonnet-4-5-20250929")
        """
        # High reasoning configuration
        # Default: claude-sonnet-4-5 (current production flagship)
        self.high_reasoning_provider = config.get(
            "high_reasoning_provider", "anthropic"
        )
        self.high_reasoning_model = config.get(
            "high_reasoning_model", "claude-sonnet-4-5-20250929"
        )

        # Mundane task configuration
        # Default: claude-haiku-4-5 (fast and cheap)
        self.mundane_provider = config.get("mundane_provider", "anthropic")
        self.mundane_model = config.get("mundane_model", "claude-haiku-4-5-20251001")

        # Fallback for when local models unavailable
        self.mundane_fallback_provider = config.get(
            "mundane_fallback_provider", "anthropic"
        )
        self.mundane_fallback_model = config.get(
            "mundane_fallback_model", "claude-3-5-haiku-20241022"
        )

        # Sensitive task configuration (HARD ROUTE to local only)
        self.sensitive_provider = config.get("sensitive_provider", "ollama")
        self.sensitive_model = config.get("sensitive_model", "llama3.3")

    def route(self, task_type: TaskType, use_fallback: bool = False) -> Tuple[str, str]:
        """
        Route a task to the appropriate model.

        Args:
            task_type: The type of task (HIGH_REASONING, MUNDANE, or SENSITIVE)
            use_fallback: If True, use fallback for MUNDANE tasks (for local model failures)
                         Note: SENSITIVE tasks never use fallback (security requirement)

        Returns:
            Tuple of (provider, model) strings
                - provider: LiteLLM provider identifier (e.g., "anthropic", "ollama")
                - model: Model identifier (e.g., "claude-4-5-sonnet-20250921", "llama3")

        Examples:
            >>> router.route(TaskType.HIGH_REASONING)
            ('anthropic', 'claude-sonnet-4-5-20250929')

            >>> router.route(TaskType.MUNDANE)
            ('anthropic', 'claude-haiku-4-5-20251001')

            >>> router.route(TaskType.MUNDANE, use_fallback=True)
            ('anthropic', 'claude-3-5-haiku-20241022')

            >>> router.route(TaskType.SENSITIVE)
            ('ollama', 'llama3.3')  # Always local, no network
        """
        if task_type == TaskType.HIGH_REASONING:
            return (self.high_reasoning_provider, self.high_reasoning_model)

        elif task_type == TaskType.MUNDANE:
            if use_fallback:
                return (self.mundane_fallback_provider, self.mundane_fallback_model)
            else:
                return (self.mundane_provider, self.mundane_model)

        elif task_type == TaskType.SENSITIVE:
            # HARD ROUTE: Always local, ignore use_fallback for security
            return (self.sensitive_provider, self.sensitive_model)

        else:
            # Should never happen with Enum type safety
            raise ValueError(f"Unknown task type: {task_type}")

    def get_model_for_task(self, task_type: TaskType, use_fallback: bool = False) -> str:
        """
        Get the full model identifier for LiteLLM.

        Args:
            task_type: The type of task
            use_fallback: If True, use fallback for MUNDANE tasks

        Returns:
            Full model string for LiteLLM (e.g., "anthropic/claude-sonnet-4-20250514")
        """
        provider, model = self.route(task_type, use_fallback)

        # LiteLLM format: "provider/model" or just "model" for some providers
        if provider in ["anthropic", "openai", "gemini"]:
            # These providers need explicit prefix
            return f"{provider}/{model}"
        else:
            # Ollama and others can use model name directly
            return model


# =============================================================================
# EXCEPTIONS
# =============================================================================

class LLMError(Exception):
    """Base exception for LLM failures."""
    pass


class ValidationError(LLMError):
    """Raised when LLM output does not match the required schema."""
    pass


class RateLimitError(LLMError):
    """Raised when rate limited by the provider."""
    pass


# =============================================================================
# STRUCTURED LLM
# =============================================================================

class StructuredLLM:
    """
    A wrapper around LiteLLM that enforces structured outputs.

    Uses the "Option 1" strategy:
    1. Inject msgspec-generated JSON Schema into system prompt
    2. Use response_format=json_object where supported
    3. Validate response with msgspec.json.decode()
    4. Retry on validation failures (up to 3 attempts)

    Usage:
        llm = StructuredLLM(model="claude-sonnet-4-5-20250929")

        class MyOutput(msgspec.Struct):
            name: str
            value: int

        result = llm.generate(
            system_prompt="You are a data extractor.",
            user_prompt="Extract the name and value from: 'Widget costs 42'",
            schema=MyOutput
        )
        # result is typed as MyOutput
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        temperature: float = 0.0,
        max_tokens: int = 16384,
    ):
        """
        Initialize the structured LLM.

        Args:
            model: LiteLLM model identifier (e.g., "gpt-4o", "claude-sonnet-4-5-20250929")
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response (Claude 4.5 supports up to 64K)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Disable LiteLLM's verbose logging
        litellm.set_verbose = False

    def _get_schema_prompt(self, schema: Type[msgspec.Struct]) -> str:
        """
        Generate a strictly typed JSON schema from the msgspec Struct.

        Uses msgspec.json.schema() to generate official JSON Schema (Draft 2020-12).
        This provides the 'Ground Truth' for the LLM to follow.
        """
        try:
            # Generate official JSON Schema
            schema_dict = msgspec.json.schema(schema)
            return json.dumps(schema_dict, indent=2)
        except Exception:
            # Fallback: Build a simplified schema hint
            # This path is rarely hit since msgspec.json.schema() is very robust
            fields = msgspec.structs.fields(schema)
            field_hints = []
            for f in fields:
                # Convert type to a clean string representation
                type_str = getattr(f.type, "__name__", str(f.type))
                # Clean up typing module prefixes
                type_str = type_str.replace("typing.", "")
                field_hints.append(f'  "{f.name}": "<{type_str}>"')
            return "{\n" + ",\n".join(field_hints) + "\n}"

    def _build_system_prompt(self, base_prompt: str, schema: Type[msgspec.Struct]) -> str:
        """Build the full system prompt with schema injection."""
        schema_json = self._get_schema_prompt(schema)

        return f"""{base_prompt}

# OUTPUT CONTRACT
You are a deterministic data generator. You DO NOT speak. You ONLY output JSON.

Your output must strictly adhere to this JSON Schema:
```json
{schema_json}
```

CRITICAL RULES:
1. Output ONLY valid JSON - no markdown, no explanation, no preamble.
2. All required fields must be present.
3. Types must match exactly (strings are strings, numbers are numbers).
4. If a field is a List, it must be a JSON array.
5. Do not include any text before or after the JSON object.
"""

    def _clean_response(self, content: str) -> str:
        """Clean LLM response of common formatting issues."""
        content = content.strip()

        # Remove markdown code blocks
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]

        if content.endswith("```"):
            content = content[:-3]

        # Remove any leading/trailing whitespace after cleanup
        return content.strip()

    def _estimate_tokens(self, system_prompt: str, user_prompt: str) -> int:
        """Estimate input tokens for rate limiting (rough: 4 chars per token)."""
        total_chars = len(system_prompt) + len(user_prompt)
        return max(100, total_chars // 4)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(ValidationError),
        reraise=True,
    )
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[T],
    ) -> T:
        """
        Generate a structured response matching the provided schema.

        Args:
            system_prompt: The base system prompt (role, context, instructions)
            user_prompt: The specific user request
            schema: A msgspec.Struct subclass defining the expected output

        Returns:
            An instance of the schema type, populated from LLM response

        Raises:
            ValidationError: If schema validation fails after 3 retries
            LLMError: If the LLM API call fails
            RateLimitError: If rate limited by provider

        Mechanism:
        1. Inject JSON Schema into System Prompt
        2. Proactive Rate Limit Check (wait if needed)
        3. Force JSON mode via response_format (where supported)
        4. Validate/Parse with msgspec (High Performance)
        5. Record usage for rate limiting
        6. Retry on Validation Error (up to 3 times)
        """
        full_system_prompt = self._build_system_prompt(system_prompt, schema)

        # Proactive rate limit protection
        guard = get_rate_limit_guard()
        estimated_tokens = self._estimate_tokens(full_system_prompt, user_prompt)
        guard.wait_if_needed(estimated_tokens)

        # Diagnostic tracking
        start_time = time.time()
        input_tokens = 0
        output_tokens = 0
        truncated = False
        error_msg = None

        try:
            # LiteLLM handles provider differences
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": full_system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                # JSON mode - widely supported (OpenAI, Anthropic, etc.)
                response_format={"type": "json_object"},
                # Ignore unsupported params for provider flexibility
                drop_params=True,
            )

            content = response.choices[0].message.content
            content = self._clean_response(content)

            # Record actual usage for rate limiting
            if response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
            guard.record_usage(input_tokens or estimated_tokens)

            # Validation Step (The "Gatekeeper")
            # msgspec is strict by default on types
            result = msgspec.json.decode(content.encode("utf-8"), type=schema)

            # Record successful call to diagnostics
            self._record_diagnostic(
                schema.__name__, start_time, input_tokens, output_tokens,
                success=True, truncated=False, error=None
            )
            return result

        except msgspec.ValidationError as e:
            error_msg = f"Schema validation failed: {e}"
            self._record_diagnostic(
                schema.__name__, start_time, input_tokens, output_tokens,
                success=False, truncated=False, error=error_msg
            )
            # This triggers the @retry decorator
            raise ValidationError(error_msg)
        except msgspec.DecodeError as e:
            # Check for truncation
            error_msg = f"JSON decode failed: {e}"
            truncated = "truncated" in str(e).lower()
            self._record_diagnostic(
                schema.__name__, start_time, input_tokens, output_tokens,
                success=False, truncated=truncated, error=error_msg
            )
            raise ValidationError(error_msg)
        except litellm.RateLimitError as e:
            # Extract retry-after from headers if available
            retry_after = getattr(e, 'retry_after', None)
            if retry_after:
                guard.set_retry_after(float(retry_after))
                time.sleep(float(retry_after))
                # Retry once after waiting
                try:
                    response = litellm.completion(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": full_system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        response_format={"type": "json_object"},
                        drop_params=True,
                    )
                    content = response.choices[0].message.content
                    content = self._clean_response(content)
                    if response.usage:
                        input_tokens = response.usage.prompt_tokens
                        output_tokens = response.usage.completion_tokens
                    guard.record_usage(input_tokens or estimated_tokens)
                    result = msgspec.json.decode(content.encode("utf-8"), type=schema)
                    self._record_diagnostic(
                        schema.__name__, start_time, input_tokens, output_tokens,
                        success=True, truncated=False, error=None
                    )
                    return result
                except Exception:
                    pass  # Fall through to raise
            self._record_diagnostic(
                schema.__name__, start_time, input_tokens, output_tokens,
                success=False, truncated=False, error=f"Rate limited: {e}"
            )
            raise RateLimitError(f"Rate limited: {e}")
        except RetryError:
            # All retries exhausted
            error_msg = f"Schema validation failed after 3 attempts for {schema.__name__}"
            self._record_diagnostic(
                schema.__name__, start_time, input_tokens, output_tokens,
                success=False, truncated=False, error=error_msg
            )
            raise ValidationError(error_msg)
        except Exception as e:
            # API outage, network error, etc.
            error_msg = f"LLM generation failed: {e}"
            self._record_diagnostic(
                schema.__name__, start_time, input_tokens, output_tokens,
                success=False, truncated=False, error=error_msg
            )
            raise LLMError(error_msg)

    def _record_diagnostic(
        self,
        schema_name: str,
        start_time: float,
        input_tokens: int,
        output_tokens: int,
        success: bool,
        truncated: bool,
        error: Optional[str],
    ) -> None:
        """Record LLM call to diagnostics (if available)."""
        try:
            from infrastructure.diagnostics import get_diagnostics
            dx = get_diagnostics()
            duration_ms = (time.time() - start_time) * 1000
            dx.record_llm_call_simple(
                schema_name=schema_name,
                duration_ms=duration_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                success=success,
                truncated=truncated,
                error=error,
            )
        except ImportError:
            pass  # Diagnostics not available
        except Exception:
            pass  # Don't let diagnostics break LLM calls

    def generate_with_history(
        self,
        messages: List[Dict[str, str]],
        schema: Type[T],
    ) -> T:
        """
        Generate with full message history (for multi-turn conversations).

        Args:
            messages: List of {"role": "...", "content": "..."} dicts
            schema: Expected output schema

        Returns:
            Parsed and validated response
        """
        if not messages:
            raise LLMError("Messages list cannot be empty")

        # Inject schema into the first system message, or prepend one
        schema_injection = self._build_system_prompt("", schema)

        processed_messages = []
        found_system = False

        for msg in messages:
            if msg["role"] == "system" and not found_system:
                # Append schema to existing system prompt
                processed_messages.append({
                    "role": "system",
                    "content": msg["content"] + "\n\n" + schema_injection,
                })
                found_system = True
            else:
                processed_messages.append(msg)

        if not found_system:
            # Prepend a system message with schema
            processed_messages.insert(0, {
                "role": "system",
                "content": schema_injection,
            })

        try:
            response = litellm.completion(
                model=self.model,
                messages=processed_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                drop_params=True,
            )

            content = response.choices[0].message.content
            content = self._clean_response(content)

            return msgspec.json.decode(content.encode("utf-8"), type=schema)

        except msgspec.ValidationError as e:
            raise ValidationError(f"Schema validation failed: {e}")
        except Exception as e:
            raise LLMError(f"LLM generation failed: {e}")


# =============================================================================
# GLOBAL SINGLETON
# =============================================================================

_llm_instance: Optional[StructuredLLM] = None
_learning_manager = None
_learning_available = False

# Try to import learning system (optional)
try:
    from infrastructure.learning import LearningManager, LearningMode, ModelRecommendation
    _learning_available = True
except ImportError:
    _learning_available = False


def get_learning_manager():
    """Get or create the learning manager (lazy initialization)."""
    global _learning_manager
    if _learning_available and _learning_manager is None:
        try:
            _learning_manager = LearningManager()
        except Exception:
            pass
    return _learning_manager


def get_llm(phase: Optional[str] = None, task_type: Optional[str] = None) -> StructuredLLM:
    """
    Get the global LLM instance, optionally with learning-based model selection.

    When learning is enabled in PRODUCTION mode:
    - Queries historical performance data
    - Recommends model based on past success rates
    - Falls back to default if no recommendation

    Args:
        phase: Optional phase name (e.g., "build", "test") for learning lookup
        task_type: Optional task type for more specific recommendations

    Configurable via environment variables:
    - PARAGON_LLM_MODEL: Model identifier (default: anthropic/claude-sonnet-4-5-20250929)
    - PARAGON_LLM_TEMPERATURE: Temperature (default: 0.0)
    - PARAGON_LEARNING_ENABLED: Enable learning-based model selection (default: "false")

    Model Options (Updated December 2025):
    - anthropic/claude-sonnet-4-5-20250929 (Current flagship, recursive self-correction)
    - anthropic/claude-haiku-4-5-20251001 (Fast, matches old Sonnet perf)
    - anthropic/claude-opus-4-5-20251101 (Premium, maximum intelligence)
    - anthropic/claude-sonnet-4-20250514 (Legacy stable)
    - openai/gpt-4o (Superior one-shot coding)
    - gemini/gemini-3-pro-preview (Google flagship)

    Note: LiteLLM requires provider prefix (anthropic/, openai/, gemini/, etc.)
    """
    global _llm_instance

    # Check if learning-based model selection is enabled
    learning_enabled = os.getenv("PARAGON_LEARNING_ENABLED", "false").lower() == "true"

    # Try to get learning-based recommendation
    recommended_model = None
    if learning_enabled and _learning_available and phase:
        try:
            manager = get_learning_manager()
            if manager:
                from agents.schemas import CyclePhase
                # Convert phase string to CyclePhase enum
                try:
                    cycle_phase = CyclePhase(phase.lower())
                    recommendation = manager.get_model_recommendation(cycle_phase, task_type)
                    if recommendation:
                        # Prepend provider if not already there
                        model_id = recommendation.model_id
                        if "/" not in model_id:
                            model_id = f"anthropic/{model_id}"
                        recommended_model = model_id
                        logger.debug(
                            f"Learning: Recommended {model_id} for {phase} "
                            f"(confidence={recommendation.confidence:.2f}, "
                            f"success_rate={recommendation.success_rate:.2%})"
                        )
                except (ValueError, KeyError):
                    pass  # Unknown phase, use default
        except Exception as e:
            logger.debug(f"Learning-based model selection failed: {e}")

    # Use recommendation or fall back to default/configured model
    if recommended_model:
        temperature = float(os.getenv("PARAGON_LLM_TEMPERATURE", "0.0"))
        return StructuredLLM(model=recommended_model, temperature=temperature)

    # Standard singleton behavior
    if _llm_instance is None:
        # Use current flagship model as default
        model = os.getenv("PARAGON_LLM_MODEL", "anthropic/claude-sonnet-4-5-20250929")
        temperature = float(os.getenv("PARAGON_LLM_TEMPERATURE", "0.0"))
        _llm_instance = StructuredLLM(model=model, temperature=temperature)
    return _llm_instance


def set_llm(llm: Optional[StructuredLLM]) -> None:
    """
    Set the global LLM instance.

    Useful for testing with mock LLMs or different configurations.
    """
    global _llm_instance
    _llm_instance = llm


def reset_llm() -> None:
    """Reset the global LLM instance (forces re-initialization on next get_llm())."""
    global _llm_instance
    _llm_instance = None
