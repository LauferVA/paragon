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
"""
import os
import json
import msgspec
from typing import Type, TypeVar, Optional, Any, Dict, List, Tuple
from enum import Enum
import litellm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
    retry_if_exception_type,
    RetryError,
)

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
        max_tokens: int = 4096,
    ):
        """
        Initialize the structured LLM.

        Args:
            model: LiteLLM model identifier (e.g., "gpt-4o", "claude-sonnet-4-5-20250929")
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response
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
        2. Force JSON mode via response_format (where supported)
        3. Validate/Parse with msgspec (High Performance)
        4. Retry on Validation Error (up to 3 times)
        """
        full_system_prompt = self._build_system_prompt(system_prompt, schema)

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

            # Validation Step (The "Gatekeeper")
            # msgspec is strict by default on types
            return msgspec.json.decode(content.encode("utf-8"), type=schema)

        except msgspec.ValidationError as e:
            # This triggers the @retry decorator
            raise ValidationError(f"Schema validation failed: {e}")
        except msgspec.DecodeError as e:
            # Invalid JSON structure
            raise ValidationError(f"JSON decode failed: {e}")
        except litellm.RateLimitError as e:
            # Extract retry-after from headers if available
            retry_after = getattr(e, 'retry_after', None)
            if retry_after:
                import time
                import logging
                logging.getLogger(__name__).warning(
                    f"Rate limited. Waiting {retry_after}s before retry..."
                )
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
                    return msgspec.json.decode(content.encode("utf-8"), type=schema)
                except Exception:
                    pass  # Fall through to raise
            raise RateLimitError(f"Rate limited: {e}")
        except RetryError:
            # All retries exhausted
            raise ValidationError(f"Schema validation failed after 3 attempts for {schema.__name__}")
        except Exception as e:
            # API outage, network error, etc.
            raise LLMError(f"LLM generation failed: {e}")

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


def get_llm() -> StructuredLLM:
    """
    Get the global LLM instance.

    Configurable via environment variables:
    - PARAGON_LLM_MODEL: Model identifier (default: anthropic/claude-sonnet-4-5-20250929)
    - PARAGON_LLM_TEMPERATURE: Temperature (default: 0.0)

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
