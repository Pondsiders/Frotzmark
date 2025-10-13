"""
PydanticAI agent setup for interactive fiction gameplay.

Defaults to OpenRouter for maximum model flexibility, but can be
configured to use other providers via environment variables.
"""

from pathlib import Path
from typing import Optional
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from .config import (
    API_KEY,
    BASE_URL,
    PROMPT_DIR,
    PROVIDER_TYPE
)


def load_system_prompt(
    context_files: Optional[list[Path]] = None,
    objective: Optional[str] = None
) -> str:
    """
    Load and assemble the system prompt from files.

    Structure:
    - prompts/preamble.md (optional): Context and instructions
    - [objective block]: Injected if objective is specified
    - [context_files]: Game-specific documentation (optional, multiple files)
    - prompts/postamble.md (optional): Additional guidance

    Args:
        context_files: List of paths to context files (e.g., manual, hints) (optional)
        objective: Specific objective to achieve (e.g., "reach the Living Room") (optional)
    """
    parts = []

    # Load preamble
    preamble_file = PROMPT_DIR / "preamble.md"
    if preamble_file.exists():
        parts.append(preamble_file.read_text().strip())

    # Inject objective block if specified
    if objective:
        objective_block = f"""# Your Objective

Your objective is:

- {objective}

After you achieve your objective, QUIT the game."""
        parts.append(objective_block)

    # Load context files if provided
    if context_files:
        for path in context_files:
            if path and path.exists():
                parts.append(path.read_text().strip())

    # Load postamble
    postamble_file = PROMPT_DIR / "postamble.md"
    if postamble_file.exists():
        parts.append(postamble_file.read_text().strip())

    return "\n\n".join(parts)


def create_agent(
    model_name: str,
    context_files: Optional[list[Path]] = None,
    temperature: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
    objective: Optional[str] = None
) -> Agent:
    """
    Create and configure the PydanticAI agent.

    By default, uses OpenRouter as the provider for maximum model
    availability. Advanced users can override PROVIDER_TYPE and BASE_URL
    to use other OpenAI-compatible endpoints (vLLM, etc.).

    Args:
        model_name: Name of the model to use (e.g., 'google/gemini-2.5-flash-lite')
        context_files: List of paths to context files (e.g., manual, hints) (optional)
        temperature: Sampling temperature (0.0 = deterministic, higher = more creative) (optional)
        reasoning_effort: Reasoning effort level for OpenRouter ('low', 'medium', 'high') (optional)
        objective: Specific objective to achieve (e.g., "reach the Living Room") (optional)
    """

    if PROVIDER_TYPE == "openrouter":
        # OpenRouter via OpenAI-compatible provider
        provider = OpenAIProvider(
            api_key=API_KEY,
            base_url=BASE_URL
        )
        model = OpenAIChatModel(model_name, provider=provider)

    else:
        # Generic OpenAI-compatible endpoint
        # (for vLLM servers, custom deployments, etc.)
        provider = OpenAIProvider(
            api_key=API_KEY,
            base_url=BASE_URL
        )
        model = OpenAIChatModel(model_name, provider=provider)

    system_prompt = load_system_prompt(context_files, objective=objective)

    # Configure model settings
    model_settings = None
    settings_kwargs = {}

    # Add temperature if specified
    if temperature is not None:
        settings_kwargs['temperature'] = temperature

    # Add reasoning config if specified (OpenRouter only)
    if reasoning_effort and PROVIDER_TYPE == "openrouter":
        settings_kwargs['extra_body'] = {
            'reasoning': {
                'effort': reasoning_effort,
                'exclude': False  # Include reasoning output in response
            }
        }

    # Create ModelSettings if we have any config
    if settings_kwargs:
        model_settings = ModelSettings(**settings_kwargs)

    agent = Agent(
        model=model,
        system_prompt=system_prompt,
        model_settings=model_settings
    )

    return agent
