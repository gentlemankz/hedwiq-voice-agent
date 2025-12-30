"""
Secrets management for Luframe Agent.

This module provides secure secret loading from either:
1. File-based secrets (production with Azure Key Vault + Docker secrets)
2. Environment variables (development with .env files)

Production flow:
1. Azure Key Vault stores secrets
2. fetch-secrets.sh downloads them to /run/secrets/ (tmpfs - RAM only)
3. Docker Compose mounts /run/secrets/ into containers
4. This module reads from those files

Development flow:
- Uses .env file directly via os.environ
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional


def get_secret(name: str, default: Optional[str] = None) -> str:
    """
    Reads a secret from a file path or falls back to environment variable.
    Supports both Docker secrets (file-based) and traditional env vars (dev).

    Args:
        name: The secret name (e.g., "LIVEKIT_API_KEY")
        default: Optional default value if secret is not found

    Returns:
        The secret value

    Raises:
        ValueError: If secret is not found and no default is provided
    """
    # Try file-based secret with _FILE suffix (Docker secrets convention)
    file_env_var = f"{name}_FILE"
    file_path = os.environ.get(file_env_var)

    if file_path:
        path = Path(file_path)
        if path.exists():
            return path.read_text().strip()

    # Try standard Docker secrets path
    # Convert LIVEKIT_API_KEY to livekit_api_key
    standard_path = Path(f"/run/secrets/{name.lower()}")
    if standard_path.exists():
        return standard_path.read_text().strip()

    # Fall back to environment variable (development)
    value = os.environ.get(name)
    if value:
        return value

    # Return default if provided
    if default is not None:
        return default

    raise ValueError(f"Secret {name} not found. Set {name} env var or {file_env_var} file path.")


def has_secret(name: str) -> bool:
    """Check if a secret exists without raising an error."""
    try:
        get_secret(name)
        return True
    except ValueError:
        return False


@lru_cache(maxsize=1)
def get_all_secrets() -> dict[str, str]:
    """
    Load all secrets once and cache them.

    Returns:
        Dictionary of all secret values
    """
    return {
        # LiveKit
        "livekit_url": get_secret("LIVEKIT_URL"),
        "livekit_api_key": get_secret("LIVEKIT_API_KEY"),
        "livekit_api_secret": get_secret("LIVEKIT_API_SECRET"),

        # Deepgram
        "deepgram_api_key": get_secret("DEEPGRAM_API_KEY"),

        # Azure OpenAI
        "azure_openai_api_key": get_secret("AZURE_OPENAI_API_KEY"),
        "azure_openai_endpoint": get_secret("AZURE_OPENAI_ENDPOINT"),
        "openai_api_version": get_secret("OPENAI_API_VERSION", "2024-10-01-preview"),
        "azure_openai_deployment": get_secret("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),

        # Internal Services
        "agent_service_url": get_secret("AGENT_SERVICE_URL", "http://localhost:8000"),
        "internal_service_token": get_secret("INTERNAL_SERVICE_TOKEN"),

        # Supabase
        "supabase_url": get_secret("SUPABASE_URL"),
        "supabase_service_role_key": get_secret("SUPABASE_SERVICE_ROLE_KEY"),
        "supabase_anon_key": get_secret("SUPABASE_ANON_KEY", ""),

        # Database
        "database_url": get_secret("DATABASE_URL"),
    }


# Convenience function for common use case
def load_secrets_to_env() -> None:
    """
    Load all file-based secrets into environment variables.
    This is useful for libraries that read directly from os.environ.

    Call this at application startup before initializing other modules.
    """
    secret_mappings = [
        ("LIVEKIT_URL", "/run/secrets/livekit_url"),
        ("LIVEKIT_API_KEY", "/run/secrets/livekit_api_key"),
        ("LIVEKIT_API_SECRET", "/run/secrets/livekit_api_secret"),
        ("DEEPGRAM_API_KEY", "/run/secrets/deepgram_api_key"),
        ("AZURE_OPENAI_API_KEY", "/run/secrets/azure_openai_api_key"),
        ("AZURE_OPENAI_ENDPOINT", "/run/secrets/azure_openai_endpoint"),
        ("OPENAI_API_VERSION", "/run/secrets/openai_api_version"),
        ("AZURE_OPENAI_DEPLOYMENT", "/run/secrets/azure_openai_deployment"),
        ("INTERNAL_SERVICE_TOKEN", "/run/secrets/internal_service_token"),
        ("SUPABASE_URL", "/run/secrets/supabase_url"),
        ("SUPABASE_SERVICE_ROLE_KEY", "/run/secrets/supabase_service_role_key"),
        ("SUPABASE_ANON_KEY", "/run/secrets/supabase_anon_key"),
        ("DATABASE_URL", "/run/secrets/database_url"),
    ]

    for env_var, file_path in secret_mappings:
        path = Path(file_path)
        if path.exists() and env_var not in os.environ:
            os.environ[env_var] = path.read_text().strip()


# Usage examples:
#
# Option 1: Direct secret access
# from secrets import get_secret
# api_key = get_secret("DEEPGRAM_API_KEY")
#
# Option 2: Load all secrets at startup
# from secrets import get_all_secrets
# secrets = get_all_secrets()
# api_key = secrets["deepgram_api_key"]
#
# Option 3: Load into environment (for libraries that use os.environ)
# from secrets import load_secrets_to_env
# load_secrets_to_env()  # Call once at startup
