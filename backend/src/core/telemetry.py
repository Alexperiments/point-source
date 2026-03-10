"""Shared Logfire configuration helpers."""

import logfire

from src.core.config import PROJECT_INFO, settings


def configure_logfire() -> None:
    """Configure Logfire from application settings."""
    token = settings.logfire_token.get_secret_value().strip()
    configure_kwargs: dict[str, object] = {
        "service_name": PROJECT_INFO["name"],
        "service_version": PROJECT_INFO["version"],
        "environment": settings.environment,
        "send_to_logfire": "if-token-present",
    }
    if token:
        configure_kwargs["token"] = token

    logfire.configure(**configure_kwargs)
