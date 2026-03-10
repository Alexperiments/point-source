"""Shared Logfire configuration helpers."""

import logfire

from src.core.config import PROJECT_INFO, settings


def configure_logfire() -> None:
    """Configure Logfire from application settings."""
    logfire.configure(
        token=settings.logfire_token.get_secret_value().strip(),
        service_name=PROJECT_INFO["name"],
        service_version=PROJECT_INFO["version"],
        environment=settings.environment,
    )
