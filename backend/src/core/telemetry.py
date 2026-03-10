"""Shared Logfire configuration helpers."""

import logfire

from src.core.config import PROJECT_INFO, settings


def configure_logfire() -> None:
    """Configure Logfire from application settings."""
    token = settings.logfire_token.get_secret_value().strip() or None
    logfire.configure(
        send_to_logfire=settings.logfire_send_to_logfire,
        token=token,
        service_name=PROJECT_INFO["name"],
        service_version=PROJECT_INFO["version"],
        environment=settings.environment,
    )
