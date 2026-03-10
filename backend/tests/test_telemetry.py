"""Tests for shared Logfire configuration."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from pydantic import SecretStr

import src.core.telemetry as telemetry


class TestConfigureLogfire:
    """Tests for configure_logfire."""

    def test_configures_logfire_with_settings_token(self, monkeypatch) -> None:
        configure = MagicMock()

        monkeypatch.setattr(telemetry.logfire, "configure", configure)
        monkeypatch.setattr(
            telemetry,
            "settings",
            SimpleNamespace(
                logfire_token=SecretStr("  test-token  "),
                environment="production",
            ),
        )

        telemetry.configure_logfire()

        configure.assert_called_once_with(
            service_name=telemetry.PROJECT_INFO["name"],
            service_version=telemetry.PROJECT_INFO["version"],
            environment="production",
            send_to_logfire="if-token-present",
            token="test-token",
        )

    def test_skips_token_when_blank(self, monkeypatch) -> None:
        configure = MagicMock()

        monkeypatch.setattr(telemetry.logfire, "configure", configure)
        monkeypatch.setattr(
            telemetry,
            "settings",
            SimpleNamespace(
                logfire_token=SecretStr("   "),
                environment="development",
            ),
        )

        telemetry.configure_logfire()

        configure.assert_called_once_with(
            service_name=telemetry.PROJECT_INFO["name"],
            service_version=telemetry.PROJECT_INFO["version"],
            environment="development",
            send_to_logfire="if-token-present",
        )
