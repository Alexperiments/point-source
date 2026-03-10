"""Tests for shared Logfire configuration."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from pydantic import SecretStr

import src.core.telemetry as telemetry


class TestConfigureLogfire:
    """Tests for configure_logfire."""

    def test_configures_logfire_from_project_settings(self, monkeypatch) -> None:
        configure = MagicMock()

        monkeypatch.setattr(telemetry.logfire, "configure", configure)
        monkeypatch.setattr(
            telemetry,
            "settings",
            SimpleNamespace(
                logfire_token=SecretStr("  test-token  "),
                logfire_send_to_logfire=False,
                environment="production",
            ),
        )

        telemetry.configure_logfire()

        configure.assert_called_once_with(
            send_to_logfire=False,
            token="test-token",
            service_name=telemetry.PROJECT_INFO["name"],
            service_version=telemetry.PROJECT_INFO["version"],
            environment="production",
        )
