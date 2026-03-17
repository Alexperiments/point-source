"""Tests for outbound email transport payloads."""

import json

from src.services.email_service import EmailMessage, PostmarkEmailService


class _DummyResponse:
    """Minimal urllib response stub."""

    status = 200

    def __enter__(self) -> "_DummyResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return b"{}"


def test_postmark_uses_html_body_without_text_body_when_html_exists(
    monkeypatch,
) -> None:
    """Avoid multipart duplication in clients that flatten alternative bodies."""
    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout):  # noqa: ANN001
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _DummyResponse()

    monkeypatch.setattr("src.services.email_service.request.urlopen", fake_urlopen)

    service = PostmarkEmailService(
        server_token="test-token",
        from_address="auth@notify.point-source.org",
        from_name="Point-source",
        message_stream="outbound",
    )

    service._send_sync(  # noqa: SLF001
        EmailMessage(
            to_email="user@example.com",
            subject="Verify your Point-source email",
            text_body="Plain text fallback",
            html_body="<p>HTML email</p>",
            tag="verify-email",
        )
    )

    payload = captured["payload"]
    assert payload["HtmlBody"] == "<p>HTML email</p>"
    assert "TextBody" not in payload


def test_postmark_uses_text_body_when_no_html_body(monkeypatch) -> None:
    """Plain text delivery should still work for non-HTML messages."""
    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout):  # noqa: ANN001
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _DummyResponse()

    monkeypatch.setattr("src.services.email_service.request.urlopen", fake_urlopen)

    service = PostmarkEmailService(
        server_token="test-token",
        from_address="auth@notify.point-source.org",
        from_name="Point-source",
        message_stream="outbound",
    )

    service._send_sync(  # noqa: SLF001
        EmailMessage(
            to_email="user@example.com",
            subject="Security notice",
            text_body="Text only email",
            tag="security",
        )
    )

    payload = captured["payload"]
    assert payload["TextBody"] == "Text only email"
    assert "HtmlBody" not in payload
