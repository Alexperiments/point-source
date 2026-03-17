"""Outbound email delivery services."""

import asyncio
import json
from dataclasses import dataclass
from urllib import error, request

from loguru import logger

from src.core.config import settings


POSTMARK_EMAIL_API_URL = "https://api.postmarkapp.com/email"


class EmailDeliveryError(Exception):
    """Raised when an outbound email cannot be delivered to the provider."""


@dataclass(slots=True)
class EmailMessage:
    """Transport-agnostic email payload."""

    to_email: str
    subject: str
    text_body: str
    html_body: str | None = None
    tag: str | None = None


class BaseEmailService:
    """Common async email service interface."""

    async def send(self, message: EmailMessage) -> None:
        """Deliver the provided message."""
        raise NotImplementedError


class ConsoleEmailService(BaseEmailService):
    """Development-friendly email sink that logs messages locally."""

    async def send(self, message: EmailMessage) -> None:
        """Log the outbound message instead of sending it to a provider."""
        logger.info(
            "Console email delivery\nTo: {to_email}\nSubject: {subject}\n\n{text}",
            to_email=message.to_email,
            subject=message.subject,
            text=message.text_body,
        )


class PostmarkEmailService(BaseEmailService):
    """Transactional email delivery via the Postmark HTTP API."""

    def __init__(
        self,
        *,
        server_token: str,
        from_address: str,
        from_name: str,
        message_stream: str,
    ) -> None:
        """Configure a Postmark-backed transactional email sender."""
        self.server_token = server_token
        self.from_address = from_address
        self.from_name = from_name
        self.message_stream = message_stream

    def _send_sync(self, message: EmailMessage) -> None:
        payload: dict[str, object] = {
            "From": f"{self.from_name} <{self.from_address}>",
            "To": message.to_email,
            "Subject": message.subject,
            "MessageStream": self.message_stream,
            "TrackLinks": "None",
            "TrackOpens": False,
        }
        if message.html_body:
            payload["HtmlBody"] = message.html_body
        else:
            payload["TextBody"] = message.text_body
        if message.tag:
            payload["Tag"] = message.tag

        req = request.Request(  # noqa: S310
            POSTMARK_EMAIL_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-Postmark-Server-Token": self.server_token,
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=15) as response:  # noqa: S310
                if response.status >= 400:
                    body = response.read().decode("utf-8", errors="replace")
                    msg = f"Postmark rejected the email request: {body}"
                    raise EmailDeliveryError(msg)
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            msg = f"Postmark delivery failed with status {exc.code}: {body}"
            raise EmailDeliveryError(msg) from exc
        except error.URLError as exc:
            msg = f"Postmark delivery failed: {exc.reason}"
            raise EmailDeliveryError(msg) from exc

    async def send(self, message: EmailMessage) -> None:
        """Send the message via Postmark in a worker thread."""
        await asyncio.to_thread(self._send_sync, message)


def create_email_service() -> BaseEmailService:
    """Instantiate the configured email delivery service."""
    if settings.email_delivery_mode == "postmark":
        return PostmarkEmailService(
            server_token=settings.postmark_server_token.get_secret_value(),
            from_address=str(settings.email_from_address),
            from_name=settings.email_from_name.strip() or "Point-source",
            message_stream=settings.postmark_message_stream.strip() or "outbound",
        )
    return ConsoleEmailService()
