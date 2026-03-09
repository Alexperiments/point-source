"""Main application entry point."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import logfire
from fastapi import FastAPI
from guard.middleware import SecurityMiddleware
from guard.models import SecurityConfig
from loguru import logger
from starlette.middleware.sessions import SessionMiddleware

from src.admin import admin
from src.api import router
from src.core.config import PROJECT_INFO, settings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SECURITY_LOG_FILE = str(PROJECT_ROOT / "security.log")
NGINX_PROXY_IP = "172.31.0.2"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    """Lifespan context manager for FastAPI app."""
    logger.info("Starting FastAPI app...")
    yield
    logger.info("Shutting down FastAPI app...")


config = SecurityConfig(
    blocked_user_agents=["curl", "wget"],
    custom_log_file=SECURITY_LOG_FILE,
    enable_redis=True,
    redis_url=settings.redis_url.unicode_string(),
    redis_prefix="myapp:",
    trusted_proxies=[NGINX_PROXY_IP],
    trusted_proxy_depth=1,
    trust_x_forwarded_proto=True,
    enable_rate_limiting=False,
    security_headers={"enabled": False},
    enable_cors=True,
    cors_allow_origins=settings.allowed_origins_list,
    cors_allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    cors_allow_headers=["*"],
    cors_allow_credentials=settings.environment != "development",
    passive_mode=True,
    log_suspicious_level="WARNING",
)

app = FastAPI(
    title=PROJECT_INFO["name"],
    version=PROJECT_INFO["version"],
    description=PROJECT_INFO["description"],
    lifespan=lifespan,
)

SecurityMiddleware.configure_cors(app, config)

logfire.configure(
    service_name=PROJECT_INFO["name"],
    service_version=PROJECT_INFO["version"],
)
logfire.instrument_pydantic_ai()
logfire.instrument_fastapi(app)

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.jwt_secret_key.get_secret_value(),
)
app.include_router(router)
admin.mount_to(app)
app.add_middleware(SecurityMiddleware, config=config)
