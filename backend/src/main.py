"""Main application entry point."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import logfire
from fastapi import FastAPI, status
from fastapi.responses import FileResponse, JSONResponse, Response
from guard.middleware import SecurityMiddleware
from guard.models import SecurityConfig
from loguru import logger
from starlette.middleware.sessions import SessionMiddleware

from src.admin import admin
from src.api import router
from src.core.config import PROJECT_INFO, settings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MONOREPO_ROOT = Path(__file__).resolve().parents[2]
SECURITY_LOG_FILE = str(PROJECT_ROOT / "security.log")
FRONTEND_DIST_DIR = MONOREPO_ROOT / "frontend" / "dist"
FRONTEND_INDEX_FILE = FRONTEND_DIST_DIR / "index.html"
RESERVED_ROUTE_PREFIXES = {
    "admin",
    "v1",
    "legacy",
    "docs",
    "redoc",
    "openapi.json",
}


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    """Lifespan context manager for FastAPI app."""
    logger.info("Starting FastAPI app...")
    yield
    logger.info("Shutting down FastAPI app...")


config = SecurityConfig(
    blocked_user_agents=["curl", "wget"],
    custom_log_file=SECURITY_LOG_FILE,
    rate_limit=100,
    rate_limit_window=60,
    enable_redis=True,
    redis_url=settings.redis_url.unicode_string(),
    redis_prefix="myapp:",
    custom_error_responses={
        429: "Rate limit exceeded. Please try again later.",
    },
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


def _frontend_not_built_response() -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": (
                "Frontend build not found. Run `cd frontend && npm run build` "
                "before serving the single-app UI from backend."
            ),
        },
    )


def _resolve_frontend_file(relative_path: str) -> Path | None:
    dist_root = FRONTEND_DIST_DIR.resolve()
    candidate = (dist_root / relative_path).resolve()

    if dist_root != candidate and dist_root not in candidate.parents:
        return None

    if candidate.is_file():
        return candidate

    return None


if not FRONTEND_INDEX_FILE.is_file():
    logger.warning(
        "Frontend build not found at '{}'. Root SPA routes will return 503 until built.",
        FRONTEND_INDEX_FILE,
    )


@app.get("/", include_in_schema=False)
async def spa_index() -> Response:
    """Serve the SPA entrypoint."""
    if FRONTEND_INDEX_FILE.is_file():
        return FileResponse(FRONTEND_INDEX_FILE)
    return _frontend_not_built_response()


@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str) -> Response:
    """Serve frontend assets and fallback to SPA for client-side routes."""
    first_segment = full_path.split("/", 1)[0]
    if first_segment in RESERVED_ROUTE_PREFIXES:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": "Not Found"},
        )

    if not FRONTEND_INDEX_FILE.is_file():
        return _frontend_not_built_response()

    requested_file = _resolve_frontend_file(full_path)
    if requested_file is not None:
        return FileResponse(requested_file)

    if Path(full_path).suffix:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": "Not Found"},
        )

    return FileResponse(FRONTEND_INDEX_FILE)
