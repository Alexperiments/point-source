"""Admin panel configuration."""

from pathlib import Path
from urllib.parse import urlsplit

import jinja2
from fastapi import status
from sqlalchemy import select
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
from starlette.status import (
    HTTP_303_SEE_OTHER,
    HTTP_400_BAD_REQUEST,
)
from starlette.templating import Jinja2Templates
from starlette_admin import BaseAdmin
from starlette_admin.auth import AdminConfig, AdminUser, AuthProvider
from starlette_admin.contrib.sqla import Admin
from starlette_admin.exceptions import FormValidationError, LoginFailed

from src.admin.views import PromptManagerView, UsersView
from src.core.database.base import async_session_factory, engine
from src.core.security import verify_password
from src.models import User


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = PROJECT_ROOT / "templates"
templates = Jinja2Templates(
    directory=[str(TEMPLATES_DIR)],
    loader=jinja2.ChoiceLoader(
        [
            jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
            jinja2.PackageLoader(
                "starlette_admin",
                "templates",
            ),
        ],
    ),
)


class AdminAuthProvider(AuthProvider):
    """Authentication provider for the admin panel."""

    async def login(
        self,
        username: str,
        password: str,
        remember_me: bool,  # noqa: ARG002
        request: Request,
        response: Response,
    ) -> Response:
        """Authenticate user against the database."""
        async with async_session_factory() as session:
            result = await session.execute(
                select(User).where(
                    User.email == username,
                    User.is_deleted.is_(False),
                ),
            )
            user = result.scalar_one_or_none()

            if not user:
                msg = "Invalid email or password"
                raise LoginFailed(msg)

            if not verify_password(password, user.hashed_password):
                msg = "Invalid email or password"
                raise LoginFailed(msg)

            if not user.is_superuser:
                msg = "Access denied. Superuser privileges required."
                raise LoginFailed(msg)

            request.session.update(
                {
                    "user_id": str(user.id),
                    "username": user.name,
                    "email": user.email,
                },
            )
            return response

    async def is_authenticated(self, request: Request) -> bool:
        """Check if user is authenticated."""
        return "user_id" in request.session

    def get_admin_config(self, request: Request) -> AdminConfig | None:
        """Get admin configuration."""
        username = request.session.get("username", "Admin")
        return AdminConfig(app_title=f"Welcome, {username}!")

    def get_admin_user(self, request: Request) -> AdminUser | None:
        """Get current admin user."""
        return AdminUser(
            username=request.session.get("username", "Admin"),
            photo_url=None,
        )

    async def logout(self, request: Request, response: Response) -> Response:
        """Log out user."""
        request.session.clear()
        return response

    async def render_login(self, request: Request, admin: BaseAdmin) -> Response:
        """Render the login page and handle form submission."""
        if request.method == "GET":
            return admin.templates.TemplateResponse(
                request=request,
                name="login.html",
                context={"_is_login_path": True},
            )

        form = await request.form()
        fallback_url = str(request.url_for(admin.route_name + ":index"))
        try:
            return await self.login(
                username=form.get("email"),
                password=form.get("password"),
                remember_me=form.get("remember_me") == "on",
                request=request,
                response=RedirectResponse(
                    _safe_redirect_target(
                        request.query_params.get("next"),
                        fallback_url,
                    ),
                    status_code=HTTP_303_SEE_OTHER,
                ),
            )
        except FormValidationError as errors:
            return admin.templates.TemplateResponse(
                request=request,
                name="login.html",
                context={"form_errors": errors, "_is_login_path": True},
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )
        except LoginFailed as error:
            return admin.templates.TemplateResponse(
                request=request,
                name="login.html",
                context={"error": error.msg, "_is_login_path": True},
                status_code=HTTP_400_BAD_REQUEST,
            )


def _safe_redirect_target(next_url: str | None, fallback_url: str) -> str:
    """Only allow redirects to local absolute paths within the current origin."""
    if not next_url:
        return fallback_url

    parsed = urlsplit(next_url)
    if parsed.scheme or parsed.netloc:
        return fallback_url
    if not next_url.startswith("/") or next_url.startswith("//"):
        return fallback_url
    return next_url


admin = Admin(
    engine=engine,
    title="Admin Panel",
    base_url="/",
    auth_provider=AdminAuthProvider(),
)

admin.add_view(UsersView(model=User, icon="fa fa-users"))
admin.add_view(PromptManagerView(templates=templates))
