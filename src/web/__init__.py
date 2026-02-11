"""Server-rendered end-user pages."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, Request, status
from pydantic import SecretStr, ValidationError
from starlette.responses import RedirectResponse, Response, StreamingResponse
from starlette.templating import Jinja2Templates

from src.core.database.base import async_session, get_async_session
from src.core.database.redis import get_redis_pool
from src.core.security import hash_password, verify_password
from src.schemas.user import UserCreate, UserLogin, UserUpdate
from src.services.auth_service import AuthService, InvalidCredentialsError
from src.services.llm_service import LLMService, LLMServiceError
from src.services.user_service import UserAlreadyExistsError, UserService


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.models.user import User


router = APIRouter()

templates = Jinja2Templates(directory="templates")


def _redirect(url: str) -> RedirectResponse:
    return RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)


async def _get_redis() -> Redis:
    return await get_redis_pool()


async def _get_session_user(
    request: Request,
    db: AsyncSession,
) -> User | None:
    token = request.session.get("access_token")
    if not token:
        return None
    auth_service = AuthService(db)
    valid, user = await auth_service.validate_token(token)
    if not valid:
        return None
    return user


@router.get("/", name="web_home")
async def home(request: Request) -> RedirectResponse:
    if request.session.get("access_token"):
        return _redirect(request.url_for("web_chat"))
    return _redirect(request.url_for("web_login"))


@router.get("/login", name="web_login")
async def login_page(request: Request) -> Response:
    return templates.TemplateResponse(
        request,
        "user_login.html",
        {"request": request, "title": "Login"},
    )


@router.post("/login", name="web_login_submit")
async def login_submit(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_async_session)],
) -> Response:
    form = await request.form()
    email = (form.get("email") or "").strip()
    password = form.get("password") or ""

    try:
        user_data = UserLogin(email=email, password=SecretStr(password))
        auth_service = AuthService(db)
        token = await auth_service.login(user_data)
        request.session["access_token"] = token.access_token
        request.session["user_email"] = email
        return _redirect(request.url_for("web_chat"))
    except (ValidationError, InvalidCredentialsError) as e:
        error = "Invalid email or password"
        if isinstance(e, ValidationError):
            error = "Please check your email and password."
        return templates.TemplateResponse(
            request,
            "user_login.html",
            {
                "request": request,
                "title": "Login",
                "error": error,
                "email": email,
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )


@router.get("/register", name="web_register")
async def register_page(request: Request) -> Response:
    return templates.TemplateResponse(
        request,
        "user_register.html",
        {"request": request, "title": "Register"},
    )


@router.post("/register", name="web_register_submit")
async def register_submit(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_async_session)],
) -> Response:
    form = await request.form()
    name = (form.get("name") or "").strip()
    email = (form.get("email") or "").strip()
    password = form.get("password") or ""

    try:
        user_in = UserCreate(name=name, email=email, password=SecretStr(password))
        auth_service = AuthService(db)
        user = await auth_service.register_user(user_in)
        token = auth_service.create_token(user)
        request.session["access_token"] = token.access_token
        request.session["user_email"] = user.email
        return _redirect(request.url_for("web_chat"))
    except UserAlreadyExistsError:
        error = "An account with that email already exists."
    except ValidationError as e:
        error = e.errors()[0].get("msg", "Invalid input")
    else:
        error = "Unable to create account."

    return templates.TemplateResponse(
        request,
        "user_register.html",
        {
            "request": request,
            "title": "Register",
            "error": error,
            "name": name,
            "email": email,
        },
        status_code=status.HTTP_400_BAD_REQUEST,
    )


@router.post("/logout", name="web_logout")
async def logout(request: Request) -> RedirectResponse:
    request.session.clear()
    return _redirect(request.url_for("web_login"))


@router.get("/chat", name="web_chat")
async def chat_page(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_async_session)],
) -> Response:
    user = await _get_session_user(request, db)
    if user is None:
        return _redirect(request.url_for("web_login"))

    return templates.TemplateResponse(
        request,
        "user_chat.html",
        {
            "request": request,
            "title": "Chat",
            "current_user": user,
        },
    )


@router.post("/chat", name="web_chat_submit")
async def chat_submit(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_async_session)],
    redis: Annotated[Redis, Depends(_get_redis)],
) -> Response:
    user = await _get_session_user(request, db)
    if user is None:
        return _redirect(request.url_for("web_login"))

    form = await request.form()
    prompt = (form.get("prompt") or "").strip()
    if not prompt:
        return templates.TemplateResponse(
            request,
            "user_chat.html",
            {
                "request": request,
                "title": "Chat",
                "current_user": user,
                "error": "Please enter a message.",
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    try:
        llm_service = LLMService(session=db, redis=redis)
        response = await llm_service.run_agent(user=user, user_prompt=prompt)
    except LLMServiceError as e:
        return templates.TemplateResponse(
            request,
            "user_chat.html",
            {
                "request": request,
                "title": "Chat",
                "current_user": user,
                "error": str(e),
                "prompt": prompt,
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    return templates.TemplateResponse(
        request,
        "user_chat.html",
        {
            "request": request,
            "title": "Chat",
            "current_user": user,
            "prompt": prompt,
            "response": response,
        },
    )


@router.post("/chat/stream", name="web_chat_stream")
async def chat_stream(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_async_session)],
) -> Response:
    user = await _get_session_user(request, db)
    if user is None:
        return _redirect(request.url_for("web_login"))

    form = await request.form()
    prompt = (form.get("prompt") or "").strip()
    if not prompt:
        return Response(
            "Please enter a message.",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    async def event_stream() -> AsyncIterator[str]:
        redis = await _get_redis()
        try:
            async with async_session() as session:
                llm_service = LLMService(session=session, redis=redis)
                async for token in llm_service.run_agent_stream(
                    user=user,
                    user_prompt=prompt,
                ):
                    payload = json.dumps({"type": "delta", "text": token})
                    yield f"data: {payload}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except LLMServiceError as e:
            payload = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {payload}\n\n"
        finally:
            await redis.aclose()

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=headers,
    )


@router.get("/profile", name="web_profile")
async def profile_page(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_async_session)],
) -> Response:
    user = await _get_session_user(request, db)
    if user is None:
        return _redirect(request.url_for("web_login"))

    return templates.TemplateResponse(
        request,
        "user_profile.html",
        {
            "request": request,
            "title": "Profile",
            "current_user": user,
        },
    )


@router.post("/profile", name="web_profile_submit")
async def profile_submit(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_async_session)],
) -> Response:
    user = await _get_session_user(request, db)
    if user is None:
        return _redirect(request.url_for("web_login"))

    form = await request.form()
    name = (form.get("name") or "").strip()
    email = (form.get("email") or "").strip()
    current_password = form.get("current_password") or ""
    new_password = form.get("new_password") or ""
    confirm_password = form.get("confirm_password") or ""

    errors: list[str] = []
    update_data = _build_profile_update_data(name, email, user)
    user, updated_profile = await _apply_profile_updates(db, user, update_data, errors)
    updated_password = await _apply_password_change(
        db,
        user,
        current_password,
        new_password,
        confirm_password,
        errors,
    )
    updated = updated_profile or updated_password

    if not errors and updated:
        auth_service = AuthService(db)
        token = auth_service.create_token(user)
        request.session["access_token"] = token.access_token
        request.session["user_email"] = user.email

    return templates.TemplateResponse(
        request,
        "user_profile.html",
        {
            "request": request,
            "title": "Profile",
            "current_user": user,
            "errors": errors,
            "success": updated and not errors,
        },
        status_code=status.HTTP_400_BAD_REQUEST if errors else status.HTTP_200_OK,
    )


def _build_profile_update_data(
    name: str,
    email: str,
    user: User,
) -> dict[str, Any]:
    update_data: dict[str, Any] = {}
    if name and name != user.name:
        update_data["name"] = name
    if email and email != user.email:
        update_data["email"] = email
    return update_data


async def _apply_profile_updates(
    db: AsyncSession,
    user: User,
    update_data: dict[str, Any],
    errors: list[str],
) -> tuple[User, bool]:
    if not update_data:
        return user, False
    try:
        user_service = UserService(db)
        user = await user_service.update_user(user.id, UserUpdate(**update_data))
    except UserAlreadyExistsError:
        errors.append("That email is already in use.")
    except ValidationError as e:
        errors.append(e.errors()[0].get("msg", "Invalid input"))
    else:
        return user, True
    return user, False


async def _apply_password_change(
    db: AsyncSession,
    user: User,
    current_password: str,
    new_password: str,
    confirm_password: str,
    errors: list[str],
) -> bool:
    if not _password_change_requested(
        current_password,
        new_password,
        confirm_password,
    ):
        return False
    if not current_password:
        errors.append("Current password is required to change your password.")
        return False
    if new_password != confirm_password:
        errors.append("New passwords do not match.")
        return False
    try:
        UserUpdate(password=SecretStr(new_password))
    except ValidationError as e:
        errors.append(e.errors()[0].get("msg", "Invalid password"))
        return False
    if not verify_password(current_password, user.hashed_password):
        errors.append("Current password is incorrect.")
        return False
    user.hashed_password = hash_password(new_password)
    await db.flush()
    return True


def _password_change_requested(
    current_password: str,
    new_password: str,
    confirm_password: str,
) -> bool:
    return any([current_password, new_password, confirm_password])


__all__ = ["router"]
