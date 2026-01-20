"""CLI for creating a superuser account (Enhanced UI)."""

import asyncio
from typing import Annotated

import typer
from pydantic import SecretStr, ValidationError
from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.theme import Theme
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.core.security import hash_password
from src.database.database import async_session_factory
from src.db_models.user import User
from src.schemas.user import UserCreate
from src.services.doc_ingestion_service import DocumentIngestionService


custom_theme = Theme(
    {
        "info": "dim cyan",
        "warning": "magenta",
        "danger": "bold red",
        "success": "bold green",
    },
)
console = Console(theme=custom_theme)

app = typer.Typer(
    help="Administrative CLI",
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@app.callback()
def main_callback() -> None:
    """Administrative CLI Entry Point."""


@app.command()
def ingest_documents(
    batch_size: Annotated[
        int,
        typer.Option(help="Batch size for document ingestion"),
    ] = 1000,
    category: Annotated[
        str | None,
        typer.Option(help="Arxiv category to filter the request."),
    ] = None,
) -> None:
    """Ingest documents from ar5iv dataset into the database."""
    console.print(
        Panel(
            Align.center(
                "[bold white]Document Ingestion Wizard[/bold white]",
            ),
            box=box.ROUNDED,
            style="bold blue",
            subtitle="Import ar5iv Papers",
        ),
    )

    with console.status(
        "[bold green]Ingesting ar5iv papers...[/bold green]",
        spinner="dots",
    ):
        try:
            count = asyncio.run(_ingest_documents(batch_size, category=category))
            console.print(
                Panel(
                    f"Successfully ingested [bold]{count}[/] documents!",
                    title="Success",
                    style="success",
                ),
            )
        except Exception as e:
            console.print(f"[bold red]Critical Error:[/bold red] {e}")
            raise typer.Exit(1) from e


async def _ingest_documents(batch_size: int, category: str | None = None) -> int:
    """Ingest documents into the database."""
    async with async_session_factory() as session:
        service = DocumentIngestionService(session)
        return await service.ingest_ar5iv_papers(
            batch_size=batch_size,
            category=category,
        )


@app.command()
def createsuperuser(
    email: Annotated[
        str | None,
        typer.Option(help="Email address for the superuser"),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option(help="Full name for the superuser"),
    ] = None,
    password: Annotated[
        str | None,
        typer.Option(help="Password for the superuser"),
    ] = None,
) -> None:
    """Create a superuser account with an interactive, beautiful UI."""
    console.print(
        Panel(
            Align.center(
                "[bold white]Admin User Creation Wizard[/bold white] 🧙‍♂️",
            ),
            box=box.ROUNDED,
            style="bold blue",
            subtitle="Secure System Access",
        ),
    )
    if not email:
        email = Prompt.ask("[bold cyan]📧 Email address[/]")

    if not name:
        name = Prompt.ask("[bold cyan]👤 Full Name[/]")

    if not password:
        while True:
            p1 = Prompt.ask("[bold cyan]🔑 Password[/]", password=True)
            p2 = Prompt.ask("[bold cyan]🔐 Confirm Password[/]", password=True)
            if p1 == p2:
                password = p1
                break
            console.print(
                "[danger]✖ Passwords do not match. Please try again.[/danger]",
            )

    try:
        asyncio.run(_create_superuser(email, name, password))
    except Exception as e:
        console.print(f"[bold red]Critical Error:[/bold red] {e}")
        raise typer.Exit(1) from e


async def _create_superuser(email: str, name: str, password: str) -> None:
    """Create superuser in the database with visual feedback."""
    try:
        user_data = UserCreate(
            email=email,
            name=name,
            password=SecretStr(password),
        )
    except ValidationError as e:
        console.print("\n[danger]Validation Failed:[/danger]")
        for error in e.errors():
            location: list[str] = error.get("loc", [])
            field = " -> ".join(str(segment) for segment in location)
            msg = error.get("msg", "Unknown error")
            console.print(f"  ❌ [bold]{field}[/]: {msg}")
        raise typer.Exit(1) from e

    if len(password.encode("utf-8")) > 72:
        console.print(Panel("Password cannot exceed 72 bytes.", style="danger"))
        raise typer.Exit(1)

    with console.status(
        "[bold green]Accessing database...[/]",
        spinner="dots",
    ) as status:
        async with async_session_factory() as session:
            try:
                status.update("[bold green]Checking existing users...[/]")
                result = await session.execute(
                    select(User).where(User.email == user_data.email),
                )
                existing = result.scalar_one_or_none()

                if existing:
                    if existing.is_superuser:
                        console.print(
                            Panel(
                                f"User [bold]{user_data.email}[/bold] is already a superuser.",
                                title="⚠ Warning",
                                style="warning",
                            ),
                        )
                        return

                    status.update("[bold blue]Promoting user to superuser...[/]")
                    existing.is_superuser = True
                    await session.commit()

                    console.print(
                        Panel(
                            f"User [bold]{user_data.email}[/bold] promoted to Superuser! 🚀",
                            title="Success",
                            style="success",
                        ),
                    )
                else:
                    status.update("[bold blue]Hashing password and saving...[/]")
                    user = User(
                        email=user_data.email,
                        name=user_data.name,
                        hashed_password=hash_password(
                            user_data.password.get_secret_value(),
                        ),
                    )
                    user.is_superuser = True
                    session.add(user)
                    await session.commit()

                    console.print(
                        Panel(
                            f"Superuser [bold]{user_data.email}[/bold] created successfully! 🎉",
                            title="Success",
                            style="success",
                        ),
                    )

            except IntegrityError as e:
                await session.rollback()
                console.print(
                    Panel(f"Database Integrity Error:\n{e!s}", style="danger"),
                )
                raise typer.Exit(1) from e
            except Exception as e:
                await session.rollback()
                console.print(Panel(f"Unexpected Error:\n{e!s}", style="danger"))
                raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
