"""Contains wrapper of logfire decorators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, LiteralString, ParamSpec, TypeVar

import logfire


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence


P = ParamSpec("P")
R = TypeVar("R")


def auto_instrument(
    *,
    span_name: LiteralString | None = None,
    extract_args: bool | Iterable[str] = False,
    tags: Sequence[str] | None = None,
    msg_template: LiteralString | None = None,
    record_return: bool = False,
    allow_generator: bool | None = None,
    new_trace: bool = False,
    **extra_kwargs: dict,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Thin wrapper around 'logfire.instrument' with opinionated defaults.

    - 'extract_args' defaults to False.
    - 'span_name' defaults to the function's '__qualname__'
      (e.g. "DocumentService._ingest_batch").
    - All async/sync/generator behavior is delegated to logfire.

    Extra keyword arguments are forwarded to 'logfire.instrument' so
    this stays compatible with future logfire options.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        effective_span_name: str | None = span_name or func.__qualname__

        # Build kwargs in the same "shape" as logfire.instrument's public API.
        instrument_kwargs: dict[str, Any] = {
            "span_name": effective_span_name,
            "extract_args": extract_args,
            "record_return": record_return,
            "new_trace": new_trace,
            **extra_kwargs,
        }

        if allow_generator is not None:
            instrument_kwargs["allow_generator"] = allow_generator
        if msg_template is not None:
            instrument_kwargs["msg_template"] = msg_template
        if tags is not None:
            instrument_kwargs["tags"] = tags

        decorated = logfire.instrument(**instrument_kwargs)
        return decorated(func)

    return decorator
