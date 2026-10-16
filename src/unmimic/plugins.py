"""Plugin interfaces and registry.

Every guard in Unmimic is a plugin that implements one of the Protocols below. The proxy
only talks to these interfaces, so new detectors and defenses can be added without
touching it. Treat the signatures as frozen: changing one means changing every plugin.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

Action = Literal["allow", "block", "flag"]


@dataclass(frozen=True)
class Verdict:
    """An input guard's decision about one request."""

    action: Action
    score: float
    reason: str = ""


@dataclass
class Context:
    """State shared by the guards during one request's pass through the proxy."""

    account_id: str
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class InputGuard(Protocol):
    """Inspects a request before it reaches the model."""

    name: str

    def check(self, request: dict[str, Any], ctx: Context) -> Verdict: ...


@runtime_checkable
class OutputGuard(Protocol):
    """Inspects and may rewrite a response, given the model's hidden reasoning."""

    name: str

    def transform(
        self, response: dict[str, Any], reasoning: str, ctx: Context
    ) -> dict[str, Any]: ...


@runtime_checkable
class AccountScorer(Protocol):
    """Scores how suspicious an account's behaviour is. Higher means more suspicious."""

    name: str

    def score(self, account_id: str, event: dict[str, Any]) -> float: ...


@runtime_checkable
class ReasoningBinder(Protocol):
    """Binds reasoning blocks to the account and conversation that produced them."""

    name: str

    def sign(self, block: str, account_id: str, conversation_id: str) -> str: ...

    def verify(self, block: str, signature: str, account_id: str, conversation_id: str) -> bool: ...


PluginKind = Literal["input", "output", "account", "binder"]

PROTOCOLS: dict[str, type] = {
    "input": InputGuard,
    "output": OutputGuard,
    "account": AccountScorer,
    "binder": ReasoningBinder,
}

T = TypeVar("T", bound=Callable[..., Any])


class PluginRegistry:
    """Maps (kind, name) to a plugin factory, usually the plugin class."""

    def __init__(self) -> None:
        self._factories: dict[str, dict[str, Callable[..., Any]]] = {k: {} for k in PROTOCOLS}

    def register(self, kind: PluginKind, name: str) -> Callable[[T], T]:
        """Class decorator: ``@registry.register("input", "stage1")``."""
        factories = self._kind(kind)

        def decorator(factory: T) -> T:
            if name in factories:
                raise ValueError(f"{kind} plugin {name!r} is already registered")
            factories[name] = factory
            return factory

        return decorator

    def create(self, kind: PluginKind, name: str, **kwargs: Any) -> Any:
        """Instantiate a registered plugin and check that it implements its Protocol."""
        factories = self._kind(kind)
        if name not in factories:
            known = ", ".join(sorted(factories)) or "none"
            raise KeyError(f"no {kind} plugin named {name!r} (registered: {known})")
        plugin = factories[name](**kwargs)
        if not isinstance(plugin, PROTOCOLS[kind]):
            raise TypeError(f"{kind} plugin {name!r} does not implement {PROTOCOLS[kind].__name__}")
        return plugin

    def names(self, kind: PluginKind) -> list[str]:
        return sorted(self._kind(kind))

    def _kind(self, kind: str) -> dict[str, Callable[..., Any]]:
        if kind not in self._factories:
            raise ValueError(f"unknown plugin kind {kind!r}; expected one of {list(PROTOCOLS)}")
        return self._factories[kind]


registry = PluginRegistry()
