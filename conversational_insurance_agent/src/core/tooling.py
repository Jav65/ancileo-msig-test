from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Protocol


class ToolHandler(Protocol):
    async def __call__(self, **kwargs: Any) -> Any: ...


class SyncToolHandler(Protocol):
    def __call__(self, **kwargs: Any) -> Any: ...


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: Dict[str, Any]
    handler: Callable[..., Any]
    is_async: bool = False

    async def arun(self, **kwargs: Any) -> Any:
        if self.is_async:
            return await self.handler(**kwargs)
        return self.handler(**kwargs)
