from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Span:
    name: str
    attributes: Dict[str, str]


class Tracer:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    @contextmanager
    def span(self, name: str, **attributes: str):
        if self.enabled:
            span = Span(name=name, attributes=dict(attributes))
            # Placeholder for Langfuse span start
            _ = span
        try:
            yield
        finally:
            if self.enabled:
                # Placeholder for Langfuse span end
                pass


def get_tracer() -> Tracer:
    # Toggle with LANGFUSE_ENABLED=1 when a real tracer is wired.
    return Tracer(enabled=False)
