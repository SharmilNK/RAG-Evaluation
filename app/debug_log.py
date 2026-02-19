from __future__ import annotations

from typing import List


_DEBUG_LOG: List[str] = []


def add_debug(message: str) -> None:
    _DEBUG_LOG.append(message)


def get_debug() -> List[str]:
    return list(_DEBUG_LOG)


def clear_debug() -> None:
    _DEBUG_LOG.clear()
